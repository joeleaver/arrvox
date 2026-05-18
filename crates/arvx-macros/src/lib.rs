//! Arrvox component proc macro.
//!
//! `#[arvx_component]` on a struct auto-generates:
//! - `ComponentMeta` impl with static `FIELDS` array
//! - `inventory::submit!(ComponentEntry { ... })` for auto-registration
//! - Type-erased `get_field`/`set_field` via field name matching
//! - Serde-based `serialize`/`deserialize_insert`
//!
//! # Attributes
//!
//! - `#[mandatory]` — component cannot be removed (Transform, EditorMetadata)
//! - `#[transient]` on a field — excluded from serialization and get/set
//! - `#[range(min, max)]` on a field — adds range metadata for the inspector
//! - `#[asset_filter("ext")]` on a field — marks as asset reference with extension

use proc_macro::TokenStream;
use quote::{format_ident, quote};
use syn::{
    parse::{Parse, ParseStream},
    parse_macro_input, Data, DeriveInput, Fields, Ident, ItemFn, LitStr, Meta, Token,
};

/// Classify a Rust type to a FieldType variant name.
fn classify_type(ty: &syn::Type) -> &'static str {
    let s = quote!(#ty).to_string().replace(' ', "");
    match s.as_str() {
        "f32" | "f64" => "Float",
        "i8" | "i16" | "i32" | "i64" | "u8" | "u16" | "u32" | "u64" | "usize" | "isize" => "Int",
        "bool" => "Bool",
        "String" => "String",
        "Vec3" | "glam::Vec3" => "Vec3",
        "[f32;3]" => "Vec3",
        "[f32;4]" => "Color",
        _ if s.starts_with("Option<") => classify_type_from_str(&s[7..s.len()-1]),
        _ => "String", // fallback
    }
}

fn classify_type_from_str(s: &str) -> &'static str {
    match s {
        "f32" | "f64" => "Float",
        "String" => "String",
        "u16" | "u32" | "i32" | "i64" => "Int",
        _ => "String",
    }
}

/// Check if a field has a given attribute name.
fn has_attr(field: &syn::Field, name: &str) -> bool {
    field.attrs.iter().any(|a| a.path().is_ident(name))
}

/// Extract `#[range(min, max)]` values.
fn extract_range(field: &syn::Field) -> Option<(f64, f64)> {
    for attr in &field.attrs {
        if attr.path().is_ident("range") {
            if let Ok(Meta::List(list)) = attr.parse_args::<Meta>() {
                // Not a clean parse — fallback to string parsing
                let _ = list;
            }
            // Parse as literal tuple: range(0.0, 100.0)
            let tokens = attr.meta.require_list().ok()?.tokens.to_string();
            let parts: Vec<&str> = tokens.split(',').collect();
            if parts.len() == 2 {
                let min: f64 = parts[0].trim().parse().ok()?;
                let max: f64 = parts[1].trim().parse().ok()?;
                return Some((min, max));
            }
        }
    }
    None
}

/// Extract `#[asset_filter("ext")]` value.
fn extract_asset_filter(field: &syn::Field) -> Option<String> {
    for attr in &field.attrs {
        if attr.path().is_ident("asset_filter") {
            let tokens = attr.meta.require_list().ok()?.tokens.to_string();
            let s = tokens.trim().trim_matches('"');
            return Some(s.to_string());
        }
    }
    None
}

#[proc_macro_attribute]
pub fn arvx_component(attr: TokenStream, item: TokenStream) -> TokenStream {
    let mut input = parse_macro_input!(item as DeriveInput);
    let struct_name = &input.ident;
    let struct_name_str = struct_name.to_string();

    // Check for #[mandatory] on the attribute
    let mandatory = !attr.is_empty() && attr.to_string().contains("mandatory");

    let fields = match &input.data {
        Data::Struct(data) => match &data.fields {
            Fields::Named(named) => &named.named,
            _ => panic!("#[arvx_component] only supports structs with named fields"),
        },
        _ => panic!("#[arvx_component] only supports structs"),
    };

    // Collect field info
    struct FieldInfo {
        name: String,
        ident: syn::Ident,
        field_type_str: String,
        transient: bool,
        range: Option<(f64, f64)>,
        asset_filter: Option<String>,
        is_option: bool,
    }

    let field_infos: Vec<FieldInfo> = fields.iter().filter_map(|f| {
        let ident = f.ident.clone()?;
        let name = ident.to_string();
        let ty = f.ty.clone();
        let transient = has_attr(f, "transient") || has_attr(f, "serde");
        let field_type_str = classify_type(&ty).to_string();
        let range = extract_range(f);
        let asset_filter = extract_asset_filter(f);
        let is_option = quote!(#ty).to_string().replace(' ', "").starts_with("Option<");
        Some(FieldInfo { name, ident, field_type_str, transient, range, asset_filter, is_option })
    }).collect();

    // Strip custom attributes (range, transient, asset_filter) from the struct
    // so the compiler doesn't reject them.
    if let Data::Struct(ref mut data) = input.data {
        if let Fields::Named(ref mut named) = data.fields {
            for field in &mut named.named {
                field.attrs.retain(|a| {
                    !a.path().is_ident("range")
                        && !a.path().is_ident("transient")
                        && !a.path().is_ident("asset_filter")
                });
            }
        }
    }

    // Generate FieldMeta static array entries
    let field_count = field_infos.len();
    let fields_static = format_ident!("{}_FIELDS", struct_name.to_string().to_uppercase());

    let meta_entries: Vec<_> = field_infos.iter().map(|fi| {
        let name = &fi.name;
        let ft = format_ident!("{}", fi.field_type_str);
        let transient = fi.transient;
        let range = match fi.range {
            Some((min, max)) => quote!(Some((#min, #max))),
            None => quote!(None),
        };
        let asset_filter = match &fi.asset_filter {
            Some(s) => quote!(Some(#s)),
            None => quote!(None),
        };
        quote! {
            arvx_engine::component_registry::FieldMeta {
                name: #name,
                field_type: arvx_engine::inspector::FieldType::#ft,
                range: #range,
                transient: #transient,
                struct_fields: None,
                asset_filter: #asset_filter,
                enum_options: None,
                scrub: false,
            }
        }
    }).collect();

    // Generate get_field match arms (skip transient)
    let get_arms: Vec<_> = field_infos.iter().filter(|fi| !fi.transient).map(|fi| {
        let name = &fi.name;
        let ident = &fi.ident;
        let ft = fi.field_type_str.as_str();
        let conversion = match ft {
            "Float" => quote!(Ok(arvx_engine::inspector::FieldValue::Float(c.#ident as f64))),
            "Int" => quote!(Ok(arvx_engine::inspector::FieldValue::Int(c.#ident as i64))),
            "Bool" => quote!(Ok(arvx_engine::inspector::FieldValue::Bool(c.#ident))),
            "String" => {
                if fi.is_option {
                    quote!(Ok(arvx_engine::inspector::FieldValue::String(c.#ident.clone().unwrap_or_default())))
                } else {
                    quote!(Ok(arvx_engine::inspector::FieldValue::String(c.#ident.clone())))
                }
            }
            "Vec3" => quote!(Ok(arvx_engine::inspector::FieldValue::Vec3(c.#ident.to_array()))),
            "Color" => quote!(Ok(arvx_engine::inspector::FieldValue::Color(c.#ident))),
            _ => quote!(Err(format!("unsupported type for field '{}'", #name))),
        };
        quote!(#name => { #conversion })
    }).collect();

    // Generate set_field match arms (skip transient)
    let set_arms: Vec<_> = field_infos.iter().filter(|fi| !fi.transient).map(|fi| {
        let name = &fi.name;
        let ident = &fi.ident;
        let ft = fi.field_type_str.as_str();
        let conversion = match ft {
            "Float" => quote! {
                if let arvx_engine::inspector::FieldValue::Float(v) = value {
                    c.#ident = v as _; Ok(())
                } else { Err("type mismatch".into()) }
            },
            "Int" => quote! {
                if let arvx_engine::inspector::FieldValue::Int(v) = value {
                    c.#ident = v as _; Ok(())
                } else { Err("type mismatch".into()) }
            },
            "Bool" => quote! {
                if let arvx_engine::inspector::FieldValue::Bool(v) = value {
                    c.#ident = v; Ok(())
                } else { Err("type mismatch".into()) }
            },
            "String" => {
                if fi.is_option {
                    quote! {
                        if let arvx_engine::inspector::FieldValue::String(v) = value {
                            c.#ident = if v.is_empty() { None } else { Some(v) }; Ok(())
                        } else { Err("type mismatch".into()) }
                    }
                } else {
                    quote! {
                        if let arvx_engine::inspector::FieldValue::String(v) = value {
                            c.#ident = v; Ok(())
                        } else { Err("type mismatch".into()) }
                    }
                }
            }
            "Vec3" => quote! {
                if let arvx_engine::inspector::FieldValue::Vec3(v) = value {
                    c.#ident = glam::Vec3::from_array(v); Ok(())
                } else { Err("type mismatch".into()) }
            },
            "Color" => quote! {
                if let arvx_engine::inspector::FieldValue::Color(v) = value {
                    c.#ident = v; Ok(())
                } else { Err("type mismatch".into()) }
            },
            _ => quote!(Err(format!("unsupported type for field '{}'", #name))),
        };
        quote!(#name => { #conversion })
    }).collect();

    let mandatory_lit = mandatory;

    // Generate remove/add_default based on mandatory
    let remove_fn = if mandatory {
        quote!(|_, _| Err(format!("{} is mandatory", #struct_name_str)))
    } else {
        quote!(|world: &mut hecs::World, entity: hecs::Entity| {
            world.remove_one::<#struct_name>(entity).map(|_| ()).map_err(|e| format!("{e}"))
        })
    };

    let output = quote! {
        #input

        static #fields_static: [arvx_engine::component_registry::FieldMeta; #field_count] = [
            #(#meta_entries),*
        ];

        inventory::submit! {
            arvx_engine::component_registry::ComponentEntry {
                name: #struct_name_str,
                meta: &#fields_static,
                mandatory: #mandatory_lit,
                has: |world: &hecs::World, entity: hecs::Entity| {
                    world.get::<&#struct_name>(entity).is_ok()
                },
                get_field: |world: &hecs::World, entity: hecs::Entity, field: &str| {
                    let c = world.get::<&#struct_name>(entity)
                        .map_err(|_| format!("no {}", #struct_name_str))?;
                    match field {
                        #(#get_arms)*
                        _ => Err(format!("unknown field '{}' on {}", field, #struct_name_str)),
                    }
                },
                set_field: |world: &mut hecs::World, entity: hecs::Entity, field: &str, value: arvx_engine::inspector::FieldValue| {
                    let mut c = world.get::<&mut #struct_name>(entity)
                        .map_err(|_| format!("no {}", #struct_name_str))?;
                    match field {
                        #(#set_arms)*
                        _ => Err(format!("field '{}' is read-only or unknown on {}", field, #struct_name_str)),
                    }
                },
                add_default: |world: &mut hecs::World, entity: hecs::Entity| {
                    world.insert_one(entity, #struct_name::default()).map_err(|e| format!("{e}"))
                },
                remove: #remove_fn,
                serialize: |world: &hecs::World, entity: hecs::Entity| -> Option<String> {
                    let c = world.get::<&#struct_name>(entity).ok()?;
                    serde_json::to_string(&*c).ok()
                },
                deserialize_insert: |world: &mut hecs::World, entity: hecs::Entity, json: &str| -> Result<(), String> {
                    let c: #struct_name = serde_json::from_str(json).map_err(|e| format!("{e}"))?;
                    world.insert_one(entity, c).map_err(|e| format!("{e}"))
                },
                on_add: None,
                on_remove: None,
            }
        }
    };

    output.into()
}

/// Register a gameplay system for automatic discovery and execution.
///
/// # Usage
///
/// ```ignore
/// use arvx_engine::arvx_system;
/// use arvx_engine::behavior::SystemContext;
///
/// #[arvx_system(phase = Update)]
/// fn spin_system(ctx: &mut SystemContext) {
///     // ...
/// }
///
/// #[arvx_system(phase = LateUpdate, after = ["movement_system"])]
/// fn camera_follow(ctx: &mut SystemContext) {
///     // ...
/// }
/// ```
///
/// # Attributes
///
/// - `phase = Update|FixedUpdate|LateUpdate` — which phase to run in (required)
/// - `after = ["name", ...]` — systems that must run before this one
/// - `before = ["name", ...]` — systems that must run after this one
#[proc_macro_attribute]
pub fn arvx_system(attr: TokenStream, item: TokenStream) -> TokenStream {
    let input_fn = parse_macro_input!(item as ItemFn);
    let fn_name = &input_fn.sig.ident;
    let fn_name_str = fn_name.to_string();

    // Parse attributes: phase = Ident, after = [...], before = [...]
    let attr_str = attr.to_string();
    let phase = extract_phase(&attr_str)
        .unwrap_or_else(|| panic!("#[arvx_system] requires `phase = Update|FixedUpdate|LateUpdate`"));
    let after = extract_string_list(&attr_str, "after");
    let before = extract_string_list(&attr_str, "before");

    let phase_ident = format_ident!("{}", phase);

    let after_tokens: Vec<_> = after.iter().map(|s| quote! { #s }).collect();
    let before_tokens: Vec<_> = before.iter().map(|s| quote! { #s }).collect();

    let output = quote! {
        #input_fn

        inventory::submit! {
            arvx_engine::behavior::SystemEntry {
                name: #fn_name_str,
                module_path: module_path!(),
                phase: arvx_engine::behavior::Phase::#phase_ident,
                after: &[#(#after_tokens),*],
                before: &[#(#before_tokens),*],
                fn_ptr: #fn_name as fn(&mut arvx_engine::behavior::SystemContext) as *const (),
            }
        }
    };

    output.into()
}

/// Extract `phase = Ident` from attribute string.
fn extract_phase(attr: &str) -> Option<String> {
    // Look for "phase = Ident" pattern.
    for part in attr.split(',') {
        let part = part.trim();
        if let Some(rest) = part.strip_prefix("phase") {
            let rest = rest.trim().strip_prefix('=')?.trim();
            // Take first word (ident).
            let ident: String = rest.chars().take_while(|c| c.is_alphanumeric() || *c == '_').collect();
            if !ident.is_empty() {
                return Some(ident);
            }
        }
    }
    None
}

/// Extract a string list like `after = ["a", "b"]` from attribute string.
fn extract_string_list(attr: &str, key: &str) -> Vec<String> {
    // Find key = [...] pattern.
    let Some(start) = attr.find(key) else { return Vec::new() };
    let rest = &attr[start + key.len()..];
    let rest = rest.trim().strip_prefix('=').unwrap_or(rest).trim();
    let Some(bracket_start) = rest.find('[') else { return Vec::new() };
    let Some(bracket_end) = rest.find(']') else { return Vec::new() };
    let inner = &rest[bracket_start + 1..bracket_end];

    inner.split(',')
        .map(|s| s.trim().trim_matches('"').trim_matches('\'').to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

// ── #[arvx_generator] ────────────────────────────────────────────────────

/// Register a generator function for automatic discovery.
///
/// # Usage
///
/// ```ignore
/// use arvx_engine::{arvx_component, arvx_generator};
/// use arvx_engine::generator::{GeneratorContext, GeneratorError};
///
/// #[arvx_component]
/// #[derive(Debug, Clone, Serialize, Deserialize)]
/// pub struct RockParams {
///     #[range(0.1, 5.0)]
///     pub radius: f32,
/// }
///
/// impl Default for RockParams {
///     fn default() -> Self { Self { radius: 1.0 } }
/// }
///
/// #[arvx_generator(name = "rock", params = RockParams)]
/// fn generate_rock(
///     params: &RockParams,
///     ctx: &mut GeneratorContext,
/// ) -> Result<(), GeneratorError> {
///     // Build voxels, call ctx.emit_child(...), etc.
///     Ok(())
/// }
/// ```
///
/// # Attributes
///
/// - `name = "..."` — unique generator name (required)
/// - `params = XParams` — param component struct (required).
///   Must be a `#[arvx_component]` with `Default + Clone`.
struct GeneratorAttrs {
    name: LitStr,
    params: Ident,
}

impl Parse for GeneratorAttrs {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let mut name: Option<LitStr> = None;
        let mut params: Option<Ident> = None;

        while !input.is_empty() {
            let key: Ident = input.parse()?;
            input.parse::<Token![=]>()?;
            match key.to_string().as_str() {
                "name" => name = Some(input.parse()?),
                "params" => params = Some(input.parse()?),
                other => {
                    return Err(syn::Error::new(
                        key.span(),
                        format!("unknown arvx_generator attribute '{other}' — expected `name` or `params`"),
                    ));
                }
            }
            if input.peek(Token![,]) {
                input.parse::<Token![,]>()?;
            }
        }

        let name = name.ok_or_else(|| {
            syn::Error::new(
                proc_macro2::Span::call_site(),
                "#[arvx_generator] requires `name = \"...\"`",
            )
        })?;
        let params = params.ok_or_else(|| {
            syn::Error::new(
                proc_macro2::Span::call_site(),
                "#[arvx_generator] requires `params = XParams`",
            )
        })?;
        Ok(GeneratorAttrs { name, params })
    }
}

#[proc_macro_attribute]
pub fn arvx_generator(attr: TokenStream, item: TokenStream) -> TokenStream {
    let attrs = parse_macro_input!(attr as GeneratorAttrs);
    let func = parse_macro_input!(item as ItemFn);

    let fn_name = &func.sig.ident;
    let gen_name = &attrs.name;
    let param_type = &attrs.params;
    let param_type_str = param_type.to_string();

    let erased_fn_name = format_ident!("__rkp_gen_{}_erased", fn_name);
    let clone_params_fn_name = format_ident!("__rkp_gen_{}_clone_params", fn_name);
    let insert_default_fn_name = format_ident!("__rkp_gen_{}_insert_default", fn_name);

    let output = quote! {
        #func

        #[doc(hidden)]
        #[allow(non_snake_case)]
        fn #erased_fn_name<'w>(
            params: &dyn ::std::any::Any,
            ctx: &mut arvx_engine::generator::GeneratorContext<'w>,
        ) -> ::std::result::Result<
            (),
            arvx_engine::generator::GeneratorError,
        > {
            let params = params.downcast_ref::<#param_type>().unwrap_or_else(|| {
                panic!(
                    "arvx_generator '{}': param type mismatch — expected {}, got {:?}",
                    #gen_name,
                    #param_type_str,
                    params.type_id(),
                )
            });
            #fn_name(params, ctx)
        }

        #[doc(hidden)]
        #[allow(non_snake_case)]
        fn #clone_params_fn_name(
            world: &hecs::World,
            entity: hecs::Entity,
        ) -> ::std::option::Option<::std::boxed::Box<dyn ::std::any::Any + Send>> {
            world.get::<&#param_type>(entity)
                .ok()
                .map(|p| ::std::boxed::Box::new((*p).clone())
                    as ::std::boxed::Box<dyn ::std::any::Any + Send>)
        }

        #[doc(hidden)]
        #[allow(non_snake_case)]
        fn #insert_default_fn_name(
            world: &mut hecs::World,
            entity: hecs::Entity,
        ) {
            let _ = world.insert_one(entity, <#param_type as ::std::default::Default>::default());
        }

        inventory::submit! {
            arvx_engine::generator::GeneratorEntry {
                name: #gen_name,
                param_component_name: #param_type_str,
                param_type_id: ::std::any::TypeId::of::<#param_type>(),
                generate_fn: #erased_fn_name,
                clone_params: #clone_params_fn_name,
                insert_default_params: #insert_default_fn_name,
            }
        }
    };

    output.into()
}
