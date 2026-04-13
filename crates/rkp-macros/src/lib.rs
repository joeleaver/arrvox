//! RKIPatch component proc macro.
//!
//! `#[rkp_component]` on a struct auto-generates:
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
use syn::{parse_macro_input, DeriveInput, Data, Fields, Lit, Meta, Expr};

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
pub fn rkp_component(attr: TokenStream, item: TokenStream) -> TokenStream {
    let mut input = parse_macro_input!(item as DeriveInput);
    let struct_name = &input.ident;
    let struct_name_str = struct_name.to_string();

    // Check for #[mandatory] on the attribute
    let mandatory = !attr.is_empty() && attr.to_string().contains("mandatory");

    let fields = match &input.data {
        Data::Struct(data) => match &data.fields {
            Fields::Named(named) => &named.named,
            _ => panic!("#[rkp_component] only supports structs with named fields"),
        },
        _ => panic!("#[rkp_component] only supports structs"),
    };

    // Collect field info
    struct FieldInfo {
        name: String,
        ident: syn::Ident,
        ty: syn::Type,
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
        Some(FieldInfo { name, ident, ty, field_type_str, transient, range, asset_filter, is_option })
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
            rkp_engine::component_registry::FieldMeta {
                name: #name,
                field_type: rkp_engine::inspector::FieldType::#ft,
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
            "Float" => quote!(Ok(rkp_engine::inspector::FieldValue::Float(c.#ident as f64))),
            "Int" => quote!(Ok(rkp_engine::inspector::FieldValue::Int(c.#ident as i64))),
            "Bool" => quote!(Ok(rkp_engine::inspector::FieldValue::Bool(c.#ident))),
            "String" => {
                if fi.is_option {
                    quote!(Ok(rkp_engine::inspector::FieldValue::String(c.#ident.clone().unwrap_or_default())))
                } else {
                    quote!(Ok(rkp_engine::inspector::FieldValue::String(c.#ident.clone())))
                }
            }
            "Vec3" => quote!(Ok(rkp_engine::inspector::FieldValue::Vec3(c.#ident.to_array()))),
            "Color" => quote!(Ok(rkp_engine::inspector::FieldValue::Color(c.#ident))),
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
                if let rkp_engine::inspector::FieldValue::Float(v) = value {
                    c.#ident = v as _; Ok(())
                } else { Err("type mismatch".into()) }
            },
            "Int" => quote! {
                if let rkp_engine::inspector::FieldValue::Int(v) = value {
                    c.#ident = v as _; Ok(())
                } else { Err("type mismatch".into()) }
            },
            "Bool" => quote! {
                if let rkp_engine::inspector::FieldValue::Bool(v) = value {
                    c.#ident = v; Ok(())
                } else { Err("type mismatch".into()) }
            },
            "String" => {
                if fi.is_option {
                    quote! {
                        if let rkp_engine::inspector::FieldValue::String(v) = value {
                            c.#ident = if v.is_empty() { None } else { Some(v) }; Ok(())
                        } else { Err("type mismatch".into()) }
                    }
                } else {
                    quote! {
                        if let rkp_engine::inspector::FieldValue::String(v) = value {
                            c.#ident = v; Ok(())
                        } else { Err("type mismatch".into()) }
                    }
                }
            }
            "Vec3" => quote! {
                if let rkp_engine::inspector::FieldValue::Vec3(v) = value {
                    c.#ident = glam::Vec3::from_array(v); Ok(())
                } else { Err("type mismatch".into()) }
            },
            "Color" => quote! {
                if let rkp_engine::inspector::FieldValue::Color(v) = value {
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

        static #fields_static: [rkp_engine::component_registry::FieldMeta; #field_count] = [
            #(#meta_entries),*
        ];

        inventory::submit! {
            rkp_engine::component_registry::ComponentEntry {
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
                set_field: |world: &mut hecs::World, entity: hecs::Entity, field: &str, value: rkp_engine::inspector::FieldValue| {
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
            }
        }
    };

    output.into()
}
