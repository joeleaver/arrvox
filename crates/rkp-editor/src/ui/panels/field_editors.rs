//! Field editors — bridges ECS inspector FieldSnapshot → prop_controls.
//!
//! The `field_editor()` function dispatches on `FieldType` and constructs
//! the appropriate prop_control with the right value Signal and callback.

use std::rc::Rc;

use rinch::prelude::*;

use rkp_engine::inspector::*;
use super::prop_controls;

/// Render the appropriate editor for a field based on its type.
pub fn field_editor(
    __scope: &mut rinch::core::dom::RenderScope,
    field: &FieldSnapshot,
    on_change: Rc<dyn Fn(FieldValue)>,
) -> rinch::core::dom::NodeHandle {
    match field.field_type {
        FieldType::Float => float_editor(__scope, field, on_change),
        FieldType::Int => int_editor(__scope, field, on_change),
        FieldType::Bool => bool_editor(__scope, field, on_change),
        FieldType::String => string_editor(__scope, field, on_change),
        FieldType::Vec3 => vec3_editor(__scope, field, on_change),
        FieldType::Color => color_editor(__scope, field, on_change),
    }
}

fn float_editor(
    __scope: &mut rinch::core::dom::RenderScope,
    field: &FieldSnapshot,
    on_change: Rc<dyn Fn(FieldValue)>,
) -> rinch::core::dom::NodeHandle {
    let val = match &field.value { FieldValue::Float(v) => *v, _ => 0.0 };

    if field.scrub {
        if let Some((min, max)) = field.range {
            let value = Signal::new(val as f32);
            let display = Memo::new(move || value.get());
            let step = ((max - min) / 200.0) as f32;
            return prop_controls::prop_scrub(
                __scope,
                &field.name,
                display,
                min as f32,
                max as f32,
                step,
                Rc::new(move |v| {
                    value.set(v);
                    on_change(FieldValue::Float(v as f64));
                }),
            );
        }
    }

    if let Some((min, max)) = field.range {
        let value = Signal::new(val);
        let step = (max - min) / 200.0;
        prop_controls::prop_slider_f64(
            __scope,
            &field.name,
            value,
            min,
            max,
            step,
            Rc::new(move |v| on_change(FieldValue::Float(v))),
        )
    } else {
        let value = Signal::new(val);
        prop_controls::prop_number_f64(
            __scope,
            &field.name,
            value,
            Rc::new(move |v| on_change(FieldValue::Float(v))),
        )
    }
}

fn int_editor(
    __scope: &mut rinch::core::dom::RenderScope,
    field: &FieldSnapshot,
    on_change: Rc<dyn Fn(FieldValue)>,
) -> rinch::core::dom::NodeHandle {
    let val = match &field.value { FieldValue::Int(v) => *v, _ => 0 };
    let value = Signal::new(val);
    prop_controls::prop_number_i64(
        __scope,
        &field.name,
        value,
        Rc::new(move |v| on_change(FieldValue::Int(v))),
    )
}

fn bool_editor(
    __scope: &mut rinch::core::dom::RenderScope,
    field: &FieldSnapshot,
    on_change: Rc<dyn Fn(FieldValue)>,
) -> rinch::core::dom::NodeHandle {
    let val = match &field.value { FieldValue::Bool(v) => *v, _ => false };
    let value = Signal::new(val);
    prop_controls::prop_checkbox(
        __scope,
        &field.name,
        value,
        Rc::new(move |v| on_change(FieldValue::Bool(v))),
    )
}

fn string_editor(
    __scope: &mut rinch::core::dom::RenderScope,
    field: &FieldSnapshot,
    on_change: Rc<dyn Fn(FieldValue)>,
) -> rinch::core::dom::NodeHandle {
    let val = match &field.value { FieldValue::String(v) => v.clone(), _ => String::new() };
    let value = Signal::new(val);

    // If this field has enum options, render as a dropdown select.
    if !field.enum_options.is_empty() {
        let options: Vec<(&str, &str)> = field.enum_options.iter()
            .map(|(v, l)| (v.as_str(), l.as_str()))
            .collect();
        prop_controls::prop_select(
            __scope,
            &field.name,
            value,
            &options,
            Rc::new(move |v| on_change(FieldValue::String(v))),
        )
    } else {
        prop_controls::prop_text(
            __scope,
            &field.name,
            value,
            Rc::new(move |v| on_change(FieldValue::String(v))),
        )
    }
}

fn vec3_editor(
    __scope: &mut rinch::core::dom::RenderScope,
    field: &FieldSnapshot,
    on_change: Rc<dyn Fn(FieldValue)>,
) -> rinch::core::dom::NodeHandle {
    let val = match &field.value { FieldValue::Vec3(v) => *v, _ => [0.0; 3] };
    let value = Signal::new(val);
    prop_controls::prop_vec3(
        __scope,
        &field.name,
        value,
        Rc::new(move |v| on_change(FieldValue::Vec3(v))),
    )
}

fn color_editor(
    __scope: &mut rinch::core::dom::RenderScope,
    field: &FieldSnapshot,
    on_change: Rc<dyn Fn(FieldValue)>,
) -> rinch::core::dom::NodeHandle {
    let val = match &field.value { FieldValue::Color(v) => *v, _ => [1.0, 1.0, 1.0, 1.0] };
    let value = Signal::new(val);
    prop_controls::prop_color(
        __scope,
        &field.name,
        value,
        Rc::new(move |v| on_change(FieldValue::Color(v))),
    )
}
