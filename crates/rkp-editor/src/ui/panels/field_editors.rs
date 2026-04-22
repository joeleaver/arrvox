//! Field editors — bridges ECS inspector FieldSnapshot → prop_controls.
//!
//! `field_editor()` dispatches on `FieldType` and constructs the
//! appropriate prop_control. The static metadata (name, type, range,
//! enum options, scrub flag) comes from the FieldSnapshot reference;
//! the live value comes from a `Memo<FieldValue>` so external updates
//! (engine writes during play, MCP, undo) flow into the input without
//! remounting the row. User edits go out through `on_change`.

use std::rc::Rc;

use rinch::prelude::*;

use rkp_engine::inspector::*;
use super::prop_controls;

/// Render the appropriate editor for a field based on its type.
pub fn field_editor(
    __scope: &mut rinch::core::dom::RenderScope,
    field: &FieldSnapshot,
    value: Memo<FieldValue>,
    on_change: Rc<dyn Fn(FieldValue)>,
) -> rinch::core::dom::NodeHandle {
    match field.field_type {
        FieldType::Float => float_editor(__scope, field, value, on_change),
        FieldType::Int => int_editor(__scope, field, value, on_change),
        FieldType::Bool => bool_editor(__scope, field, value, on_change),
        FieldType::String => string_editor(__scope, field, value, on_change),
        FieldType::Vec3 => vec3_editor(__scope, field, value, on_change),
        FieldType::Color => color_editor(__scope, field, value, on_change),
    }
}

fn float_editor(
    __scope: &mut rinch::core::dom::RenderScope,
    field: &FieldSnapshot,
    value: Memo<FieldValue>,
    on_change: Rc<dyn Fn(FieldValue)>,
) -> rinch::core::dom::NodeHandle {
    let value_f64 = Memo::new(move || match value.get() {
        FieldValue::Float(v) => v,
        _ => 0.0,
    });

    if field.scrub {
        if let Some((min, max)) = field.range {
            let value_f32 = Memo::new(move || value_f64.get() as f32);
            let step = ((max - min) / 200.0) as f32;
            return prop_controls::prop_scrub(
                __scope,
                &field.name,
                value_f32,
                min as f32,
                max as f32,
                step,
                Rc::new(move |v| on_change(FieldValue::Float(v as f64))),
            );
        }
    }

    if let Some((min, max)) = field.range {
        let step = (max - min) / 200.0;
        prop_controls::prop_slider_f64_memo(
            __scope,
            &field.name,
            value_f64,
            min,
            max,
            step,
            Rc::new(move |v| on_change(FieldValue::Float(v))),
        )
    } else {
        prop_controls::prop_number_f64_memo(
            __scope,
            &field.name,
            value_f64,
            Rc::new(move |v| on_change(FieldValue::Float(v))),
        )
    }
}

fn int_editor(
    __scope: &mut rinch::core::dom::RenderScope,
    field: &FieldSnapshot,
    value: Memo<FieldValue>,
    on_change: Rc<dyn Fn(FieldValue)>,
) -> rinch::core::dom::NodeHandle {
    let value_i64 = Memo::new(move || match value.get() {
        FieldValue::Int(v) => v,
        _ => 0,
    });
    prop_controls::prop_number_i64_memo(
        __scope,
        &field.name,
        value_i64,
        Rc::new(move |v| on_change(FieldValue::Int(v))),
    )
}

fn bool_editor(
    __scope: &mut rinch::core::dom::RenderScope,
    field: &FieldSnapshot,
    value: Memo<FieldValue>,
    on_change: Rc<dyn Fn(FieldValue)>,
) -> rinch::core::dom::NodeHandle {
    let value_b = Memo::new(move || matches!(value.get(), FieldValue::Bool(true)));
    prop_controls::prop_checkbox_memo(
        __scope,
        &field.name,
        value_b,
        Rc::new(move |v| on_change(FieldValue::Bool(v))),
    )
}

fn string_editor(
    __scope: &mut rinch::core::dom::RenderScope,
    field: &FieldSnapshot,
    value: Memo<FieldValue>,
    on_change: Rc<dyn Fn(FieldValue)>,
) -> rinch::core::dom::NodeHandle {
    let value_s = Memo::new(move || match value.get() {
        FieldValue::String(v) => v,
        _ => String::new(),
    });

    if !field.enum_options.is_empty() {
        let options: Vec<(&str, &str)> = field.enum_options.iter()
            .map(|(v, l)| (v.as_str(), l.as_str()))
            .collect();
        prop_controls::prop_select(
            __scope,
            &field.name,
            value_s,
            &options,
            Rc::new(move |v| on_change(FieldValue::String(v))),
        )
    } else {
        prop_controls::prop_text_memo(
            __scope,
            &field.name,
            value_s,
            Rc::new(move |v| on_change(FieldValue::String(v))),
        )
    }
}

fn vec3_editor(
    __scope: &mut rinch::core::dom::RenderScope,
    field: &FieldSnapshot,
    value: Memo<FieldValue>,
    on_change: Rc<dyn Fn(FieldValue)>,
) -> rinch::core::dom::NodeHandle {
    let value_v = Memo::new(move || match value.get() {
        FieldValue::Vec3(v) => v,
        _ => [0.0; 3],
    });
    prop_controls::prop_vec3(
        __scope,
        &field.name,
        value_v,
        Rc::new(move |v| on_change(FieldValue::Vec3(v))),
    )
}

fn color_editor(
    __scope: &mut rinch::core::dom::RenderScope,
    field: &FieldSnapshot,
    value: Memo<FieldValue>,
    on_change: Rc<dyn Fn(FieldValue)>,
) -> rinch::core::dom::NodeHandle {
    let value_c = Memo::new(move || match value.get() {
        FieldValue::Color(v) => v,
        _ => [1.0, 1.0, 1.0, 1.0],
    });
    prop_controls::prop_color(
        __scope,
        &field.name,
        value_c,
        Rc::new(move |v| on_change(FieldValue::Color(v))),
    )
}
