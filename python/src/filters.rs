use crate::conversions::pyobject_to_json;
use omendb_lib::vector::MetadataFilter;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use serde_json::Value as JsonValue;

/// Parse Python filter dict to Rust MetadataFilter
pub(crate) fn parse_filter(filter: &Bound<'_, PyDict>) -> PyResult<MetadataFilter> {
    // Handle special logical operators first
    if let Some(and_value) = filter.get_item("$and")? {
        // $and expects an array of filter dicts
        let and_list = and_value
            .cast::<PyList>()
            .map_err(|_| PyValueError::new_err("$and must be an array of filters"))?;

        let mut sub_filters = Vec::new();
        for item in and_list.iter() {
            let sub_dict = item
                .cast::<PyDict>()
                .map_err(|_| PyValueError::new_err("Each $and element must be a dict"))?;
            sub_filters.push(parse_filter(sub_dict)?);
        }

        return Ok(MetadataFilter::And(sub_filters));
    }

    if let Some(or_value) = filter.get_item("$or")? {
        // $or expects an array of filter dicts
        let or_list = or_value
            .cast::<PyList>()
            .map_err(|_| PyValueError::new_err("$or must be an array of filters"))?;

        let mut sub_filters = Vec::new();
        for item in or_list.iter() {
            let sub_dict = item
                .cast::<PyDict>()
                .map_err(|_| PyValueError::new_err("Each $or element must be a dict"))?;
            sub_filters.push(parse_filter(sub_dict)?);
        }

        return Ok(MetadataFilter::Or(sub_filters));
    }

    if let Some(not_value) = filter.get_item("$not")? {
        let not_dict = not_value
            .cast::<PyDict>()
            .map_err(|_| PyValueError::new_err("$not must be a filter dict"))?;
        let inner = parse_filter(not_dict)?;
        return Ok(MetadataFilter::Not(Box::new(inner)));
    }

    // Parse regular field filters
    let mut filters = Vec::new();

    for (key, value) in filter.iter() {
        let key_str: String = key.extract()?;

        // Check if value is an operator dict like {"$gt": 5}
        if let Ok(op_dict) = value.cast::<PyDict>() {
            for (op, op_value) in op_dict.iter() {
                let op_str: String = op.extract()?;
                match op_str.as_str() {
                    "$eq" => {
                        let json_value = pyobject_to_json(&op_value)?;
                        filters.push(MetadataFilter::Eq(key_str.clone(), json_value));
                    }
                    "$ne" => {
                        let json_value = pyobject_to_json(&op_value)?;
                        filters.push(MetadataFilter::Ne(key_str.clone(), json_value));
                    }
                    "$gt" => {
                        let num: f64 = op_value.extract()?;
                        filters.push(MetadataFilter::Gt(key_str.clone(), num));
                    }
                    "$gte" => {
                        let num: f64 = op_value.extract()?;
                        filters.push(MetadataFilter::Gte(key_str.clone(), num));
                    }
                    "$lt" => {
                        let num: f64 = op_value.extract()?;
                        filters.push(MetadataFilter::Lt(key_str.clone(), num));
                    }
                    "$lte" => {
                        let num: f64 = op_value.extract()?;
                        filters.push(MetadataFilter::Lte(key_str.clone(), num));
                    }
                    "$in" => {
                        let list = op_value.cast::<PyList>()?;
                        let json_vals: Result<Vec<JsonValue>, _> =
                            list.iter().map(|obj| pyobject_to_json(&obj)).collect();
                        filters.push(MetadataFilter::In(key_str.clone(), json_vals?));
                    }
                    "$contains" => {
                        let substr: String = op_value.extract()?;
                        filters.push(MetadataFilter::Contains(key_str.clone(), substr));
                    }
                    _ => {
                        return Err(PyValueError::new_err(format!(
                            "Unknown filter operator: {}",
                            op_str
                        )));
                    }
                }
            }
        } else {
            // Direct equality: {"field": value}
            let json_value = pyobject_to_json(&value)?;
            filters.push(MetadataFilter::Eq(key_str, json_value));
        }
    }

    if filters.len() == 1 {
        Ok(filters.pop().unwrap())
    } else {
        Ok(MetadataFilter::And(filters))
    }
}
