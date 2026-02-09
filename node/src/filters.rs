use napi::bindgen_prelude::*;
use omendb_lib::vector::MetadataFilter;
use serde_json::Value as JsonValue;

/// Parse a numeric comparison operator ($gt, $gte, $lt, $lte)
pub(crate) fn parse_numeric_op(op: &str, key: &str, value: &JsonValue) -> Result<MetadataFilter> {
    let num = value
        .as_f64()
        .ok_or_else(|| Error::new(Status::InvalidArg, format!("{} requires a number", op)))?;
    Ok(match op {
        "$gt" => MetadataFilter::Gt(key.to_string(), num),
        "$gte" => MetadataFilter::Gte(key.to_string(), num),
        "$lt" => MetadataFilter::Lt(key.to_string(), num),
        "$lte" => MetadataFilter::Lte(key.to_string(), num),
        _ => unreachable!(),
    })
}

/// Parse JavaScript filter object into MetadataFilter
/// Supports: equality, $gt, $gte, $lt, $lte, $in, $contains, $and, $or
pub(crate) fn parse_filter(filter: &JsonValue) -> Result<MetadataFilter> {
    let obj = filter
        .as_object()
        .ok_or_else(|| Error::new(Status::InvalidArg, "Filter must be an object"))?;

    if let Some(and_value) = obj.get("$and") {
        let arr = and_value
            .as_array()
            .ok_or_else(|| Error::new(Status::InvalidArg, "$and must be an array"))?;
        let sub_filters: Result<Vec<MetadataFilter>> = arr.iter().map(parse_filter).collect();
        return Ok(MetadataFilter::And(sub_filters?));
    }

    if let Some(or_value) = obj.get("$or") {
        let arr = or_value
            .as_array()
            .ok_or_else(|| Error::new(Status::InvalidArg, "$or must be an array"))?;
        let sub_filters: Result<Vec<MetadataFilter>> = arr.iter().map(parse_filter).collect();
        return Ok(MetadataFilter::Or(sub_filters?));
    }

    if let Some(not_value) = obj.get("$not") {
        let inner = parse_filter(not_value)?;
        return Ok(MetadataFilter::Not(Box::new(inner)));
    }

    let mut filters = Vec::new();

    for (key, value) in obj {
        if let Some(op_obj) = value.as_object() {
            for (op, op_value) in op_obj {
                let filter = match op.as_str() {
                    "$eq" => MetadataFilter::Eq(key.clone(), op_value.clone()),
                    "$ne" => MetadataFilter::Ne(key.clone(), op_value.clone()),
                    "$gt" | "$gte" | "$lt" | "$lte" => parse_numeric_op(op, key, op_value)?,
                    "$in" => {
                        let arr = op_value.as_array().ok_or_else(|| {
                            Error::new(Status::InvalidArg, "$in requires an array")
                        })?;
                        MetadataFilter::In(key.clone(), arr.clone())
                    }
                    "$contains" => {
                        let s = op_value.as_str().ok_or_else(|| {
                            Error::new(Status::InvalidArg, "$contains requires a string")
                        })?;
                        MetadataFilter::Contains(key.clone(), s.to_string())
                    }
                    _ => {
                        return Err(Error::new(
                            Status::InvalidArg,
                            format!("Unknown filter operator: {}", op),
                        ));
                    }
                };
                filters.push(filter);
            }
        } else {
            filters.push(MetadataFilter::Eq(key.clone(), value.clone()));
        }
    }

    if filters.len() == 1 {
        Ok(filters.into_iter().next().expect("checked len == 1"))
    } else {
        Ok(MetadataFilter::And(filters))
    }
}
