#![allow(clippy::too_many_arguments)]
#![allow(clippy::type_complexity)]

mod collections;
mod conversions;
mod database;
mod filters;
mod hybrid;
mod iterators;
mod lifecycle;
mod open;
mod parsing;
mod search;

use pyo3::prelude::*;

#[pymodule]
fn omendb(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(open::open, m)?)?;
    m.add_class::<database::VectorDatabase>()?;
    m.add_class::<iterators::VectorDatabaseIdIterator>()?;
    Ok(())
}
