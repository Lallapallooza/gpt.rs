use gpt_rs::backend::shape_helpers::{
    checked_element_count_or_error, contiguous_strides_or_error, static_dims_or_error,
};
use gpt_rs::backend::spec::{DimSymbol, Dimension, Shape};

#[test]
fn static_dims_or_error_returns_dims_for_static_shape() {
    let shape = Shape::new(vec![Dimension::Static(2), Dimension::Static(3)]);
    let dims = static_dims_or_error::<String, _>(&shape, |symbol| symbol.as_str().to_string())
        .unwrap_or_else(|err| panic!("unexpected error: {err}"));
    assert_eq!(dims, vec![2, 3]);
}

#[test]
fn static_dims_or_error_maps_dynamic_symbol() {
    let shape = Shape::new(vec![
        Dimension::Static(2),
        Dimension::Dynamic(DimSymbol::new("B")),
        Dimension::Static(4),
    ]);
    let err = static_dims_or_error::<String, _>(&shape, |symbol| symbol.as_str().to_string())
        .expect_err("dynamic shape should return an error");
    assert_eq!(err, "B");
}

#[test]
fn checked_element_count_or_error_reports_overflow() {
    let err = checked_element_count_or_error(&[usize::MAX, 2], || "overflow".to_string())
        .expect_err("overflow should be reported");
    assert_eq!(err, "overflow");
}

#[test]
fn contiguous_strides_or_error_returns_row_major_layout() {
    let strides = contiguous_strides_or_error::<String, _>(&[2, 3, 4], || "overflow".to_string())
        .unwrap_or_else(|err| panic!("unexpected error: {err}"));
    assert_eq!(strides, vec![12, 4, 1]);
}
