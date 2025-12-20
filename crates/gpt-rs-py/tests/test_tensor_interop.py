"""Tests for Tensor NumPy and PyTorch interoperability."""

import numpy as np
import pytest


def test_from_numpy_f32():
    """Test Tensor.from_numpy with f32 dtype validation (TASK-002)."""
    import gpt_rs

    arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    tensor = gpt_rs.Tensor.from_numpy(arr)

    assert tensor.shape == [2, 2]
    assert tensor.dtype == "f32"


def test_to_numpy_preserves_data():
    """Test Tensor.to_numpy round-trip (TASK-003)."""
    import gpt_rs

    arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    tensor = gpt_rs.Tensor.from_numpy(arr)
    result = tensor.to_numpy()

    np.testing.assert_array_equal(result, arr)
    assert result.dtype == np.float32


def test_from_numpy_fortran_order_preserves_data():
    """Ensure Fortran-contiguous arrays round-trip without reordering."""
    import gpt_rs

    arr = np.asfortranarray(np.arange(12, dtype=np.float32).reshape(3, 4))
    assert arr.flags["F_CONTIGUOUS"] and not arr.flags["C_CONTIGUOUS"]

    tensor = gpt_rs.Tensor.from_numpy(arr)
    result = tensor.to_numpy()

    np.testing.assert_array_equal(result, arr)


def test_from_numpy_transposed_view_preserves_data():
    """Ensure non-contiguous transposed views round-trip without reordering."""
    import gpt_rs

    base = np.arange(12, dtype=np.float32).reshape(3, 4)
    arr = base.T
    assert not arr.flags["C_CONTIGUOUS"]

    tensor = gpt_rs.Tensor.from_numpy(arr)
    result = tensor.to_numpy()

    np.testing.assert_array_equal(result, arr)


def test_from_torch_cpu():
    """Test Tensor.from_torch CPU conversion (TASK-004)."""
    import gpt_rs

    try:
        import torch
    except ImportError:
        pytest.skip("PyTorch not installed")

    t = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    tensor = gpt_rs.Tensor.from_torch(t)

    assert tensor.shape == [2, 2]
    assert tensor.dtype == "f32"


def test_to_torch_dtype_mapping():
    """Test Tensor.to_torch CPU export (TASK-005)."""
    import gpt_rs

    try:
        import torch
    except ImportError:
        pytest.skip("PyTorch not installed")

    arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    tensor = gpt_rs.Tensor.from_numpy(arr)
    result = tensor.to_torch()

    assert result.shape == (2, 2)
    assert result.dtype == torch.float32
    torch.testing.assert_close(result, torch.tensor([[1.0, 2.0], [3.0, 4.0]]))


def test_unsupported_dtype_raises():
    """Test Tensor dtype validation rejecting unsupported types (TASK-006)."""
    import gpt_rs

    # complex64 should be rejected
    arr_complex = np.array([1 + 2j, 3 + 4j], dtype=np.complex64)
    with pytest.raises(TypeError, match="unsupported|complex"):
        gpt_rs.Tensor.from_numpy(arr_complex)

    # int8 should be rejected (only f32, f64, i32, i64 supported)
    arr_int8 = np.array([1, 2, 3], dtype=np.int8)
    with pytest.raises(TypeError, match="unsupported|int8"):
        gpt_rs.Tensor.from_numpy(arr_int8)


def test_tensor_properties():
    """Test Tensor.shape, dtype, backend properties (part of TASK-011)."""
    import gpt_rs

    arr = np.array([[[1.0, 2.0]], [[3.0, 4.0]]], dtype=np.float32)
    tensor = gpt_rs.Tensor.from_numpy(arr)

    assert tensor.shape == [2, 1, 2]
    assert tensor.dtype == "f32"
    assert isinstance(tensor.backend, str)


def test_copy_semantics():
    """Test that from_numpy copies data (not zero-copy view)."""
    import gpt_rs

    original = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    tensor = gpt_rs.Tensor.from_numpy(original)

    # Modify original array
    original[0, 0] = 999.0

    # Tensor should be unaffected
    result = tensor.to_numpy()
    assert result[0, 0] == 1.0, "Tensor should not be affected by changes to source array"


def test_backend_selection():
    """Test backend selection affects tensor creation."""
    import gpt_rs

    # Test CPU backend
    gpt_rs.set_backend("cpu")
    current = gpt_rs.get_backend()
    assert current in ["cpu", "cpu-portable"]

    arr = np.array([1.0, 2.0], dtype=np.float32)
    tensor = gpt_rs.Tensor.from_numpy(arr)
    assert tensor.backend in ["cpu", "cpu-portable"]


def test_list_backends():
    """Test listing available backends."""
    import gpt_rs

    backends = gpt_rs.list_backends()
    assert isinstance(backends, list)
    assert len(backends) > 0
    assert any("cpu" in b for b in backends)
