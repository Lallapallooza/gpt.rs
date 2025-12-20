"""Test functional operations (softmax, gelu, layer_norm, matmul, etc.)."""

import gpt_rs
import numpy as np
import pytest


class TestFunctionalOps:
    """Test gpt_rs.functional module operations."""

    @pytest.fixture(autouse=True)
    def setup_backend(self):
        """Ensure CPU backend is active for all tests."""
        gpt_rs.set_backend("cpu")

    def test_softmax_last_dim_2d(self):
        """Test softmax on 2D array."""
        x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        tensor_x = gpt_rs.Tensor.from_numpy(x)

        result = gpt_rs.functional.softmax_last_dim(tensor_x)
        result_np = result.to_numpy()

        # Compare with NumPy implementation
        expected = np.exp(x) / np.exp(x).sum(axis=-1, keepdims=True)
        np.testing.assert_allclose(result_np, expected, atol=1e-6)

    def test_softmax_last_dim_1d(self):
        """Test softmax on 1D array."""
        x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        tensor_x = gpt_rs.Tensor.from_numpy(x)

        result = gpt_rs.functional.softmax_last_dim(tensor_x)
        result_np = result.to_numpy()

        expected = np.exp(x) / np.exp(x).sum()
        np.testing.assert_allclose(result_np, expected, atol=1e-6)

    def test_softmax_last_dim_3d(self):
        """Test softmax on 3D array."""
        x = np.random.randn(2, 3, 4).astype(np.float32)
        tensor_x = gpt_rs.Tensor.from_numpy(x)

        result = gpt_rs.functional.softmax_last_dim(tensor_x)
        result_np = result.to_numpy()

        expected = np.exp(x) / np.exp(x).sum(axis=-1, keepdims=True)
        np.testing.assert_allclose(result_np, expected, atol=1e-6)

    def test_softmax_sums_to_one(self):
        """Test that softmax output sums to 1 along last dimension."""
        x = np.random.randn(3, 5).astype(np.float32)
        tensor_x = gpt_rs.Tensor.from_numpy(x)

        result = gpt_rs.functional.softmax_last_dim(tensor_x)
        result_np = result.to_numpy()

        sums = result_np.sum(axis=-1)
        np.testing.assert_allclose(sums, np.ones(3), atol=1e-6)

    def test_gelu_basic(self):
        """Test GELU activation function."""
        x = np.array([[-1.0, 0.0, 1.0], [2.0, -2.0, 0.5]], dtype=np.float32)
        tensor_x = gpt_rs.Tensor.from_numpy(x)

        result = gpt_rs.functional.gelu(tensor_x)
        result_np = result.to_numpy()

        # GELU(x) ≈ x * Φ(x) where Φ is standard normal CDF
        # Approximation: 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
        import scipy.special

        expected = x * scipy.special.ndtr(x)

        np.testing.assert_allclose(result_np, expected, atol=1e-4)

    def test_gelu_zero(self):
        """Test GELU at zero."""
        x = np.array([0.0], dtype=np.float32)
        tensor_x = gpt_rs.Tensor.from_numpy(x)

        result = gpt_rs.functional.gelu(tensor_x)
        result_np = result.to_numpy()

        assert abs(result_np[0]) < 1e-6, "GELU(0) should be close to 0"

    def test_matmul_2d(self):
        """Test matrix multiplication for 2D tensors."""
        a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        b = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)

        tensor_a = gpt_rs.Tensor.from_numpy(a)
        tensor_b = gpt_rs.Tensor.from_numpy(b)

        result = gpt_rs.functional.matmul(tensor_a, tensor_b)
        result_np = result.to_numpy()

        expected = np.matmul(a, b)
        np.testing.assert_allclose(result_np, expected, atol=1e-6)

    def test_matmul_3d(self):
        """Test batched matrix multiplication."""
        a = np.random.randn(2, 3, 4).astype(np.float32)
        b = np.random.randn(2, 4, 5).astype(np.float32)

        tensor_a = gpt_rs.Tensor.from_numpy(a)
        tensor_b = gpt_rs.Tensor.from_numpy(b)

        result = gpt_rs.functional.matmul(tensor_a, tensor_b)
        result_np = result.to_numpy()

        expected = np.matmul(a, b)
        np.testing.assert_allclose(result_np, expected, atol=1e-5)

    def test_matmul_requires_rank2(self):
        """Test that matmul requires rank-2 or rank-3 tensors."""
        a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        b = np.array([1.0, 0.0, -1.0], dtype=np.float32)

        tensor_a = gpt_rs.Tensor.from_numpy(a)
        tensor_b = gpt_rs.Tensor.from_numpy(b)

        # Should raise error for rank-1 tensor
        with pytest.raises(ValueError, match="rank-2 or rank-3"):
            gpt_rs.functional.matmul(tensor_a, tensor_b)

    def test_add_bias_2d(self):
        """Test adding bias to 2D tensor."""
        x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        bias = np.array([0.1, 0.2, 0.3], dtype=np.float32)

        tensor_x = gpt_rs.Tensor.from_numpy(x)
        tensor_bias = gpt_rs.Tensor.from_numpy(bias)

        result = gpt_rs.functional.add_bias(tensor_x, tensor_bias)
        result_np = result.to_numpy()

        expected = x + bias
        np.testing.assert_allclose(result_np, expected, atol=1e-6)

    def test_add_bias_3d(self):
        """Test adding bias to 3D tensor."""
        x = np.random.randn(2, 3, 4).astype(np.float32)
        bias = np.random.randn(4).astype(np.float32)

        tensor_x = gpt_rs.Tensor.from_numpy(x)
        tensor_bias = gpt_rs.Tensor.from_numpy(bias)

        result = gpt_rs.functional.add_bias(tensor_x, tensor_bias)
        result_np = result.to_numpy()

        expected = x + bias
        np.testing.assert_allclose(result_np, expected, atol=1e-6)

    def test_layer_norm_2d(self):
        """Test layer normalization on 2D tensor."""
        x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        weight = np.ones(3, dtype=np.float32)
        bias = np.zeros(3, dtype=np.float32)
        eps = 1e-5

        tensor_x = gpt_rs.Tensor.from_numpy(x)
        tensor_weight = gpt_rs.Tensor.from_numpy(weight)
        tensor_bias = gpt_rs.Tensor.from_numpy(bias)

        result = gpt_rs.functional.layer_norm(tensor_x, tensor_weight, tensor_bias, eps)
        result_np = result.to_numpy()

        # Compute expected layer norm
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        expected = (x - mean) / np.sqrt(var + eps) * weight + bias

        np.testing.assert_allclose(result_np, expected, atol=1e-5)

    def test_layer_norm_with_learnable_params(self):
        """Test layer normalization with non-trivial weight and bias."""
        x = np.random.randn(2, 4).astype(np.float32)
        weight = np.array([1.0, 2.0, 1.5, 0.5], dtype=np.float32)
        bias = np.array([0.1, -0.1, 0.2, -0.2], dtype=np.float32)
        eps = 1e-5

        tensor_x = gpt_rs.Tensor.from_numpy(x)
        tensor_weight = gpt_rs.Tensor.from_numpy(weight)
        tensor_bias = gpt_rs.Tensor.from_numpy(bias)

        result = gpt_rs.functional.layer_norm(tensor_x, tensor_weight, tensor_bias, eps)
        result_np = result.to_numpy()

        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        expected = (x - mean) / np.sqrt(var + eps) * weight + bias

        np.testing.assert_allclose(result_np, expected, atol=1e-5)

    def test_layer_norm_zero_mean_unit_var(self):
        """Test that layer norm produces zero mean and unit variance (before affine transform)."""
        x = np.random.randn(3, 5).astype(np.float32) * 10 + 50  # arbitrary mean and scale
        weight = np.ones(5, dtype=np.float32)
        bias = np.zeros(5, dtype=np.float32)
        eps = 1e-5

        tensor_x = gpt_rs.Tensor.from_numpy(x)
        tensor_weight = gpt_rs.Tensor.from_numpy(weight)
        tensor_bias = gpt_rs.Tensor.from_numpy(bias)

        result = gpt_rs.functional.layer_norm(tensor_x, tensor_weight, tensor_bias, eps)
        result_np = result.to_numpy()

        # Check mean is close to 0 and variance close to 1
        mean = result_np.mean(axis=-1)
        var = result_np.var(axis=-1)

        np.testing.assert_allclose(mean, np.zeros(3), atol=1e-5)
        np.testing.assert_allclose(var, np.ones(3), atol=1e-4)


def test_embedding_lookup_basic():
    """Test embedding lookup functionality with rank-1 indices."""
    gpt_rs.set_backend("cpu")

    # Embedding table: 5 words, 3 dimensions
    table = np.array(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0], [13.0, 14.0, 15.0]],
        dtype=np.float32,
    )

    # Indices to look up (rank-1)
    indices = np.array([0, 2, 4], dtype=np.int32)

    tensor_table = gpt_rs.Tensor.from_numpy(table)
    tensor_indices = gpt_rs.Tensor.from_numpy(indices)

    result = gpt_rs.functional.embedding_lookup(tensor_table, tensor_indices)
    result_np = result.to_numpy()

    expected = table[[0, 2, 4]]
    np.testing.assert_array_equal(result_np, expected)


def test_embedding_lookup_rank2():
    """Test embedding lookup with rank-2 indices (legacy format)."""
    gpt_rs.set_backend("cpu")

    table = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)

    # Rank-2 indices: [seq_len, 1]
    indices = np.array([[0], [2], [1]], dtype=np.int32)

    tensor_table = gpt_rs.Tensor.from_numpy(table)
    tensor_indices = gpt_rs.Tensor.from_numpy(indices)

    result = gpt_rs.functional.embedding_lookup(tensor_table, tensor_indices)
    result_np = result.to_numpy()

    expected = table[[0, 2, 1]]
    np.testing.assert_array_equal(result_np, expected)
