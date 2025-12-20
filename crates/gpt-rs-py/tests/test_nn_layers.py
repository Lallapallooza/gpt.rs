"""Test neural network layers (Embedding, Linear, LayerNorm, FeedForward)."""

import gpt_rs
import numpy as np
import pytest


class TestNNLayers:
    """Test gpt_rs.nn module layers."""

    @pytest.fixture(autouse=True)
    def setup_backend(self):
        """Ensure CPU backend is active for all tests."""
        gpt_rs.set_backend("cpu")

    def test_linear_no_bias(self):
        """Test Linear layer without bias."""
        # Weight: (2, 3) - maps 2D input to 3D output
        weight = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        tensor_weight = gpt_rs.Tensor.from_numpy(weight)

        linear = gpt_rs.nn.Linear(tensor_weight, bias=None)

        # Input: (1, 2)
        x = np.array([[1.0, 0.5]], dtype=np.float32)
        tensor_x = gpt_rs.Tensor.from_numpy(x)

        result = linear.forward(tensor_x)
        result_np = result.to_numpy()

        # Expected: x @ weight = [[1*1 + 0.5*4, 1*2 + 0.5*5, 1*3 + 0.5*6]]
        expected = np.array([[3.0, 4.5, 6.0]], dtype=np.float32)
        np.testing.assert_allclose(result_np, expected, atol=1e-6)

    def test_linear_with_bias(self):
        """Test Linear layer with bias."""
        weight = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        bias = np.array([0.1, 0.2], dtype=np.float32)

        tensor_weight = gpt_rs.Tensor.from_numpy(weight)
        tensor_bias = gpt_rs.Tensor.from_numpy(bias)

        linear = gpt_rs.nn.Linear(tensor_weight, bias=tensor_bias)

        x = np.array([[2.0, 3.0]], dtype=np.float32)
        tensor_x = gpt_rs.Tensor.from_numpy(x)

        result = linear.forward(tensor_x)
        result_np = result.to_numpy()

        # x @ weight + bias = [[2*1 + 3*3 + 0.1, 2*2 + 3*4 + 0.2]]
        expected = np.array([[11.1, 16.2]], dtype=np.float32)
        np.testing.assert_allclose(result_np, expected, atol=1e-5)

    def test_linear_batched(self):
        """Test Linear layer with batched input."""
        weight = np.random.randn(4, 5).astype(np.float32)
        bias = np.random.randn(5).astype(np.float32)

        tensor_weight = gpt_rs.Tensor.from_numpy(weight)
        tensor_bias = gpt_rs.Tensor.from_numpy(bias)

        linear = gpt_rs.nn.Linear(tensor_weight, bias=tensor_bias)

        # Batched input: (3, 4)
        x = np.random.randn(3, 4).astype(np.float32)
        tensor_x = gpt_rs.Tensor.from_numpy(x)

        result = linear.forward(tensor_x)
        result_np = result.to_numpy()

        expected = x @ weight + bias
        np.testing.assert_allclose(result_np, expected, atol=1e-5)

    def test_layer_norm_forward(self):
        """Test LayerNorm layer."""
        normalized_shape = 4
        weight = np.ones(normalized_shape, dtype=np.float32)
        bias = np.zeros(normalized_shape, dtype=np.float32)
        eps = 1e-5

        tensor_weight = gpt_rs.Tensor.from_numpy(weight)
        tensor_bias = gpt_rs.Tensor.from_numpy(bias)

        layer_norm = gpt_rs.nn.LayerNorm(tensor_weight, tensor_bias, eps=eps)

        x = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], dtype=np.float32)
        tensor_x = gpt_rs.Tensor.from_numpy(x)

        result = layer_norm.forward(tensor_x)
        result_np = result.to_numpy()

        # Compute expected
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        expected = (x - mean) / np.sqrt(var + eps) * weight + bias

        np.testing.assert_allclose(result_np, expected, atol=1e-5)

    def test_layer_norm_with_learnable_params(self):
        """Test LayerNorm with non-identity weight and bias."""
        weight = np.array([2.0, 1.0, 0.5, 1.5], dtype=np.float32)
        bias = np.array([0.1, -0.1, 0.2, -0.2], dtype=np.float32)
        eps = 1e-5

        tensor_weight = gpt_rs.Tensor.from_numpy(weight)
        tensor_bias = gpt_rs.Tensor.from_numpy(bias)

        layer_norm = gpt_rs.nn.LayerNorm(tensor_weight, tensor_bias, eps=eps)

        x = np.random.randn(2, 4).astype(np.float32)
        tensor_x = gpt_rs.Tensor.from_numpy(x)

        result = layer_norm.forward(tensor_x)
        result_np = result.to_numpy()

        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        expected = (x - mean) / np.sqrt(var + eps) * weight + bias

        np.testing.assert_allclose(result_np, expected, atol=1e-5)

    def test_embedding_forward(self):
        """Test Embedding layer with rank-1 indices."""
        # vocab_size=5, embedding_dim=3
        weight = np.array(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0],
                [10.0, 11.0, 12.0],
                [13.0, 14.0, 15.0],
            ],
            dtype=np.float32,
        )

        tensor_weight = gpt_rs.Tensor.from_numpy(weight)
        embedding = gpt_rs.nn.Embedding(tensor_weight)

        # Look up indices [0, 2, 4] (rank-1)
        indices = np.array([0, 2, 4], dtype=np.int32)
        tensor_indices = gpt_rs.Tensor.from_numpy(indices)

        result = embedding.forward(tensor_indices)
        result_np = result.to_numpy()

        expected = weight[[0, 2, 4]]
        np.testing.assert_array_equal(result_np, expected)

    def test_feedforward_no_bias(self):
        """Test FeedForward layer without bias."""
        d_model = 4
        d_ff = 6

        w_in = np.random.randn(d_model, d_ff).astype(np.float32)
        w_out = np.random.randn(d_ff, d_model).astype(np.float32)

        tensor_w_in = gpt_rs.Tensor.from_numpy(w_in)
        tensor_w_out = gpt_rs.Tensor.from_numpy(w_out)

        ffn = gpt_rs.nn.FeedForward(tensor_w_in, tensor_w_out, b_in=None, b_out=None)

        x = np.random.randn(2, d_model).astype(np.float32)
        tensor_x = gpt_rs.Tensor.from_numpy(x)

        result = ffn.forward(tensor_x)
        result_np = result.to_numpy()

        # Expected: GELU(x @ w_in) @ w_out
        import scipy.special

        hidden = x @ w_in
        hidden_gelu = hidden * scipy.special.ndtr(hidden)
        expected = hidden_gelu @ w_out

        np.testing.assert_allclose(result_np, expected, atol=1e-4)

    def test_feedforward_with_bias(self):
        """Test FeedForward layer with bias."""
        d_model = 3
        d_ff = 4

        w_in = np.random.randn(d_model, d_ff).astype(np.float32)
        w_out = np.random.randn(d_ff, d_model).astype(np.float32)
        b_in = np.random.randn(d_ff).astype(np.float32)
        b_out = np.random.randn(d_model).astype(np.float32)

        tensor_w_in = gpt_rs.Tensor.from_numpy(w_in)
        tensor_w_out = gpt_rs.Tensor.from_numpy(w_out)
        tensor_b_in = gpt_rs.Tensor.from_numpy(b_in)
        tensor_b_out = gpt_rs.Tensor.from_numpy(b_out)

        ffn = gpt_rs.nn.FeedForward(tensor_w_in, tensor_w_out, b_in=tensor_b_in, b_out=tensor_b_out)

        x = np.random.randn(2, d_model).astype(np.float32)
        tensor_x = gpt_rs.Tensor.from_numpy(x)

        result = ffn.forward(tensor_x)
        result_np = result.to_numpy()

        # Expected: GELU(x @ w_in + b_in) @ w_out + b_out
        import scipy.special

        hidden = x @ w_in + b_in
        hidden_gelu = hidden * scipy.special.ndtr(hidden)
        expected = hidden_gelu @ w_out + b_out

        np.testing.assert_allclose(result_np, expected, atol=1e-4)

    def test_feedforward_batched_2d(self):
        """Test FeedForward with batched 2D input."""
        batch_size = 5
        d_model = 4
        d_ff = 6

        w_in = np.random.randn(d_model, d_ff).astype(np.float32)
        w_out = np.random.randn(d_ff, d_model).astype(np.float32)

        tensor_w_in = gpt_rs.Tensor.from_numpy(w_in)
        tensor_w_out = gpt_rs.Tensor.from_numpy(w_out)

        ffn = gpt_rs.nn.FeedForward(tensor_w_in, tensor_w_out, b_in=None, b_out=None)

        # Batched 2D input
        x = np.random.randn(batch_size, d_model).astype(np.float32)
        tensor_x = gpt_rs.Tensor.from_numpy(x)

        result = ffn.forward(tensor_x)
        result_np = result.to_numpy()

        # Expected: GELU(x @ w_in) @ w_out
        import scipy.special

        hidden = x @ w_in
        hidden_gelu = hidden * scipy.special.ndtr(hidden)
        expected = hidden_gelu @ w_out

        np.testing.assert_allclose(result_np, expected, atol=1e-4)
