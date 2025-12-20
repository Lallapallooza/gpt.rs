#!/usr/bin/env python3
"""
Basic usage examples for gpt-rs-py.

Demonstrates core functionality without requiring PyTorch.

Requirements:
    pip install numpy gpt-rs

Usage:
    python examples/basic_usage.py
"""

import numpy as np

try:
    import gpt_rs
except ImportError:
    print("Error: gpt_rs not installed.")
    print("\nTo install:")
    print("  pip install maturin")
    print("  cd crates/gpt-rs-py")
    print("  maturin develop")
    exit(1)


def example_tensor_creation():
    """Example: Creating and converting tensors."""
    print("\n=== Tensor Creation ===")

    # Create NumPy array
    np_array = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    print(f"NumPy array shape: {np_array.shape}")
    print(f"NumPy array:\n{np_array}")

    # Convert to gpt-rs tensor
    tensor = gpt_rs.Tensor.from_numpy(np_array)
    print(f"\ngpt-rs tensor shape: {tensor.shape}")
    print(f"gpt-rs tensor dtype: {tensor.dtype}")
    print(f"gpt-rs tensor backend: {tensor.backend}")

    # Convert back to NumPy
    result = tensor.numpy()
    print(f"\nBack to NumPy:\n{result}")


def example_backend_selection():
    """Example: Selecting and listing backends."""
    print("\n=== Backend Selection ===")

    # List available backends
    backends = gpt_rs.list_backends()
    print(f"Available backends: {backends}")

    # Set backend
    gpt_rs.set_backend("cpu")
    current = gpt_rs.get_backend()
    print(f"Current backend: {current}")


def example_functional_ops():
    """Example: Using functional operations."""
    print("\n=== Functional Operations ===")

    gpt_rs.set_backend("cpu")

    # Softmax
    x = gpt_rs.Tensor.from_numpy(np.array([[1.0, 2.0, 3.0]], dtype=np.float32))
    softmax_out = gpt_rs.functional.softmax_last_dim(x)
    print(f"Input: {x.numpy()}")
    print(f"Softmax: {softmax_out.numpy()}")
    print(f"Softmax sum: {softmax_out.numpy().sum():.6f} (should be 1.0)")

    # GELU
    x = gpt_rs.Tensor.from_numpy(np.array([[-2.0, -1.0, 0.0, 1.0, 2.0]], dtype=np.float32))
    gelu_out = gpt_rs.functional.gelu(x)
    print(f"\nGELU input: {x.numpy()}")
    print(f"GELU output: {gelu_out.numpy()}")

    # Matrix multiplication
    a = gpt_rs.Tensor.from_numpy(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
    b = gpt_rs.Tensor.from_numpy(np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32))
    matmul_out = gpt_rs.functional.matmul(a, b)
    print(f"\nA @ B =\n{matmul_out.numpy()}")


def example_layer_norm():
    """Example: Layer normalization."""
    print("\n=== Layer Normalization ===")

    gpt_rs.set_backend("cpu")

    # Create input
    x = gpt_rs.Tensor.from_numpy(np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32))

    # Create weight (gamma) and bias (beta)
    weight = gpt_rs.Tensor.from_numpy(np.ones(4, dtype=np.float32))
    bias = gpt_rs.Tensor.from_numpy(np.zeros(4, dtype=np.float32))

    # Apply layer norm
    output = gpt_rs.functional.layer_norm(x, weight, bias, eps=1e-5)

    print(f"Input: {x.numpy()}")
    print(f"Output: {output.numpy()}")
    print(f"Output mean: {output.numpy().mean():.6f} (should be ~0)")
    print(f"Output std: {output.numpy().std():.6f} (should be ~1)")


def example_linear_layer():
    """Example: Linear layer."""
    print("\n=== Linear Layer ===")

    gpt_rs.set_backend("cpu")

    # Input: batch_size=2, input_dim=4
    x = gpt_rs.Tensor.from_numpy(np.random.randn(2, 4).astype(np.float32))

    # Weight: input_dim=4, output_dim=3
    weight = gpt_rs.Tensor.from_numpy(np.random.randn(4, 3).astype(np.float32))

    # Bias: output_dim=3
    bias = gpt_rs.Tensor.from_numpy(np.random.randn(3).astype(np.float32))

    # Create layer
    linear = gpt_rs.nn.Linear(weight, bias)

    # Forward pass
    output = linear.forward(x)

    print(f"Input shape: {x.shape}")
    print(f"Weight shape: {weight.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output:\n{output.numpy()}")


def example_embedding():
    """Example: Embedding lookup."""
    print("\n=== Embedding Layer ===")

    gpt_rs.set_backend("cpu")

    # Embedding table: vocab_size=100, embed_dim=16
    np.random.seed(42)
    embedding_table = gpt_rs.Tensor.from_numpy(np.random.randn(100, 16).astype(np.float32))

    # Create embedding layer
    embedding = gpt_rs.nn.Embedding(embedding_table)

    # Token indices
    indices = gpt_rs.Tensor.from_numpy(np.array([0, 5, 10, 99], dtype=np.int32))

    # Lookup embeddings
    embeddings = embedding.forward(indices)

    print(f"Embedding table shape: {embedding_table.shape}")
    print(f"Indices: {indices.numpy()}")
    print(f"Output embeddings shape: {embeddings.shape}")
    print(f"First embedding vector: {embeddings.numpy()[0][:5]}... (first 5 dims)")


def example_feedforward():
    """Example: Feed-forward network."""
    print("\n=== Feed-Forward Network ===")

    gpt_rs.set_backend("cpu")

    np.random.seed(42)

    # Input: batch_size=2, dim=8
    x = gpt_rs.Tensor.from_numpy(np.random.randn(2, 8).astype(np.float32))

    # FFN expands to 4x hidden size then projects back
    hidden_size = 32
    w_in = gpt_rs.Tensor.from_numpy(np.random.randn(8, hidden_size).astype(np.float32))
    b_in = gpt_rs.Tensor.from_numpy(np.random.randn(hidden_size).astype(np.float32))
    w_out = gpt_rs.Tensor.from_numpy(np.random.randn(hidden_size, 8).astype(np.float32))
    b_out = gpt_rs.Tensor.from_numpy(np.random.randn(8).astype(np.float32))

    # Create FFN layer
    ffn = gpt_rs.nn.FeedForward(w_in, w_out, b_in, b_out)

    # Forward pass
    output = ffn.forward(x)

    print(f"Input shape: {x.shape}")
    print(f"Hidden size: {hidden_size}")
    print(f"Output shape: {output.shape}")
    print(f"Output:\n{output.numpy()}")


def example_combined_operations():
    """Example: Combining multiple operations."""
    print("\n=== Combined Operations ===")

    gpt_rs.set_backend("cpu")
    np.random.seed(42)

    # Simulate a simple transformer block component:
    # 1. Linear projection
    # 2. Layer norm
    # 3. GELU

    batch_size = 2
    seq_len = 4
    dim = 8

    # Input sequence
    x = gpt_rs.Tensor.from_numpy(np.random.randn(batch_size * seq_len, dim).astype(np.float32))

    # Linear layer
    weight = gpt_rs.Tensor.from_numpy(np.random.randn(dim, dim).astype(np.float32))
    bias = gpt_rs.Tensor.from_numpy(np.random.randn(dim).astype(np.float32))
    linear = gpt_rs.nn.Linear(weight, bias)

    # Apply linear
    x = linear.forward(x)
    print(f"After linear: shape={x.shape}")

    # Layer norm
    ln_weight = gpt_rs.Tensor.from_numpy(np.ones(dim, dtype=np.float32))
    ln_bias = gpt_rs.Tensor.from_numpy(np.zeros(dim, dtype=np.float32))
    x = gpt_rs.functional.layer_norm(x, ln_weight, ln_bias)
    print(f"After layer norm: shape={x.shape}")

    # GELU activation
    x = gpt_rs.functional.gelu(x)
    print(f"After GELU: shape={x.shape}")

    print(f"Final output sample:\n{x.numpy()[:2]}")


def main():
    """Run all examples."""
    print("=" * 60)
    print("gpt-rs Basic Usage Examples")
    print("=" * 60)

    example_tensor_creation()
    example_backend_selection()
    example_functional_ops()
    example_layer_norm()
    example_linear_layer()
    # example_embedding()  # FIXME: embedding_lookup has backend issues
    example_feedforward()
    example_combined_operations()

    print("\n" + "=" * 60)
    print("All examples complete (embedding skipped - known issue)!")
    print("=" * 60)


if __name__ == "__main__":
    main()
