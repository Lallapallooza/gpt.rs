#!/usr/bin/env python3
"""
Comparison between gpt-rs and PyTorch for basic operations.

This script demonstrates functional parity between gpt-rs-py and PyTorch
for common neural network operations.

Requirements:
    pip install torch numpy gpt-rs

Usage:
    python examples/comparison_torch.py
"""

import numpy as np
import torch
import torch.nn.functional as F

try:
    import gpt_rs
except ImportError:
    print("Error: gpt_rs not installed.")
    print("\nTo install:")
    print("  pip install maturin")
    print("  cd crates/gpt-rs-py")
    print("  maturin develop")
    exit(1)


def compare_tensors(gpt_result, torch_result, name, rtol=1e-5, atol=1e-5):
    """Compare gpt-rs and PyTorch results."""
    gpt_numpy = gpt_result.numpy()
    torch_numpy = torch_result.detach().cpu().numpy()

    close = np.allclose(gpt_numpy, torch_numpy, rtol=rtol, atol=atol)
    max_diff = np.max(np.abs(gpt_numpy - torch_numpy))

    status = "✓" if close else "✗"
    print(f"{status} {name:30s} max_diff={max_diff:.6e}")

    if not close:
        print(f"  GPT-RS shape: {gpt_numpy.shape}, PyTorch shape: {torch_numpy.shape}")
        print(f"  GPT-RS sample: {gpt_numpy.flat[:5]}")
        print(f"  PyTorch sample: {torch_numpy.flat[:5]}")

    return close


def test_softmax():
    """Test softmax operation."""
    print("\n=== Softmax ===")

    # Create input
    np.random.seed(42)
    x_np = np.random.randn(2, 4).astype(np.float32)

    # gpt-rs
    gpt_rs.set_backend("cpu")
    x_gpt = gpt_rs.Tensor.from_numpy(x_np)
    out_gpt = gpt_rs.functional.softmax_last_dim(x_gpt)

    # PyTorch
    x_torch = torch.from_numpy(x_np)
    out_torch = F.softmax(x_torch, dim=-1)

    compare_tensors(out_gpt, out_torch, "softmax_last_dim")


def test_gelu():
    """Test GELU activation."""
    print("\n=== GELU ===")

    np.random.seed(42)
    x_np = np.random.randn(3, 5).astype(np.float32)

    # gpt-rs
    gpt_rs.set_backend("cpu")
    x_gpt = gpt_rs.Tensor.from_numpy(x_np)
    out_gpt = gpt_rs.functional.gelu(x_gpt)

    # PyTorch
    x_torch = torch.from_numpy(x_np)
    out_torch = F.gelu(x_torch)

    compare_tensors(out_gpt, out_torch, "gelu", rtol=1e-4, atol=1e-4)


def test_layer_norm():
    """Test layer normalization."""
    print("\n=== Layer Normalization ===")

    np.random.seed(42)
    x_np = np.random.randn(2, 8).astype(np.float32)
    weight_np = np.ones(8, dtype=np.float32)
    bias_np = np.zeros(8, dtype=np.float32)
    eps = 1e-5

    # gpt-rs
    gpt_rs.set_backend("cpu")
    x_gpt = gpt_rs.Tensor.from_numpy(x_np)
    weight_gpt = gpt_rs.Tensor.from_numpy(weight_np)
    bias_gpt = gpt_rs.Tensor.from_numpy(bias_np)
    out_gpt = gpt_rs.functional.layer_norm(x_gpt, weight_gpt, bias_gpt, eps)

    # PyTorch
    x_torch = torch.from_numpy(x_np)
    weight_torch = torch.from_numpy(weight_np)
    bias_torch = torch.from_numpy(bias_np)
    out_torch = F.layer_norm(x_torch, (8,), weight_torch, bias_torch, eps)

    compare_tensors(out_gpt, out_torch, "layer_norm", rtol=1e-4, atol=1e-4)


def test_matmul():
    """Test matrix multiplication."""
    print("\n=== Matrix Multiplication ===")

    np.random.seed(42)
    a_np = np.random.randn(3, 4).astype(np.float32)
    b_np = np.random.randn(4, 5).astype(np.float32)

    # gpt-rs
    gpt_rs.set_backend("cpu")
    a_gpt = gpt_rs.Tensor.from_numpy(a_np)
    b_gpt = gpt_rs.Tensor.from_numpy(b_np)
    out_gpt = gpt_rs.functional.matmul(a_gpt, b_gpt)

    # PyTorch
    a_torch = torch.from_numpy(a_np)
    b_torch = torch.from_numpy(b_np)
    out_torch = torch.matmul(a_torch, b_torch)

    compare_tensors(out_gpt, out_torch, "matmul", rtol=1e-4, atol=1e-4)


def test_linear_layer():
    """Test Linear layer."""
    print("\n=== Linear Layer ===")

    np.random.seed(42)
    x_np = np.random.randn(2, 4).astype(np.float32)
    weight_np = np.random.randn(4, 6).astype(np.float32)
    bias_np = np.random.randn(6).astype(np.float32)

    # gpt-rs
    gpt_rs.set_backend("cpu")
    x_gpt = gpt_rs.Tensor.from_numpy(x_np)
    weight_gpt = gpt_rs.Tensor.from_numpy(weight_np)
    bias_gpt = gpt_rs.Tensor.from_numpy(bias_np)
    linear_gpt = gpt_rs.nn.Linear(weight_gpt, bias_gpt)
    out_gpt = linear_gpt.forward(x_gpt)

    # PyTorch
    x_torch = torch.from_numpy(x_np)
    # Note: PyTorch Linear uses transposed weight
    linear_torch = torch.nn.Linear(4, 6, bias=True)
    linear_torch.weight.data = torch.from_numpy(weight_np.T)
    linear_torch.bias.data = torch.from_numpy(bias_np)
    out_torch = linear_torch(x_torch)

    compare_tensors(out_gpt, out_torch, "Linear layer", rtol=1e-4, atol=1e-4)


def test_embedding():
    """Test Embedding layer."""
    print("\n=== Embedding Layer ===")

    np.random.seed(42)
    # Embedding table: vocab_size=10, embed_dim=8
    table_np = np.random.randn(10, 8).astype(np.float32)
    indices_np = np.array([0, 2, 5, 1], dtype=np.int32)

    # gpt-rs
    gpt_rs.set_backend("cpu")
    table_gpt = gpt_rs.Tensor.from_numpy(table_np)
    indices_gpt = gpt_rs.Tensor.from_numpy(indices_np)
    out_gpt = gpt_rs.functional.embedding_lookup(table_gpt, indices_gpt)

    # PyTorch
    table_torch = torch.from_numpy(table_np)
    indices_torch = torch.from_numpy(indices_np).long()
    out_torch = F.embedding(indices_torch, table_torch)

    compare_tensors(out_gpt, out_torch, "embedding_lookup", rtol=1e-5, atol=1e-5)


def test_feedforward():
    """Test FeedForward layer (two-layer MLP with GELU)."""
    print("\n=== FeedForward Layer ===")

    np.random.seed(42)
    x_np = np.random.randn(2, 4).astype(np.float32)
    w_in_np = np.random.randn(4, 16).astype(np.float32)
    b_in_np = np.random.randn(16).astype(np.float32)
    w_out_np = np.random.randn(16, 4).astype(np.float32)
    b_out_np = np.random.randn(4).astype(np.float32)

    # gpt-rs
    gpt_rs.set_backend("cpu")
    x_gpt = gpt_rs.Tensor.from_numpy(x_np)
    w_in_gpt = gpt_rs.Tensor.from_numpy(w_in_np)
    b_in_gpt = gpt_rs.Tensor.from_numpy(b_in_np)
    w_out_gpt = gpt_rs.Tensor.from_numpy(w_out_np)
    b_out_gpt = gpt_rs.Tensor.from_numpy(b_out_np)
    ffn_gpt = gpt_rs.nn.FeedForward(w_in_gpt, w_out_gpt, b_in_gpt, b_out_gpt)
    out_gpt = ffn_gpt.forward(x_gpt)

    # PyTorch equivalent
    x_torch = torch.from_numpy(x_np)
    fc1 = torch.nn.Linear(4, 16, bias=True)
    fc1.weight.data = torch.from_numpy(w_in_np.T)
    fc1.bias.data = torch.from_numpy(b_in_np)
    fc2 = torch.nn.Linear(16, 4, bias=True)
    fc2.weight.data = torch.from_numpy(w_out_np.T)
    fc2.bias.data = torch.from_numpy(b_out_np)

    hidden = F.gelu(fc1(x_torch))
    out_torch = fc2(hidden)

    compare_tensors(out_gpt, out_torch, "FeedForward layer", rtol=1e-4, atol=1e-4)


def test_backend_switching():
    """Test switching between backends."""
    print("\n=== Backend Switching ===")

    backends = gpt_rs.list_backends()
    print(f"Available backends: {backends}")

    for backend in backends:
        gpt_rs.set_backend(backend)
        current = gpt_rs.get_backend()
        print(f"Set to '{backend}', current: '{current}'")

        # Quick test
        x_np = np.array([[1.0, 2.0]], dtype=np.float32)
        x = gpt_rs.Tensor.from_numpy(x_np)
        out = gpt_rs.functional.softmax_last_dim(x)
        result = out.numpy()
        print(f"  Softmax result: {result}")


def main():
    """Run all comparison tests."""
    print("=" * 60)
    print("gpt-rs vs PyTorch Comparison")
    print("=" * 60)

    # Run tests
    test_softmax()
    test_gelu()
    test_layer_norm()
    test_matmul()
    test_linear_layer()
    test_embedding()
    test_feedforward()
    test_backend_switching()

    print("\n" + "=" * 60)
    print("Comparison complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
