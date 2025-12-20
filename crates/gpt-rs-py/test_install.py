#!/usr/bin/env python3
"""
Quick installation test for gpt-rs-py.

This script verifies that gpt-rs-py is correctly installed and functional.

Usage:
    python test_install.py
"""

import sys


def test_import():
    """Test that gpt_rs can be imported."""
    try:
        import gpt_rs  # noqa: F401

        print("✓ Successfully imported gpt_rs")
        return True
    except ImportError as e:
        print(f"✗ Failed to import gpt_rs: {e}")
        print("\nTo install, run:")
        print("  cd crates/gpt-rs-py")
        print("  maturin develop")
        return False


def test_backend():
    """Test backend functionality."""
    try:
        import gpt_rs

        backends = gpt_rs.list_backends()
        print(f"✓ Available backends: {backends}")

        if not backends:
            print("✗ No backends available")
            return False

        gpt_rs.set_backend(backends[0])
        current = gpt_rs.get_backend()
        print(f"✓ Set backend to: {current}")

        return True
    except Exception as e:
        print(f"✗ Backend test failed: {e}")
        return False


def test_tensor_ops():
    """Test basic tensor operations."""
    try:
        import gpt_rs
        import numpy as np

        gpt_rs.set_backend("cpu")

        # Create tensor
        x = gpt_rs.Tensor.from_numpy(np.array([[1.0, 2.0]], dtype=np.float32))
        print(f"✓ Created tensor with shape: {x.shape}")

        # Softmax
        out = gpt_rs.functional.softmax_last_dim(x)
        result = out.numpy()
        print(f"✓ Softmax output: {result}")

        # Verify softmax sums to 1
        if abs(result.sum() - 1.0) < 1e-5:
            print("✓ Softmax verification passed")
            return True
        else:
            print(f"✗ Softmax sum is {result.sum()}, expected 1.0")
            return False

    except Exception as e:
        print(f"✗ Tensor operations test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_nn_layer():
    """Test NN layer."""
    try:
        import gpt_rs
        import numpy as np

        gpt_rs.set_backend("cpu")

        # Create linear layer
        np.random.seed(42)
        weight = gpt_rs.Tensor.from_numpy(np.random.randn(2, 3).astype(np.float32))
        bias = gpt_rs.Tensor.from_numpy(np.random.randn(3).astype(np.float32))
        linear = gpt_rs.nn.Linear(weight, bias)

        # Forward pass
        x = gpt_rs.Tensor.from_numpy(np.random.randn(1, 2).astype(np.float32))
        output = linear.forward(x)

        print(f"✓ Linear layer forward pass: input {x.shape} -> output {output.shape}")

        return True
    except Exception as e:
        print(f"✗ NN layer test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run all installation tests."""
    print("=" * 60)
    print("gpt-rs-py Installation Test")
    print("=" * 60)
    print()

    tests = [
        ("Import", test_import),
        ("Backend", test_backend),
        ("Tensor Operations", test_tensor_ops),
        ("NN Layer", test_nn_layer),
    ]

    results = []
    for name, test_func in tests:
        print(f"\nTest: {name}")
        print("-" * 60)
        success = test_func()
        results.append((name, success))
        print()

    print("=" * 60)
    print("Test Summary")
    print("=" * 60)

    all_passed = True
    for name, success in results:
        status = "PASS" if success else "FAIL"
        symbol = "✓" if success else "✗"
        print(f"{symbol} {name:20s} {status}")
        if not success:
            all_passed = False

    print()
    if all_passed:
        print("✓ All tests passed! gpt-rs-py is correctly installed.")
        return 0
    else:
        print("✗ Some tests failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
