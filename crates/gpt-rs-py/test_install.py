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


def test_runtime_api():
    """Test that runtime APIs exist and basic errors surface correctly."""
    try:
        import gpt_rs

        assert hasattr(gpt_rs, "load_model")
        assert hasattr(gpt_rs, "LoadedModel")
        assert hasattr(gpt_rs, "Tokenizer")

        # We don't ship checkpoints with the Python package; just validate error plumbing.
        try:
            _ = gpt_rs.load_model("this_file_should_not_exist.bin")
            print("✗ Expected load_model() to fail for a missing file")
            return False
        except Exception:
            pass

        info = gpt_rs.version_info()
        print(f"✓ version_info: {info}")

        feats = gpt_rs.backend_features()
        print(f"✓ backend_features: {feats}")

        return True
    except Exception as e:
        print(f"✗ Runtime API test failed: {e}")
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
        ("Runtime API", test_runtime_api),
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
