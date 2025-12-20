# gpt-rs-py: Python Bindings for gpt-rs

Pure Rust GPT implementation with Python bindings via PyO3. Provides high-performance tensor operations and neural network layers without external ML framework dependencies.

## Quickstart

```bash
# 1. Install maturin
pip install maturin

# 2. Build and install (from this directory: crates/gpt-rs-py)
maturin develop

# 3. Test it works
python test_install.py

# 4. Try examples
python examples/basic_usage.py
```

## Features

- **Dynamic Backend Selection**: Choose between CPU and Faer backends at runtime
- **Zero-Copy Interop**: Efficient NumPy array integration
- **Functional Operations**: softmax, GELU, layer norm, matmul, embedding lookup, and more
- **NN Layers**: Embedding, Linear, LayerNorm, FeedForward
- **Type-Safe**: Full Rust implementation with Python ergonomics
- **Intuitive API**: Accepts natural rank-1 indices for embedding lookup (e.g., `[0, 1, 2]`)

## Installation

### Prerequisites

```bash
# Install maturin (PyO3 build tool)
pip install maturin

# Or with uv (faster)
uv pip install maturin
```

### Development Installation (Recommended for now)

```bash
# From the gpt-rs root directory
cd crates/gpt-rs-py

# Install in development mode
maturin develop

# Or with the faer backend
maturin develop --features faer

# Or with the C backend (requires conversion-c)
maturin develop --features conversion-c

# Or release mode for better performance
maturin develop --release
```

### From Wheel (Future)

```bash
# Build the wheel
cd crates/gpt-rs-py
maturin build --release

# Install the wheel (replace with actual version/platform)
pip install ../../target/wheels/gpt_rs-*.whl
```

**Note**: Wheel distribution via PyPI is not yet available. Use development installation for now.

### Quick Test

```bash
# After installation, verify it works
python test_install.py

# Run basic examples
python examples/basic_usage.py

# Compare with PyTorch (requires: pip install torch)
python examples/comparison_torch.py
```

## Quick Start

```python
import numpy as np
import gpt_rs

# Set backend (cpu, faer, or c)
gpt_rs.set_backend("cpu")

# Create tensors from NumPy arrays
x = gpt_rs.Tensor.from_numpy(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))

# Use functional operations
softmax_out = gpt_rs.functional.softmax_last_dim(x)
gelu_out = gpt_rs.functional.gelu(x)

# Create NN layers
weight = gpt_rs.Tensor.from_numpy(np.random.randn(2, 3).astype(np.float32))
bias = gpt_rs.Tensor.from_numpy(np.random.randn(3).astype(np.float32))

linear = gpt_rs.nn.Linear(weight, bias)
output = linear.forward(x)

# Convert back to NumPy
result = output.numpy()
print(result.shape)  # (2, 3)
print(output.shape)  # [2, 3] - Note: property, not method!
```

## API Reference

### Backend Management

```python
# List available backends
backends = gpt_rs.list_backends()  # ['cpu', 'cpu-portable', 'faer', 'c']

# Set active backend
gpt_rs.set_backend("cpu")

# Get current backend
current = gpt_rs.get_backend()
```

### Tensor Operations

```python
# Create from NumPy (always copies data)
tensor = gpt_rs.Tensor.from_numpy(np_array)

# Convert to NumPy
np_array = tensor.numpy()

# Properties (note: these are properties, not methods!)
shape = tensor.shape      # list of dimensions [2, 3, 4]
dtype = tensor.dtype      # 'f32', 'i32', etc.
backend = tensor.backend  # backend name
```

### Functional Operations

All operations in `gpt_rs.functional`:

- `softmax_last_dim(tensor)` - Softmax activation
- `gelu(tensor)` - GELU activation
- `layer_norm(tensor, weight, bias, eps=1e-5)` - Layer normalization
- `matmul(a, b)` - Matrix multiplication
- `add_bias(tensor, bias)` - Add bias to tensor
- `embedding_lookup(table, indices)` - Embedding table lookup

### NN Layers

All layers in `gpt_rs.nn`:

**Embedding**
```python
embedding = gpt_rs.nn.Embedding(weight)
output = embedding.forward(indices)
```

**Linear**
```python
linear = gpt_rs.nn.Linear(weight, bias=None)
output = linear.forward(x)
```

**LayerNorm**
```python
layer_norm = gpt_rs.nn.LayerNorm(weight, bias, eps=1e-5)
output = layer_norm.forward(x)
```

**FeedForward**
```python
ffn = gpt_rs.nn.FeedForward(w_in, w_out, b_in=None, b_out=None)
output = ffn.forward(x)
```

## Examples

See `examples/` directory:
- `comparison_torch.py` - Side-by-side comparison with PyTorch
- `basic_usage.py` - Basic tensor operations
- `nn_layers.py` - Neural network layer usage

## Development

### Building

```bash
# Debug build
maturin develop

# Release build
maturin build --release

# With features
maturin develop --features faer
maturin develop --features conversion-c
```

### Testing

```bash
# Run Rust tests
cargo test -p gpt-rs-py

# Run Python tests (once implemented)
pytest tests/
```

## Performance

gpt-rs-py is designed for:
- **CPU inference**: Optimized for CPU-only environments
- **Research**: Experimenting with transformer architectures
- **Education**: Understanding GPT internals without framework complexity

For production GPU workloads, consider PyTorch or JAX.

## Architecture Notes

The Python bindings use the backend registry system for dynamic backend selection. While the backend is selected at runtime using `gpt_rs.set_backend()`, the functional operations require compile-time knowledge of backend types. The bindings handle this by:

1. Using `gpt_rs.backend.registry` for type-erased backend storage
2. Dispatching to typed operations based on the currently selected backend name
3. Supporting "cpu", "faer" (when compiled with `--features faer`), and "c" (when compiled with `--features conversion-c`)

This approach provides runtime flexibility while maintaining type safety and avoiding hardcoded backend assumptions throughout the codebase.

## License

Same as parent gpt-rs project.
