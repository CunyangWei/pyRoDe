# RoDe SpMM Python Bindings

This directory contains Python bindings for the RoDe SpMM (Sparse Matrix-Matrix Multiplication) library, which allows efficient sparse-dense matrix multiplication on CUDA devices.

## Building the Python Module

### Prerequisites

- CUDA toolkit (10.0 or newer)
- CMake (3.12 or newer)
- Python 3.6 or newer
- PyTorch (for tensor operations)
- pybind11

### Build Instructions

1. Make sure pybind11 is installed. You can install it using:

```bash
pip install pybind11
```

2. Build the library and its Python bindings:

```bash
mkdir build
cd build
cmake ..
make -j
```

3. The built module will be in the build directory. You can add this to your Python path or copy it to your project directory.

## Usage

### Basic Usage

```python
import torch
import rodespmm

# Create a CSR sparse matrix
rows = 1000
cols = 1000
nonzeros = 10000

# Create CSR format tensors (can be on CPU or GPU)
row_offsets = torch.tensor([...], dtype=torch.int32, device="cuda")
column_indices = torch.tensor([...], dtype=torch.int32, device="cuda")
values = torch.tensor([...], dtype=torch.float32, device="cuda")

# Create the sparse matrix object
# If your tensors are on GPU, use device="GPU"
# If your tensors are on CPU, use device="CPU"
sp_mat = rodespmm.CudaSpMat(
    row_offsets, column_indices, values, rows, cols, seg_length=512, device="GPU"
)

# Create a dense matrix B on GPU
B = torch.randn((cols, 128), dtype=torch.float32, device="cuda")

# Perform sparse-dense matrix multiplication
C = sp_mat.spmm(B)  # C is a torch.Tensor on GPU
```

### API Reference

#### `CudaSpMat` Class

Constructor:
```python
spmat = rodespmm.CudaSpMat(
    row_offsets,    # CSR row offsets (torch.Tensor or numpy.ndarray)
    column_indices, # CSR column indices (torch.Tensor or numpy.ndarray)
    values,         # CSR values (torch.Tensor or numpy.ndarray)
    rows,           # Number of rows in the sparse matrix
    cols,           # Number of columns in the sparse matrix
    seg_length=512, # Segment length for RoDe SpMM algorithm
    device="GPU"    # "GPU" if input tensors are on GPU, "CPU" if on CPU
)
```

Methods:
- `spmm(B)`: Perform sparse-dense matrix multiplication with dense matrix B

Properties:
- `rows`: Number of rows in the sparse matrix
- `cols`: Number of columns in the sparse matrix

## Notes on Zero-Copy

The Python bindings use the CUDA array interface (`__cuda_array_interface__`) to access the GPU memory directly without copying data. However, for the sparse matrix creation, if the device is "GPU", the data still needs to be copied to the host first for the `RowDivide2Segment` operation, which can only run on the CPU. The final sparse matrix data is always stored on the GPU for efficient computation. 