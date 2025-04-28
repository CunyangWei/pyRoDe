import torch
import numpy as np
import rodespmm
import scipy.io
import os
import sys
import argparse
from memory_profiler import memory_usage
import time
import cupy as cp
device="cuda"

def csr_to_torch_sparse(row_offsets, column_indices, values, rows, cols, device):
    """
    Convert CSR format to torch sparse tensor format
    """
    # Convert row_offsets to row indices
    row_indices = []
    for i in range(len(row_offsets) - 1):
        row_indices.extend([i] * (row_offsets[i + 1] - row_offsets[i]))
    
    # Create indices tensor (2, nnz)
    indices = torch.stack([
        torch.tensor(row_indices, dtype=torch.long, device=device),
        torch.tensor(column_indices, dtype=torch.long, device=device)
    ])
    
    # Create values tensor
    values_tensor = torch.tensor(values, dtype=torch.float32, device=device)
    
    # Create sparse tensor
    return torch.sparse_coo_tensor(indices, values_tensor, (rows, cols))

def verify_with_torch_sparse(sp_mat, row_offsets, column_indices, values, rows, cols, B, device):
    """
    Verify RoDe SPMM results with torch.sparse.mm
    """
    print("Verifying with torch.sparse.mm...")
    
    # Convert CSR to torch sparse format
    torch_sparse = csr_to_torch_sparse(row_offsets, column_indices, values, rows, cols, device)
    
    # Run torch.sparse.mm
    torch_result = torch.sparse.mm(torch_sparse, B)
    
    # Run rodespmm
    rode_result = sp_mat.spmm(B)
    
    
    
    # Compare results
    max_diff = torch.max(torch.abs(torch_result - rode_result)).item()
    avg_diff = torch.mean(torch.abs(torch_result - rode_result)).item()
    
    print(f"Verification results:")
    print(f"  Max absolute difference: {max_diff:.5e}")
    print(f"  Average absolute difference: {avg_diff:.5e}")
    
    # Check if differences are within acceptable tolerance
    tolerance = 1e-5
    if max_diff <= tolerance:
        print(f"✅ Verification PASSED! (max difference within tolerance of {tolerance})")
    else:
        print(f"❌ Verification FAILED! (max difference exceeds tolerance of {tolerance})")
    
    return torch_result, rode_result

def test_spmm_gpu():
    
    # Load tensors from .pt files
    dataset_path = "/pscratch/sd/c/cunyang/RoDe/dataset"
    
    # Load the sparse matrix components
    row_offsets_tensor = torch.load(os.path.join(dataset_path, "shard_0_2449029_1224515_row_offsets.pt"))
    column_indices_tensor = torch.load(os.path.join(dataset_path, "shard_0_2449029_1224515_column_indices.pt"))
    values_tensor = torch.load(os.path.join(dataset_path, "shard_0_2449029_1224515_values.pt"))
    
    # Move tensors to the device
    row_offsets_tensor = row_offsets_tensor.to(device=device, dtype=torch.int32)
    column_indices_tensor = column_indices_tensor.to(device=device, dtype=torch.int32)
    values_tensor = values_tensor.to(device=device, dtype=torch.float32)
    
    # Extract dimensions from filename
    rows, cols = 2449029, 1224515
    
    # Convert tensors to numpy arrays for verification functions
    row_offsets = row_offsets_tensor.cpu().numpy()
    column_indices = column_indices_tensor.cpu().numpy()
    values = values_tensor.cpu().numpy()
    
    # Convert to tensors
    # row_offsets_tensor = torch.tensor(row_offsets, dtype=torch.int32, device=device)
    # column_indices_tensor = torch.tensor(column_indices, dtype=torch.int32, device=device)
    # values_tensor = torch.tensor(values, dtype=torch.float32, device=device)
    # Create the CudaSpMat from GPU tensors
    sp_mat = rodespmm.CudaSpMat(row_offsets_tensor, column_indices_tensor, values_tensor, rows, cols, device="GPU")
    
    print(f"Created sparse matrix on GPU with shape ({sp_mat.rows}, {sp_mat.cols})")
    
    # Create a dense matrix B - use a small number of columns to avoid kernel config issues
    n_cols = 128  # Using 32 columns should be compatible with the kernels
    # B = torch.ones((cols, n_cols), dtype=torch.float32, device=device, requires_grad=True)
    B = torch.ones((cols, n_cols), dtype=torch.float32, device=device)
    print("B: ", B)
    # B_detached = B.detach()
    # B = torch.ones((1, 1), dtype=torch.float32, device=device)
    # B_cupy = cp.from_dlpack(torch.utils.dlpack.to_dlpack(B))
    # Perform the sparse-dense matrix multiplication
    print("Performing SPMM...")
    print("B.type: ", B.type())
    # exit()
    C = sp_mat.spmm(B)
    
    print("Result of sparse-dense matrix multiplication completed successfully")
    print("Shape of result:", C.shape)
    
    # Basic verification - check a few values
    # print("Basic verification - checking first row values:")
    # print("First row of result:", C[0, :5].cpu().numpy())  # Just print a few values
    
    # Verify with torch.sparse.mm
    torch_result, _ = verify_with_torch_sparse(sp_mat, row_offsets, column_indices, values, rows, cols, B, device)
    
    # Compare first few values
    # print("First row of torch.sparse.mm result:", torch_result[0, :5].cpu().numpy())
    
    print("Test completed without errors.")

if __name__ == "__main__":
    test_spmm_gpu()
