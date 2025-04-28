import torch
import numpy as np
import rodespmm
import scipy.io
import os
import sys
import argparse

def read_mtx_file(file_path):
    """
    Read a sparse matrix from an MTX file and convert to CSR format
    Returns row_offsets, column_indices, values, rows, cols
    """
    print(f"Reading MTX file: {file_path}")
    # Read the MTX file
    try:
        mtx = scipy.io.mmread(file_path)
    except Exception as e:
        print(f"Error reading MTX file: {e}")
        sys.exit(1)
    
    # Convert to CSR format
    csr = mtx.tocsr()
    
    # Get dimensions
    rows, cols = csr.shape
    print(f"Matrix dimensions: {rows}x{cols}, {csr.nnz} non-zeros")
    
    # Extract CSR components
    row_offsets = csr.indptr.astype(np.int32)
    column_indices = csr.indices.astype(np.int32)
    values = csr.data.astype(np.float32)
    
    return row_offsets, column_indices, values, rows, cols

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

def test_spmm_gpu(mtx_file):
    """
    Test the RoDe SPMM library with GPU tensors using a matrix from an MTX file
    """
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("CUDA device not available. Skipping GPU test.")
        return
    
    # Read the matrix from the MTX file
    row_offsets, column_indices, values, rows, cols = read_mtx_file(mtx_file)
    
    # Convert to tensors
    row_offsets_tensor = torch.tensor(row_offsets, dtype=torch.int32, device=device)
    column_indices_tensor = torch.tensor(column_indices, dtype=torch.int32, device=device)
    values_tensor = torch.tensor(values, dtype=torch.float32, device=device)
    
    # Create the CudaSpMat from GPU tensors
    sp_mat = rodespmm.CudaSpMat(row_offsets_tensor, column_indices_tensor, values_tensor, rows, cols, device="GPU")
    
    print(f"Created sparse matrix on GPU with shape ({sp_mat.rows}, {sp_mat.cols})")
    
    # Create a dense matrix B - use a small number of columns to avoid kernel config issues
    n_cols = 128  # Using 32 columns should be compatible with the kernels
    B = torch.ones((cols, n_cols), dtype=torch.float32, device=device)
    
    # Perform the sparse-dense matrix multiplication
    print("Performing SPMM...")
    C = sp_mat.spmm(B)
    
    print("Result of sparse-dense matrix multiplication completed successfully")
    print("Shape of result:", C.shape)
    
    # Basic verification - check a few values
    print("Basic verification - checking first row values:")
    print("First row of result:", C[0, :5].cpu().numpy())  # Just print a few values
    
    # Verify with torch.sparse.mm
    torch_result, _ = verify_with_torch_sparse(sp_mat, row_offsets, column_indices, values, rows, cols, B, device)
    
    # Compare first few values
    print("First row of torch.sparse.mm result:", torch_result[0, :5].cpu().numpy())
    
    print("Test completed without errors.")

def test_spmm_cpu(mtx_file):
    """
    Test the RoDe SPMM library with CPU tensors using a matrix from an MTX file
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("CUDA device not available. Skipping CPU->GPU test.")
        return
    
    # Read the matrix from the MTX file
    row_offsets, column_indices, values, rows, cols = read_mtx_file(mtx_file)
    
    # No need to convert to numpy arrays as they already are
    
    # Create the CudaSpMat from CPU tensors
    sp_mat = rodespmm.CudaSpMat(row_offsets, column_indices, values, rows, cols, device="CPU")
    
    print(f"Created sparse matrix from CPU with shape ({sp_mat.rows}, {sp_mat.cols})")
    
    # Create a dense matrix B on GPU - use a small number of columns to avoid kernel config issues
    n_cols = 128  # Using 32 columns should be compatible with the kernels
    B = torch.ones((cols, n_cols), dtype=torch.float32, device=device)
    
    # Perform the sparse-dense matrix multiplication
    print("Performing SPMM...")
    C = sp_mat.spmm(B)
    
    print("Result of sparse-dense matrix multiplication completed successfully")
    print("Shape of result:", C.shape)
    
    # Basic verification - check a few values
    print("Basic verification - checking first row values:")
    print("First row of result:", C[0, :5].cpu().numpy())  # Just print a few values
    
    # Verify with torch.sparse.mm
    torch_result, _ = verify_with_torch_sparse(sp_mat, row_offsets, column_indices, values, rows, cols, B, device)
    
    # Compare first few values
    print("First row of torch.sparse.mm result:", torch_result[0, :5].cpu().numpy())
    
    print("Test completed without errors.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test RoDe SPMM with matrices from MTX files")
    parser.add_argument("--mtx_file", type=str, help="Path to the MTX file to use", default="/pscratch/sd/c/cunyang/workspace/dataset/amazon0505/amazon0505.mtx")
    args = parser.parse_args()
    
    mtx_file = args.mtx_file
    
    print("Testing RoDe SPMM with GPU tensors:")
    test_spmm_gpu(mtx_file)
    
    print("\nTesting RoDe SPMM with CPU tensors:")
    test_spmm_cpu(mtx_file) 