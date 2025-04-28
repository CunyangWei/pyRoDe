import torch
import numpy as np
import sys

print(f"Python version: {sys.version}")
print("Importing rodespmm module...")
import rodespmm
print("Module imported successfully!")

# Try to create a minimal sparse matrix
print("Creating a minimal sparse matrix for testing...")
try:
    # CPU data
    rows = 2
    cols = 2
    row_offsets = np.array([0, 1, 2], dtype=np.int32)
    column_indices = np.array([0, 1], dtype=np.int32)
    values = np.array([1.0, 1.0], dtype=np.float32)
    
    print("Creating CudaSpMat with CPU data...")
    sp_mat = rodespmm.CudaSpMat(row_offsets, column_indices, values, rows, cols, device="CPU")
    print(f"Successfully created CudaSpMat: rows={sp_mat.rows}, cols={sp_mat.cols}")
    
    # Only test GPU if available
    if torch.cuda.is_available():
        print("\nCUDA is available! Testing with GPU data...")
        # GPU data
        device = torch.device("cuda")
        d_row_offsets = torch.tensor([0, 1, 2], dtype=torch.int32, device=device)
        d_column_indices = torch.tensor([0, 1], dtype=torch.int32, device=device)
        d_values = torch.tensor([1.0, 1.0], dtype=torch.float32, device=device)
        
        print("Creating CudaSpMat with GPU data...")
        sp_mat_gpu = rodespmm.CudaSpMat(d_row_offsets, d_column_indices, d_values, rows, cols, device="GPU")
        print(f"Successfully created CudaSpMat on GPU: rows={sp_mat_gpu.rows}, cols={sp_mat_gpu.cols}")
    else:
        print("CUDA is not available. Skipping GPU test.")
        
except Exception as e:
    print(f"Error: {e}") 