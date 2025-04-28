#include <algorithm>
#include <cstdint>
#include <cstring>
#include <numeric>
// #define GLOG_USE_GLOG_EXPORT
// #include <glog/logging.h>
#include "matrix_utils.h"
#include <iostream>

#define CHECK_GE(val, ref) do {                           \
  if (!((val) >= (ref))) {                              \
      fprintf(stderr, "CHECK_GE failed: %s >= %s, "     \
              "but %d vs %d\n", #val, #ref, (val), (ref)); \
      abort();                                        \
  }                                                   \
} while (0)

#define CHECK_EQ(val, ref) do {                           \
  if (!((val) == (ref))) {                              \
      fprintf(stderr, "CHECK_EQ failed: %s == %s, "     \
              "but %d vs %d\n", #val, #ref, (val), (ref)); \
      abort();                                        \
  }                                                   \
} while (0)
#define METHOD_V1

namespace SPC {

namespace {

/**
 * @brief Helper to convert float data to half precision data.
 */
__global__ void ConvertKernel(const float *in_f, half2 *out, int n) {
  const float2 *in = reinterpret_cast<const float2 *>(in_f);
  n /= 2;
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx >= n) return;
  out[idx] = __float22half2_rn(in[idx]);
}

__global__ void ConvertKernel(const int *in_i, short2 *out, int n) {
  const int2 *in = reinterpret_cast<const int2 *>(in_i);
  n /= 2;
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx >= n) return;
  int2 a = in[idx];
  short2 b;
  b.x = static_cast<short>(a.x);
  b.y = static_cast<short>(a.y);
  out[idx] = b;
}

__global__ void ConvertKernel(const half2 *in, float *out_f, int n) {
  float2 *out = reinterpret_cast<float2 *>(out_f);
  n /= 2;
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx >= n) return;
  out[idx] = __half22float2(in[idx]);
}

__global__ void ConvertKernel(const short2 *in, int *out_i, int n) {
  int2 *out = reinterpret_cast<int2 *>(out_i);
  n /= 2;
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx >= n) return;
  short2 a = in[idx];
  int2 b;
  b.x = static_cast<int>(a.x);
  b.y = static_cast<int>(a.y);
  out[idx] = b;
}

__global__ void ConvertKernel(const float *in, double *out_i, int n) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx >= n) return;
  out_i[idx] = (double) in[idx];
}

__global__ void ConvertKernel(const double *in, float *out_i, int n) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx >= n) return;
  out_i[idx] = (float) in[idx];
}

void PadSparseMatrix(const std::vector<int> &row_offsets,
                     const std::vector<float> &values,
                     const std::vector<int> &column_indices, int row_padding,
                     std::vector<int> *row_offsets_out,
                     std::vector<float> *values_out,
                     std::vector<int> *column_indices_out) {
  // CHECK_GE(row_padding, 0) << "Row padding factor must be greater than zero.";
  CHECK_GE(row_padding, 0);
  if (row_padding < 2) {
    // For row padding to the nearest 1 element, copy the input to the
    // output and return early. We also execute this code path for
    // `row_padding` == 0, which indicates no padding is to be added.
    row_offsets_out->assign(row_offsets.begin(), row_offsets.end());
    values_out->assign(values.begin(), values.end());
    column_indices_out->assign(column_indices.begin(), column_indices.end());
    return;
  }
  row_offsets_out->push_back(0);

  int offset = 0;
  for (int i = 0; i < row_offsets.size() - 1; ++i) {
    // Copy the existing values and column indices for this row to
    // the output.
    int row_length = row_offsets[i + 1] - row_offsets[i];
    values_out->resize(values_out->size() + row_length);
    column_indices_out->resize(column_indices_out->size() + row_length);
    std::copy(values.begin() + row_offsets[i],
              values.begin() + row_offsets[i + 1],
              values_out->begin() + offset);
    std::copy(column_indices.begin() + row_offsets[i],
              column_indices.begin() + row_offsets[i + 1],
              column_indices_out->begin() + offset);
    offset += row_length;

    // Calculate the number of zeros that need to be inserted in
    // this row to reach the desired padding factor.
    int residue = offset % row_padding;
    int to_add = (row_padding - residue) % row_padding;
    for (; to_add > 0; --to_add) {
      values_out->push_back(0.0);

      // NOTE: When we pad with zeros the column index that we assign
      // the phantom zero needs to be a valid column index s.t. we
      // don't index out-of-range into the dense rhs matrix when
      // computing spmm. Here we set all padding column-offsets to
      // the same column as the final non-padding weight in the row.
      column_indices_out->push_back(column_indices_out->back());
      ++offset;
    }
    row_offsets_out->push_back(offset);
  }
}

}  // namespace

template <typename In, typename Out>
cudaError_t Convert(const In *in, Out *out, int n) {
  if (n == 0) return cudaSuccess;
  // CHECK_EQ(n % 2, 0) << "Number of elements must be multiple of 2.";
  CHECK_EQ(n % 2, 0);

  int threads_per_block = 64;
  int blocks_per_grid = (n + threads_per_block - 1) / threads_per_block;
  ConvertKernel<<<blocks_per_grid, threads_per_block, 0, 0>>>(in, out, n);
  return cudaGetLastError();
}

template<>
cudaError_t Convert(const float *in, float *out, int n) {
  return cudaMemcpy(out, in, n * sizeof(float), cudaMemcpyDeviceToDevice);
}

template<>
cudaError_t Convert(const int *in, int *out, int n) {
  return cudaMemcpy(out, in, n * sizeof(int), cudaMemcpyDeviceToDevice);
}

template<>
cudaError_t Convert(const float *in, double *out, int n) {
  if (n == 0) return cudaSuccess;

  int threads_per_block = 64;
  int blocks_per_grid = (n + threads_per_block - 1) / threads_per_block;
  ConvertKernel<<<blocks_per_grid, threads_per_block, 0, 0>>>(in, out, n);
  return cudaGetLastError();
}


void IdentityRowSwizzle(int rows, const int * /* unused */, int *row_indices) {
  std::iota(row_indices, row_indices + rows, 0);
}

void SortedRowSwizzle(int rows, const int *row_offsets, int *row_indices) {
  // Create our unsorted row indices.
  std::vector<int> swizzle_staging(rows);
  std::iota(swizzle_staging.begin(), swizzle_staging.end(), 0);

  // Argsort the row indices based on their length.
  std::sort(swizzle_staging.begin(), swizzle_staging.end(),
            [&row_offsets](int idx_a, int idx_b) {
              int length_a = row_offsets[idx_a + 1] - row_offsets[idx_a];
              int length_b = row_offsets[idx_b + 1] - row_offsets[idx_b];
              return length_a > length_b;
            });

  // Copy the ordered row indices to the output.
  std::memcpy(row_indices, swizzle_staging.data(), sizeof(int) * rows);
}

template <typename Value>
CudaSparseMatrix<Value>::CudaSparseMatrix(const SparseMatrix &sparse_matrix) {
  // The number of nonzeros in each row must be divisible by the number of
  // elements per scalar for the specified data type.
  for (int i = 0; i < sparse_matrix.Rows(); ++i) {
    int nnz = sparse_matrix.RowOffsets()[i + 1] - sparse_matrix.RowOffsets()[i];
    // CHECK_EQ(nnz % TypeUtils<Value>::kElementsPerScalar, 0)
    //     << "The number of elements in each row must be divisible by "
    //     << "the number of elements per scalar value for the specified "
    //     << "data type.";
    CHECK_EQ(nnz % TypeUtils<Value>::kElementsPerScalar, 0);
  }
  InitFromSparseMatrix(sparse_matrix);
}

template <typename Value>
void CudaSparseMatrix<Value>::InitFromSparseMatrix(
    const SparseMatrix &sparse_matrix) {
  // Copy the sparse matrix meta-data.
  rows_ = sparse_matrix.Rows();
  columns_ = sparse_matrix.Columns();
  nonzeros_ = sparse_matrix.Nonzeros();
  pad_rows_to_ = sparse_matrix.PadRowsTo();
  num_elements_with_padding_ = sparse_matrix.NumElementsWithPadding();
  weight_distribution_ = sparse_matrix.WeightDistribution();
  row_swizzle_ = sparse_matrix.RowSwizzle();

  // Allocate memory on the GPU for our matrix.
  float *values_float = nullptr;
  int *column_indices_int = nullptr;
  CUDA_CALL(
      cudaMalloc(&values_float, sizeof(float) * num_elements_with_padding_));
  CUDA_CALL(cudaMalloc(&column_indices_int,
                       sizeof(int) * num_elements_with_padding_));
  CUDA_CALL(cudaMalloc(&row_offsets_, sizeof(int) * (rows_ + 1)));
  CUDA_CALL(cudaMalloc(&row_indices_, sizeof(int) * rows_));

  // Copy the results to the GPU.
  CUDA_CALL(cudaMemcpy(values_float, sparse_matrix.Values(),
                       sizeof(float) * num_elements_with_padding_,
                       cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(column_indices_int, sparse_matrix.ColumnIndices(),
                       sizeof(int) * num_elements_with_padding_,
                       cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(row_offsets_, sparse_matrix.RowOffsets(),
                       sizeof(int) * (rows_ + 1), cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(row_indices_, sparse_matrix.RowIndices(),
                       sizeof(int) * rows_, cudaMemcpyHostToDevice));
  CUDA_CALL(cudaStreamSynchronize(nullptr));

  // Allocate memory for the values and indices in the target datatype.
  int elements =
      num_elements_with_padding_ / TypeUtils<Value>::kElementsPerScalar;
  CUDA_CALL(cudaMalloc(&values_, sizeof(Value) * elements));
  CUDA_CALL(cudaMalloc(&column_indices_, sizeof(Index) * elements));

  // Convert to the target datatype.
  CUDA_CALL(Convert(values_float, values_, num_elements_with_padding_));
  CUDA_CALL(
      Convert(column_indices_int, column_indices_, num_elements_with_padding_));
  CUDA_CALL(cudaStreamSynchronize(nullptr));

  // Free the temporary memory.
  CUDA_CALL(cudaFree(values_float));
  CUDA_CALL(cudaFree(column_indices_int));


  nr1 = sparse_matrix.nr1;
  nr2 = sparse_matrix.nr2;

  row_indices_1 = nullptr;
  row_indices_2 = nullptr;
  if(nr1 > 0){
    CUDA_CALL(cudaMalloc((void**)&row_indices_1,sizeof(int)*nr1));
    CUDA_CALL(cudaMemcpy(row_indices_1,sparse_matrix.row_indices_1,sizeof(int)*nr1,cudaMemcpyHostToDevice));
  }
  if(nr2 > 0) {
    CUDA_CALL(cudaMalloc((void**)&row_indices_2,sizeof(int)*nr2));
    CUDA_CALL(cudaMemcpy(row_indices_2,sparse_matrix.row_indices_2,sizeof(int)*nr2,cudaMemcpyHostToDevice));
  }

  seg_row_indices = nullptr;
  seg_st_offsets = nullptr;
  seg_row_indices_residue = nullptr;

  n_segs = sparse_matrix.n_segs;
  
  if(n_segs > 0) {
    CUDA_CALL(cudaMalloc((void**)&seg_row_indices,sizeof(int)*n_segs));
    CUDA_CALL(cudaMemcpy(seg_row_indices,sparse_matrix.seg_row_indices,sizeof(int)*n_segs,cudaMemcpyHostToDevice));
  
    CUDA_CALL(cudaMalloc((void**)&seg_st_offsets,sizeof(int)*(n_segs+1)));
    CUDA_CALL(cudaMemcpy(seg_st_offsets,sparse_matrix.seg_st_offsets,sizeof(int)*(n_segs+1),cudaMemcpyHostToDevice));
  }

  n_segs_residue = sparse_matrix.n_segs_residue;
  if(n_segs_residue > 0) {
    CUDA_CALL(cudaMalloc((void**)&seg_row_indices_residue,sizeof(int)*n_segs_residue));
    CUDA_CALL(cudaMemcpy(seg_row_indices_residue,sparse_matrix.seg_row_indices_residue,sizeof(int)*n_segs_residue,cudaMemcpyHostToDevice));
  }
  
}

// Explicit instantiations for template functions and classes.
template class CudaSparseMatrix<float>;
template class CudaSparseMatrix<half2>;
template class CudaSparseMatrix<double>;

int compare0(const void *a, const void *b)
{
  if (((struct v_struct *)a)->row - ((struct v_struct *)b)->row > 0) return 1;
  if (((struct v_struct *)a)->row - ((struct v_struct *)b)->row < 0) return -1;
  if (((struct v_struct *)a)->col - ((struct v_struct *)b)->col > 0) return 1;
  if (((struct v_struct *)a)->col - ((struct v_struct *)b)->col < 0) return -1;
  return 0;
}

__host__ SparseMatrix::SparseMatrix(const std::string& file_path, Swizzle row_swizzle, int pad_rows_to) {
  srand(0);

  row_swizzle_ = row_swizzle;
  pad_rows_to_ = pad_rows_to;

  char buf[300];
  int nflag, sflag;
  bool is_coordinate = false;

  FILE *fp = fopen(file_path.c_str(), "r");
  if (!fp) {
    fprintf(stderr, "Error: Cannot open file %s\n", file_path.c_str());
    exit(1);
  }
  
  fgets(buf, 300, fp);
  
  // Check format type: coordinate (sparse matrix) or array (dense matrix)
  if (strstr(buf, "coordinate") != NULL) {
    is_coordinate = true;
  } else if (strstr(buf, "array") != NULL) {
    is_coordinate = false;
    fprintf(stderr, "Error: Array format is not supported, only coordinate format for sparse matrices.\n");
    exit(1);
  } else {
    fprintf(stderr, "Warning: Format not specified in header, assuming coordinate format.\n");
    is_coordinate = true;  
  }
  
  // Check symmetry
  if (strstr(buf, "symmetric") != NULL) {
    sflag = 1; 
  } else if (strstr(buf, "Hermitian") != NULL) {
    sflag = 1; 
  } else if (strstr(buf, "skew-symmetric") != NULL) {
    sflag = 2; 
  } else {
    sflag = 0; 
  }

  // Check data type
  if (strstr(buf, "pattern") != NULL) {
    nflag = 0; 
  } else if (strstr(buf, "complex") != NULL) {
    nflag = -1; 
  } else if (strstr(buf, "real") != NULL || strstr(buf, "double") != NULL || strstr(buf, "integer") != NULL) {
    nflag = 1; 
  } else {
    nflag = 1; 
    fprintf(stderr, "Warning: Field type not specified, assuming real values.\n");
  }

  #ifdef SYM
      sflag = 1;
  #endif

  // Skip comment lines (lines starting with %)
  int pre_count = 0;
  while (1) {
      pre_count++;
      if (fgets(buf, 300, fp) == NULL) {
          fprintf(stderr, "Error: Unexpected end of file while reading comments.\n");
          fclose(fp);
          exit(1);
      }
      if (strstr(buf, "%") == NULL) break;
  }
  fclose(fp);

  int i;
  fp = fopen(file_path.c_str(), "r");
  if (!fp) {
    fprintf(stderr, "Error: Cannot reopen file %s\n", file_path.c_str());
    exit(1);
  }
  
  for (i = 0; i < pre_count; i++)
    fgets(buf, 300, fp);

  // Read matrix dimensions
  if (is_coordinate) {
    if (fscanf(fp, "%d %d %d", &rows_, &columns_, &nonzeros_) != 3) {
      fprintf(stderr, "Error: Failed to read matrix dimensions and nonzeros.\n");
      fclose(fp);
      exit(1);
    }
  } else {
    if (fscanf(fp, "%d %d", &rows_, &columns_) != 2) {
      fprintf(stderr, "Error: Failed to read matrix dimensions.\n");
      fclose(fp);
      exit(1);
    }
    nonzeros_ = rows_ * columns_; 
  }

  // If it is symmetric/Hermitian/skew-symmetric, need to copy data to the upper triangle
  if (sflag > 0) {
    nonzeros_ *= 2; 
  }

  struct v_struct *temp_v = (struct v_struct *)malloc(sizeof(struct v_struct) * (nonzeros_ + 1));
  if (!temp_v) {
    fprintf(stderr, "Error: Memory allocation failed.\n");
    fclose(fp);
    exit(1);
  }
  
  int actual_nonzeros = 0; 
  
  // Read non-zero elements
  for (i = 0; i < (sflag > 0 ? nonzeros_ / 2 : nonzeros_); i++) {
    int row, col;
    float val;
    
    if (is_coordinate) {
      // Coordinate format: each line contains row, column, and optional value
      if (fscanf(fp, "%d %d", &row, &col) != 2) {
        fprintf(stderr, "Error: Failed to read matrix entry %d.\n", i);
        free(temp_v);
        fclose(fp);
        exit(1);
      }
      
      // MTX format is 1-indexed, convert to 0-indexed
      row--; 
      col--;
      
      // Check index range
      if (row < 0 || row >= rows_ || col < 0 || col >= columns_) {
        fprintf(stderr, "Error: Index out of range (%d, %d) for matrix size %d x %d\n", 
                row + 1, col + 1, rows_, columns_);
        free(temp_v);
        fclose(fp);
        exit(1);
      }
      
      // Read or generate values based on data type
      if (nflag == 0) { 
        val = (float)(rand() % 1048576) / 1048576; 
      } else if (nflag == 1) { 
        if (fscanf(fp, " %f ", &val) != 1) {
          fprintf(stderr, "Error: Failed to read value for entry %d.\n", i);
          free(temp_v);
          fclose(fp);
          exit(1);
        }
      } else { 
        float real_part, imag_part;
        if (fscanf(fp, " %f %f ", &real_part, &imag_part) != 2) {
          fprintf(stderr, "Error: Failed to read complex value for entry %d.\n", i);
          free(temp_v);
          fclose(fp);
          exit(1);
        }
        val = real_part; 
      }
      
      #ifdef SIM_VALUE
        val = 1.0f;
      #endif
      
      temp_v[actual_nonzeros].row = row;
      temp_v[actual_nonzeros].col = col;
      temp_v[actual_nonzeros].val = val;
      actual_nonzeros++;
      
      // For symmetric matrices, add corresponding symmetric elements (but skip duplicate diagonal elements)
      if (sflag > 0 && row != col) {
        temp_v[actual_nonzeros].row = col;
        temp_v[actual_nonzeros].col = row;
        temp_v[actual_nonzeros].val = (sflag == 2) ? -val : val;
        actual_nonzeros++;
      }
    }
  }
  
  fclose(fp);
  nonzeros_ = actual_nonzeros; 
  
  // Sort non-zero elements by row and column
  qsort(temp_v, nonzeros_, sizeof(struct v_struct), compare0);

  // Remove duplicate elements
  int *loc = (int *)malloc(sizeof(int) * (nonzeros_ + 1));
  if (!loc) {
    fprintf(stderr, "Error: Memory allocation failed for loc array.\n");
    free(temp_v);
    exit(1);
  }
  
  memset(loc, 0, sizeof(int) * (nonzeros_ + 1));
  loc[0] = 1;
  for (i = 1; i < nonzeros_; i++) {
    if (temp_v[i].row == temp_v[i-1].row && temp_v[i].col == temp_v[i-1].col)
      loc[i] = 0; 
    else 
      loc[i] = 1;
  }
  
  // Calculate new positions for each unique element
  for (i = 1; i <= nonzeros_; i++)
    loc[i] += loc[i-1];
  
  for (i = nonzeros_; i >= 1; i--)
    loc[i] = loc[i-1];
  
  loc[0] = 0;
  
  // Rearrange elements to remove duplicates
  for (i = 0; i < nonzeros_; i++) {
    temp_v[loc[i]].row = temp_v[i].row;
    temp_v[loc[i]].col = temp_v[i].col;
    temp_v[loc[i]].val = temp_v[i].val;
  }
  
  nonzeros_ = loc[nonzeros_]; 
  temp_v[nonzeros_].row = rows_;
  free(loc);

  // Convert to CSR format
  std::vector<float> values_staging(nonzeros_);
  std::vector<int> row_offsets_staging(rows_ + 1);
  std::vector<int> column_indices_staging(nonzeros_);

  std::fill(row_offsets_staging.begin(), row_offsets_staging.end(), 0);
  
  for (i = 0; i < nonzeros_; i++) {
    column_indices_staging[i] = temp_v[i].col;
    values_staging[i] = temp_v[i].val;
    row_offsets_staging[1 + temp_v[i].row] = i + 1;
  }

  // Handle empty rows
  for (i = 1; i < rows_; i++) {
    if (row_offsets_staging[i] == 0) 
      row_offsets_staging[i] = row_offsets_staging[i-1];
  }
  
  row_offsets_staging[rows_] = nonzeros_;
  free(temp_v);

  // Apply padding
  std::vector<int> row_offsets_staging1, column_indices_staging1;
  std::vector<float> values_staging1;
  PadSparseMatrix(row_offsets_staging, values_staging, column_indices_staging, pad_rows_to,
                  &row_offsets_staging1, &values_staging1,
                  &column_indices_staging1);

  num_elements_with_padding_ = row_offsets_staging1[rows_];

  // Allocate memory and copy data
  values_ = new float[num_elements_with_padding_];
  column_indices_ = new int[num_elements_with_padding_];
  row_offsets_ = new int[rows_ + 1];

  std::memcpy(values_, values_staging1.data(),
              num_elements_with_padding_ * sizeof(float));
  std::memcpy(column_indices_, column_indices_staging1.data(),
              num_elements_with_padding_ * sizeof(int));
  std::memcpy(row_offsets_, row_offsets_staging1.data(),
              (rows_ + 1) * sizeof(int));

  // Set row indices
  row_indices_ = new int[rows_];
  if (row_swizzle_ == IDENTITY) {
    IdentityRowSwizzle(rows_, row_offsets_, row_indices_);
  } else {
    SortedRowSwizzle(rows_, row_offsets_, row_indices_);
  }

  // Initialize other data structures
  row_indices_1 = new int[rows_];
  row_indices_2 = new int[rows_];

  seg_row_indices = nullptr;
  seg_st_offsets  = nullptr;
  n_segs = 0;

  seg_row_indices_residue = nullptr;
  n_segs_residue = 0;
}

__host__ SparseMatrix::SparseMatrix(const int* row_offsets_input, const int* column_indices_input, 
                                  const float* values_input, int rows, int columns, int nonzeros, 
                                  Swizzle row_swizzle, int pad_rows_to) {
  // Set matrix properties
  rows_ = rows;
  columns_ = columns;
  nonzeros_ = nonzeros;
  row_swizzle_ = row_swizzle;
  pad_rows_to_ = pad_rows_to;

  std::cout << "rows: " << rows_ << " columns: " << columns_ << " nonzeros: " << nonzeros_ << std::endl;
  
  // Convert CSR data to vectors for padding
  std::vector<int> row_offsets_staging(row_offsets_input, row_offsets_input + rows + 1);
  std::cout << "row_offsets_staging: " << row_offsets_staging.size() << std::endl;
  std::vector<int> column_indices_staging(column_indices_input, column_indices_input + nonzeros);
  std::cout << "row_offsets_staging: " << row_offsets_staging.size() << std::endl;
  std::vector<float> values_staging(values_input, values_input + nonzeros);
  std::cout << "values_staging: " << values_staging.size() << std::endl;
  
  // Apply padding
  std::vector<int> row_offsets_staging1, column_indices_staging1;
  std::vector<float> values_staging1;
  PadSparseMatrix(row_offsets_staging, values_staging, column_indices_staging, pad_rows_to,
                 &row_offsets_staging1, &values_staging1, &column_indices_staging1);
  std::cout << "!!!!!!!!!!!!!!!!!!!!!!! " << std::endl;
  
  
  num_elements_with_padding_ = row_offsets_staging1[rows_];
  
  // Allocate memory and copy data
  values_ = new float[num_elements_with_padding_];
  column_indices_ = new int[num_elements_with_padding_];
  row_offsets_ = new int[rows_ + 1];
  std::cout << "Allocate memory and copy data " << std::endl;
  
  std::memcpy(values_, values_staging1.data(), num_elements_with_padding_ * sizeof(float));
  std::memcpy(column_indices_, column_indices_staging1.data(), num_elements_with_padding_ * sizeof(int));
  std::memcpy(row_offsets_, row_offsets_staging1.data(), (rows_ + 1) * sizeof(int));

  std::cout << "memcpy!!!!!!!!!!!!!!!! " << std::endl;
  
  // Set row indices
  row_indices_ = new int[rows_];
  if (row_swizzle_ == IDENTITY) {
    IdentityRowSwizzle(rows_, row_offsets_, row_indices_);
  } else {
    SortedRowSwizzle(rows_, row_offsets_, row_indices_);
  }
  
  std::cout << "sort!!!!!!!!!!!!!!! " << std::endl;

  // Initialize other data structures
  row_indices_1 = new int[rows_];
  row_indices_2 = new int[rows_];
  nr1 = 0;
  nr2 = 0;
  
  seg_row_indices = nullptr;
  seg_st_offsets = nullptr;
  n_segs = 0;
  
  seg_row_indices_residue = nullptr;
  n_segs_residue = 0;
}

void SparseMatrix::RowDivide2Segment(int SegmentLength,int vectorLen,int KBLOCK) {
  int nr = 0;
  std::vector<int> row_indices_staging;
  std::vector<int> st_offsets_staging;
  std::vector<int> row_indices_residue_staging;
  std::cout << "SegmentLength: " << SegmentLength << " vectorLen: " << vectorLen << " KBLOCK: " << KBLOCK << std::endl;
  std::cout << "rows_: " << rows_ << " row_offsets_: " << row_offsets_[rows_] << std::endl;

  for(int i=0; i < rows_; ++i) {
    int row_offset = row_offsets_[i];
    int n_padding = row_offset % vectorLen;
    int nnz = row_offsets_[i+1] - row_offset + n_padding;
    if(i <10) {
      std::cout << "i: " << i << " nnz: " << nnz << " row_offset: " << row_offset << " n_padding: " << n_padding << std::endl;
    }
    
    if(nnz > SegmentLength) {
      row_indices_staging.push_back(i);
      st_offsets_staging.push_back(row_offset);
      row_offset = (row_offset + SegmentLength) - n_padding;

      nnz -= SegmentLength;
    }

    while(nnz > SegmentLength) {
      row_indices_staging.push_back(i);
      st_offsets_staging.push_back(row_offset);

      row_offset += SegmentLength;
      nnz -= SegmentLength;
    }

    if(nnz > 0) {
      if(nnz >= KBLOCK){
        row_indices_staging.push_back(i);
        st_offsets_staging.push_back(row_offset);
      }
      if( nnz % KBLOCK) {
        row_indices_residue_staging.push_back(i);
      }
    }
    if(i <10) {
      std::cout << "row_indices_staging: " << row_indices_staging.size() << " st_offsets_staging: " << st_offsets_staging.size() << " row_indices_residue_staging: " << row_indices_residue_staging.size() << std::endl;
    }
  }
  std::cout << "row_indices_staging: " << row_indices_staging.size() << " st_offsets_staging: " << st_offsets_staging.size() << " row_indices_residue_staging: " << row_indices_residue_staging.size() << std::endl;

  st_offsets_staging.push_back(row_offsets_[rows_]);

  if(n_segs > 0) {
    delete[] seg_row_indices;
    delete[] seg_st_offsets;
  }
  if(n_segs_residue > 0) {
    delete[] seg_row_indices_residue;
  }

  std::cout << "11111111111111111" << std::endl;

  n_segs = row_indices_staging.size();
  seg_row_indices = new int[n_segs];
  seg_st_offsets = new int[n_segs+1];

  std::cout << "222222222222222222222222" << std::endl;

  n_segs_residue = row_indices_residue_staging.size();
  seg_row_indices_residue = new int[n_segs_residue];

  std::cout << "3333333333333333333333" << std::endl;

  std::memcpy(seg_row_indices,row_indices_staging.data(),sizeof(int)*n_segs);
  std::memcpy(seg_st_offsets,st_offsets_staging.data(),sizeof(int)*(n_segs+1));
  std::memcpy(seg_row_indices_residue,row_indices_residue_staging.data(),sizeof(int)*n_segs_residue);

  std::cout << "n_segs: " << n_segs << " n_segs_residue: " << n_segs_residue << std::endl;
}
}  


