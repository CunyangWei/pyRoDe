#ifndef MATRIX_UTILS_H_ 
#define MATRIX_UTILS_H_

#include <vector>
#include <iostream>
#include <string>

#include "basic_utils.h"
// #include "absl/random/random.h"

namespace SPC {

template <typename In, typename Out>
cudaError_t Convert(const In *in, Out *out, int n);

void IdentityRowSwizzle(int rows, const int* row_offsets, int* row_indices);
void SortedRowSwizzle(int rows, const int* row_offsets, int* row_indices);

enum Swizzle {
  IDENTITY = 0,
  SORTED = 1,
  LD_BALANCE = 2,
};

enum ElementDistribution {
  RANDOM_UNIFORM = 0,
  PERFECT_UNIFORM = 1
};

template <typename Value>
class CudaSparseMatrix;

class SparseMatrix {
 public:
 __host__ SparseMatrix(const std::string& file_path, Swizzle row_swizzle, int pad_rows_to);
 __host__ SparseMatrix(const int* row_offsets, const int* column_indices, const float* values, 
                     int rows, int columns, int nonzeros, Swizzle row_swizzle, int pad_rows_to);
  ~SparseMatrix() {
    delete[] values_;
    delete[] row_offsets_;
    delete[] column_indices_;
    delete[] row_indices_;

    if(row_indices_1 != nullptr)
      delete[] row_indices_1;
    if(row_indices_2 != nullptr)
      delete[] row_indices_2;

    if(seg_row_indices)
      delete[] seg_row_indices;
    if(seg_st_offsets)
      delete[] seg_st_offsets;

    if(seg_row_indices_residue) 
      delete[] seg_row_indices_residue;
  }
  SparseMatrix(const SparseMatrix&) = delete;
  SparseMatrix& operator=(const SparseMatrix&) = delete;
  SparseMatrix(SparseMatrix&&) = delete;
  SparseMatrix& operator=(SparseMatrix&&) = delete;

  const float* Values() const { return values_; }
  float* Values() { return values_; }
  const int* RowOffsets() const { return row_offsets_; }
  int* RowOffsets() { return row_offsets_; }
  const int* ColumnIndices() const { return column_indices_; }
  int* ColumnIndices() { return column_indices_; }
  const int* RowIndices() const { return row_indices_; }
  int* RowIndices() { return row_indices_; }
  int Rows() const { return rows_; }
  int Columns() const { return columns_; }
  int Nonzeros() const { return nonzeros_; }
  int PadRowsTo() const { return pad_rows_to_; }
  int NumElementsWithPadding() const { return num_elements_with_padding_; }
  void RowDivide2Segment(int SegmentLength, int vectorLen, int KBLOCK);
  ElementDistribution WeightDistribution() const { return weight_distribution_; }
  Swizzle RowSwizzle() const { return row_swizzle_; }

  int* row_indices_1;
  int* row_indices_2;
  int nr1 = 0, nr2 = 0;
  int* seg_row_indices;
  int* seg_st_offsets;
  int n_segs;
  int* seg_row_indices_residue;
  int n_segs_residue;
  
 protected:
  SparseMatrix() : values_(nullptr), row_offsets_(nullptr), column_indices_(nullptr),
                   row_indices_(nullptr), rows_(0), columns_(0), nonzeros_(0),
                   pad_rows_to_(0), num_elements_with_padding_(0),
                   weight_distribution_(RANDOM_UNIFORM), row_swizzle_(IDENTITY) {}
  float* values_;
  int* row_offsets_;
  int* column_indices_;
  int* row_indices_;
  int rows_, columns_, nonzeros_;
  int pad_rows_to_, num_elements_with_padding_;
  ElementDistribution weight_distribution_;
  Swizzle row_swizzle_;
};

template <typename Value>
class CudaSparseMatrix {
 public:
   explicit CudaSparseMatrix(const SparseMatrix& sparse_matrix);
   ~CudaSparseMatrix() {
    CUDA_CALL(cudaFree(values_));
    CUDA_CALL(cudaFree(row_offsets_));
    CUDA_CALL(cudaFree(column_indices_));
    CUDA_CALL(cudaFree(row_indices_));
    if(row_indices_1 != nullptr) {
      CUDA_CALL(cudaFree(row_indices_1));
    }
    if(row_indices_2 != nullptr)
      CUDA_CALL(cudaFree(row_indices_2));
    if(seg_row_indices != nullptr) {
      CUDA_CALL(cudaFree(seg_row_indices));
    }
    if(seg_st_offsets != nullptr) {
      CUDA_CALL(cudaFree(seg_st_offsets));
    }
    if(seg_row_indices_residue != nullptr) {
      CUDA_CALL(cudaFree(seg_row_indices_residue));
    }
  }
  CudaSparseMatrix(const CudaSparseMatrix&) = delete;
  CudaSparseMatrix& operator=(const CudaSparseMatrix&) = delete;
  CudaSparseMatrix(CudaSparseMatrix&&) = delete;
  CudaSparseMatrix& operator=(CudaSparseMatrix&&) = delete;

  typedef typename Value2Index<Value>::Index Index;
  const Value* Values() const { return values_; }
  Value* Values() { return values_; }
  const int* RowOffsets() const { return row_offsets_; }
  int* RowOffsets() { return row_offsets_; }
  const Index* ColumnIndices() const { return column_indices_; }
  Index* ColumnIndices() { return column_indices_; }
  const int* RowIndices() const { return row_indices_; }
  int* RowIndices() { return row_indices_; }
  int Rows() const { return rows_; }
  int Columns() const { return columns_; }
  int Nonzeros() const { return nonzeros_; }
  int PadRowsTo() const { return pad_rows_to_; }
  int NumElementsWithPadding() const { return num_elements_with_padding_; }
  ElementDistribution WeightDistribution() const { return weight_distribution_; }
  Swizzle RowSwizzle() const { return row_swizzle_; }
  int * row_indices_1;
  int * row_indices_2;
  int nr1;
  int nr2;
  int * seg_row_indices;
  int * seg_st_offsets;
  int n_segs;
  int * seg_row_indices_residue;
  int n_segs_residue;
  
 protected:
  CudaSparseMatrix() : values_(nullptr), row_offsets_(nullptr), column_indices_(nullptr),
                       row_indices_(nullptr), rows_(0), columns_(0), nonzeros_(0),
                       pad_rows_to_(0), num_elements_with_padding_(0),
                       weight_distribution_(RANDOM_UNIFORM), row_swizzle_(IDENTITY) {}
  Value* values_;
  int* row_offsets_;
  Index* column_indices_;
  int* row_indices_;
  int rows_, columns_, nonzeros_;
  int pad_rows_to_, num_elements_with_padding_;
  ElementDistribution weight_distribution_;
  Swizzle row_swizzle_;
  void InitFromSparseMatrix(const SparseMatrix& sparse_matrix);
};

inline std::vector<float> ToVector(const SparseMatrix& sparse_matrix) {
  int num = sparse_matrix.NumElementsWithPadding();
  std::vector<float> out(sparse_matrix.Values(), sparse_matrix.Values() + num);
  return out;
}

struct v_struct {
  int row, col;
  float val;
};

}  // namespace SPC
#endif
