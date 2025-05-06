#include "cuda_runtime.h"
#include "matrix_utils.h"
#include "cusparse.h"

#include "RoDeSpmm.h"

#include <sys/io.h>
#include <iostream>
#include <stdio.h>
#include <unistd.h>
#include <dirent.h>
#include <stdlib.h>
#include <sys/stat.h>

using namespace std;
using namespace SPC;

#define SEG_LENGTH 512

#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
void check(cudaError_t err, const char *const func, const char *const file,
           const int line)
{
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA Error: %s at %s: %d\n", cudaGetErrorString(err), file, line);
        fprintf(stderr, "%s %s\n", cudaGetErrorString(err), func);
        // We don't exit when we encounter CUDA errors in this example.
        // std::exit(EXIT_FAILURE);
    }
}

// Add validation mode macro definition
#define VALIDATE_RESULTS 0 // Set to 1 to enable result validation

void ValidateResults(float *h_C, float *h_C_ref, int64_t size)
{
    float max_diff = 0.0f;
    float avg_diff = 0.0f;
    int diff_count = 0;

    for (int64_t i = 0; i < size; i++)
    {
        float diff = fabs(h_C[i] - h_C_ref[i]);
        if (diff > 1e-5)
        {
            diff_count++;
            if (diff > max_diff)
                max_diff = diff;
        }
        avg_diff += diff;
    }
    avg_diff /= size;

    if (diff_count > 0)
    {
        printf("Validation Result: Differences found\n");
        printf("Maximum Difference: %f\n", max_diff);
        printf("Average Difference: %f\n", avg_diff);
        printf("Number of Differences: %d\n", diff_count);
    }
    else
    {
        printf("Validation Result: Perfect match\n");
    }
}

int main(int argc, char **argv)
{
    if (argc < 3)
    {
        cout << "Usage: " << argv[0] << " BN file_path" << endl;
        return 0;
    }
    int64_t n = atoi(argv[1]);
    string file_path = argv[2];
#if VALIDATE_RESULTS
    int ITER = 1;
    int WARM = 0;
#else
    int ITER = 10;
    int WARM = 10;
#endif

    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    double gflops = 0.0f;

    SPC::SparseMatrix sm1(file_path, SPC::SORTED, 1);
    sm1.RowDivide2Segment(SEG_LENGTH, 4, 32);
    SPC::CudaSparseMatrix<float> c_sm(sm1);

    std::cout << "Sparse matrix info:" << std::endl;
    std::cout << "  Rows: " << c_sm.Rows() << std::endl;
    std::cout << "  Columns: " << c_sm.Columns() << std::endl;
    std::cout << "  Non-zeros: " << c_sm.Nonzeros() << std::endl;
    std::cout << "  Segments: " << c_sm.n_segs << std::endl;
    std::cout << "  Residue segments: " << c_sm.n_segs_residue << std::endl;

    int64_t m = c_sm.Rows(); int64_t k = c_sm.Columns();

    int64_t denseB_size = (int64_t)n * (int64_t)k;
    float *h_DenseMatB = (float *)malloc(sizeof(float) * denseB_size);
    for (int64_t i = 0; i < denseB_size; i++)
    {
        h_DenseMatB[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    }

    float *d_B;
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_B, sizeof(float) * denseB_size));
    CHECK_CUDA_ERROR(cudaMemcpy(d_B, h_DenseMatB, sizeof(float) * denseB_size, cudaMemcpyHostToDevice));

    float *d_C;
    cudaMalloc((void **)&d_C, (int64_t)sizeof(float) * (int64_t)m * (int64_t)n);

    float *d_C2;
    cudaMalloc((void **)&d_C2, (int64_t)sizeof(float) * (int64_t)m * (int64_t)n);

    float *diff;
    cudaMalloc((void **)&diff, sizeof(float) * 1);

    float tot_ms;
    cudaEvent_t event1, event2;
    cudaEventCreate(&event1);
    cudaEventCreate(&event2);

    cusparseStatus_t status;
    cusparseHandle_t handle;
    cusparseCreate(&handle);
    cusparseSpMatDescr_t matA;
    cusparseDnMatDescr_t matB, matC;
    status = cusparseCreateCsr(&matA, m, k, c_sm.Nonzeros(),
                    c_sm.RowOffsets(), c_sm.ColumnIndices(), c_sm.Values(),
                    CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                    CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);

    void *dBuffer = NULL;
    size_t bufferSize = 0;

    // Create dense matrix B
    int ldb = n;
    int ldc = n;
	float alpha = 1.0f,beta = 0.0f;

    cudaDeviceSynchronize();
    for(int i = 0; i < WARM; i++)
    {
        status = cusparseCreateDnMat(&matB, k, n, ldb, d_B, CUDA_R_32F, CUSPARSE_ORDER_ROW);
        status = cusparseCreateDnMat(&matC, m, n, ldc, d_C, CUDA_R_32F, CUSPARSE_ORDER_ROW);
        status = cusparseSpMM_bufferSize(
                                handle,
                                CUSPARSE_OPERATION_NON_TRANSPOSE,
                                CUSPARSE_OPERATION_NON_TRANSPOSE,
                                &alpha, matA, matB, &beta, matC, CUDA_R_32F,
                                CUSPARSE_SPMM_ALG_DEFAULT, &bufferSize);

                                cudaMalloc(&dBuffer, bufferSize);

        status = cusparseSpMM(handle,CUSPARSE_OPERATION_NON_TRANSPOSE,
                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                    &alpha, matA, matB, &beta, matC, CUDA_R_32F,
                    CUSPARSE_SPMM_ALG_DEFAULT, dBuffer);

        
        cusparseDestroyDnMat(matB);
        cusparseDestroyDnMat(matC);
        cudaFree(dBuffer);
    }

    cudaDeviceSynchronize();

    cudaEventRecord(event1, 0);

    for(int i = 0; i < ITER; i++)
    {
        cusparseCreateDnMat(&matB, k, n, ldb, d_B, CUDA_R_32F, CUSPARSE_ORDER_ROW);
        cusparseCreateDnMat(&matC, m, n, ldc, d_C, CUDA_R_32F, CUSPARSE_ORDER_ROW);

        cusparseSpMM_bufferSize(
                                handle,
                                CUSPARSE_OPERATION_NON_TRANSPOSE,
                                CUSPARSE_OPERATION_NON_TRANSPOSE,
                                &alpha, matA, matB, &beta, matC, CUDA_R_32F,
                                CUSPARSE_SPMM_ALG_DEFAULT, &bufferSize);
        cudaMalloc(&dBuffer, bufferSize);

        cusparseSpMM(handle,CUSPARSE_OPERATION_NON_TRANSPOSE,
                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                    &alpha, matA, matB, &beta, matC, CUDA_R_32F,
                    CUSPARSE_SPMM_ALG_DEFAULT, dBuffer);
        
        cusparseDestroyDnMat(matB);
        cusparseDestroyDnMat(matC);
        cudaFree(dBuffer);
    }
    
    cudaEventRecord(event2, 0);

    cudaEventSynchronize(event1);
    cudaEventSynchronize(event2);
    cudaEventElapsedTime(&tot_ms, event1, event2);
    cudaDeviceSynchronize();

    gflops = (double)ITER * (double)c_sm.Nonzeros() * 2 * n / tot_ms / 1000000;
    // printf(", %f, %f", tot_ms / (float)ITER, gflops);
    printf("cuda spmm time: %f, gflops: %f\n", tot_ms / (float)ITER, gflops);
    cudaDeviceSynchronize();

#if VALIDATE_RESULTS
    float *h_C = (float *)malloc(sizeof(float) * m * n);
    cudaMemcpy(h_C, d_C, sizeof(float) * m * n, cudaMemcpyDeviceToHost);
#endif
    cudaFree(d_C);
    cusparseDestroySpMat(matA);
    cusparseDestroy(handle);

    gflops = 0;
    tot_ms = 0;

    cudaEventDestroy(event1);
    cudaEventDestroy(event2);
    cudaEventCreate(&event1);
    cudaEventCreate(&event2);

    cudaDeviceSynchronize();
    cudaEventRecord(event1, 0);

    if (n < 128)
    {
        for (int i = 0; i < ITER; ++i)
            RoDeSpmm_n32(c_sm.n_segs, c_sm.n_segs_residue, c_sm.Columns(), n,
                         c_sm.Values(), c_sm.ColumnIndices(), c_sm.RowOffsets(),
                         c_sm.seg_row_indices, c_sm.seg_row_indices_residue, c_sm.seg_st_offsets,
                         d_B, d_C2, stream1, stream2);
    }
    else
    {
        for (int i = 0; i < ITER; ++i)
            RoDeSpmm_n128(c_sm.n_segs, c_sm.n_segs_residue, c_sm.Columns(), n,
                          c_sm.Values(), c_sm.ColumnIndices(), c_sm.RowOffsets(),
                          c_sm.seg_row_indices, c_sm.seg_row_indices_residue, c_sm.seg_st_offsets,
                          d_B, d_C2, stream1, stream2);
    }

    cudaEventRecord(event2, 0);

    cudaEventSynchronize(event1);
    cudaEventSynchronize(event2);
    cudaEventElapsedTime(&tot_ms, event1, event2);
    cudaDeviceSynchronize();

    gflops = (double)ITER * (double)c_sm.Nonzeros() * 2 * n / tot_ms / 1000000;

    printf("RoDe spmm time: %f, gflops: %f\n", tot_ms / (float)ITER, gflops);

#if VALIDATE_RESULTS
    float *h_C2 = (float *)malloc(sizeof(float) * m * n);
    cudaMemcpy(h_C2, d_C2, sizeof(float) * m * n, cudaMemcpyDeviceToHost);
    ValidateResults(h_C, h_C2, m * n);
    free(h_C);
    free(h_C2);
#endif

    cudaFree(d_C2);
    cudaFree(d_B);
    free(h_DenseMatB);

    cudaEventDestroy(event1);
    cudaEventDestroy(event2);
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    cudaFree(diff);

    return 0;
}

