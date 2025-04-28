#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cuda_runtime.h>
#include "../utils/matrix_utils.h"
#include "../RoDe_SpMM/RoDeSpmm.h"
#include <iostream>

namespace py = pybind11;

// Helper function to get raw pointer from a tensor with CUDA array interface
template <typename T>
T* get_cuda_ptr(py::object tensor) {
    try {
        if (!py::hasattr(tensor, "__cuda_array_interface__")) {
            throw std::runtime_error("Tensor does not have __cuda_array_interface__");
        }
        
        py::dict cuda_array_interface = tensor.attr("__cuda_array_interface__").cast<py::dict>();
        if (!cuda_array_interface.contains("data")) {
            throw std::runtime_error("__cuda_array_interface__ missing 'data' field");
        }
        
        py::tuple data = cuda_array_interface["data"].cast<py::tuple>();
        if (py::len(data) < 1) {
            throw std::runtime_error("__cuda_array_interface__ 'data' field is empty");
        }
        
        uintptr_t ptr = data[0].cast<uintptr_t>();
        return reinterpret_cast<T*>(ptr);
    } catch (const py::error_already_set& e) {
        std::cerr << "Error getting CUDA pointer: " << e.what() << std::endl;
        throw;
    } catch (const std::exception& e) {
        std::cerr << "Error getting CUDA pointer: " << e.what() << std::endl;
        throw;
    }
}

// Helper function to check if a tensor is on CUDA
bool is_cuda(py::object tensor) {
    try {
        if (!py::hasattr(tensor, "device")) {
            return false;
        }
        
        auto device = tensor.attr("device");
        if (!py::hasattr(device, "type")) {
            return false;
        }
        
        auto device_type = device.attr("type").cast<std::string>();
        return device_type == "cuda";
    } catch (const std::exception& e) {
        std::cerr << "Error checking if tensor is on CUDA: " << e.what() << std::endl;
        return false;
    }
}

class CudaSpMat {
private:
    SPC::SparseMatrix* host_sm;
    SPC::CudaSparseMatrix<float>* device_sm;
    int rows, cols;
    bool owns_data;
    cudaStream_t stream1, stream2;
    int* dummy_seg_indices; // Dummy data for n_segs=0 case

public:
    CudaSpMat(py::object row_offsets, py::object column_indices, py::object values, 
              int rows_, int cols_, int seg_length = 512, std::string device = "GPU") : 
        rows(rows_), cols(cols_), owns_data(true), dummy_seg_indices(nullptr) {

        // Create CUDA streams
        cudaError_t stream1_err = cudaStreamCreate(&stream1);
        cudaError_t stream2_err = cudaStreamCreate(&stream2);
        
        if (stream1_err != cudaSuccess || stream2_err != cudaSuccess) {
            throw std::runtime_error("Failed to create CUDA streams");
        }

        bool res = is_cuda(row_offsets);
        std::cout << "row_offsets is_cuda: " << res << std::endl;
        res = is_cuda(column_indices);
        std::cout << "column_indices is_cuda: " << res << std::endl;
        res = is_cuda(values);
        std::cout << "values is_cuda: " << res << std::endl;

        bool is_gpu_data = (device == "GPU") && 
                          is_cuda(row_offsets) && 
                          is_cuda(column_indices) && 
                          is_cuda(values);

        // Get the sizes
        int nonzeros = py::len(values);
        
        if (is_gpu_data) {
            try {
                // Get raw pointers from tensors
                int* d_row_offsets = get_cuda_ptr<int>(row_offsets);
                int* d_column_indices = get_cuda_ptr<int>(column_indices);
                float* d_values = get_cuda_ptr<float>(values);
                
                // Copy data to host (needed for SparseMatrix construction)
                // Calculate memory requirements
                size_t row_offsets_size = (rows + 1) * sizeof(int);
                size_t column_indices_size = nonzeros * sizeof(int);
                size_t values_size = nonzeros * sizeof(float);
                size_t total_requested = row_offsets_size + column_indices_size + values_size;
                
                std::cout << "Memory requested: " << total_requested << " bytes" << std::endl;
                
                int* h_row_offsets = nullptr;
                int* h_column_indices = nullptr;
                float* h_values = nullptr;
                
                try {
                    h_row_offsets = new int[rows + 1];
                    std::cout << "Allocated h_row_offsets" << std::endl;
                    h_column_indices = new int[nonzeros];
                    std::cout << "Allocated h_column_indices" << std::endl;
                    h_values = new float[nonzeros];
                    std::cout << "Allocated h_values" << std::endl;
                } catch (const std::bad_alloc& e) {
                    std::cerr << "Memory allocation failed! Requested: " << total_requested / 1024 /1024 / 1024 << " GB" << std::endl;
                    throw std::runtime_error("Failed to allocate host memory for sparse matrix");
                }

                std::cout << "Allocated host memory" << std::endl;
                cudaError_t err1 = cudaMemcpy(h_row_offsets, d_row_offsets, (rows + 1) * sizeof(int), cudaMemcpyDeviceToHost);
                if (err1 != cudaSuccess) {
                    delete[] h_row_offsets;
                    delete[] h_column_indices;
                    delete[] h_values;
                    throw std::runtime_error("Failed to copy data from GPU to host");
                }
                std::cout << "Copied row_offsets" << std::endl;
                cudaError_t err2 = cudaMemcpy(h_column_indices, d_column_indices, nonzeros * sizeof(int), cudaMemcpyDeviceToHost);
                if (err2 != cudaSuccess) {
                    delete[] h_row_offsets;
                    delete[] h_column_indices;
                    delete[] h_values;
                    throw std::runtime_error("Failed to copy data from GPU to host");
                }
                std::cout << "Copied column_indices" << std::endl;
                cudaError_t err3 = cudaMemcpy(h_values, d_values, nonzeros * sizeof(float), cudaMemcpyDeviceToHost);
                if (err3 != cudaSuccess) {
                    delete[] h_row_offsets;
                    delete[] h_column_indices;
                    delete[] h_values;
                    throw std::runtime_error("Failed to copy data from GPU to host");
                }
                std::cout << "Copied values" << std::endl;
                

                if (err1 != cudaSuccess || err2 != cudaSuccess || err3 != cudaSuccess) {
                    delete[] h_row_offsets;
                    delete[] h_column_indices;
                    delete[] h_values;
                    throw std::runtime_error("Failed to copy data from GPU to host");
                }
                
                // Create sparse matrix on host
                host_sm = new SPC::SparseMatrix(h_row_offsets, h_column_indices, h_values, 
                                              rows, cols, nonzeros, SPC::SORTED, 1);
                std::cout << "Create sparse matrix on host" << std::endl;
                // Free temporary host memory
                delete[] h_row_offsets;
                delete[] h_column_indices;
                delete[] h_values;
            } catch (const std::exception& e) {
                std::cerr << "Error in GPU data path: " << e.what() << std::endl;
                throw;
            }
        } else {
            // For CPU data, directly use the numpy arrays
            try {
                
                py::buffer_info row_offsets_buffer = py::buffer(row_offsets).request();
                py::buffer_info column_indices_buffer = py::buffer(column_indices).request();
                py::buffer_info values_buffer = py::buffer(values).request();
                
                if (row_offsets_buffer.format != py::format_descriptor<int>::format() ||
                    column_indices_buffer.format != py::format_descriptor<int>::format()) {
                    throw std::runtime_error("row_offsets and column_indices must be int32 arrays");
                }
                
                if (values_buffer.format != py::format_descriptor<float>::format()) {
                    throw std::runtime_error("values must be float32 array");
                }
                
                int* h_row_offsets = static_cast<int*>(row_offsets_buffer.ptr);
                int* h_column_indices = static_cast<int*>(column_indices_buffer.ptr);
                float* h_values = static_cast<float*>(values_buffer.ptr);
                
                // Create sparse matrix on host
                host_sm = new SPC::SparseMatrix(h_row_offsets, h_column_indices, h_values, 
                                              rows, cols, nonzeros, SPC::SORTED, 1);
            } catch (const std::exception& e) {
                std::cerr << "Error in CPU data path: " << e.what() << std::endl;
                throw std::runtime_error(std::string("Error accessing CPU arrays: ") + e.what());
            }
        }
        std::cout << "start devide" << std::endl;
        // Divide into segments
        try {

            host_sm->RowDivide2Segment(seg_length, 4, 32);
            std::cout << "devided" << std::endl;
            device_sm = new SPC::CudaSparseMatrix<float>(*host_sm);
            std::cout << "created device_sm" << std::endl;
            
            // Print some information about the sparse matrix
            // std::cout << "Sparse matrix info:" << std::endl;
            // std::cout << "  Rows: " << device_sm->Rows() << std::endl;
            // std::cout << "  Columns: " << device_sm->Columns() << std::endl;
            // std::cout << "  Non-zeros: " << device_sm->Nonzeros() << std::endl;
            // std::cout << "  Segments: " << device_sm->n_segs << std::endl;
            // std::cout << "  Residue segments: " << device_sm->n_segs_residue << std::endl;

        } catch (const std::exception& e) {
            delete host_sm;
            if (dummy_seg_indices != nullptr) {
                cudaFree(dummy_seg_indices);
            }
            std::cerr << "Error initializing sparse matrix: " << e.what() << std::endl;
            throw std::runtime_error(std::string("Error initializing sparse matrix: ") + e.what());
        }
        std::cout << "DONE!!!!!!!!!!!!!" << std::endl;
    }
    
    ~CudaSpMat() {
        if (owns_data) {
            delete host_sm;
            delete device_sm;
        }
        if (dummy_seg_indices != nullptr) {
            cudaFree(dummy_seg_indices);
        }
        cudaStreamDestroy(stream1);
        cudaStreamDestroy(stream2);
    }
    
    py::object spmm(py::object B) {
        if (!is_cuda(B)) {
            throw std::runtime_error("Input tensor B must be a CUDA tensor");
        }
        
        // Get dimensions for B and output C
        py::tuple B_shape = B.attr("shape").cast<py::tuple>();
        if (B_shape.size() != 2) {
            throw std::runtime_error("Input tensor B must be 2-dimensional");
        }
        
        int B_rows = B_shape[0].cast<int>();
        int B_cols = B_shape[1].cast<int>();
        
        if (B_rows != cols) {
            throw std::runtime_error("Dimension mismatch: Input matrix B rows (" + 
                                    std::to_string(B_rows) + 
                                    ") must match sparse matrix columns (" + 
                                    std::to_string(cols) + ")");
        }
        
        // Get raw pointer from B
        float* d_B = get_cuda_ptr<float>(B);
        
        // Import torch to create output tensor
        py::object torch = py::module::import("torch");
        
        // Create output tensor C on GPU
        py::object C = torch.attr("zeros")(py::make_tuple(rows, B_cols), 
                                         py::arg("dtype") = torch.attr("float32"),
                                         py::arg("device") = B.attr("device"));
        
        // Get raw pointer from C
        float* d_C = get_cuda_ptr<float>(C);
        
        // Perform SPMM operation

        // std::cout << "SPMM operation info:" << std::endl;
        // std::cout << "  Matrix B shape: " << B_rows << "x" << B_cols << std::endl;
        // std::cout << "  Output C shape: " << rows << "x" << B_cols << std::endl;
        // std::cout << "  Using kernel: " << (B_cols < 128 ? "RoDeSpmm_n32" : "RoDeSpmm_n128") << std::endl;
        // std::cout << "  n_segs: " << device_sm->n_segs << std::endl;
        // std::cout << "  n_segs_residue: " << device_sm->n_segs_residue << std::endl;
        // std::cout << "  columns: " << device_sm->Columns() << std::endl;

        // Ensure n_segs is at least 1 for the kernel call to be valid
        // This is a workaround for the case where n_segs = 0
        int n_segs = device_sm->n_segs > 0 ? device_sm->n_segs : 1;
        
        // If n_segs was 0, we need to create a dummy seg_row_indices array to avoid passing nullptr
        int* seg_row_indices = device_sm->seg_row_indices;
        if (device_sm->n_segs == 0 && device_sm->seg_row_indices == nullptr) {
            // Allocate a dummy array with 1 element if n_segs was 0
            if (dummy_seg_indices == nullptr) {
                cudaError_t err = cudaMalloc(&dummy_seg_indices, sizeof(int));
                if (err != cudaSuccess) {
                    throw std::runtime_error("CUDA error during allocation of dummy indices: " + 
                                           std::string(cudaGetErrorString(err)));
                }
                int zero = 0;
                cudaMemcpy(dummy_seg_indices, &zero, sizeof(int), cudaMemcpyHostToDevice);
            }
            seg_row_indices = dummy_seg_indices;
        }

        // Call the appropriate SPMM kernel based on B_cols
        cudaError_t err;
        // if (B_cols < 128) {
        //     std::cout << "Calling RoDeSpmm_n32..." << std::endl;
        //     RoDeSpmm_n32(n_segs, device_sm->n_segs_residue, device_sm->Columns(), B_cols,
        //                 device_sm->Values(), device_sm->ColumnIndices(), device_sm->RowOffsets(),
        //                 seg_row_indices, device_sm->seg_row_indices_residue, device_sm->seg_st_offsets,
        //                 d_B, d_C, stream1, stream2);
        // } else {
            // std::cout << "Calling RoDeSpmm_n128..." << std::endl;
            RoDeSpmm_n128(device_sm->n_segs, device_sm->n_segs_residue, device_sm->Columns(), B_cols,
                            device_sm->Values(), device_sm->ColumnIndices(), device_sm->RowOffsets(),
                            device_sm->seg_row_indices, device_sm->seg_row_indices_residue, device_sm->seg_st_offsets,
                            d_B, d_C, stream1, stream2);
        // }
        
        // Check for any CUDA errors
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("CUDA error during SPMM: ") + cudaGetErrorString(err));
        }
        
        // Synchronize to ensure the computation is complete
        cudaStreamSynchronize(stream1);
        cudaStreamSynchronize(stream2);
        // std::cout << "SPMM completed successfully" << std::endl;

        
        return C;
    }
    
    int get_rows() const { return rows; }
    int get_cols() const { return cols; }
};

PYBIND11_MODULE(rodespmm, m) {
    m.doc() = "Python bindings for RoDe SpMM library";
    
    py::class_<CudaSpMat>(m, "CudaSpMat")
        .def(py::init<py::object, py::object, py::object, int, int, int, std::string>(),
             py::arg("row_offsets"), 
             py::arg("column_indices"), 
             py::arg("values"), 
             py::arg("rows"), 
             py::arg("cols"), 
             py::arg("seg_length") = 512,
             py::arg("device") = "GPU")
        .def("spmm", &CudaSpMat::spmm)
        .def_property_readonly("rows", &CudaSpMat::get_rows)
        .def_property_readonly("cols", &CudaSpMat::get_cols);
} 