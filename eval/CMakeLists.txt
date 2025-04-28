include_directories(${UTILS_LIBS})

include_directories(${RoDe_SPMM_LIBS})
include_directories(${RoDe_SDDMM_LIBS})

add_executable(eval_spmm_f32_inputN eval_spmm_f32_inputN.cu)
target_link_libraries(eval_spmm_f32_inputN myutils RoDe_spmm cudart cusparse)

add_executable(eval_spmm_mtx_reader eval_spmm_mtx_reader.cu)
target_link_libraries(eval_spmm_mtx_reader myutils RoDe_spmm cudart cusparse)
