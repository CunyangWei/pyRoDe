aux_source_directory(. DIR_OUR_SPMM_SRCS)

include_directories(${UTILS_LIBS})
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_library(RoDe_spmm ${DIR_OUR_SPMM_SRCS})
target_link_libraries(RoDe_spmm myutils)