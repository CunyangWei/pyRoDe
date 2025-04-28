file(GLOB UTILS_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cc"
)
add_library(myutils SHARED ${UTILS_SOURCES})
set_target_properties(myutils PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(myutils ${SPC_LIBS})
