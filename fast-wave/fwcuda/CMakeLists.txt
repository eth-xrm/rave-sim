add_library(fwcuda kernels.cu)

target_include_directories(fwcuda PUBLIC ${CMAKE_CURRENT_LIST_DIR})
target_link_libraries(fwcuda PUBLIC CUDA::cufft)
set_target_properties(fwcuda PROPERTIES CUDA_ARCHITECTURES "70;75")
target_compile_features(fwcuda PUBLIC cxx_std_17)
