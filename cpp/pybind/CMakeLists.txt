# setup PYTHON_EXECUTABLE

find_package (Python COMPONENTS Interpreter Development)
message(STATUS "Python_EXECUTABLE: " ${Python_EXECUTABLE})

add_subdirectory(pybind11)

pybind11_add_module(pysumopt MODULE pysumopt.cc)

target_link_libraries(pysumopt PUBLIC sumopt Eigen3::Eigen)
#target_compile_features(pysumopt PUBLIC cxx_std_17)
