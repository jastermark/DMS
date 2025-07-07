cmake_minimum_required(VERSION 3.10)

project(sumopt)

find_package(Eigen3 REQUIRED 3.4.0)

find_package(Ceres 2.2.0 REQUIRED EXACT)
include_directories(${CERES_INCLUDE_DIRS})
include_directories("/usr/include/")
message(CERES_INCLUDE_DIRS="${CERES_INCLUDE_DIRS}")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)


if(MSVC)
	add_definitions(-D_USE_MATH_DEFINES)
	#add_compile_options(/bigobj /fp:fast)	
	add_compile_options(/bigobj)	
else()
	add_compile_options(-Wall -Werror -Wno-array-bounds -Wno-sign-compare -Wno-unused-variable -fPIC -O3 -march=native)
endif()

# python bindings
add_subdirectory(pybind)




include_directories(src)
add_subdirectory(src)

