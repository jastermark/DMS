

set(SOURCES
	sumopt.cc	
	baselines.cc
	essential.cc
	solvers.cc
	ransac.cc
	sampling.cc
	clustering.cc
)

add_library(sumopt STATIC ${SOURCES})
target_link_libraries(sumopt PRIVATE Eigen3::Eigen ${CERES_LIBRARIES})
#target_compile_options(sumopt PUBLIC -Wall -Werror -Wno-sign-compare -Wno-unused-variable -fPIC -march=native)
