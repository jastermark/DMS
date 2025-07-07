#ifndef RELPOSEBA_PYBIND_HELPERS_H_
#define RELPOSEBA_PYBIND_HELPERS_H_

#include <pybind11/eigen.h>
#include <pybind11/iostream.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "../src/types.h"
using namespace sumopt;
namespace py = pybind11;

static std::string toString(const Eigen::MatrixXd &mat) {
    std::stringstream ss;
    ss << mat;
    return ss.str();
}

void reorder_matrix(const Eigen::Matrix<double, 9, 9> &M, Eigen::Matrix<double, 9, 9> &M_reordered) {
    M_reordered.col(0) = M.col(0);
    M_reordered.col(1) = M.col(3);
    M_reordered.col(2) = M.col(6);  
    M_reordered.col(3) = M.col(1); 
    M_reordered.col(4) = M.col(4); 
    M_reordered.col(5) = M.col(7);
    M_reordered.col(6) = M.col(2);
    M_reordered.col(7) = M.col(5);
    M_reordered.col(8) = M.col(8);
}
void reorder_matrices(const std::vector<Eigen::Matrix<double,9,9>> &M, std::vector<Eigen::Matrix<double,9,9>> &M_reordered) {
    M_reordered.resize(M.size());
    for (size_t i = 0; i < M.size(); ++i) {
        reorder_matrix(M[i], M_reordered[i]);
    }
}

template <typename T> void update(const py::dict &input, const std::string &name, T &value) {
    if (input.contains(name)) {
        value = input[name.c_str()].cast<T>();
    }
}
template <> void update(const py::dict &input, const std::string &name, bool &value) {
    if (input.contains(name)) {
        py::object input_value = input[name.c_str()];
        value = (py::str(input_value).is(py::str(Py_True)));
    }
}

void update_ransac_options(const py::dict &input, RansacOptions &ransac_opt) {
    update(input, "score_approx", ransac_opt.score_approx);
    update(input, "score_centers", ransac_opt.score_centers);
    update(input, "threshold_center", ransac_opt.threshold_center);
    update(input, "max_iterations", ransac_opt.max_iterations);
    update(input, "min_iterations", ransac_opt.min_iterations);
    update(input, "dyn_num_trials_mult", ransac_opt.dyn_num_trials_mult);
    update(input, "success_prob", ransac_opt.success_prob);
    update(input, "seed", ransac_opt.seed);
    update(input, "progressive_sampling", ransac_opt.progressive_sampling);
    update(input, "max_prosac_iterations", ransac_opt.max_prosac_iterations);
}



void write_to_dict(const RansacStats &stats, py::dict &dict) {
    dict["refinements"] = stats.refinements;
    dict["iterations"] = stats.iterations;
    dict["num_inliers"] = stats.num_inliers;
    dict["inlier_ratio"] = stats.inlier_ratio;
    dict["model_score"] = stats.model_score;
}
void write_to_dict(const BundleStats &stats, py::dict &dict) {
    dict["iterations"] = stats.iterations;
    dict["cost"] = stats.cost;
    dict["initial_cost"] = stats.initial_cost;
    dict["invalid_steps"] = stats.invalid_steps;
    dict["grad_norm"] = stats.grad_norm;
    dict["step_norm"] = stats.step_norm;
    dict["lambda"] = stats.lambda;
}

std::vector<bool> convert_inlier_vector(const std::vector<char> &inliers) {
    std::vector<bool> inliers_bool(inliers.size());
    for (size_t k = 0; k < inliers.size(); ++k) {
        inliers_bool[k] = static_cast<bool>(inliers[k]);
    }
    return inliers_bool;
}
#endif