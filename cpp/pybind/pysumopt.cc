#include "../src/alignment.h"

#include "pybind11_extension.h"
#include "../src/baselines.h"
#include "../src/sumopt.h"
#include "../src/solvers.h"
#include "../src/ransac.h"
#include "../src/clustering.h"

#include "helpers.h"
#include <iostream>
#include <pybind11/eigen.h>
#include <pybind11/iostream.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <tuple>

/*
static std::string toString(const Eigen::MatrixXd& mat){
    std::stringstream ss;
    ss << mat;
    return ss.str();
}
*/



namespace py = pybind11;

using namespace sumopt;


std::pair<CameraPose, py::dict> lm_sampson_distance_wrapper(const std::vector<Eigen::Vector2d>& x1, const std::vector<Eigen::Vector2d>& x2, CameraPose &init_pose, double thr) {
    CameraPose refined;
    refined.q = init_pose.q;
    refined.t = init_pose.t;
    BundleStats stats = lm_sampson(x1, x2, &refined, thr);
    py::dict info;
    write_to_dict(stats, info);
    return std::make_pair(refined, info);
}



std::pair<CameraPose, py::dict> lm_sampson_approx_ceres_wrapper(const std::vector<Eigen::Matrix<double,9,9>>& M, const std::vector<Eigen::Vector2d>& x01, const std::vector<Eigen::Vector2d>& x02, CameraPose &init_pose, const std::vector<double> &thresholds, bool save_iterations) {
    CameraPose refined;
    refined.q = init_pose.q;
    refined.t = init_pose.t;
    std::vector<CameraPose> intermediate_poses;
    std::vector<double> intermediate_costs;

    std::vector<Eigen::Matrix<double,9,9>> M_reordered;
    reorder_matrices(M, M_reordered);

    if(save_iterations) {   
        lm_sampson_approx_ceres(M_reordered, x01, x02, &refined, thresholds, &intermediate_poses, &intermediate_costs);
    } else {
        lm_sampson_approx_ceres(M_reordered, x01, x02, &refined, thresholds);
    }
    py::dict info;
    info["costs"] = intermediate_costs;
    info["poses"] = intermediate_poses;
    return std::make_pair(refined, info);
}

std::pair<CameraPose, py::dict> lm_sampson_approx_wrapper(const std::vector<Eigen::Matrix<double,9,9>>& M, const std::vector<Eigen::Vector2d>& x01, const std::vector<Eigen::Vector2d>& x02, CameraPose &init_pose, const std::vector<double> &thresholds) {
    CameraPose refined;
    refined.q = init_pose.q;
    refined.t = init_pose.t;
    std::vector<Eigen::Matrix<double,9,9>> M_reordered;
    reorder_matrices(M, M_reordered);

    BundleStats stats = lm_sampson_approx(M_reordered, x01, x02, &refined, thresholds);
    py::dict info;
    write_to_dict(stats, info);
    return std::make_pair(refined, info);
}


std::pair<Eigen::Matrix3d, py::dict> lm_fundamental_sampson_approx_wrapper(const std::vector<Eigen::Matrix<double,9,9>>& M, const std::vector<Eigen::Vector2d>& x01, const std::vector<Eigen::Vector2d>& x02, Eigen::Matrix3d &Fmat, const std::vector<double> &thresholds) {
    std::vector<Eigen::Matrix<double,9,9>> M_reordered;
    reorder_matrices(M, M_reordered);

    BundleOptions opt;
    opt.gradient_tol = 1e-12;
    opt.step_tol = 1e-12;
    Eigen::Matrix3d refined = Fmat;
    BundleStats stats = lm_fundamental_approx(M_reordered, x01, x02, &refined, thresholds, opt);
    py::dict info;
    write_to_dict(stats, info);
    return std::make_pair(refined, info);
}


std::pair<CameraPose, py::dict> lm_sampson_approx_w_centers_wrapper(const std::vector<Eigen::Matrix<double,9,9>>& M, const std::vector<Eigen::Vector2d>& x01, const std::vector<Eigen::Vector2d>& x02, CameraPose &init_pose, const std::vector<double> &thresholds, const double threshold_center) {
    CameraPose refined;
    refined.q = init_pose.q;
    refined.t = init_pose.t;
    std::vector<Eigen::Matrix<double,9,9>> M_reordered;
    reorder_matrices(M, M_reordered);

    BundleStats stats = lm_sampson_approx_w_centers(M_reordered, x01, x02, &refined, thresholds, threshold_center);
    py::dict info;

    write_to_dict(stats, info);
    return std::make_pair(refined, info);
}


std::vector<CameraPose> relpose_5pt_wrapper(const std::vector<Eigen::Vector3d> &x1,const std::vector<Eigen::Vector3d> &x2) {
    std::vector<CameraPose> poses;
    relpose_5pt(x1, x2, &poses);
    return poses;
}

std::vector<CameraPose> relpose_sum_wrapper(const Eigen::Matrix<double,9,9> &M, const Eigen::Vector3d &x01, const Eigen::Vector3d &x02) {
    std::vector<CameraPose> poses;
    Eigen::Matrix<double,9,9> M_reordered;
    reorder_matrix(M, M_reordered);
    relpose_sum(M_reordered, x01, x02, &poses);
    return poses;
}


std::pair<CameraPose, py::dict> ransac_relpose_wrapper(const std::vector<Eigen::Vector2d>& x1_sample, const std::vector<Eigen::Vector2d>& x2_sample, const std::vector<Eigen::Matrix<double,9,9>>& M, const std::vector<Eigen::Vector2d>& x01, const std::vector<Eigen::Vector2d>& x02, const std::vector<double> &thresholds, py::dict opt) {
    
    RansacOptions ransac_opt;
    update_ransac_options(opt, ransac_opt);
    
    CameraPose output;
    std::vector<Eigen::Matrix<double,9,9>> M_reordered;
    reorder_matrices(M, M_reordered);
    std::vector<char> inliers;
    RansacStats stats = ransac_relpose(x1_sample, x2_sample, M_reordered, x01, x02, thresholds, ransac_opt, &output, &inliers);
    py::dict info;
    write_to_dict(stats, info);
    info["inliers"] = convert_inlier_vector(inliers);
    return std::make_pair(output, info);
}


std::pair<CameraPose, py::dict> cluster_and_ransac_relpose_wrapper(const std::vector<Eigen::Vector2d>& x1, const std::vector<Eigen::Vector2d>& x2, int K, double threshold, int cluster_max_iter, py::dict opt) {
    std::vector<Eigen::Matrix<double,9,9>> M;
    std::vector<Eigen::Vector2d> centroid1, centroid2;
    std::vector<int> labels;
    std::vector<int> num_pts;
    cluster_and_summarize(x1, x2, K, cluster_max_iter, &labels, &M, &centroid1, &centroid2, &num_pts);

    
    RansacOptions ransac_opt;
    update_ransac_options(opt, ransac_opt);

    CameraPose output;
    std::vector<Eigen::Matrix<double,9,9>> M_reordered;
    reorder_matrices(M, M_reordered);
    std::vector<char> inliers;

    std::vector<double> thresholds;
    thresholds.resize(K);
    for(int i = 0; i < K; i++) {
        thresholds[i] = threshold * std::sqrt(num_pts[i]);
    }

    RansacStats stats = ransac_relpose(x1, x2, M_reordered, centroid1, centroid2, thresholds, ransac_opt, &output, &inliers);
    py::dict info;
    write_to_dict(stats, info);
    info["inliers"] = convert_inlier_vector(inliers);
    return std::make_pair(output, info);
}




std::pair<CameraPose, py::dict> ransac_relpose_exhaustive_wrapper(const std::vector<Eigen::Matrix<double,9,9>>& M, const std::vector<Eigen::Vector2d>& x01, const std::vector<Eigen::Vector2d>& x02, const std::vector<double> &thresholds, py::dict opt, bool single_refinement) {
    
    RansacOptions ransac_opt;
    update_ransac_options(opt, ransac_opt);
    
    CameraPose output;
    std::vector<Eigen::Matrix<double,9,9>> M_reordered;
    reorder_matrices(M, M_reordered);
    std::vector<char> inliers;
    RansacStats stats = ransac_relpose_exhaustive(M_reordered, x01, x02, thresholds, ransac_opt, &output, &inliers, single_refinement);
    py::dict info;
    write_to_dict(stats, info);
    info["inliers"] = convert_inlier_vector(inliers);
    return std::make_pair(output, info);
}


std::tuple<std::vector<int>, std::vector<Eigen::Vector2d>, std::vector<Eigen::Vector2d>, py::dict> kmeans_wrapper(const std::vector<Eigen::Vector2d>& x1, const std::vector<Eigen::Vector2d>& x2, int K, int max_iter) {

    std::vector<Eigen::Vector2d> centroid1, centroid2;
    std::vector<int> labels;

    int iters = kmeans(K, x1, x2, &labels, &centroid1, &centroid2, max_iter);

    py::dict info;
    info["iters"] = iters;

    return std::make_tuple(labels, centroid1, centroid2, info);
}

std::tuple<std::vector<Eigen::Matrix<double,9,9>>, std::vector<Eigen::Vector2d>, std::vector<Eigen::Vector2d>, std::vector<int>, std::vector<int>> cluster_and_summarize_wrapper(const std::vector<Eigen::Vector2d>& x1, const std::vector<Eigen::Vector2d>& x2, int K, int max_iter) {
    std::vector<Eigen::Matrix<double,9,9>> M;
    std::vector<Eigen::Vector2d> centroid1, centroid2;
    std::vector<int> labels;
    std::vector<int> num_pts;

    cluster_and_summarize(x1, x2, K, max_iter, &labels, &M, &centroid1, &centroid2, &num_pts);
    return std::make_tuple(M, centroid1, centroid2,labels, num_pts);
}


py::dict evaluate_approx(const std::vector<Eigen::Matrix<double,9,9>>& M, const std::vector<Eigen::Vector2d>& x01, const std::vector<Eigen::Vector2d>& x02, const CameraPose &pose, const std::vector<double> &thresholds, int trials) {
    std::vector<Eigen::Matrix<double,9,9>> M_reordered;
    reorder_matrices(M, M_reordered);

    std::vector<double> runtimes(trials, 0);
    double score;
    size_t inlier_count;
    for(int i = 0; i < trials; ++i) {
        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
        score = compute_sampson_msac_score(pose, M_reordered, x01, x02, thresholds, &inlier_count);
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        runtimes[i] = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    }    

    std::sort(runtimes.begin(), runtimes.end());
    double median_runtime = runtimes[trials/2];
    py::dict info;
    info["score"] = score;
    info["inlier_count"] = inlier_count;
    info["time"] = median_runtime;
    return info;
}

py::dict evaluate_sampson(const std::vector<Eigen::Vector2d>& x1, const std::vector<Eigen::Vector2d>& x2, const CameraPose &pose, const double threshold, int trials) {
    
    size_t inlier_count;

    std::vector<double> runtimes(trials, 0);

    double score;
    for(int i = 0; i < trials; ++i) {
        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
        score = compute_sampson_msac_score(pose, x1, x2, threshold * threshold, &inlier_count);
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        runtimes[i] = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    }

    std::sort(runtimes.begin(), runtimes.end());
    double median_runtime = runtimes[trials/2];

    py::dict info;
    info["score"] = score;
    info["inlier_count"] = inlier_count;
    info["time"] = median_runtime;
    return info;
}

PYBIND11_MODULE(pysumopt, m) {

    m.doc() = "Optimization for keypoint match summarization.";

    py::class_<sumopt::CameraPose>(m, "CameraPose")
        .def(py::init<>())
        .def_readwrite("q", &sumopt::CameraPose::q)
        .def_readwrite("t", &sumopt::CameraPose::t)
        .def_property("R", &sumopt::CameraPose::R,
                      [](sumopt::CameraPose &self, Eigen::Matrix3d R_new) { self.q = sumopt::rotmat_to_quat(R_new); })
        .def_property("Rt", &sumopt::CameraPose::Rt,
                      [](sumopt::CameraPose &self, Eigen::Matrix<double, 3, 4> Rt_new) {
                          self.q = sumopt::rotmat_to_quat(Rt_new.leftCols<3>());
                          self.t = Rt_new.col(3);
                      })
        .def("center", &sumopt::CameraPose::center, "Returns the camera center (c=-R^T*t).")
        .def("__repr__", [](const sumopt::CameraPose &a) {
            return "[q: " + toString(a.q.transpose()) + ", " + "t: " + toString(a.t.transpose()) + "]";
        });

    // Sampson error
    m.def("lm_sampson", &lm_sampson_distance_wrapper, py::arg("x1"), py::arg("x2"),  py::arg("init_pose"), py::arg("threshold"), "Sampson refinement",
          py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>());
    m.def("lm_sampson_approx", &lm_sampson_approx_wrapper, py::arg("M"), py::arg("x01"), py::arg("x02"),  py::arg("init_pose"), py::arg("thresholds"), "Sampson refinement",
          py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>());
    m.def("lm_sampson_approx_w_centers", &lm_sampson_approx_w_centers_wrapper, py::arg("M"), py::arg("x01"), py::arg("x02"),  py::arg("init_pose"), py::arg("thresholds"), py::arg("threshold_center"), "Sampson refinement",
          py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>());
    m.def("lm_fundamental_sampson_approx", &lm_fundamental_sampson_approx_wrapper, py::arg("M"), py::arg("x01"), py::arg("x02"),  py::arg("init_Fmat"), py::arg("thresholds"), "Sampson refinement of fundamental matrix",
          py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>());
    
    m.def("lm_sampson_approx_ceres", &lm_sampson_approx_ceres_wrapper, py::arg("M"), py::arg("x01"), py::arg("x02"),  py::arg("init_pose"), py::arg("thresholds"), py::arg("save_iterations") = false, "Sampson refinement",
          py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>());

    // Solvers
     m.def("relpose_5pt", &relpose_5pt_wrapper, py::arg("x1"), py::arg("x2"), "Classic 5 point solver",
          py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>());
    m.def("relpose_sum", &relpose_sum_wrapper, py::arg("M"), py::arg("x01"), py::arg("x02"), "Relpose solver from cluster",
          py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>());

     m.def("ransac_relpose", &ransac_relpose_wrapper, py::arg("x1_sample"), py::arg("x2_sample"), py::arg("M"), py::arg("x01"), py::arg("x02"), py::arg("thresholds"), py::arg("opt") = py::dict(), "LO-RANSAC. Scoring with summarized correspondences. Generates models with points from (x1_sample,x2_sample).",
          py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>());
     m.def("ransac_relpose_exhaustive", &ransac_relpose_exhaustive_wrapper, py::arg("M"), py::arg("x01"), py::arg("x02"), py::arg("thresholds"), py::arg("opt") = py::dict(), py::arg("single_refinement") = false, "LO-RANSAC. Scoring with summarized correspondences. Generates models with points from (x1_sample,x2_sample).",
          py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>());

    m.def("kmeans", &kmeans_wrapper, py::arg("x1"), py::arg("x2"), py::arg("K"), py::arg("max_iter"), "K-means clustering",
          py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>());

    m.def("cluster_and_summarize", &cluster_and_summarize_wrapper, py::arg("x1"), py::arg("x2"), py::arg("K"), py::arg("max_iter"), "Cluster and summarize",
          py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>());

    m.def("cluster_and_ransac_relpose", &cluster_and_ransac_relpose_wrapper, py::arg("x1"), py::arg("x2"), py::arg("K"), py::arg("threshold"), py::arg("cluster_max_iter"), py::arg("opt") = py::dict(), "Cluster and summarize, and then run RANSAC",
          py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>());


    m.def("evaluate_approx", &evaluate_approx, py::arg("M"), py::arg("x01"), py::arg("x02"), py::arg("pose"), py::arg("thresholds"), py::arg("trials") = 1, "Evaluate approx",
          py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>());
    
    m.def("evaluate_sampson", &evaluate_sampson, py::arg("x1"), py::arg("x2"), py::arg("pose"), py::arg("threshold"), py::arg("trials") = 1, "Evaluate sampson error",
          py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>());

    m.attr("__version__") = std::string("0.0.1");
}
