#pragma once

#include "quaternion.h"
#include "alignment.h"

#include "alignment.h"

#include <Eigen/Dense>
#include <vector>

namespace sumopt {

// Consistent with poselib::CameraPose
struct alignas(32) CameraPose {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    // Rotation is represented as a unit quaternion
    // with real part first, i.e. QW, QX, QY, QZ
    Eigen::Vector4d q;
    Eigen::Vector3d t;

    // Constructors (Defaults to identity camera)
    CameraPose() : q(1.0, 0.0, 0.0, 0.0), t(0.0, 0.0, 0.0) {}
    CameraPose(const Eigen::Vector4d &qq, const Eigen::Vector3d &tt) : q(qq), t(tt) {}
    CameraPose(const Eigen::Matrix3d &R, const Eigen::Vector3d &tt) : q(rotmat_to_quat(R)), t(tt) {}

    // Helper functions
    inline Eigen::Matrix3d R() const { return quat_to_rotmat(q); }
    inline Eigen::Matrix<double, 3, 4> Rt() const {
        Eigen::Matrix<double, 3, 4> tmp;
        tmp.block<3, 3>(0, 0) = quat_to_rotmat(q);
        tmp.col(3) = t;
        return tmp;
    }
    inline Eigen::Vector3d rotate(const Eigen::Vector3d &p) const { return quat_rotate(q, p); }
    inline Eigen::Vector3d derotate(const Eigen::Vector3d &p) const { return quat_rotate(quat_conj(q), p); }
    inline Eigen::Vector3d apply(const Eigen::Vector3d &p) const { return rotate(p) + t; }

    inline Eigen::Vector3d center() const { return -derotate(t); }
};


struct RansacOptions {
    bool score_approx = true;
    bool score_centers = false;
    double threshold_center = 1.0;
    size_t max_iterations = 100000;
    size_t min_iterations = 1000;
    double dyn_num_trials_mult = 3.0;
    double success_prob = 0.9999;
    double max_reproj_error = 12.0;  // used for 2D-3D matches
    double max_epipolar_error = 1.0; // used for 2D-2D matches
    unsigned long seed = 0;
    // If we should use PROSAC sampling. Assumes data is sorted
    bool progressive_sampling = false;
    size_t max_prosac_iterations = 100000;
    // Whether we should use real focal length checking: https://arxiv.org/abs/2311.16304
    // Assumes that principal points of both cameras are at origin.
    bool real_focal_check = false;
    // Minimum (effective) field-of-view to accept when estimating focal length
    // in degrees. Effective means based on the image points supplied
    // and not on the actual image size.
    // Setting to 0 (or negative) disables checking.
    double min_fov = 5.0; // circa 500mm lens 35mm-equivalent
};

struct RansacStats {
    size_t refinements = 0;
    size_t iterations = 0;
    size_t num_inliers = 0;
    double inlier_ratio = 0;
    double model_score = std::numeric_limits<double>::max();
};

struct BundleOptions {
    size_t max_iterations = 100;
    double gradient_tol = 1e-10;
    double step_tol = 1e-8;
    double initial_lambda = 1e-3;
    double min_lambda = 1e-10;
    double max_lambda = 1e10;
    bool verbose = false;
};

struct BundleStats {
    size_t iterations = 0;
    double initial_cost;
    double cost;
    double lambda;
    size_t invalid_steps;
    double step_norm;
    double grad_norm;
};


}