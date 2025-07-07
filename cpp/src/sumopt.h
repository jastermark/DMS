#pragma once
#include "alignment.h"

#include <Eigen/Dense>
#include "types.h"

namespace sumopt {


	BundleStats lm_sampson_approx(const std::vector<Eigen::Matrix<double,9,9>>& M,
		const std::vector<Eigen::Vector2d>& x01, const std::vector<Eigen::Vector2d>& x02,
		CameraPose* pose, const std::vector<double> &thresholds, const BundleOptions &opt = BundleOptions());
	
	BundleStats lm_sampson_approx_w_centers(const std::vector<Eigen::Matrix<double,9,9>>& M,
		const std::vector<Eigen::Vector2d>& x01, const std::vector<Eigen::Vector2d>& x02,
		CameraPose* pose, const std::vector<double> &thresholds, double threshold_center, const BundleOptions &opt = BundleOptions());

	BundleStats lm_sampson(const std::vector<Eigen::Vector2d>& x01, const std::vector<Eigen::Vector2d>& x02,
		CameraPose* pose, double threshold, const BundleOptions &opt = BundleOptions());


	BundleStats lm_fundamental_approx(const std::vector<Eigen::Matrix<double,9,9>>& M,
		const std::vector<Eigen::Vector2d>& x01, const std::vector<Eigen::Vector2d>& x02,
		Eigen::Matrix3d* Fmat, const std::vector<double> &thresholds, const BundleOptions &opt = BundleOptions());
	

	void lm_sampson_approx_ceres(const std::vector<Eigen::Matrix<double,9,9>>& M,
		const std::vector<Eigen::Vector2d>& x01, const std::vector<Eigen::Vector2d>& x02,
		CameraPose* pose, const std::vector<double> &thresholds,
		std::vector<CameraPose> *intermediate_poses = nullptr, std::vector<double> *intermediate_cost = nullptr);


}