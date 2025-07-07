#ifndef RANSAC_H_
#define RANSAC_H_
#include "alignment.h"

#include "types.h"
namespace sumopt {


RansacStats ransac_relpose_exhaustive(const std::vector<Eigen::Matrix<double, 9, 9>> &M, const std::vector<Eigen::Vector2d> &x1_center, const std::vector<Eigen::Vector2d> &x2_center, const std::vector<double> &thresholds, const RansacOptions &opt, CameraPose *output, std::vector<char> *inliers, bool single_refinement=false);


RansacStats ransac_relpose(const std::vector<Eigen::Vector2d> &x1_sample, const std::vector<Eigen::Vector2d> &x2_sample, const std::vector<Eigen::Matrix<double, 9, 9>> &M, const std::vector<Eigen::Vector2d> &x1_center, const std::vector<Eigen::Vector2d> &x2_center, const std::vector<double> &thresholds, const RansacOptions &opt, CameraPose *output, std::vector<char> *inliers);
double compute_sampson_msac_score(const CameraPose &pose, const std::vector<Eigen::Matrix<double, 9, 9>> &M, const std::vector<Eigen::Vector2d> &x1, const std::vector<Eigen::Vector2d> &x2, const std::vector<double> &thresholds, size_t *inlier_count);
int get_inliers(const CameraPose &pose,const std::vector<Eigen::Matrix<double, 9, 9>> &M, const std::vector<Eigen::Vector2d> &x1, const std::vector<Eigen::Vector2d> &x2, const std::vector<double> &thresholds, std::vector<char> *inliers);
double compute_sampson_msac_score(const CameraPose &pose, const std::vector<Eigen::Vector2d> &x1,
                                  const std::vector<Eigen::Vector2d> &x2, double sq_threshold, size_t *inlier_count);
}

#endif