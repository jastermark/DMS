#pragma once 
#include "alignment.h"

#include "types.h"

namespace sumopt {
int relpose_5pt(const std::vector<Eigen::Vector3d> &x1, const std::vector<Eigen::Vector3d> &x2,
                std::vector<CameraPose> *output);

int relpose_sum(const Eigen::Matrix<double, 9, 9> &M,
                const Eigen::Vector3d &x01,
                const Eigen::Vector3d &x02,
                std::vector<CameraPose> *output);

}