#pragma once

#include "alignment.h"
#include <Eigen/Dense>
#include "types.h"

namespace sumopt {

	void lm_sampson_distance(const std::vector<Eigen::Vector2d>& x1, const std::vector<Eigen::Vector2d>& x2, CameraPose* pose, double threshold);

}