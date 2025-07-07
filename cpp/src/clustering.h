#pragma once
#include "alignment.h"
#include <Eigen/Dense>
#include <vector>

namespace sumopt {   
    int kmeans(int k, 
                const std::vector<Eigen::Vector2d>& x1,
                const std::vector<Eigen::Vector2d> &x2,
                std::vector<int>* labels,
                std::vector<Eigen::Vector2d>* centroid1,
                std::vector<Eigen::Vector2d>* centroid2,
                int max_iter = 5);

    int cluster_and_summarize(const std::vector<Eigen::Vector2d>& x1,
                              const std::vector<Eigen::Vector2d>& x2,
                              int K,
                              int max_iter,
                              std::vector<int>* labels,
                              std::vector<Eigen::Matrix<double,9,9>>* M,
                              std::vector<Eigen::Vector2d>* centroid1,
                              std::vector<Eigen::Vector2d>* centroid2,
                              std::vector<int> *num_pts);
}