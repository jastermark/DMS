#include "clustering.h"

namespace sumopt {
int kmeans(int K, 
            const std::vector<Eigen::Vector2d>& x1,
            const std::vector<Eigen::Vector2d> &x2,
            std::vector<int>* labels,
            std::vector<Eigen::Vector2d>* centroid1,
            std::vector<Eigen::Vector2d>* centroid2,
            int max_iter) {
    labels->resize(x1.size(), -1);
    centroid1->resize(K);
    centroid2->resize(K);

    // Initialize centroids
    for(int i = 0; i < K; i++) {
        (*centroid1)[i] = x1[i * x1.size() / K];
        (*centroid2)[i] = x2[i * x2.size() / K];
    }

    int iter;
    for(iter = 0; iter < max_iter; ++iter) {
        // Assign points to clusters
        bool updated = false;
        for(int i = 0; i < x1.size(); i++) {
            double min_dist = std::numeric_limits<double>::max();
            int min_idx = -1;
            for(int j = 0; j < K; j++) {
                double dist = (x1[i] - (*centroid1)[j]).squaredNorm() + (x2[i] - (*centroid2)[j]).squaredNorm();
                if(dist < min_dist) {
                    min_dist = dist;
                    min_idx = j;
                }
            }
            //if((*labels)[i] != min_idx) {
            //    updated = true;
            //}
            (*labels)[i] = min_idx;
        }
        // Update centroids
        for(int i = 0; i < K; i++) {
            (*centroid1)[i].setZero();
            (*centroid2)[i].setZero();
        }

        std::vector<int> count(K, 0);
        for(int i = 0; i < x1.size(); i++) {
            (*centroid1)[(*labels)[i]] += x1[i];
            (*centroid2)[(*labels)[i]] += x2[i];
            count[(*labels)[i]]++;
        }

        for(int i = 0; i < K; i++) {
            if(count[i] > 0) {
                (*centroid1)[i] /= count[i];
                (*centroid2)[i] /= count[i];
            }
        }

        //if(!updated) {
        //    break;
        //}
    }
    return iter;
}

 int cluster_and_summarize(const std::vector<Eigen::Vector2d>& pts1,
                              const std::vector<Eigen::Vector2d>& pts2,
                              int K,
                              int max_iter,
                              std::vector<int>* labels,
                              std::vector<Eigen::Matrix<double,9,9>>* M,
                              std::vector<Eigen::Vector2d>* centroid1,
                              std::vector<Eigen::Vector2d>* centroid2,
                              std::vector<int> *num_pts) {
    int iters = kmeans(K, pts1, pts2, labels, centroid1, centroid2, max_iter);
    M->resize(K);
    for(int i = 0; i < K; i++) {
        M->at(i).setZero();
    }

    Eigen::Matrix<double,9,1> C;
    num_pts->resize(K, 0);
    for(int j = 0; j < pts1.size(); j++) {
        double x1 = pts1[j](0);
        double y1 = pts1[j](1);
        double x2 = pts2[j](0);
        double y2 = pts2[j](1);
        C << x2*x1, x2*y1, x2, y2*x1, y2*y1, y2, x1, y1, 1.0;
        //C << x1*x2, x1*y2, x1, y1*x2, y1*y2, y1, x2, y2, 1.0;
   
        M->at((*labels)[j]) += C * C.transpose();
        (*num_pts)[(*labels)[j]]++;
    }

    for(int i = 0; i < K; i++) {
        (*M)[i] = (*M)[i].llt().matrixL().transpose();
    }                      

    return iters;
}
} // namespace sumopt