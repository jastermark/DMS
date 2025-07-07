#include "sumopt.h"
#include <ceres/ceres.h>
#include "jacobian_impl.h"
#include "lm_impl.h"

namespace sumopt {

    struct SampsonApproxError {
        SampsonApproxError(const Eigen::Matrix<double,9,9> &M, const Eigen::Vector2d& x01, const Eigen::Vector2d& x02, double threshold) : M_(M), x1_(x01), x2_(x02), thr(threshold) {}


        template <typename T>
        bool operator()(const T* const qvec, const T* tvec, T* residuals) const {
            Eigen::Quaternion<T> q;
            q.coeffs() << qvec[0], qvec[1], qvec[2], qvec[3];

            Eigen::Matrix<T, 3, 1> t{ tvec[0],tvec[1],tvec[2] };

            Eigen::Matrix<T, 3, 3> E;
            E << T(0.0), -t(2), t(1),
                t(2), T(0.0), -t(0),
                -t(1), t(0), T(0.0);

            E = E * q.toRotationMatrix();

            Eigen::Matrix<T, 2, 1> n1 = E.template block<2, 3>(0, 0) * x1_.cast<T>().homogeneous();
            Eigen::Matrix<T, 2, 1> n2 = E.transpose().template block<2, 3>(0, 0) * x2_.cast<T>().homogeneous();
            T denom = ceres::sqrt(n1.dot(n1) + n2.dot(n2));

            Eigen::Matrix<T, 9, 1> res = M_ * Eigen::Map<Eigen::Matrix<T, 9, 1>>(E.data()) / denom;
            T res2 = res.transpose() * res;

            if(thr < 0 || res2 <= thr * thr) {
                for(int i = 0; i < 9; ++i) {
                    residuals[i] = res[i];
                }
            } else {
                for(int i = 0; i < 9; ++i) {
                    residuals[i] = T(thr / 3.0); // divide by 3 here to make sure ||r|| = thr
                }
            }
            
            return true;
        }

        // Factory function
        static ceres::CostFunction* CreateCost(const Eigen::Matrix<double,9,9> &M, const Eigen::Vector2d& x1, const Eigen::Vector2d& x2, double threshold) {
            return (new ceres::AutoDiffCostFunction<SampsonApproxError, 9, 4, 3>(
                new SampsonApproxError(M, x1, x2, threshold)));
        }

        const Eigen::Matrix<double,9,9> &M_;
        const Eigen::Vector2d &x1_, &x2_;
        double thr;
    };


class IterationCallbackImpl : public ceres::IterationCallback {
public:
    IterationCallbackImpl(CameraPose *p, std::vector<Eigen::Vector3d> *n, std::vector<CameraPose> *v,
        std::vector<std::vector<Eigen::Vector3d>> *nn, std::vector<double> *c) : pose(p), normals(n), intermediate_poses(v), intermediate_normals(nn), costs(c) {}
    ~IterationCallbackImpl() override = default;

    ceres::CallbackReturnType operator()(const ceres::IterationSummary &summary) {
        if(pose != nullptr && intermediate_poses != nullptr) {
            intermediate_poses->push_back(*pose);
        }
        if(costs != nullptr) {
            costs->push_back(summary.cost);
        }
        if(normals != nullptr && intermediate_normals != nullptr) {
            intermediate_normals->push_back(*normals);
        }
        return ceres::SOLVER_CONTINUE;
    }

    CameraPose *pose = nullptr;
    std::vector<Eigen::Vector3d> *normals = nullptr;
    std::vector<CameraPose> *intermediate_poses = nullptr;
    std::vector<std::vector<Eigen::Vector3d>> *intermediate_normals = nullptr;
    std::vector<double> *costs = nullptr;
};

void lm_sampson_approx_ceres(const std::vector<Eigen::Matrix<double,9,9>>& M, const std::vector<Eigen::Vector2d>& x01, const std::vector<Eigen::Vector2d>& x02,
                CameraPose* pose, const std::vector<double> &thresholds, std::vector<CameraPose> *intermediate_poses, std::vector<double> *intermediate_cost) {
        if (M.size() == 0)
            return;

        bool save_iterations = intermediate_poses || intermediate_cost;

        ceres::Problem problem;

        Eigen::Quaterniond q(pose->R());

        for (int i = 0; i < M.size(); ++i) {
            ceres::CostFunction* cost = SampsonApproxError::CreateCost(M[i], x01[i], x02[i], thresholds[i]);
            problem.AddResidualBlock(cost, nullptr, q.coeffs().data(), pose->t.data());
        }
        problem.AddParameterBlock(q.coeffs().data(), 4, new ceres::EigenQuaternionManifold());
        problem.AddParameterBlock(pose->t.data(), 3, new ceres::SphereManifold<3>());

        ceres::Solver::Options options;
        options.linear_solver_type = ceres::DENSE_QR;
        options.update_state_every_iteration = save_iterations;
        options.minimizer_progress_to_stdout = false;
        options.gradient_tolerance = 1e-12;
        options.parameter_tolerance = 1e-12;
        options.function_tolerance = 1e-12;
        ceres::Solver::Summary summary;

        if(save_iterations) {
            IterationCallbackImpl callback(pose, nullptr, intermediate_poses, nullptr, intermediate_cost);
            options.callbacks.push_back(&callback);
            ceres::Solve(options, &problem, &summary);
        } else {
            ceres::Solve(options, &problem, &summary);
        }

        pose->q = rotmat_to_quat(q.toRotationMatrix());
        //std::cout << summary.FullReport() << std::endl;

}


void print_iteration(const BundleStats &stats) {
    if (stats.iterations == 0) {
        std::cout << "initial_cost=" << stats.initial_cost << "\n";
    }
    std::cout << "iter=" << stats.iterations << ", cost=" << stats.cost << ", step=" << stats.step_norm
              << ", grad=" << stats.grad_norm << ", lambda=" << stats.lambda << "\n";
}
BundleStats lm_sampson_approx(const std::vector<Eigen::Matrix<double,9,9>>& M, const std::vector<Eigen::Vector2d>& x01, const std::vector<Eigen::Vector2d>& x02,
                CameraPose* pose, const std::vector<double> &thresholds, const BundleOptions &opt) {
        if (M.size() == 0)
            return BundleStats();
        
        SummarizedRelativePoseJacobianAccumulator acc(M, x01, x02, thresholds);

        if(opt.verbose) {
            return lm_impl<decltype(acc)>(acc, pose, opt, print_iteration);
        } else {
            return lm_impl<decltype(acc)>(acc, pose, opt, nullptr);
        }
}

BundleStats lm_sampson_approx_w_centers(const std::vector<Eigen::Matrix<double,9,9>>& M,
		const std::vector<Eigen::Vector2d>& x01, const std::vector<Eigen::Vector2d>& x02,
		CameraPose* pose, const std::vector<double> &thresholds, double threshold_center, const BundleOptions &opt) {

        if (M.size() == 0)
            return BundleStats();
        
        SummarizedRelativePoseJacobianAccumulator summarized_acc(M, x01, x02, thresholds);
        RelativePoseJacobianAccumulator relative_acc(x01, x02, threshold_center);
        HybridAccumulator acc(summarized_acc, relative_acc);

        if(opt.verbose) {
            return lm_impl<decltype(acc)>(acc, pose, opt, print_iteration);
        } else {
            return lm_impl<decltype(acc)>(acc, pose, opt, nullptr);
        }
}
	
BundleStats lm_sampson(const std::vector<Eigen::Vector2d>& x01, const std::vector<Eigen::Vector2d>& x02,
		CameraPose* pose, double threshold, const BundleOptions &opt) {

        if (x01.size() == 0)
            return BundleStats();
        
        RelativePoseJacobianAccumulator acc(x01, x02, threshold);

        if(opt.verbose) {
            return lm_impl<decltype(acc)>(acc, pose, opt, print_iteration);
        } else {
            return lm_impl<decltype(acc)>(acc, pose, opt, nullptr);
        }
}

BundleStats lm_fundamental_approx(const std::vector<Eigen::Matrix<double,9,9>>& M,
		const std::vector<Eigen::Vector2d>& x01, const std::vector<Eigen::Vector2d>& x02,
		Eigen::Matrix3d* Fmat, const std::vector<double> &thresholds, const BundleOptions &opt) {
        if (M.size() == 0)
            return BundleStats();
        
        SummarizedFundamentalJacobianAccumulator acc(M, x01, x02, thresholds);
        FactorizedFundamentalMatrix FF(*Fmat);
        BundleStats stats;
        if(opt.verbose) {
            stats = lm_impl<decltype(acc)>(acc, &FF, opt, print_iteration);
        } else {
            stats = lm_impl<decltype(acc)>(acc, &FF, opt, nullptr);
        }
        *Fmat = FF.F();
        return stats;
}


}