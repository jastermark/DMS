#include "baselines.h"
#include <Eigen/Dense>
#include <ceres/ceres.h>


namespace sumopt {

    struct SampsonError {
        SampsonError(const Eigen::Vector2d& x1, const Eigen::Vector2d& x2, double threshold) : x1_(x1), x2_(x2), thr(threshold) {}


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

            
            T ep = x2_.cast<T>().homogeneous().dot(E * x1_.cast<T>().homogeneous());

            Eigen::Matrix<T, 2, 1> n1 = E.template block<2, 3>(0, 0) * x1_.cast<T>().homogeneous();
            Eigen::Matrix<T, 2, 1> n2 = E.transpose().template block<2, 3>(0, 0) * x2_.cast<T>().homogeneous();

            T res = ep / ceres::sqrt(n1.dot(n1) + n2.dot(n2));

            if(thr < 0 || ceres::abs(res) <= thr) {
                residuals[0] = res;
            } else {
                residuals[0] = T(thr);
            }
            
            return true;
        }

        // Factory function
        static ceres::CostFunction* CreateCost(const Eigen::Vector2d& x1, const Eigen::Vector2d& x2, double threshold) {
            return (new ceres::AutoDiffCostFunction<SampsonError, 1, 4, 3>(
                new SampsonError(x1, x2, threshold)));
        }

        Eigen::Vector2d x1_, x2_;
        double thr;
    };

	void lm_sampson_distance(const std::vector<Eigen::Vector2d>& x1, const std::vector<Eigen::Vector2d>& x2, CameraPose* pose, double threshold) {
        if (x1.size() < 5)
            return;

        ceres::Problem problem;

        Eigen::Quaterniond q(pose->R());

        for (int i = 0; i < x1.size(); ++i) {
            ceres::CostFunction* cost = SampsonError::CreateCost(x1[i], x2[i], threshold);
            problem.AddResidualBlock(cost, nullptr, q.coeffs().data(), pose->t.data());
        }
        problem.AddParameterBlock(q.coeffs().data(), 4, new ceres::EigenQuaternionManifold());
        problem.AddParameterBlock(pose->t.data(), 3, new ceres::SphereManifold<3>());

        ceres::Solver::Options options;
        options.linear_solver_type = ceres::DENSE_QR;
        options.minimizer_progress_to_stdout = false;
        options.gradient_tolerance = 1e-12;
        options.parameter_tolerance = 1e-12;
        options.function_tolerance = 1e-12;
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);

        pose->q = rotmat_to_quat(q.toRotationMatrix());
        //std::cout << summary.FullReport() << std::endl;
	}

}
