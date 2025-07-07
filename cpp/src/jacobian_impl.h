// Copyright (c) 2021, Viktor Larsson
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of the copyright holder nor the
//       names of its contributors may be used to endorse or promote products
//       derived from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL COPYRIGHT HOLDERS OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#ifndef POSELIB_JACOBIAN_IMPL_H_
#define POSELIB_JACOBIAN_IMPL_H_
#include "alignment.h"

#include "essential.h"
#include "types.h"
#include "ransac_impl.h"

namespace sumopt {

class SummarizedRelativePoseJacobianAccumulator {
  public:
    SummarizedRelativePoseJacobianAccumulator(const std::vector<Eigen::Matrix<double,9,9>> &mats,
                                    const std::vector<Eigen::Vector2d> &x1, const std::vector<Eigen::Vector2d> &x2,
                                    const std::vector<double> &thrs)
        : M(mats), x01(x1), x02(x2), thresholds(thrs) {}

    double residual(const CameraPose &pose) const {
        Eigen::Matrix3d E;
        essential_from_motion(pose, &E);
        Eigen::Vector<double, 9> E_vec = Eigen::Map<Eigen::Vector<double, 9>>(E.data());

        double cost = 0.0;
        for (size_t k = 0; k < M.size(); ++k) {

            double C2 = (M[k] * E_vec).squaredNorm();
            double nJc_sq = (E.block<2, 3>(0, 0) * x01[k].homogeneous()).squaredNorm() +
                            (E.block<3, 2>(0, 0).transpose() * x02[k].homogeneous()).squaredNorm();

            double r2 = C2 / nJc_sq;
            cost += std::min(r2, thresholds[k] * thresholds[k]);
        }

        return cost;
    }

    size_t accumulate(const CameraPose &pose, Eigen::Matrix<double, 5, 5> &JtJ, Eigen::Matrix<double, 5, 1> &Jtr) {
        // We start by setting up a basis for the updates in the translation (orthogonal to t)
        // We find the minimum element of t and cross product with the corresponding basis vector.
        // (this ensures that the first cross product is not close to the zero vector)
        if (std::abs(pose.t.x()) < std::abs(pose.t.y())) {
            // x < y
            if (std::abs(pose.t.x()) < std::abs(pose.t.z())) {
                tangent_basis.col(0) = pose.t.cross(Eigen::Vector3d::UnitX()).normalized();
            } else {
                tangent_basis.col(0) = pose.t.cross(Eigen::Vector3d::UnitZ()).normalized();
            }
        } else {
            // x > y
            if (std::abs(pose.t.y()) < std::abs(pose.t.z())) {
                tangent_basis.col(0) = pose.t.cross(Eigen::Vector3d::UnitY()).normalized();
            } else {
                tangent_basis.col(0) = pose.t.cross(Eigen::Vector3d::UnitZ()).normalized();
            }
        }
        tangent_basis.col(1) = tangent_basis.col(0).cross(pose.t).normalized();

        Eigen::Matrix3d E, R;
        R = pose.R();
        essential_from_motion(pose, &E);
        Eigen::Vector<double, 9> E_vec = Eigen::Map<Eigen::Vector<double, 9>>(E.data());

        // Matrices contain the jacobians of E w.r.t. the rotation and translation parameters
        Eigen::Matrix<double, 9, 3> dR;
        Eigen::Matrix<double, 9, 2> dt;

        // Each column is vec(E*skew(e_k)) where e_k is k:th basis vector
        dR.block<3, 1>(0, 0).setZero();
        dR.block<3, 1>(0, 1) = -E.col(2);
        dR.block<3, 1>(0, 2) = E.col(1);
        dR.block<3, 1>(3, 0) = E.col(2);
        dR.block<3, 1>(3, 1).setZero();
        dR.block<3, 1>(3, 2) = -E.col(0);
        dR.block<3, 1>(6, 0) = -E.col(1);
        dR.block<3, 1>(6, 1) = E.col(0);
        dR.block<3, 1>(6, 2).setZero();

        // Each column is vec(skew(tangent_basis[k])*R)
        dt.block<3, 1>(0, 0) = tangent_basis.col(0).cross(R.col(0));
        dt.block<3, 1>(0, 1) = tangent_basis.col(1).cross(R.col(0));
        dt.block<3, 1>(3, 0) = tangent_basis.col(0).cross(R.col(1));
        dt.block<3, 1>(3, 1) = tangent_basis.col(1).cross(R.col(1));
        dt.block<3, 1>(6, 0) = tangent_basis.col(0).cross(R.col(2));
        dt.block<3, 1>(6, 1) = tangent_basis.col(1).cross(R.col(2));

        size_t num_residuals = 0;
        for (size_t k = 0; k < M.size(); ++k) {
            Eigen::Vector<double, 9> Me = M[k] * E_vec;
            double C = Me.norm();

            // J_C is the Jacobian of the epipolar constraint w.r.t. the image points
            Eigen::Vector4d J_C;
            J_C << E.block<3, 2>(0, 0).transpose() * x02[k].homogeneous(), E.block<2, 3>(0, 0) * x01[k].homogeneous();
            
            Eigen::Matrix<double,9,1> denom_de;
            denom_de << x01[k](0)*(E_vec(6) + E_vec(0)*x01[k](0) + E_vec(3)*x01[k](1)) + x02[k](0)*(E_vec(2) + E_vec(0)*x02[k](0) + E_vec(1)*x02[k](1)), x01[k](0)*(E_vec(7) + E_vec(1)*x01[k](0) + E_vec(4)*x01[k](1)) + x02[k](1)*(E_vec(2) + E_vec(0)*x02[k](0) + E_vec(1)*x02[k](1)), E_vec(2) + E_vec(0)*x02[k](0) + E_vec(1)*x02[k](1), x01[k](1)*(E_vec(6) + E_vec(0)*x01[k](0) + E_vec(3)*x01[k](1)) + x02[k](0)*(E_vec(5) + E_vec(3)*x02[k](0) + E_vec(4)*x02[k](1)), x01[k](1)*(E_vec(7) + E_vec(1)*x01[k](0) + E_vec(4)*x01[k](1)) + x02[k](1)*(E_vec(5) + E_vec(3)*x02[k](0) + E_vec(4)*x02[k](1)), E_vec(5) + E_vec(3)*x02[k](0) + E_vec(4)*x02[k](1), E_vec(6) + E_vec(0)*x01[k](0) + E_vec(3)*x01[k](1), E_vec(7) + E_vec(1)*x01[k](0) + E_vec(4)*x01[k](1), 0; 
            // Transposed version of the above
            //denom_de << x01[k](0)*(E_vec(2) + E_vec(0)*x01[k](0) + E_vec(1)*x01[k](1)) + x02[k](0)*(E_vec(6) + E_vec(0)*x02[k](0) + E_vec(3)*x02[k](1)), x01[k](1)*(E_vec(2) + E_vec(0)*x01[k](0) + E_vec(1)*x01[k](1)) + x02[k](0)*(E_vec(7) + E_vec(1)*x02[k](0) + E_vec(4)*x02[k](1)), E_vec(2) + E_vec(0)*x01[k](0) + E_vec(1)*x01[k](1), x01[k](0)*(E_vec(5) + E_vec(3)*x01[k](0) + E_vec(4)*x01[k](1)) + x02[k](1)*(E_vec(6) + E_vec(0)*x02[k](0) + E_vec(3)*x02[k](1)), x01[k](1)*(E_vec(5) + E_vec(3)*x01[k](0) + E_vec(4)*x01[k](1)) + x02[k](1)*(E_vec(7) + E_vec(1)*x02[k](0) + E_vec(4)*x02[k](1)), E_vec(5) + E_vec(3)*x01[k](0) + E_vec(4)*x01[k](1), E_vec(6) + E_vec(0)*x02[k](0) + E_vec(3)*x02[k](1), E_vec(7) + E_vec(1)*x02[k](0) + E_vec(4)*x02[k](1), 0;

            const double nJ_C = J_C.norm();
            const double inv_nJ_C = 1.0 / nJ_C;
            const double r = C * inv_nJ_C;

            // Compute weight from robust loss function (used in the IRLS)
            if(r > thresholds[k]) {
                continue;
            }
            num_residuals++;

            // Compute Jacobian of Summarized Sampson error w.r.t the fundamental/essential matrix (3x3)
            Eigen::Matrix<double, 9, 9> dF = M[k] * inv_nJ_C;

            // Compute derivative w.r.t. the denominator
            dF -= (Me) * (denom_de.transpose()) * inv_nJ_C * inv_nJ_C * inv_nJ_C;

            // and then w.r.t. the pose parameters (rotation + tangent basis for translation)
            Eigen::Matrix<double, 9, 5> J;
            J.block<9, 3>(0, 0) = dF * dR;
            J.block<9, 2>(0, 3) = dF * dt;

            // Accumulate into JtJ and Jtr
            JtJ += J.transpose() * J;
            Jtr += J.transpose() * (Me * inv_nJ_C);
        }
        return num_residuals;
    }

    CameraPose step(Eigen::Matrix<double, 5, 1> dp, const CameraPose &pose) const {
        CameraPose pose_new;
        pose_new.q = quat_step_post(pose.q, dp.block<3, 1>(0, 0));
        pose_new.t = pose.t + tangent_basis * dp.block<2, 1>(3, 0);
        return pose_new;
    }
    typedef CameraPose param_t;
    static constexpr size_t num_params = 5;

  private:
    const std::vector<Eigen::Matrix<double,9,9>> &M;
    const std::vector<Eigen::Vector2d> &x01;
    const std::vector<Eigen::Vector2d> &x02;
    const std::vector<double> &thresholds;
    Eigen::Matrix<double, 3, 2> tangent_basis;
};

class RelativePoseJacobianAccumulator {
  public:
    RelativePoseJacobianAccumulator(const std::vector<Eigen::Vector2d> &points2D_1, const std::vector<Eigen::Vector2d> &points2D_2, double thr)
        : x1(points2D_1), x2(points2D_2), threshold(thr) {}

    double residual(const CameraPose &pose) const {
        Eigen::Matrix3d E;
        essential_from_motion(pose, &E);

        double t2 = threshold * threshold;

        double cost = 0.0;
        for (size_t k = 0; k < x1.size(); ++k) {
            double C = x2[k].homogeneous().dot(E * x1[k].homogeneous());
            double nJc_sq = (E.block<2, 3>(0, 0) * x1[k].homogeneous()).squaredNorm() +
                            (E.block<3, 2>(0, 0).transpose() * x2[k].homogeneous()).squaredNorm();

            double r2 = (C * C) / nJc_sq;
            cost += std::min(r2, t2);
        }

        return cost;
    }

    size_t accumulate(const CameraPose &pose, Eigen::Matrix<double, 5, 5> &JtJ, Eigen::Matrix<double, 5, 1> &Jtr) {
        // We start by setting up a basis for the updates in the translation (orthogonal to t)
        // We find the minimum element of t and cross product with the corresponding basis vector.
        // (this ensures that the first cross product is not close to the zero vector)
        if (std::abs(pose.t.x()) < std::abs(pose.t.y())) {
            // x < y
            if (std::abs(pose.t.x()) < std::abs(pose.t.z())) {
                tangent_basis.col(0) = pose.t.cross(Eigen::Vector3d::UnitX()).normalized();
            } else {
                tangent_basis.col(0) = pose.t.cross(Eigen::Vector3d::UnitZ()).normalized();
            }
        } else {
            // x > y
            if (std::abs(pose.t.y()) < std::abs(pose.t.z())) {
                tangent_basis.col(0) = pose.t.cross(Eigen::Vector3d::UnitY()).normalized();
            } else {
                tangent_basis.col(0) = pose.t.cross(Eigen::Vector3d::UnitZ()).normalized();
            }
        }
        tangent_basis.col(1) = tangent_basis.col(0).cross(pose.t).normalized();

        Eigen::Matrix3d E, R;
        R = pose.R();
        essential_from_motion(pose, &E);

        // Matrices contain the jacobians of E w.r.t. the rotation and translation parameters
        Eigen::Matrix<double, 9, 3> dR;
        Eigen::Matrix<double, 9, 2> dt;

        // Each column is vec(E*skew(e_k)) where e_k is k:th basis vector
        dR.block<3, 1>(0, 0).setZero();
        dR.block<3, 1>(0, 1) = -E.col(2);
        dR.block<3, 1>(0, 2) = E.col(1);
        dR.block<3, 1>(3, 0) = E.col(2);
        dR.block<3, 1>(3, 1).setZero();
        dR.block<3, 1>(3, 2) = -E.col(0);
        dR.block<3, 1>(6, 0) = -E.col(1);
        dR.block<3, 1>(6, 1) = E.col(0);
        dR.block<3, 1>(6, 2).setZero();

        // Each column is vec(skew(tangent_basis[k])*R)
        dt.block<3, 1>(0, 0) = tangent_basis.col(0).cross(R.col(0));
        dt.block<3, 1>(0, 1) = tangent_basis.col(1).cross(R.col(0));
        dt.block<3, 1>(3, 0) = tangent_basis.col(0).cross(R.col(1));
        dt.block<3, 1>(3, 1) = tangent_basis.col(1).cross(R.col(1));
        dt.block<3, 1>(6, 0) = tangent_basis.col(0).cross(R.col(2));
        dt.block<3, 1>(6, 1) = tangent_basis.col(1).cross(R.col(2));

        size_t num_residuals = 0;
        for (size_t k = 0; k < x1.size(); ++k) {
            double C = x2[k].homogeneous().dot(E * x1[k].homogeneous());

            // J_C is the Jacobian of the epipolar constraint w.r.t. the image points
            Eigen::Vector4d J_C;
            J_C << E.block<3, 2>(0, 0).transpose() * x2[k].homogeneous(), E.block<2, 3>(0, 0) * x1[k].homogeneous();
            const double nJ_C = J_C.norm();
            const double inv_nJ_C = 1.0 / nJ_C;
            const double r = C * inv_nJ_C;

            if (std::abs(r) > threshold) {
                continue;
            }
            num_residuals++;

            // Compute Jacobian of Sampson error w.r.t the fundamental/essential matrix (3x3)
            Eigen::Matrix<double, 1, 9> dF;
            dF << x1[k](0) * x2[k](0), x1[k](0) * x2[k](1), x1[k](0), x1[k](1) * x2[k](0), x1[k](1) * x2[k](1),
                x1[k](1), x2[k](0), x2[k](1), 1.0;
            const double s = C * inv_nJ_C * inv_nJ_C;
            dF(0) -= s * (J_C(2) * x1[k](0) + J_C(0) * x2[k](0));
            dF(1) -= s * (J_C(3) * x1[k](0) + J_C(0) * x2[k](1));
            dF(2) -= s * (J_C(0));
            dF(3) -= s * (J_C(2) * x1[k](1) + J_C(1) * x2[k](0));
            dF(4) -= s * (J_C(3) * x1[k](1) + J_C(1) * x2[k](1));
            dF(5) -= s * (J_C(1));
            dF(6) -= s * (J_C(2));
            dF(7) -= s * (J_C(3));
            dF *= inv_nJ_C;

            // and then w.r.t. the pose parameters (rotation + tangent basis for translation)
            Eigen::Matrix<double, 1, 5> J;
            J.block<1, 3>(0, 0) = dF * dR;
            J.block<1, 2>(0, 3) = dF * dt;

            // Accumulate into JtJ and Jtr
            Jtr += C * inv_nJ_C * J.transpose();
            JtJ(0, 0) += (J(0) * J(0));
            JtJ(1, 0) += (J(1) * J(0));
            JtJ(1, 1) += (J(1) * J(1));
            JtJ(2, 0) += (J(2) * J(0));
            JtJ(2, 1) += (J(2) * J(1));
            JtJ(2, 2) += (J(2) * J(2));
            JtJ(3, 0) += (J(3) * J(0));
            JtJ(3, 1) += (J(3) * J(1));
            JtJ(3, 2) += (J(3) * J(2));
            JtJ(3, 3) += (J(3) * J(3));
            JtJ(4, 0) += (J(4) * J(0));
            JtJ(4, 1) += (J(4) * J(1));
            JtJ(4, 2) += (J(4) * J(2));
            JtJ(4, 3) += (J(4) * J(3));
            JtJ(4, 4) += (J(4) * J(4));

        }
        return num_residuals;
    }

    CameraPose step(Eigen::Matrix<double, 5, 1> dp, const CameraPose &pose) const {
        CameraPose pose_new;
        pose_new.q = quat_step_post(pose.q, dp.block<3, 1>(0, 0));
        pose_new.t = pose.t + tangent_basis * dp.block<2, 1>(3, 0);
        return pose_new;
    }
    typedef CameraPose param_t;
    static constexpr size_t num_params = 5;

  private:
    const std::vector<Eigen::Vector2d> &x1;
    const std::vector<Eigen::Vector2d> &x2;
    const double threshold;
    Eigen::Matrix<double, 3, 2> tangent_basis;
};

class HybridAccumulator {
  public:
    HybridAccumulator(SummarizedRelativePoseJacobianAccumulator &summarized_accum,
                      RelativePoseJacobianAccumulator &relative_accum)
        : summarized_acc(summarized_accum), relative_acc(relative_accum) {}

    double residual(const CameraPose &pose) const {
        return summarized_acc.residual(pose) + relative_acc.residual(pose);
    }

    size_t accumulate(const CameraPose &pose, Eigen::Matrix<double, 5, 5> &JtJ,
                      Eigen::Matrix<double, 5, 1> &Jtr) {
        return summarized_acc.accumulate(pose, JtJ, Jtr) + relative_acc.accumulate(pose, JtJ, Jtr);
    }

    CameraPose step(Eigen::Matrix<double, 5, 1> dp, const CameraPose &pose) const {
        return relative_acc.step(dp, pose);
    }
    typedef CameraPose param_t;
    static constexpr size_t num_params = 5;

  private:
    SummarizedRelativePoseJacobianAccumulator &summarized_acc;
    RelativePoseJacobianAccumulator &relative_acc;
};


// This is the SVD factorization proposed by Bartoli and Sturm in
// Non-Linear Estimation of the Fundamental Matrix With Minimal Parameters, PAMI 2004
// Though we do different updates (lie vs the euler angles used in the original paper)
struct FactorizedFundamentalMatrix {
    FactorizedFundamentalMatrix() {}
    FactorizedFundamentalMatrix(const Eigen::Matrix3d &F) {
        Eigen::JacobiSVD<Eigen::Matrix3d> svd(F, Eigen::ComputeFullV | Eigen::ComputeFullU);
        Eigen::Matrix3d U = svd.matrixU();
        Eigen::Matrix3d V = svd.matrixV();
        if (U.determinant() < 0) {
            U = -U;
        }
        if (V.determinant() < 0) {
            V = -V;
        }
        qU = rotmat_to_quat(U);
        qV = rotmat_to_quat(V);
        Eigen::Vector3d s = svd.singularValues();
        sigma = s(1) / s(0);
    }
    Eigen::Matrix3d F() const {
        Eigen::Matrix3d U = quat_to_rotmat(qU);
        Eigen::Matrix3d V = quat_to_rotmat(qV);
        return U.col(0) * V.col(0).transpose() + sigma * U.col(1) * V.col(1).transpose();
    }

    Eigen::Vector4d qU, qV;
    double sigma;
};

// Fundaemental matrix
class SummarizedFundamentalJacobianAccumulator {
  public:
    SummarizedFundamentalJacobianAccumulator(const std::vector<Eigen::Matrix<double,9,9>> &mats,
                                    const std::vector<Eigen::Vector2d> &x1, const std::vector<Eigen::Vector2d> &x2,
                                    const std::vector<double> &thrs)
        : M(mats), x01(x1), x02(x2), thresholds(thrs) {}

    double residual(const FactorizedFundamentalMatrix &FF) const {
        Eigen::Matrix3d F = FF.F();
        Eigen::Vector<double, 9> F_vec = Eigen::Map<Eigen::Vector<double, 9>>(F.data());

        double cost = 0.0;
        for (size_t k = 0; k < M.size(); ++k) {

            double C2 = (M[k] * F_vec).squaredNorm();
            double nJc_sq = (F.block<2, 3>(0, 0) * x01[k].homogeneous()).squaredNorm() +
                            (F.block<3, 2>(0, 0).transpose() * x02[k].homogeneous()).squaredNorm();

            double r2 = C2 / nJc_sq;
            cost += std::min(r2, thresholds[k] * thresholds[k]);
        }

        return cost;
    }

    size_t accumulate(const FactorizedFundamentalMatrix &FF, Eigen::Matrix<double, 7, 7> &JtJ, Eigen::Matrix<double, 7, 1> &Jtr) {
        Eigen::Matrix3d F = FF.F();

        // Matrices contain the jacobians of F w.r.t. the factorized fundamental matrix (U,V,sigma)
        const Eigen::Matrix3d U = quat_to_rotmat(FF.qU);
        const Eigen::Matrix3d V = quat_to_rotmat(FF.qV);

        const Eigen::Matrix3d d_sigma = U.col(1) * V.col(1).transpose();
        Eigen::Matrix<double, 9, 7> dF_dparams;
        dF_dparams << 0, F(2, 0), -F(1, 0), 0, F(0, 2), -F(0, 1), d_sigma(0, 0), -F(2, 0), 0, F(0, 0), 0, F(1, 2),
            -F(1, 1), d_sigma(1, 0), F(1, 0), -F(0, 0), 0, 0, F(2, 2), -F(2, 1), d_sigma(2, 0), 0, F(2, 1), -F(1, 1),
            -F(0, 2), 0, F(0, 0), d_sigma(0, 1), -F(2, 1), 0, F(0, 1), -F(1, 2), 0, F(1, 0), d_sigma(1, 1), F(1, 1),
            -F(0, 1), 0, -F(2, 2), 0, F(2, 0), d_sigma(2, 1), 0, F(2, 2), -F(1, 2), F(0, 1), -F(0, 0), 0, d_sigma(0, 2),
            -F(2, 2), 0, F(0, 2), F(1, 1), -F(1, 0), 0, d_sigma(1, 2), F(1, 2), -F(0, 2), 0, F(2, 1), -F(2, 0), 0,
            d_sigma(2, 2);

        Eigen::Vector<double, 9> F_vec = Eigen::Map<Eigen::Vector<double, 9>>(F.data());

        size_t num_residuals = 0;
        for (size_t k = 0; k < M.size(); ++k) {
            Eigen::Vector<double, 9> Me = M[k] * F_vec;
            double C = Me.norm();

            // J_C is the Jacobian of the epipolar constraint w.r.t. the image points
            Eigen::Vector4d J_C;
            J_C << F.block<3, 2>(0, 0).transpose() * x02[k].homogeneous(), F.block<2, 3>(0, 0) * x01[k].homogeneous();
            
            Eigen::Matrix<double,9,1> denom_df;
            denom_df << x01[k](0)*(F_vec(6) + F_vec(0)*x01[k](0) + F_vec(3)*x01[k](1)) + x02[k](0)*(F_vec(2) + F_vec(0)*x02[k](0) + F_vec(1)*x02[k](1)), x01[k](0)*(F_vec(7) + F_vec(1)*x01[k](0) + F_vec(4)*x01[k](1)) + x02[k](1)*(F_vec(2) + F_vec(0)*x02[k](0) + F_vec(1)*x02[k](1)), F_vec(2) + F_vec(0)*x02[k](0) + F_vec(1)*x02[k](1), x01[k](1)*(F_vec(6) + F_vec(0)*x01[k](0) + F_vec(3)*x01[k](1)) + x02[k](0)*(F_vec(5) + F_vec(3)*x02[k](0) + F_vec(4)*x02[k](1)), x01[k](1)*(F_vec(7) + F_vec(1)*x01[k](0) + F_vec(4)*x01[k](1)) + x02[k](1)*(F_vec(5) + F_vec(3)*x02[k](0) + F_vec(4)*x02[k](1)), F_vec(5) + F_vec(3)*x02[k](0) + F_vec(4)*x02[k](1), F_vec(6) + F_vec(0)*x01[k](0) + F_vec(3)*x01[k](1), F_vec(7) + F_vec(1)*x01[k](0) + F_vec(4)*x01[k](1), 0; 

            const double nJ_C = J_C.norm();
            const double inv_nJ_C = 1.0 / nJ_C;
            const double r = C * inv_nJ_C;

            // Compute weight from robust loss function (used in the IRLS)
            if(r > thresholds[k]) {
                continue;
            }
            num_residuals++;

            // Compute Jacobian of Summarized Sampson error w.r.t the fundamental/essential matrix (3x3)
            Eigen::Matrix<double, 9, 9> dF = M[k] * inv_nJ_C;

            // Compute derivative w.r.t. the denominator
            dF -= (Me) * (denom_df.transpose()) * inv_nJ_C * inv_nJ_C * inv_nJ_C;

            // and then w.r.t. the pose parameters (rotation + tangent basis for translation)
            Eigen::Matrix<double, 9, 7> J = dF * dF_dparams;

            // Accumulate into JtJ and Jtr
            JtJ += J.transpose() * J;
            Jtr += J.transpose() * (Me * inv_nJ_C);
        }
        return num_residuals;
    }

    FactorizedFundamentalMatrix step(Eigen::Matrix<double, 7, 1> dp, const FactorizedFundamentalMatrix &F) const {
        FactorizedFundamentalMatrix F_new;
        F_new.qU = quat_step_pre(F.qU, dp.block<3, 1>(0, 0));
        F_new.qV = quat_step_pre(F.qV, dp.block<3, 1>(3, 0));
        F_new.sigma = F.sigma + dp(6);
        return F_new;
    }
    typedef FactorizedFundamentalMatrix param_t;
    static constexpr size_t num_params = 7;

  private:
    const std::vector<Eigen::Matrix<double,9,9>> &M;
    const std::vector<Eigen::Vector2d> &x01;
    const std::vector<Eigen::Vector2d> &x02;
    const std::vector<double> &thresholds;
};


} // namespace poselib

#endif
