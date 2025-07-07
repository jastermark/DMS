#include "ransac.h"
#include "essential.h"
#include "sampling.h"
#include "sumopt.h"
#include "ransac_impl.h"
#include "solvers.h"
namespace sumopt {

class SummarizedRelativePoseEstimator {
  public:
    SummarizedRelativePoseEstimator(const RansacOptions &ransac_opt,
                          const std::vector<Eigen::Vector2d> &x1_sample,
                          const std::vector<Eigen::Vector2d> &x2_sample,
                          const std::vector<Eigen::Matrix<double, 9, 9>> &M,
                          const std::vector<Eigen::Vector2d> &x1_center,
                          const std::vector<Eigen::Vector2d> &x2_center,
                          const std::vector<double> &thrs)
        : num_data((ransac_opt.score_centers ? x1_center.size() : 0)+(ransac_opt.score_approx ? x1_center.size() : 0)), opt(ransac_opt), x1_sample(x1_sample), x2_sample(x2_sample), M(M), x1_center(x1_center), x2_center(x2_center),
            thresholds(thrs),
          sampler(x1_sample.size(), sample_sz, opt.seed, opt.progressive_sampling, opt.max_prosac_iterations) {
        x1s.resize(sample_sz);
        x2s.resize(sample_sz);
        sample.resize(sample_sz);
    }

    void generate_models(std::vector<CameraPose> *models) {
        sampler.generate_sample(&sample);
        for (size_t k = 0; k < sample_sz; ++k) {
            x1s[k] = x1_sample[sample[k]].homogeneous().normalized();
            x2s[k] = x2_sample[sample[k]].homogeneous().normalized();
        }
        relpose_5pt(x1s, x2s, models);
    }

    double score_model(const CameraPose &pose, size_t *inlier_count) const {
        size_t approx_inliers = 0;
        size_t center_inliers = 0;

        double score = 0.0;
        *inlier_count = 0;
        if(opt.score_centers) {
            double msac_center = compute_sampson_msac_score(pose, x1_center, x2_center, opt.threshold_center * opt.threshold_center, &center_inliers);
            score += msac_center;
            *inlier_count += center_inliers;
        }
        if(opt.score_approx) {
            double msac_approx = compute_sampson_msac_score(pose, M, x1_center, x2_center, thresholds, &approx_inliers);
            score += msac_approx;
            *inlier_count += approx_inliers;
        }
        return score;
    }
    void refine_model(CameraPose *pose) const {
        BundleOptions bundle_opt;
        bundle_opt.max_iterations = 25;
        if(opt.score_centers && opt.score_approx) {
            lm_sampson_approx_w_centers(M, x1_center, x2_center, pose, thresholds, opt.threshold_center, bundle_opt);
        } else if(opt.score_centers && !opt.score_approx) {
            lm_sampson(x1_center, x2_center, pose, opt.threshold_center, bundle_opt);
        } else if(opt.score_approx && !opt.score_centers) {
            lm_sampson_approx(M, x1_center, x2_center, pose, thresholds, bundle_opt);
        }
    }

    const size_t sample_sz = 5;
    const size_t num_data;

  private:
    const RansacOptions &opt;

    // Points used for sampling models
    const std::vector<Eigen::Vector2d> &x1_sample;
    const std::vector<Eigen::Vector2d> &x2_sample;

    // Approximation of dense matches
    const std::vector<Eigen::Matrix<double, 9, 9>> &M;
    const std::vector<Eigen::Vector2d> &x1_center;
    const std::vector<Eigen::Vector2d> &x2_center;
    const std::vector<double> &thresholds;

    RandomSampler sampler;
    // pre-allocated vectors for sampling
    std::vector<Eigen::Vector3d> x1s, x2s;
    std::vector<size_t> sample;
};


RansacStats ransac_relpose(const std::vector<Eigen::Vector2d> &x1_sample, const std::vector<Eigen::Vector2d> &x2_sample, const std::vector<Eigen::Matrix<double, 9, 9>> &M, const std::vector<Eigen::Vector2d> &x1_center, const std::vector<Eigen::Vector2d> &x2_center, const std::vector<double> &thresholds, const RansacOptions &opt, CameraPose *output, std::vector<char> *inliers) {
    output->q << 1.0, 0.0, 0.0, 0.0;
    output->t << 0.0, 0.0, 0.0;
    SummarizedRelativePoseEstimator estimator(opt, x1_sample, x2_sample, M, x1_center, x2_center, thresholds);
    RansacStats stats = ransac<SummarizedRelativePoseEstimator>(estimator, opt, output);
    get_inliers(*output, M, x1_center, x2_center, thresholds, inliers);
    return stats;
}

double compute_sampson_msac_score(const CameraPose &pose, const std::vector<Eigen::Matrix<double, 9, 9>> &M, const std::vector<Eigen::Vector2d> &x1, const std::vector<Eigen::Vector2d> &x2, const std::vector<double> &thresholds, size_t *inlier_count) {
    *inlier_count = 0;
    Eigen::Matrix3d E;
    essential_from_motion(pose, &E);
    Eigen::Vector<double, 9> E_vec = Eigen::Map<Eigen::Vector<double, 9>>(E.data());

    double score = 0.0;
    for (size_t k = 0; k < M.size(); ++k) {
        const double sq_threshold = thresholds[k] * thresholds[k];
        
        const double nMe2 = (M[k] * E_vec).squaredNorm();

        const double denom2 = (E.block<2, 3>(0, 0) * x1[k].homogeneous()).squaredNorm() +
                            (E.block<3, 2>(0, 0).transpose() * x2[k].homogeneous()).squaredNorm();

        const double r2 = nMe2 / denom2;
        
        if (r2 < sq_threshold) {
            bool cheirality =
                check_cheirality(pose, x1[k].homogeneous().normalized(), x2[k].homogeneous().normalized(), 0.01);
            if (cheirality) {
                (*inlier_count)++;
                score += r2;
            } else {
                score += sq_threshold;
            }
        } else {
            score += sq_threshold;
        }
    }
    return score;
}
int get_inliers(const CameraPose &pose, const std::vector<Eigen::Matrix<double, 9, 9>> &M, const std::vector<Eigen::Vector2d> &x1, const std::vector<Eigen::Vector2d> &x2, const std::vector<double> &thresholds, std::vector<char> *inliers) {
    int inlier_count = 0;
    Eigen::Matrix3d E;
    essential_from_motion(pose, &E);
    Eigen::Vector<double, 9> E_vec = Eigen::Map<Eigen::Vector<double, 9>>(E.data());

    double score = 0.0;
    inliers->resize(M.size());
    for (size_t k = 0; k < M.size(); ++k) {
        const double sq_threshold = thresholds[k] * thresholds[k];
        
        const double nMe = (M[k] * E_vec).norm();

        const double denom2 = (E.block<2, 3>(0, 0) * x1[k].homogeneous()).squaredNorm() +
                            (E.block<3, 2>(0, 0).transpose() * x2[k].homogeneous()).squaredNorm();

        const double r2 = nMe * nMe / denom2;
        
        if (r2 < sq_threshold) {
            bool cheirality =
                check_cheirality(pose, x1[k].homogeneous().normalized(), x2[k].homogeneous().normalized(), 0.01);
            if (cheirality) {
                inlier_count++;
                (*inliers)[k] = true;
            } else {
                (*inliers)[k] = false;
            }
        } else {
            (*inliers)[k] = false;
        }
    }
    return inlier_count;
}


// Returns MSAC score of the Sampson error (checks cheirality of points as well)
double compute_sampson_msac_score(const CameraPose &pose, const std::vector<Eigen::Vector2d> &x1,
                                  const std::vector<Eigen::Vector2d> &x2, double sq_threshold, size_t *inlier_count) {
    *inlier_count = 0;
    Eigen::Matrix3d E;
    essential_from_motion(pose, &E);

    // For some reason this is a lot faster than just using nice Eigen expressions...
    const double E0_0 = E(0, 0), E0_1 = E(0, 1), E0_2 = E(0, 2);
    const double E1_0 = E(1, 0), E1_1 = E(1, 1), E1_2 = E(1, 2);
    const double E2_0 = E(2, 0), E2_1 = E(2, 1), E2_2 = E(2, 2);

    double score = 0.0;
    for (size_t k = 0; k < x1.size(); ++k) {
        const double x1_0 = x1[k](0), x1_1 = x1[k](1);
        const double x2_0 = x2[k](0), x2_1 = x2[k](1);

        const double Ex1_0 = E0_0 * x1_0 + E0_1 * x1_1 + E0_2;
        const double Ex1_1 = E1_0 * x1_0 + E1_1 * x1_1 + E1_2;
        const double Ex1_2 = E2_0 * x1_0 + E2_1 * x1_1 + E2_2;

        const double Ex2_0 = E0_0 * x2_0 + E1_0 * x2_1 + E2_0;
        const double Ex2_1 = E0_1 * x2_0 + E1_1 * x2_1 + E2_1;
        // const double Ex2_2 = E0_2 * x2_0 + E1_2 * x2_1 + E2_2;

        const double C = x2_0 * Ex1_0 + x2_1 * Ex1_1 + Ex1_2;
        const double Cx = Ex1_0 * Ex1_0 + Ex1_1 * Ex1_1;
        const double Cy = Ex2_0 * Ex2_0 + Ex2_1 * Ex2_1;
        const double r2 = C * C / (Cx + Cy);

        if (r2 < sq_threshold) {
            bool cheirality =
                check_cheirality(pose, x1[k].homogeneous().normalized(), x2[k].homogeneous().normalized(), 0.01);
            if (cheirality) {
                (*inlier_count)++;
                score += r2;
            } else {
                score += sq_threshold;
            }
        } else {
            score += sq_threshold;
        }
    }
    return score;
}


RansacStats ransac_relpose_exhaustive(const std::vector<Eigen::Matrix<double, 9, 9>> &M, const std::vector<Eigen::Vector2d> &x1_center, const std::vector<Eigen::Vector2d> &x2_center, const std::vector<double> &thresholds, const RansacOptions &opt, CameraPose *best_model, std::vector<char> *inliers, bool single_refinement) {
    best_model->q << 1.0, 0.0, 0.0, 0.0;
    best_model->t << 0.0, 0.0, 0.0;
    RansacStats stats;
    stats.num_inliers = 0;
    stats.model_score = std::numeric_limits<double>::max();
    stats.iterations = 0;
    stats.refinements = 0;

    size_t inlier_count = 0;
    size_t best_minimal_inlier_count = 0;
    double best_minimal_msac_score = std::numeric_limits<double>::max();

    SummarizedRelativePoseEstimator estimator(opt, x1_center, x2_center, M, x1_center, x2_center, thresholds);

    std::vector<CameraPose> models;
    for (size_t i = 0; i < M.size(); ++i) {
        models.clear();
        relpose_sum(M[i], x1_center[i].homogeneous().normalized(), x2_center[i].homogeneous().normalized(), &models);

        int best_model_ind = -1;
        for (size_t i = 0; i < models.size(); ++i) {
            double score_msac = estimator.score_model(models[i], &inlier_count);
            bool more_inliers = inlier_count > best_minimal_inlier_count;
            bool better_score = score_msac < best_minimal_msac_score;

            if (more_inliers || better_score) {
                if (more_inliers) {
                    best_minimal_inlier_count = inlier_count;
                }
                if (better_score) {
                    best_minimal_msac_score = score_msac;
                }
                best_model_ind = i;

                // check if we should update best model already
                if (score_msac < stats.model_score) {
                    stats.model_score = score_msac;
                    *best_model = models[i];
                    stats.num_inliers = inlier_count;
                }
            }
        }
        if(single_refinement)
            continue;
        if (best_model_ind == -1)
            continue;

        // Refinement
        CameraPose refined_model = models[best_model_ind];
        estimator.refine_model(&refined_model);
        stats.refinements++;
        double refined_msac_score = estimator.score_model(refined_model, &inlier_count);
        if (refined_msac_score < stats.model_score) {
            stats.model_score = refined_msac_score;
            stats.num_inliers = inlier_count;
            *best_model = refined_model;
        }
    }

    // Final refinement
    if(single_refinement)
        estimator.refine_model(best_model);
    get_inliers(*best_model, M, x1_center, x2_center, thresholds, inliers);
    return stats;
}


}