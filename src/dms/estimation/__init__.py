from .robust import run_pysumopt_estimation, run_poselib_estimation, \
                                        run_poselib_fundamental
from .refinement import run_pysumopt_refinement_dense, run_pysumopt_refinement_centers, \
                                            run_pysumopt_refinement_approx, run_poselib_refinement, \
                                            run_pysumopt_refinement_fundamental


def estimate_relative_pose(data, sampling='center', scoring='center', refinement='center', disable_LO=False, seed=0):

    assert sampling in ['dense', 'center', 'approx']
    assert scoring in ['dense', 'center', 'approx']
    assert refinement in ['dense', 'center', 'approx']
    assert not (disable_LO and not (sampling == 'approx')), 'Cannot disable LO without exhaustive sampling'

    # Run RANSAC
    if sampling == 'dense' and scoring == 'dense':
        ransac_result, runtime = run_pysumopt_estimation(data, sampling, scoring, disable_LO, seed)
    elif sampling == 'dense' and scoring == 'center':
        ransac_result, runtime = run_pysumopt_estimation(data, sampling, scoring, disable_LO, seed)
    elif sampling == 'dense' and scoring == 'approx':
        ransac_result, runtime = run_pysumopt_estimation(data, sampling, scoring, disable_LO, seed)

    elif sampling == 'center' and scoring == 'dense':
        raise NotImplementedError('Center sampling with dense scoring not implemented')  # This makes no sense
    elif sampling == 'center' and scoring == 'center':
        ransac_result, runtime = run_pysumopt_estimation(data, sampling, scoring, disable_LO, seed)
    elif sampling == 'center' and scoring == 'approx':
        ransac_result, runtime = run_pysumopt_estimation(data, sampling, scoring, disable_LO, seed)

    elif sampling == 'approx' and scoring == 'dense':
       raise NotImplementedError('Approximate ransac with dense scoring not implemented')  # Makes no sense
    elif sampling == 'approx' and scoring == 'center':
        ransac_result, runtime = run_pysumopt_estimation(data, sampling, scoring, disable_LO, seed)
    elif sampling == 'approx' and scoring == 'approx':
        ransac_result, runtime = run_pysumopt_estimation(data, sampling, scoring, disable_LO, seed)

    if refinement == scoring:
        # If refinement was done with estimator, we are done
        return ransac_result, runtime

    # Run separate refinement
    if refinement == 'dense':
        refinement_result, refinement_runtime = run_pysumopt_refinement_dense(data, ransac_result)
    elif refinement == 'center':
        refinement_result, refinement_runtime = run_pysumopt_refinement_centers(data, ransac_result)
    elif refinement == 'approx':
        refinement_result, refinement_runtime = run_pysumopt_refinement_approx(data, ransac_result)

    return refinement_result, runtime + refinement_runtime


def estimate_fundamental(data, sampling='center', scoring='center', refinement='center', seed=0):

    assert sampling in ['dense', 'center']
    assert scoring == sampling
    assert refinement in ['dense', 'center', 'approx']

    # Run RANSAC
    ransac_result, runtime = run_poselib_fundamental(data, sparsified=(sampling == 'center'), seed=seed)

    if refinement == sampling:
        # If refinement was done with estimator, we are done
        return ransac_result, runtime
    
    # Run separate refinement
    if refinement == 'approx':
        refinement_result, refinement_runtime = run_pysumopt_refinement_fundamental(data, ransac_result)
    else:
        raise NotImplementedError('Dense/Center refinement for fundamental matrix not implemented')
    
    return refinement_result, runtime + refinement_runtime


def run_poselib_relative_pose(data, sampling_and_scoring='center', refinement='center'):

    assert sampling_and_scoring in ['dense', 'center'], 'Invalid poselib sampling_and_scoring method: {}'.format(sampling_and_scoring)
    assert refinement in ['dense', 'center'], 'Invalid poselib refinement method: {}'.format(refinement)

    sparsified_ransac = (sampling_and_scoring == 'center')
    sparsified_refinement = (refinement == 'center')

    ransac_result, runtime = run_poselib_estimation(data, sparsified=sparsified_ransac)

    if refinement == sampling_and_scoring:
        # If refinement was done with estimator, we are done
        return ransac_result, runtime
    
    refinement_result, refinement_runtime = run_poselib_refinement(data, ransac_result, sparsified=sparsified_refinement)

    return refinement_result, runtime + refinement_runtime