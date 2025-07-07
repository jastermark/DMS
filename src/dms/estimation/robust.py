import datetime
import numpy as np

import poselib
import pysumopt

CAM = {
    'model': 'SIMPLE_PINHOLE',
    'params': [1.0, 0.0, 0.0],
    'width': -1,
    'height': -1,
}


def run_poselib_estimation(data, sparsified=False):

    ransac_opt ={
        'max_epipolar_error': data['threshold_E'],
        'min_iterations': 100,
        'max_iterations': 10000,
        'success_prob': 0.9999,
    }
    bundle_opt = {
        'loss_type': 'TRUNCATED',
        'loss_scale': data['threshold_E'],
        # 'max_iterations': 100 if refine else 0,  # Toggle bundle adjustment
    }

    if sparsified:
        x1_calibrated = np.array([e.x01 for e in data['ecorrs']])
        x2_calibrated = np.array([e.x02 for e in data['ecorrs']])
    else:
        x1_calibrated = data['x1c']
        x2_calibrated = data['x2c']


    tt1 = datetime.datetime.now()
    pose, info = poselib.estimate_relative_pose(x1_calibrated, x2_calibrated, CAM, CAM, ransac_opt, bundle_opt)
    tt2 = datetime.datetime.now()

    result = {
        'R': pose.R,
        't': pose.t,
    }
    return result, (tt2 - tt1).total_seconds()


def run_poselib_fundamental(data, sparsified=False, seed=0):
    ransac_opt = {
        'max_epipolar_error': data['threshold_E'],
        'min_iterations': 100,
        'max_iterations': 10000,
        'success_probability': 0.9999,
        'seed': seed,
    }
    bundle_opt = {
        'loss_type': 'TRUNCATED',
        'loss_scale': data['threshold_E'],
        # 'max_iterations': 0,
    }

    if sparsified:
        x1_normalized = np.array([e.x01 for e in data['ecorrs']])
        x2_normalized = np.array([e.x02 for e in data['ecorrs']])
    else:
        x1_normalized = data['x1c']
        x2_normalized = data['x2c']

    tt1 = datetime.datetime.now()
    fundamental, info = poselib.estimate_fundamental(x1_normalized, x2_normalized, ransac_opt, bundle_opt)
    tt2 = datetime.datetime.now()

    # Un-normalize estimated fundamental matrix
    # Unless we failed to estimate it (then N might be rank-deficient)
    if not np.allclose(fundamental, np.zeros((3, 3))):
        N1 = data['N1']
        N2 = data['N2']
        fundamental = np.linalg.inv(N2).T @ fundamental @ np.linalg.inv(N1)

    return fundamental, (tt2 - tt1).total_seconds()


def run_pysumopt_estimation(data, sampling='dense', scoring='center', disable_LO=False, seed=0):

    assert sampling in ['dense', 'center', 'approx'], f'Unknown sampling method: {sampling}'
    assert scoring in ['dense', 'center', 'approx'], f'Unknown scoring method: {scoring}'
    if disable_LO:
        assert sampling == 'approx', 'Cannot disable LO without exhaustive sampling'

    ecorrs = data['ecorrs']
    thr = data['threshold_E']
    exhaustive_sampling = False

    if len(ecorrs) == 0 and (
            sampling in ['center', 'approx'] or scoring in ['center', 'approx']
        ):
        # Fallback to dense method
        sampling = 'dense'
        scoring = 'dense'
        print('Warning: No ecorrs, falling back to dense RANSAC')

    ransac_opt ={
        # 'max_epipolar_error': data['threshold_E'],
        'threshold_center': thr,
        'min_iterations': 100,
        'max_iterations': 10000,
        'success_prob': 0.9999,
        'score_centers': (scoring == 'center' or scoring == 'dense'),
        'score_approx': scoring == 'approx',
        'seed': seed,
    }
    bundle_opt = {
        'loss_type': 'TRUNCATED',
        'loss_scale': data['threshold_E'],
        # 'max_iterations': 100 if refine else 0,  # Toggle bundle adjustment
    }

    M = [c.M for c in ecorrs]
    thrs = [np.sqrt(c.num_pts) * thr for c in ecorrs]

    # Set up matches
    x1_dense = data['x1c']
    x2_dense = data['x2c']
    dense_matches = (x1_dense, x2_dense)

    x1_sparsified = np.array([e.x01 for e in data['ecorrs']])
    x2_sparsified = np.array([e.x02 for e in data['ecorrs']])
    sparsified_matches = (x1_sparsified, x2_sparsified)

    x1_centroids = np.array([e.centroid1 for e in data['ecorrs']])
    x2_centroids = np.array([e.centroid2 for e in data['ecorrs']])
    centroids = (x1_centroids, x2_centroids)

    if sampling == 'dense':
        (x1_sampling, x2_sampling) = dense_matches
    elif sampling in 'center':
        (x1_sampling, x2_sampling) = sparsified_matches
    elif sampling == 'approx':
        # (x1_sampling, x2_sampling) = centroids
        (x1_sampling, x2_sampling) = sparsified_matches
        exhaustive_sampling = True
    else:
        raise ValueError(f'Unknown sampling method: {sampling}')

    if scoring == 'dense':
        (x1_scoring, x2_scoring) = dense_matches
    elif scoring == 'center':
        (x1_scoring, x2_scoring) = sparsified_matches
    elif scoring == 'approx':
        # (x1_scoring, x2_scoring) = centroids
        (x1_scoring, x2_scoring) = sparsified_matches
    else:
        raise ValueError(f'Unknown scoring method: {scoring}')

    if exhaustive_sampling:
        # Exhaustive search
        tt1 = datetime.datetime.now()
        pose, info = pysumopt.ransac_relpose_exhaustive(M, x1_sampling, x2_sampling, thrs, ransac_opt, disable_LO)
        tt2 = datetime.datetime.now()
    else:
        tt1 = datetime.datetime.now()
        pose, info = pysumopt.ransac_relpose(x1_sampling, x2_sampling, M, x1_scoring, x2_scoring, thrs, ransac_opt)
        tt2 = datetime.datetime.now()


    result = {
        'R': pose.R,
        't': pose.t,
    }
    return result, (tt2 - tt1).total_seconds()
