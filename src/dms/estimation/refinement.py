import datetime
import numpy as np
from copy import deepcopy

import poselib
import pysumopt

from .. import geometry

CAM = {
    'model': 'SIMPLE_PINHOLE',
    'params': [1.0, 0.0, 0.0],
    'width': -1,
    'height': -1,
}


def run_pysumopt_refinement_approx(data, init_pose):

    thr = data['threshold_E']

    if len(data['ecorrs']) == 0:
        # Fallback to dense method
        print('Warning: No ecorrs, falling back to dense refinement')
        return run_pysumopt_refinement_dense(data, init_pose)

    E0 = geometry.essential_from_pose(init_pose['R'], init_pose['t'])
    if np.allclose(E0, np.zeros((3, 3))):
        # Initial pose is degenerate. Refinement can not be done.
        return init_pose, 0.0

    # Copy to avoid modifying the original data, in case we run more methods
    ecorrs = deepcopy(data['ecorrs'])

    # Filter outliers
    if data['filter_outliers']:
        for c in ecorrs:
            c.filter_outliers(E0, thr)

    ecorrs = [c for c in ecorrs if c.M is not None]

    if len(ecorrs) == 0:
        # No inlier clusters remain, refinement can not be done
        return init_pose, 0.0

    if 'reweigh' in data and data['reweigh'] is not None:
        # Create reweighed M matrices
        weighing = data['reweigh']
        M = [c.get_summarized_dlt(weighing, E0) for c in ecorrs]
    else:
        M = [c.M for c in ecorrs]

    # x01 = [c.centroid1 for c in ecorrs]
    # x02 = [c.centroid2 for c in ecorrs]
    x01 = [c.x01 for c in ecorrs]
    x02 = [c.x02 for c in ecorrs]
    thrs = [np.sqrt(c.num_pts) * thr for c in ecorrs]

    initial_pose = pysumopt.CameraPose()
    initial_pose.R = init_pose['R']
    initial_pose.t = init_pose['t']

    tt1 = datetime.datetime.now()
    pose_refined, info = pysumopt.lm_sampson_approx(M, x01, x02, initial_pose, thrs)
    tt2 = datetime.datetime.now()

    opt = {
        'R': pose_refined.R,
        't': pose_refined.t,
    }
    return opt, (tt2 - tt1).total_seconds()


def run_pysumopt_refinement_centers(data, init_pose):
    ecorrs = data['ecorrs']
    thr = data['threshold_E']

    if len(ecorrs) == 0 and (
            sampling in ['center', 'approx'] or scoring in ['center', 'approx']
        ):
        # Fallback to dense method
        sampling = 'dense'
        scoring = 'dense'
        print('Warning: No ecorrs, falling back to dense RANSAC')

    x01 = [c.x01 for c in ecorrs]
    x02 = [c.x02 for c in ecorrs]
    
    initial_pose = pysumopt.CameraPose()
    initial_pose.R = init_pose['R']
    initial_pose.t = init_pose['t']

    tt1 = datetime.datetime.now()
    pose_refined, info = pysumopt.lm_sampson(x01, x02, initial_pose, thr)
    tt2 = datetime.datetime.now()

    opt = {
        'R': pose_refined.R,
        't': pose_refined.t,
    }
    return opt, (tt2 - tt1).total_seconds()


def run_pysumopt_refinement_dense(data, init_pose):
    thr = data['threshold_E']
    
    x1 = data['x1c']
    x2 = data['x2c']
    
    initial_pose = pysumopt.CameraPose()
    initial_pose.R = init_pose['R']
    initial_pose.t = init_pose['t']

    tt1 = datetime.datetime.now()
    pose_refined, info = pysumopt.lm_sampson(x1, x2, initial_pose, thr)
    tt2 = datetime.datetime.now()

    opt = {
        'R': pose_refined.R,
        't': pose_refined.t,
    }
    return opt, (tt2 - tt1).total_seconds()


def run_pysumopt_refinement_fundamental(data, init_F):

    thr = data['threshold_E']

    if  np.allclose(init_F, np.zeros((3, 3))):
        # Initial F is invalid. Refinement can not be done.
        return init_F, 0.0

    if len(data['ecorrs']) == 0:
        # Fallback to dense method
        print('Warning: No ecorrs, falling back to dense refinement')
        # return init_F, 0.0
        return run_poselib_refinement_fundamental(data, init_F)

    # Copy to avoid modifying the original data, in case we run more methods
    ecorrs = deepcopy(data['ecorrs'])

    # Filter outliers
    if data['filter_outliers']:
        raise NotImplementedError('Filtering outliers for fundamental matrix not implemented')

    ecorrs = [c for c in ecorrs if c.M is not None]
    if len(ecorrs) == 0:
        # No inlier clusters remain, refinement can not be done
        return init_F, 0.0

    if 'reweigh' in data and data['reweigh'] is not None:
        # Create reweighed M matrices
        # weighing = data['reweigh']
        # M = [c.get_summarized_dlt(weighing, E0) for c in ecorrs]
        raise NotImplementedError('Reweighing for fundamental matrix not implemented')
    else:
        M = [c.M for c in ecorrs]

    x01 = [c.x01 for c in ecorrs]
    x02 = [c.x02 for c in ecorrs]
    thrs = [np.sqrt(c.num_pts) * thr for c in ecorrs]

    # Normalize initial fundamental matrix
    N1 = data['N1']
    N2 = data['N2']
    init_Fn = N2.T @ init_F @ N1

    tt1 = datetime.datetime.now()
    refined_Fn, info = pysumopt.lm_fundamental_sampson_approx(M, x01, x02, init_Fn, thrs)
    tt2 = datetime.datetime.now()

    # Un-normalize estimated fundamental matrix
    refined_F = np.linalg.inv(N2).T @ refined_Fn @ np.linalg.inv(N1)

    # print(info['iterations'])
    return refined_F, (tt2 - tt1).total_seconds()


def run_poselib_refinement_fundamental(data, init_F, sparsified=False):

    bundle_opt = {
        'loss_type': 'TRUNCATED',
        'loss_scale': data['threshold_E']
    }

    if sparsified:
        x1_normalized = np.array([e.x01 for e in data['ecorrs']])
        x2_normalized = np.array([e.x02 for e in data['ecorrs']])
    else:
        x1_normalized = data['x1c']
        x2_normalized = data['x2c']

    # Normalize initial fundamental matrix
    N1 = data['N1']
    N2 = data['N2']
    init_Fn = N2.T @ init_F @ N1


    tt1 = datetime.datetime.now()
    refined_Fn, info = poselib.refine_fundamental(x1_normalized, x2_normalized, init_Fn, bundle_opt)
    tt2 = datetime.datetime.now()

    # Un-normalize estimated fundamental matrix
    refined_F = np.linalg.inv(N2).T @ refined_Fn @ np.linalg.inv(N1)
    return refined_F, (tt2 - tt1).total_seconds()


def run_poselib_refinement(data, init_pose, sparsified=False):

    bundle_opt = {
        'loss_type': 'TRUNCATED',
        'loss_scale': data['threshold_E']
    }

    if sparsified:
        x1_calibrated = np.array([e.x01 for e in data['ecorrs']])
        x2_calibrated = np.array([e.x02 for e in data['ecorrs']])
    else:
        x1_calibrated = data['x1c']
        x2_calibrated = data['x2c']

    initial_pose = poselib.CameraPose()
    initial_pose.R = init_pose['R']
    initial_pose.t = init_pose['t']

    tt1 = datetime.datetime.now()
    pose_refined, info = poselib.refine_relative_pose(x1_calibrated, x2_calibrated, initial_pose, CAM, CAM, bundle_opt)
    tt2 = datetime.datetime.now()

    opt = {
        'R': pose_refined.R,
        't': pose_refined.t,
    }
    return opt, (tt2 - tt1).total_seconds()
