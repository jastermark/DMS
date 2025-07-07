import numpy as np

DEFAULT_AUC_THRESHOLDS = [5, 10, 20]


def rotation_angle(R):
    # Computes angle of rotation from rotation matrix
    return np.rad2deg(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))


def angle(v1, v2):
    # Computes angle between two vectors
    v1 = v1.squeeze()
    v2 = v2.squeeze()
    assert v1.shape == v2.shape and len(v1.shape) == 1

    if np.linalg.norm(v1) == 0:
        return 180
    n = np.linalg.norm(v1) * np.linalg.norm(v2)
    return np.rad2deg(np.arccos(np.clip(np.dot(v1, v2) / n, -1.0, 1.0)))


def calculate_pose_errors(pose, image_pair_data):
    R_gt, t_gt = image_pair_data['R_gt'], image_pair_data['t_gt']
    R_est, t_est = pose['R'], pose['t']

    errors = {}
    errors['R'] = rotation_angle(R_est @ R_gt.T)
    errors['t'] = angle(t_est, t_gt)
    errors['max'] = np.max((errors['R'], errors['t']))
    return errors


def calculate_multipose_errors(poses, image_pair_data):
    # Calculate pose errors for multiple estimates
    errors = {'R': [], 't': [], 'max': []}
    for pose in poses:
        pose_errors = calculate_pose_errors(pose, image_pair_data)
        for error_key, err in pose_errors.items():
            errors[error_key].append(err)
    return errors


def compute_auc(errors, thresholds=DEFAULT_AUC_THRESHOLDS):
    sort_idx = np.argsort(errors)
    errors = np.array(errors.copy())[sort_idx]
    recall = (np.arange(len(errors)) + 1) / len(errors)
    errors = np.r_[0., errors]
    recall = np.r_[0., recall]
    aucs = []
    for t in thresholds:
        last_index = np.searchsorted(errors, t)
        r = np.r_[recall[:last_index], recall[last_index-1]]
        e = np.r_[errors[:last_index], t]
        aucs.append(np.trapz(r, x=e)/t)
    return aucs


def compute_average_auc(errors, thresholds=DEFAULT_AUC_THRESHOLDS):
    # Compute AUC for multiple trials -- assume each row is a trial
    all_aucs = [compute_auc(errs_row, thresholds) for errs_row in errors]
    mean_auc = np.mean(all_aucs, axis=0)
    stddevs = np.std(all_aucs, axis=0)
    return mean_auc, stddevs
