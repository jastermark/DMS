import numpy as np


def to_homogeneous(x):
    # x is (..., N, 2) or (..., N, 3)
    pts_shape = x.shape[:-1]
    h_coord = np.ones((*pts_shape, 1))
    return np.concatenate([x, h_coord], axis=-1)


def from_homogeneous(x):
    return x[..., 0:-1] / x[..., -1:]


def calibrate_pts(pts, K):
    pts_calib = np.asarray(pts.copy())
    pts_calib[:,0] -= K[0,2]
    pts_calib[:,1] -= K[1,2]
    pts_calib[:,0] /= K[0,0]
    pts_calib[:,1] /= K[1,1]
    return pts_calib


def uncalibrate_pts(pts, K):
    pts = to_homogeneous(pts)
    pts = K @ pts.T
    pts = from_homogeneous(pts.T)
    return pts


def setup_fundamental_dlt_matrix(x1, x2, row_major=True):
    # Builds matrix such that
    # x2'*E*x1 = M * E.flatten() 
    # if row_major=False it is columnwise (matlab) ordering
    # x2'*E*x1 = M * E.T.flatten() 
    
    M = []
    for (p1,p2) in zip(x1,x2):
        if row_major:
            M.append([p2[0]*p1[0], p2[0]*p1[1], p2[0], p2[1]*p1[0], p2[1]*p1[1], p2[1], p1[0], p1[1], 1.0])
        else:
            M.append([p1[0]*p2[0], p1[0]*p2[1], p1[0], p1[1]*p2[0], p1[1]*p2[1], p1[1], p2[0], p2[1], 1.0])
    return np.array(M)


def crossmat(t):
    t = t.flatten()
    return np.array([[0, -t[2], t[1]],
                     [t[2], 0, -t[0]],
                     [-t[1], t[0], 0]])


def essential_from_pose(R, t):
    t = t.flatten()
    return crossmat(t) @ R
