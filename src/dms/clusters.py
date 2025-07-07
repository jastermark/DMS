import numpy as np

from . import geometry

EPSILON = 1e-6


class ClusterInitializedCorrespondence:

    def __init__(self, cluster, K1, K2, weighing=None) -> None:
        self.K1 = K1
        self.K2 = K2
        self.x1 = cluster['x1'][()]
        self.x2 = cluster['x2'][()]

        # Calibrated points
        self.x1c = geometry.calibrate_pts(self.x1, K1)
        self.x2c = geometry.calibrate_pts(self.x2, K2)
        self.size = len(self.x1c)
        self.num_pts = self.size

        # Summarize
        self.centroid1 = np.mean(self.x1c, axis=0)
        self.centroid2 = np.mean(self.x2c, axis=0)
        self.x01, self.x02 = self._get_closest_match(self.centroid1, self.centroid2)
        self.M = self.get_summarized_dlt(weighing)

        # Uncalibrated
        self.centroid1_pixels = geometry.uncalibrate_pts(self.centroid1, K1)
        self.centroid2_pixels = geometry.uncalibrate_pts(self.centroid2, K2)
        self.x01_pixels = geometry.uncalibrate_pts(self.x01, K1)
        self.x02_pixels = geometry.uncalibrate_pts(self.x02, K2)

    @classmethod
    def from_dense_cluster(cls, cluster, K1, K2, weighing=None):
        ''' Create correspondence from cluster of dense correspondences. '''
        # TODO: Rewrite __init__ and use this method instead to initialize from dense cluster.
        return cls(cluster, K1, K2, weighing)

    @classmethod
    def from_summarized_cluster(cls, centroid1, centroid2, M, K1, K2):
        ''' Create correspondence from already summarized cluster. '''
        raise NotImplementedError("Not implemented yet.")

    def _representative_match(self, x1c, x2c):
        x = np.concatenate([x1c, x2c], axis=1)
        m = np.mean(x, axis=0)
        return m[0:2], m[2:4]

    def _get_closest_match(self, c1, c2):
        # Finds the closest match to point x (calibrated)
        ind = np.argmin(np.linalg.norm(self.x1c - c1, axis=1) + np.linalg.norm(self.x2c - c2, axis=1))
        x01 = self.x1c[ind]
        x02 = self.x2c[ind]
        return x01, x02

    def get_summarized_dlt(self, weighing=None, E0=None):
        raise NotImplementedError("DLT summarization requires specific correspondence.")

    def get_inlier_mask(self, E, threshold):
        # Find inliers based on Sampson Error for a given Essential matrix
        residuals = geometry.sampson_error(E, self.x1c, self.x2c)
        return (residuals <= threshold)

    def filter_outliers(self, E, threshold):
        # Filter out outliers based on Sampson error
        inlier_mask = self.get_inlier_mask(E, threshold)

        self.x1c = self.x1c[inlier_mask]
        self.x2c = self.x2c[inlier_mask]
        self.size = np.sum(inlier_mask)
        self.num_pts = self.size

        # Summarize
        self.M = self.get_summarized_dlt(E0=E)
        if self.M is None:
            # Consider the correspondence an outlier if it has too few inliers
            return

        self.centroid1 = np.mean(self.x1c, axis=0)
        self.centroid2 = np.mean(self.x2c, axis=0)
        self.x01, self.x02 = self._get_closest_match(self.centroid1, self.centroid2)

        # Uncalibrated
        self.centroid1_pixels = geometry.uncalibrate_pts(self.centroid1, self.K1)
        self.centroid2_pixels = geometry.uncalibrate_pts(self.centroid2, self.K2)
        self.x01_pixels = geometry.uncalibrate_pts(self.x01, self.K1)
        self.x02_pixels = geometry.uncalibrate_pts(self.x02, self.K2)


class ClusterCorrespondence(ClusterInitializedCorrespondence):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    @staticmethod
    def _reduce(M, use_cholesky=True):
        # Reduce the DLT matrix to 9x9
        if use_cholesky and M.shape[0] >= 9:
            try:
                return np.linalg.cholesky(M.T @ M).T
            except np.linalg.LinAlgError:
                # If Cholesky fails, fallback to QR decomposition
                print("Cholesky decomposition failed, falling back to QR decomposition.")

        # Fallback to QR decomposition
        M_reduced = np.linalg.qr(M, mode='r')
        if M_reduced.shape[0] < 9:
            M_reduced = np.pad(M_reduced, ((0, 9 - M_reduced.shape[0]), (0, 0)), mode='constant')
        return M_reduced

    def get_summarized_dlt(self, weighing=None, E0=None):
        if self.num_pts == 0:
            return None

        # Get reduced DLT matrix for E
        M_full = geometry.setup_fundamental_dlt_matrix(self.x1c, self.x2c)
        if weighing is not None:
            raise NotImplementedError("Weighing is not implemented.")
        M_reduced = self._reduce(M_full)
        return M_reduced

    def summarized_sampson_error(self, E):
        e = E.flatten()
        numerator = np.linalg.norm(self.M @ e)

        x01 = geometry.to_homogeneous(self.x01)
        x02 = geometry.to_homogeneous(self.x02)
        denominator = np.sqrt(np.linalg.norm(E[0:2,:] @ x01) ** 2 + np.linalg.norm(E[:,0:2].T @ x02) ** 2) 
        return numerator / denominator

    def true_sampson_residuals(self, E):
        residuals = geometry.sampson_error(E, self.x1c, self.x2c)
        return residuals

    def true_sampson_error(self, E):
        residuals = geometry.sampson_error(E, self.x1c, self.x2c)
        return np.sqrt(np.sum(residuals ** 2))
