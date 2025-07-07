import cv2
import os
import numpy as np
import h5py

from .camera import CameraPose
from . import geometry


def h5_to_camera_dict(data):
    camera_dict = {}
    camera_dict['model'] = data['model'].asstr()[0]
    camera_dict['width'] = int(data['width'][0])
    camera_dict['height'] = int(data['height'][0])
    camera_dict['params'] = data['params'][:]
    return camera_dict

def camera_dict_to_calib_matrix(cam):
    if cam['model'] == 'PINHOLE':
        p = cam['params']
        return np.array([[p[0], 0.0, p[2]], [0.0, p[1], p[3]], [0.0, 0.0, 1.0]])
    else:
        raise Exception('nyi model in camera_dict_to_calib_matrix')
        

def filter_small_clusters(clusters, min_size=15):
    return [c for c in clusters if 'x1' in c and len(c['x1']) >= min_size]


class ImageLoader:
    """ Object for loading an image the first time it is needded.
    Loads using OpenCV (supports *.ppm). """

    def __init__(self, path, grayscale=False) -> None:
        self.path = path
        self._loadflag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
        self._loaded_image = None

    def _load(self):
        # Only load the first time this is called - after that save in memory

        if self._loaded_image is not None:
            return self._loaded_image

        image  = cv2.imread(self.path, self._loadflag)
        self._loaded_image = image
        return image

    @staticmethod
    def convert_opencv_to_numpy(image):
        """_summary_

        Args:
            image (uint8-array): (H, W, C) BGR image

        Returns:
            (float32-array): (H, W, C) RGB image
        """
        cv_rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return cv_rgb_image.astype("float32") / 255.

    @staticmethod
    def convert_pytorch_to_numpy(tensor):
        """_summary_

        Args:
            tensor (float32-tensor): (C, H, W) or (1, C, H, W) RGB image

        Returns:
            (float32-array): (H, W, C) RGB image
        """
        squeezed_tensor = tensor.squeeze()
        numpy_image = squeezed_tensor.numpy()
        channel_last_image = numpy_image.transpose([1, 2, 0])
        return channel_last_image

    def as_opencv(self):
        # Loads image as uint8 BGR (OpenCV standard).
        return self._load()

    def as_numpy(self):
        # Loads image and converts to float32 Numpy array (supports *.ppm)
        cv_bgr_image = self._load()
        return self.convert_opencv_to_numpy(cv_bgr_image)


class ImagePair:
    def __init__(self, image_dir, data) -> None:
        self.data = data
        self.image_dir = image_dir
        self.key = self.data.name.replace('/', '')
        self.has_ground_truth = 'R' in self.data and 't' in self.data
 
    def R(self):
        if not self.has_ground_truth:
            return None

        R_gt = self.data['R'][:].astype('float64')
        u,s,vt = np.linalg.svd(R_gt)
        R_gt = u @ vt
        return R_gt

    def t(self, normalize=True):
        if not self.has_ground_truth:
            return None

        t_gt = self.data['t'][:].astype('float64')
        if normalize:
            t_gt = t_gt / np.linalg.norm(t_gt)
        return t_gt

    def relative_pose(self):
        return CameraPose(self.R(), self.t())
    
    def essential_matrix(self):
        R = self.R()
        t = self.t()
        T = np.array([[0.0, -t[2], t[1]], [t[2], 0.0, -t[0]], [-t[1], t[0], 0.0]])
        E = T @ R
        return E

    def fundamental_matrix(self):
        (K1, K2) = self.calib_matrices()
        E = self.essential_matrix()
        F = np.linalg.inv(K2).T @ E @ np.linalg.inv(K1)
        return F

    def image_names(self):
        return [self.data['name1'].asstr()[0], self.data['name2'].asstr()[0]]

    def image_paths(self):
        return [os.path.join(self.image_dir, name) for name in self.image_names()]
    
    def images(self, mode="numpy", grayscale=False):
        image_paths = self.image_paths()
        if mode == "numpy":
            return (ImageLoader(image_paths[0], grayscale).as_numpy(), ImageLoader(image_paths[1], grayscale).as_numpy())
        elif mode == "opencv":
            raise NotImplementedError("OpenCV mode not implemented")
        elif mode == "torch":
            raise NotImplementedError("Torch mode not implemented")
        else:
            raise ValueError("Invalid image mode")

    def num_matches(self):
        return len(self.data['x1'])

    def matches(self, calibrated=False, confidence=None):
        x1 = self.data['x1'][:]
        x2 = self.data['x2'][:]
        if confidence is not None:
            mask = self.inlier_mask(confidence, method='confidence')
            x1 = x1[mask]
            x2 = x2[mask]
        if not calibrated:
            return (x1, x2)
        else:
            K1, K2 = self.calib_matrices()
            x1c = geometry.calibrate_pts(x1, K1)
            x2c = geometry.calibrate_pts(x2, K2)
            return (x1c, x2c)

    def inlier_mask(self, threshold, method='sampson'):
        if method == 'confidence':
            assert 'confidence' in self.data, "Confidence values not found in dataset"
            return self.data['confidence'][:] >= threshold
        if method == 'sampson':
            return self.sampson_inlier_mask(threshold)
        else:
            raise ValueError(f"Invalid inlier method: {method}")

    def sampson_inlier_mask(self, threshold):
        x1 = self.data['x1'][:]
        x2 = self.data['x2'][:]
        F = self.fundamental_matrix()
        residuals = geometry.sampson_error(F, x1, x2)
        return residuals < threshold

    def cameras(self):
        return (h5_to_camera_dict(self.data['camera1']), h5_to_camera_dict(self.data['camera2']))

    def has_calibration(self):
        if 'camera1' in self.data and 'camera2' in self.data:
            return 'model' in self.data['camera1'] and 'model' in self.data['camera2']
        return False

    def calib_matrices(self):
        if self.has_calibration():
            cam1_dict = h5_to_camera_dict(self.data['camera1'])
            cam2_dict = h5_to_camera_dict(self.data['camera2'])
            return (camera_dict_to_calib_matrix(cam1_dict), camera_dict_to_calib_matrix(cam2_dict))
        else:
            # return self.normalization_matrices()
            return (np.eye(3), np.eye(3))

    def normalization_matrices(self):
        x1, x2 = self.matches()
        N1 = geometry.normalization_matrix(x1)
        N2 = geometry.normalization_matrix(x2)
        return (N1, N2)

    def clusters(self, min_size=None):
        if 'clusters' in self.data:
            clusters_group = self.data['clusters']
            clusters = [c for c in clusters_group.values()]
            if min_size is not None:
                return filter_small_clusters(clusters, min_size)
            return clusters
        else:
            return None


class EvaluationDataset:

    def __init__(self, h5_path, im_path) -> None:
        assert os.path.exists(h5_path), "Dataset file not found: " + h5_path
        assert os.path.exists(im_path), "Image directory not found"
        self.data = h5py.File(h5_path, 'r')
        self.image_dir = im_path
        self.pairs = list(self.data.keys())

    def __iter__(self):
        for i in range(len(self.pairs)):
            yield self[i]
    
    def __len__(self):
        return len(self.data.items())
    
    def __getitem__(self, idx):
        return ImagePair(self.image_dir, self.data[self.pairs[idx]])

    def close(self):
        # Close the h5 file
        self.data.close()
