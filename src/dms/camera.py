import numpy as np
from scipy.spatial.transform import Rotation


class CameraPose:

    NUM_DOF = 5  # 3 for rotation, 2 for translation

    def __init__(self, rotmat=None, tvec=None) -> None:

        # Default pose is identity
        if rotmat is None:
            rotmat = np.eye(3)
        if tvec is None:
            tvec = np.zeros((3, 1))

        self._rotation = Rotation.from_matrix(rotmat)

        self.R = self.rotmat()  # matrix
        self.t = tvec.squeeze()  # vector

    @staticmethod
    def rotmat_from_parameters(parameters):
        assert len(parameters) == 3
        return Rotation.from_mrp(parameters).as_matrix()

    @staticmethod
    def tmat_from_parameters(parameters):
        # Using convention: theta = inclination, phi = azimuth from
        # https://en.wikipedia.org/wiki/Spherical_coordinate_system#Cartesian_coordinates
        assert len(parameters) == 2
        theta, phi = parameters[0], parameters[1]
        return np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])

    @staticmethod
    def from_parameters(parameters):
        R = CameraPose.rotmat_from_parameters(parameters[:3])
        t = CameraPose.tmat_from_parameters(parameters[3:])
        return R, t

    def set_parameters(self, parameters):
        assert len(parameters) == self.NUM_DOF
        self._rotation = Rotation.from_mrp(parameters[:3])
        self.t = self.tmat_from_parameters(parameters[3:]).squeeze()

    def get_parameters(self):
        R_parameters = self._rotation.as_mrp()
        # Using convention: theta = inclination, phi = azimuth from
        # https://en.wikipedia.org/wiki/Spherical_coordinate_system#Cartesian_coordinates
        t_normalized = self.t / np.linalg.norm(self.t)
        # theta = np.arccos(t_normalized[2])
        # phi = np.arctan2(t_normalized[1], t_normalized[0])
        # t_parameters = np.array([theta, phi])
        t_parameters = self.cart2sph(*t_normalized)
        return np.concatenate([R_parameters, t_parameters])

    @staticmethod
    def cart2sph(x, y, z):
        xy = np.sqrt(x**2 + y**2) # sqrt(x² + y²)       x_2 = x**2   y_2 = y**2   z_2 = z**2
        # r = np.sqrt(x_2 + y_2 + z_2) # r = sqrt(x² + y² + z²)
        
        phi = np.arctan2(y, x) 
        theta = np.arctan2(xy, z) 
        return theta, phi
 

    def rotmat(self):
        return self._rotation.as_matrix()

    def tmat(self):
        # Return translation as a column matrix
        return self.t.reshape(3, 1)

    def rquat(self):
        return self._rotation.as_quat()
