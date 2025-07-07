import PIL
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
from matplotlib import colormaps

from . import geometry
from .dataloader import ImageLoader

COLOR_DEFAULT = mcolors.to_rgb("blue")


class MatchPlotter:

    MARKER_SIZE = 5
    LINES = 0

    # Color options
    GRADIENT = 1
    GRADIENT_INLIERS = 2

    def __init__(self, im1=None, im2=None, matches=None, inlier_mask=None, style=LINES, grayscale=False) -> None:
        self._im1 = None
        self._im2 = None
        self._x1 = None
        self._x2 = None
        self._inlier_mask = None
        self.masks = None
        self.style = style
        self.grayscale = grayscale

        if im1 is not None:
            self.add_image(im1, index=0)
        if im2 is not None:
            self.add_image(im2, index=1)
        if matches is not None:
            self.set_matches(matches, inlier_mask)

        self._fig = None

    def add_image(self, image_name, index=None):
        if index is None:
            if self._im1 is None:
                index = 0
            elif self._im2 is None:
                index = 1
            else:
                raise ValueError("Can not add image to plotter; all images are set. \
                                 If you want to replace an image, please specify index.")
        if isinstance(image_name, str):
            image = ImageLoader(image_name, self.grayscale).as_numpy()
        elif isinstance(image_name, PIL.Image.Image):
            image = np.array(image_name)
            if self.grayscale:
                raise ValueError("Grayscale conversion not implemented for PIL-images.")
        else:
            image = image_name
            if self.grayscale:
                raise ValueError("Grayscale conversion not implemented for numpy arrays.")

        if index == 0:
            self._im1 = image
        elif index == 1:
            self._im2 = image
        else:
            raise ValueError("Invalid index")
    
    # def add_images(self, image_names):
    #     for i, image_name in enumerate(image_names):
    #         self.add_image(image_name, index=i)

    # def set_cameras(self, K1, K2):
    #     self.K1 = K1
    #     self.K2 = K2

    def set_matches(self, matches, inlier_mask=None, calibrated=False):
        x1, x2 = matches
        if calibrated:
            # Project back to image plane
            x1 = geometry.uncalibrate_pts(x1, self.K1)
            x2 = geometry.uncalibrate_pts(x2, self.K2)
        self._x1 = x1
        self._x2 = x2
        self._inlier_mask = inlier_mask

    # def add_match(self, match):
    #     # Add a single match to existing matches
    #     new_x1, new_x2 = match
    #     if self._x1 is None:
    #         self._x1 = new_x1.copy().reshape(1, -1)
    #         self._x2 = new_x2.copy().reshape(1, -1)
    #     else:
    #         self._x1 = np.vstack([self._x1, new_x1])
    #         self._x2 = np.vstack([self._x2, new_x2])

    # def add_matches(self, matches):
    #     # Add matches to existing matches
    #     raise NotImplementedError()

    # def add_masks(self, masks):
    #     if self.masks is None:
    #         self.masks = masks.copy()
    #     else:
    #         self.masks += masks

    def draw_images(self):
        assert not self._im1 is None, "Image 1 not found"
        assert not self._im2 is None, "Image 2 not found"

        # Concatenate images
        h1, w1 = self._im1.shape[:2]
        h2, w2 = self._im2.shape[:2]
        img = np.zeros((max(h1, h2), w1 + w2, 3), dtype=self._im1.dtype)
        img[:h1, :w1, :] = self._im1
        img[:h2, w1:, :] = self._im2

        # Plot image
        self._fig = plt.figure(figsize=(20, 20))
        plt.imshow(img)
        plt.axis('off')

    # def draw_inlier_matches(self):
    #     inlier_x1 = self._x1[self._inlier_mask]
    #     inlier_x2 = self._x2[self._inlier_mask]
    #     self.draw_matches(inlier_x1, inlier_x2)

    # def draw_keypoints(self, x):
    #     if self.masks is None or len(self.masks) == 0:
    #         plt.scatter(x[:,0], x[:,1], s=3, color=COLOR_DEFAULT)
    #     else:
    #         for i, mask in enumerate(self.masks):
    #             color = ind2color(i)
    #             plt.scatter(x[mask,0], x[mask,1], s=self.MARKER_SIZE, color=color)
    #         unassigned_mask = ~sum(self.masks).astype(bool)
    #         plt.scatter(x[unassigned_mask,0], x[unassigned_mask,1], s=self.MARKER_SIZE, color=COLOR_DEFAULT)

    # def draw_all_keypoints(self):
    #     if self._x1 is not None:
    #         # Keypoints in image 1
    #         self.draw_keypoints(self._x1)
        
    #     if self._x2 is not None and self._im1 is not None:
    #         # Keypoints in image 2 (offset by image 1)
    #         h, w = self._im1.shape[:2]
    #         x2 = self._x2.copy()
    #         x2[:,0] += w
    #         self.draw_keypoints(x2)

    def draw_matches_gradient(self, x1, x2, cmap='Spectral'):
        cmap = colormaps[cmap]
        _, w = self._im1.shape[:2]

        ind = np.argsort(x1[:, 0] + x1[:, 1])
        for k, (z1, z2) in enumerate(zip(x1[ind], x2[ind])):
            c = cmap(k/len(ind))
            plt.plot(z1[0], z1[1], color=c, alpha=0.8, marker='.', markersize=5)
            plt.plot(z2[0]+w, z2[1], color=c, alpha=0.8, marker='.', markersize=5)

    def _draw_matches_binary_color(self, x1, x2, inlier_mask=None, color=None, lines=True):
        if x1 is None or x2 is None:
            return
        if self._im1 is None:
            return
        if color is None and inlier_mask is None:
            color = COLOR_DEFAULT

        if inlier_mask is None:
            colors = [color] * len(x1)
        else:
            colors = ['green' if m else 'red' for m in inlier_mask]

        # Offset x2 by image 1
        x2 = x2.copy()
        _, w = self._im1.shape[:2]
        x2[:,0] += w

        for start, stop, c in zip(x1, x2, colors):
            if lines:
                x_interval = [start[0], stop[0]]
                y_interval = [start[1], stop[1]]
                plt.plot(x_interval, y_interval, color=c, alpha=0.8)
            else:
                plt.plot(start[0], start[1], color=c, alpha=0.8, marker='.', markersize=5)
                plt.plot(stop[0], stop[1], color=c, alpha=0.8, marker='.', markersize=5)

    def draw_matches(self, x1, x2, color=None):
        if self.style == self.LINES:
            self._draw_matches_binary_color(x1, x2, self._inlier_mask, color=color)
        elif self.style == self.GRADIENT:
            self.draw_matches_gradient(x1, x2, cmap='Spectral')
        elif self.style == self.GRADIENT_INLIERS:
            self._draw_matches_binary_color(x1[~self._inlier_mask], x2[~self._inlier_mask], color='black', lines=False)
            self.draw_matches_gradient(x1[self._inlier_mask], x2[self._inlier_mask], cmap='Spectral')
        elif self.style == self.DOUBLE_GRADIENT:
            self.draw_matches_gradient(x1[~self._inlier_mask], x2[~self._inlier_mask], cmap='Reds')
            self.draw_matches_gradient(x1[self._inlier_mask], x2[self._inlier_mask], cmap='Greens')
        else:
            raise ValueError("Invalid style", self.style)

    def draw(self, title=None, show_matches=True, color=None):
        self.draw_images()

        if show_matches:
            self.draw_matches(self._x1, self._x2, color) 
        else:
            self.draw_all_keypoints()

        if title is not None:
            title = f"{len(self._x1)} matches"
        plt.title(title)
        plt.show()
