import abc
import enum
from typing import Any, Literal, Protocol
import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern
import pywt
from skimage.feature import daisy

from libs_week3.color_conversion import ColorConversion, ColorSpace
from libs_week3.preprocessing import ImagePreprocessStep


def flatten_list(l):
    res = []

    for sublist in l:
        for item in sublist:
            res.append(item)

    return res


def bgr_to_cmyk(bgr_image):
    bgr = bgr_image
    r = bgr[..., 2]
    g = bgr[..., 1]
    b = bgr[..., 0]
    
    k = 1 - np.maximum.reduce([r, g, b])
    
    denom = 1 - k
    denom[denom == 0] = 1
    
    c = (1 - r - k) / denom
    m = (1 - g - k) / denom
    y = (1 - b - k) / denom
    
    cmyk = np.stack((c, m, y, k), axis=-1)
    
    cmyk[np.isnan(cmyk)] = 0

    return cmyk

# https://stackoverflow.com/questions/5595425/how-to-compare-floats-for-almost-equality-in-python
def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
    return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)


def get_colorspace_ranges(color_space: 'ColorSpace') -> list[tuple[float, float]]:
    """
    Get the appropriate value ranges for each channel in a color space.
    Returns a list of (min, max) tuples, one per channel.

    Note: These ranges are for NORMALIZED (0-1) images after dividing by 255.
    OpenCV cvtColor works on uint8 [0-255] or float32 [0-1] differently.
    """
    if color_space == ColorSpace.RGB:
        return [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]

    elif color_space == ColorSpace.HSV:
        # OpenCV HSV on float32 [0-1]: H=[0,1], S=[0,1], V=[0,1]
        return [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]

    elif color_space == ColorSpace.LAB:
        # OpenCV LAB on float32 input [0-1]:
        # L in [0, 100], a in [-127, 127], b in [-127, 127]
        # These are the ACTUAL ranges, not normalized to [0,1]!
        return [(0.0, 100.0), (-127.0, 127.0), (-127.0, 127.0)]

    elif color_space == ColorSpace.LUV:
        # OpenCV LUV on float32: L=[0,100], u=[-134,220], v=[-140,122] (approx)
        return [(0.0, 100.0), (-134.0, 220.0), (-140.0, 122.0)]

    elif color_space == ColorSpace.YCRCB:
        # OpenCV YCrCb on float32 [0-1]: Y=[0,1], Cr=[0,1], Cb=[0,1]
        return [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]

    elif color_space == ColorSpace.HLS:
        # OpenCV HLS on float32: H=[0,1], L=[0,1], S=[0,1]
        return [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]

    elif color_space == ColorSpace.YUV:
        # OpenCV YUV on float32: Y=[0,1], U=[-0.436,0.436], V=[-0.615,0.615] (approx)
        return [(0.0, 1.0), (-0.5, 0.5), (-0.5, 0.5)]

    elif color_space == ColorSpace.XYZ:
        # OpenCV XYZ on float32: X=[0,~1], Y=[0,~1], Z=[0,~1]
        return [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]

    elif color_space == ColorSpace.CMYK:
        # Our custom CMYK: all in [0,1]
        return [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]

    else:
        # Default: assume [0, 1] for all channels
        return [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]



class KeypointDescriptorMaker(abc.ABC):
    #keypoint descriptor class
    
    @abc.abstractmethod
    def detect_and_compute(self, image: np.ndarray, mask: np.ndarray | None = None) -> tuple[list, np.ndarray]:
        
        #Detect keypoints and compute descriptors.
        #Returns: keypoints: List of cv2.KeyPoint objects, descriptors: Array of shape (n_keypoints, descriptor_dim)
        pass
    
    @abc.abstractmethod
    def to_dict(self) -> dict[str, Any]:
        pass


class ORBDescriptor(KeypointDescriptorMaker):
    def __init__(self, n_features: int = 500, scale_factor: float = 1.2, n_levels: int = 8):
        self.n_features = n_features
        self.scale_factor = scale_factor
        self.n_levels = n_levels
        self.orb = cv2.ORB_create(nfeatures=n_features, scaleFactor=scale_factor, nlevels=n_levels)
        
    def detect_and_compute(self, image: np.ndarray, mask: np.ndarray | None = None) -> tuple[list, np.ndarray]:
        if len(image.shape) == 3:
            gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            gray = (image * 255).astype(np.uint8)
        
        keypoints, descriptors = self.orb.detectAndCompute(gray, mask)
        return keypoints, descriptors
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "ORB",
            "n_features": self.n_features,
            "scale_factor": self.scale_factor,
            "n_levels": self.n_levels
        }

class DaisyDescriptor(KeypointDescriptorMaker):
    def __init__(self, step: int = 4, radius: int = 15, rings: int = 3, histograms: int = 8, orientations: int = 8):
        self.step = step
        self.radius = radius
        self.rings = rings
        self.histograms = histograms
        self.orientations = orientations
        
    def detect_and_compute(self, image: np.ndarray, mask: np.ndarray | None = None) -> tuple[list, np.ndarray]:
        if len(image.shape) == 3:
            gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            gray = (image * 255).astype(np.uint8)
        
        descriptor = daisy(gray, step=self.step, radius=self.radius, rings=self.rings, histograms=self.histograms, orientations=self.orientations, visualize=False)
        
        rows, cols, desc_dim = descriptor.shape
        
        keypoints = []
        descriptors = []
        
        for r in range(rows):
            for c in range(cols):
                keypoint = cv2.KeyPoint(x=c * self.step, y=r * self.step, size=float(self.radius))
                keypoints.append(keypoint)
                descriptors.append(descriptor[r, c, :])
        
        if len(descriptors) == 0:
            return [], None
        
        descriptors = np.array(descriptors, dtype=np.float32)
        return keypoints, descriptors
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "DAISY",
            "step": self.step,
            "radius": self.radius,
            "rings": self.rings,
            "histograms": self.histograms,
            "orientations": self.orientations
        }
        
class SIFTDescriptor(KeypointDescriptorMaker):
    def __init__(self, n_features: int = 0, n_octave_layers: int = 3, contrast_threshold: float = 0.04, edge_threshold: float = 10, sigma: float = 1.6):
        self.n_features = n_features
        self.n_octave_layers = n_octave_layers
        self.contrast_threshold = contrast_threshold
        self.edge_threshold = edge_threshold
        self.sigma = sigma
        self.sift = cv2.SIFT_create(nfeatures=n_features, nOctaveLayers=n_octave_layers, contrastThreshold=contrast_threshold, edgeThreshold=edge_threshold, sigma=sigma)
        
    def detect_and_compute(self, image: np.ndarray, mask: np.ndarray | None = None) -> tuple[list, np.ndarray]:
        if len(image.shape) == 3:
            gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            gray = (image * 255).astype(np.uint8)
        
        keypoints, descriptors = self.sift.detectAndCompute(gray, mask)
        return keypoints, descriptors
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "SIFT",
            "n_features": self.n_features,
            "n_octave_layers": self.n_octave_layers,
            "contrast_threshold": self.contrast_threshold,
            "edge_threshold": self.edge_threshold,
            "sigma": self.sigma
        }


class DescriptorMatcher:
    
    def __init__(self, matcher_type: str = 'BF', norm_type: int = cv2.NORM_L2, cross_check: bool = False, ratio_test_threshold: float = 0.75):
        #matcher_type: 'BF' or 'FLANN'
        #norm_type: cv2.NORM_L2 for sift and daisy, cv2.NORM_HAMMING for orb
        self.matcher_type = matcher_type
        self.norm_type = norm_type
        self.cross_check = cross_check
        self.ratio_test_threshold = ratio_test_threshold
        
        if matcher_type == "BF":
            self.matcher = cv2.BFMatcher(normType=norm_type, crossCheck=cross_check)
        else:
            if norm_type == cv2.NORM_HAMMING:
                index_params = dict(algorithm=6, table_number=12, key_size=20, multi_probe_level=2)
                search_params = dict(checks=50)
            else:
                index_params = dict(algorithm=1, trees=5)
                search_params = dict(checks=50)
            self.matcher = cv2.FlannBasedMatcher(indexParams=index_params, searchParams=search_params)
            
    def match(self, descriptors1: np.ndarray, descriptors2: np.ndarray) -> list:
        
        if descriptors1 is None or descriptors2 is None or len(descriptors1) == 0 or len(descriptors2) == 0:
            return []
        matches = self.matcher.knnMatch(descriptors1, descriptors2, k=2)
        good_matches = []
        for m_n in matches:
            if len(m_n) != 2:
                continue
            m, n = m_n
            if m.distance < self.ratio_test_threshold * n.distance:
                good_matches.append(m)
        return good_matches
    
    def discard_painting(self, num_matches: int, threshold: int = 10) -> bool:
        #return True if the number of matches is less than the threshold
        return num_matches < threshold
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "matcher_type": self.matcher_type,
            "norm_type": self.norm_type,
            "cross_check": self.cross_check,
            "ratio_test_threshold": self.ratio_test_threshold
        }
            
        
        

if __name__ == "__main__":
    pass
    


        