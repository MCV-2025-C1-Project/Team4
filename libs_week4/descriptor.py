import abc
import enum
from typing import Any, Literal, Protocol
import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern
import pywt
from skimage.feature import daisy, hog
from sklearn.decomposition import PCA
from cv2 import xfeatures2d

# from libs_week3.color_conversion import ColorConversion, ColorSpace
# from libs_week3.preprocessing import ImagePreprocessStep

class ColorSpace(enum.Enum):
    RGB = 'RGB'
    # GRAY = 'GRAY'
    HSV = 'HSV'
    LAB = 'LAB'
    YCRCB = 'YCRCB'
    HLS = 'HLS'
    CMYK = 'CMYK'
    LUV = 'LUV'
    XYZ = 'XYZ'
    YUV = 'YUV'
    
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


class KeypointFinder(abc.ABC):
    """
    Detect keypoints only (no descriptors).
    """
    @abc.abstractmethod
    def detect(self, image: np.ndarray, mask: np.ndarray | None = None) -> list[cv2.KeyPoint]:
        pass

    @abc.abstractmethod
    def to_dict(self) -> dict[str, Any]:
        pass


class HarrisFinder(KeypointFinder):
    def __init__(self, block_size: int = 2, ksize: int = 3, k: float = 0.04, thresh: float = 0.01):
        self.block_size = block_size
        self.ksize = ksize
        self.k = k
        self.thresh = thresh

    def detect(self, image: np.ndarray, mask: np.ndarray | None = None) -> list[cv2.KeyPoint]:
        if len(image.shape) == 3:
            gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            gray = (image * 255).astype(np.uint8)

        gray_f = np.float32(gray)
        dst = cv2.cornerHarris(gray_f, blockSize=self.block_size, ksize=self.ksize, k=self.k)
        # dilate to mark the corners
        dst_dilated = cv2.dilate(dst, None)
        # threshold to get strong corners
        corners = (dst_dilated > self.thresh * dst_dilated.max()).astype(np.uint8)
        # find centroids (connected components)
        contours, _ = cv2.findContours(corners, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        keypoints = []
        for cnt in contours:
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
            keypoints.append(cv2.KeyPoint(float(cx), float(cy), size= self.ksize*2))
        return keypoints

    def to_dict(self) -> dict[str, Any]:
        return {"type": "HARRIS", "block_size": self.block_size, "ksize": self.ksize, "k": self.k, "thresh": self.thresh}


class HarrisLaplacianFinder(KeypointFinder):
    def __init__(self, scales: list[float] = None, harris_kwargs: dict | None = None, laplacian_k: int = 3, harris_thresh: float = 0.01):
        # scales: gaussian sigma values to search across
        self.scales = scales if scales is not None else [1.0, 1.6, 2.0, 2.8]
        self.harris_kwargs = harris_kwargs or {}
        self.laplacian_k = laplacian_k
        self.harris_thresh = harris_thresh

    def detect(self, image: np.ndarray, mask: np.ndarray | None = None) -> list[cv2.KeyPoint]:
        if len(image.shape) == 3:
            gray0 = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            gray0 = (image * 255).astype(np.uint8)

        # For each scale compute Harris response on the blurred image and keep candidates where
        # Laplacian has local maxima (approx).
        candidates = []
        laplacian_responses = []

        for sigma in self.scales:
            ksize = int(max(3, round(sigma * 3) | 1))  # odd kernel
            blurred = cv2.GaussianBlur(gray0, (ksize, ksize), sigmaX=sigma)
            harris = cv2.cornerHarris(np.float32(blurred), blockSize=self.harris_kwargs.get("block_size", 2),
                                     ksize=self.harris_kwargs.get("ksize", 3),
                                     k=self.harris_kwargs.get("k", 0.04))
            # threshold to extract points
            thresh_mask = (harris > self.harris_thresh * harris.max()).astype(np.uint8)
            contours, _ = cv2.findContours(thresh_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # Laplacian on the blurred image
            lap = cv2.Laplacian(blurred, cv2.CV_32F, ksize=self.laplacian_k)
            for cnt in contours:
                M = cv2.moments(cnt)
                if M["m00"] == 0:
                    continue
                cx = M["m10"] / M["m00"]
                cy = M["m01"] / M["m00"]
                candidates.append((cx, cy, sigma))
                laplacian_responses.append(lap[int(round(cy)), int(round(cx))])

        # Keep candidates where the laplacian response is a local maxima in scale-space (simple filtering)
        # We sort by absolute laplacian magnitude and pick top ones to avoid massive number of keypoints.
        if not candidates:
            return []
        candidates = np.array(candidates, dtype=float)
        laplacian_responses = np.array(laplacian_responses, dtype=float)
        # pick top N by abs(laplacian)
        N = min(500, len(candidates))
        idxs = np.argsort(-np.abs(laplacian_responses))[:N]
        keypoints = []
        for i in idxs:
            x, y, s = candidates[i]
            keypoints.append(cv2.KeyPoint(float(x), float(y), size=float(s * 6.0)))
        return keypoints

    def to_dict(self) -> dict[str, Any]:
        return {"type": "HARRIS_LAPLACIAN", "scales": self.scales, "harris_kwargs": self.harris_kwargs}


class SIFTFinder(KeypointFinder):
    def __init__(self, n_features: int = 0, n_octave_layers: int = 3, contrast_threshold: float = 0.04, edge_threshold: float = 10, sigma: float = 1.6):
        try:
            self.sift = cv2.SIFT_create(nfeatures=n_features, nOctaveLayers=n_octave_layers,
                                       contrastThreshold=contrast_threshold, edgeThreshold=edge_threshold, sigma=sigma)
        except AttributeError:
            # older OpenCV naming
            self.sift = cv2.xfeatures2d.SIFT_create(nfeatures=n_features, nOctaveLayers=n_octave_layers,
                                                   contrastThreshold=contrast_threshold, edgeThreshold=edge_threshold, sigma=sigma)

    def detect(self, image: np.ndarray, mask: np.ndarray | None = None) -> list[cv2.KeyPoint]:
        if len(image.shape) == 3:
            gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            gray = (image * 255).astype(np.uint8)
        return self.sift.detect(gray, mask)

    def to_dict(self) -> dict[str, Any]:
        return {"type": "SIFT", "n_features": self.sift.getNFeatures(), "n_octave_layers": self.sift.getNOctaveLayers(),
                "contrast_threshold": self.sift.getContrastThreshold(), "edge_threshold": self.sift.getEdgeThreshold(), "sigma": self.sift.getSigma()}

class KeypointDescriptorMaker(abc.ABC):
    #keypoint descriptor class
    
    @abc.abstractmethod
    def detect_and_compute(self, image: np.ndarray, mask: np.ndarray | None = None) -> tuple[list, np.ndarray]:
        
        #Detect keypoints and compute descriptors.
        #Returns: keypoints: List of cv2.KeyPoint objects, descriptors: Array of shape (n_keypoints, descriptor_dim)
        pass
    
    def detect(self, image: np.ndarray, mask: np.ndarray | None = None) -> list[cv2.KeyPoint]:
        if len(image.shape) == 3:
            gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            gray = (image * 255).astype(np.uint8)
        keypoints = self.finder.detect(gray, mask)
        return keypoints 
    
    @abc.abstractmethod
    def to_dict(self) -> dict[str, Any]:
        pass


class ORBDescriptor(KeypointDescriptorMaker):
    def __init__(self, finder: KeypointFinder = None, n_features: int = 500, scale_factor: float = 1.2, n_levels: int = 8):
        self.finder = finder
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
    
    def detect(self, image: np.ndarray, mask: np.ndarray | None = None) -> list[cv2.KeyPoint]:
        if len(image.shape) == 3:
            gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            gray = (image * 255).astype(np.uint8)
        keypoints = self.finder.detect(gray, mask)
        return keypoints
    
    def compute(self, image: np.ndarray, keypoints: list[cv2.KeyPoint], mask: np.ndarray | None = None) -> np.ndarray:
        if len(image.shape) == 3:
            gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            gray = (image * 255).astype(np.uint8)
        _, descriptors = self.orb.compute(gray, keypoints)
        return descriptors
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "ORB",
            "n_features": self.n_features,
            "scale_factor": self.scale_factor,
            "n_levels": self.n_levels
        }

class DaisyDescriptor(KeypointDescriptorMaker):
    def __init__(self, finder: KeypointFinder = None, step: int = 4, radius: int = 15, rings: int = 3, histograms: int = 8, orientations: int = 8):
        self.finder = finder
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
    
    def detect(self, image: np.ndarray, mask: np.ndarray | None = None) -> list[cv2.KeyPoint]:
        if len(image.shape) == 3:
            gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            gray = (image * 255).astype(np.uint8)
        
        keypoints = self.finder.detect(gray, mask)
        
        return keypoints
    
    def compute(self, image: np.ndarray, keypoints: list[cv2.KeyPoint], mask: np.ndarray | None = None) -> np.ndarray:
        if len(image.shape) == 3:
            gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            gray = (image * 255).astype(np.uint8)
        
        descriptor = daisy(gray, step=self.step, radius=self.radius, rings=self.rings, histograms=self.histograms, orientations=self.orientations, visualize=False)
        
        
        descriptors = [descriptor[kp[0], kp[1], :] for kp in keypoints]

        
        if len(descriptors) == 0:
            return None
        
        descriptors = np.array(descriptors, dtype=np.float32)
        return descriptors
    
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
    def __init__(self, finder: KeypointFinder = None, n_features: int = 0, n_octave_layers: int = 3, contrast_threshold: float = 0.04, edge_threshold: float = 10, sigma: float = 1.6):
        self.finder = finder
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
    
    def detect(self, image: np.ndarray, mask: np.ndarray | None = None) -> list[cv2.KeyPoint]:
        if len(image.shape) == 3:
            gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            gray = (image * 255).astype(np.uint8)
        keypoints = self.finder.detect(gray, mask)
        return keypoints    
    
    def compute(self, image: np.ndarray, keypoints: list[cv2.KeyPoint], mask: np.ndarray | None = None) -> np.ndarray:
        if len(image.shape) == 3:
            gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            gray = (image * 255).astype(np.uint8)
        _, descriptors = self.sift.compute(gray, keypoints)
        return descriptors
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "SIFT",
            "n_features": self.n_features,
            "n_octave_layers": self.n_octave_layers,
            "contrast_threshold": self.contrast_threshold,
            "edge_threshold": self.edge_threshold,
            "sigma": self.sigma
        }
class BRISKDescriptor(KeypointDescriptorMaker):
    def __init__(self, finder: KeypointFinder = None, thresh: int = 30, octaves: int = 3, pattern_scale: float = 1.0):
        self.finder = finder
        self.thresh = thresh
        self.octaves = octaves
        self.pattern_scale = pattern_scale
        self.brisk = cv2.BRISK_create(thresh=thresh, octaves=octaves, patternScale=pattern_scale)
        
    def detect_and_compute(self, image: np.ndarray, mask: np.ndarray | None = None) -> tuple[list, np.ndarray]:
        if len(image.shape) == 3:
            gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            gray = (image * 255).astype(np.uint8)
        
        keypoints, descriptors = self.brisk.detectAndCompute(gray, mask)
        return keypoints, descriptors
    
    def detect(self, image: np.ndarray, mask: np.ndarray | None = None) -> list[cv2.KeyPoint]:
        if len(image.shape) == 3:
            gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            gray = (image * 255).astype(np.uint8)
        keypoints = self.finder.detect(gray, mask)
        return keypoints  
    
    def compute(self, image: np.ndarray, keypoints: list[cv2.KeyPoint]) -> np.ndarray:
        if len(image.shape) == 3:
            gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            gray = (image * 255).astype(np.uint8)
        _, descriptors = self.brisk.compute(gray, keypoints)
        return descriptors
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "BRISK",
            "thresh": self.thresh,
            "octaves": self.octaves,
            "pattern_scale": self.pattern_scale
        }
    
    
class AKAZEDescriptor(KeypointDescriptorMaker):
    def __init__(self, finder: KeypointFinder = None, descriptor_type: int = cv2.AKAZE_DESCRIPTOR_MLDB, 
                 descriptor_size: int = 0, 
                 descriptor_channels: int = 3, 
                 threshold: float = 0.001, 
                 n_octaves: int = 4, 
                 n_octave_layers: int = 4, 
                 diffusivity: int = cv2.KAZE_DIFF_PM_G2):
        
        self.descriptor_type = descriptor_type
        self.descriptor_size = descriptor_size
        self.descriptor_channels = descriptor_channels
        self.threshold = threshold
        self.n_octaves = n_octaves
        self.n_octave_layers = n_octave_layers
        self.diffusivity = diffusivity
        self.akaze = cv2.AKAZE_create(
            descriptor_type=descriptor_type,
            descriptor_size=descriptor_size,
            descriptor_channels=descriptor_channels,
            threshold=threshold,
            nOctaves=n_octaves,
            nOctaveLayers=n_octave_layers,
            diffusivity=diffusivity
        )
        
    def detect_and_compute(self, image: np.ndarray, mask: np.ndarray | None = None) -> tuple[list, np.ndarray]:
        if len(image.shape) == 3:
            gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            gray = (image * 255).astype(np.uint8)
        
        keypoints, descriptors = self.akaze.detectAndCompute(gray, mask)
        return keypoints, descriptors

    def detect(self, image: np.ndarray, mask: np.ndarray | None = None) -> list[cv2.KeyPoint]:
        if len(image.shape) == 3:
            gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            gray = (image * 255).astype(np.uint8)
        keypoints = self.finder.detect(gray, mask)
        return keypoints  
    
    def compute(self, image: np.ndarray, keypoints: list[cv2.KeyPoint]) -> tuple[list, np.ndarray]:
        if len(image.shape) == 3:
            gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            gray = (image * 255).astype(np.uint8)
        [setattr(kp, 'class_id', 0) for kp in keypoints] # in akaze compute func the class_id of the keypoints can't be -1(the default value). It needs to be changed to a different value than -1.
        _, descriptors = self.akaze.compute(gray, keypoints)
        return descriptors

    
    
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "AKAZE",
            "descriptor_type": self.descriptor_type,
            "descriptor_size": self.descriptor_size,
            "descriptor_channels": self.descriptor_channels,
            "threshold": self.threshold,
            "n_octaves": self.n_octaves,
            "n_octave_layers": self.n_octave_layers,
            "diffusivity": self.diffusivity
        }

class SURFDescriptor(KeypointDescriptorMaker):
    """
    Note: SURF is part of the opencv-contrib-python package. 
    Ensure you have it installed for this class to work.
    """
    def __init__(self, finder: KeypointFinder = None, hessian_threshold: float = 100, n_octaves: int = 4, 
                n_octave_layers: int = 3, extended: bool = False, upright: bool = False):
        self.finder = finder
        self.hessian_threshold = hessian_threshold
        self.n_octaves = n_octaves
        self.n_octave_layers = n_octave_layers
        self.extended = extended
        self.upright = upright
        self.surf = xfeatures2d.SURF_create(
            hessianThreshold=hessian_threshold,
            nOctaves=n_octaves,
            nOctaveLayers=n_octave_layers,
            extended=extended,
            upright=upright
        )

    def detect_and_compute(self, image: np.ndarray, mask: np.ndarray | None = None) -> tuple[list, np.ndarray]:
        if len(image.shape) == 3:
            gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            gray = (image * 255).astype(np.uint8)
            
        keypoints, descriptors = self.surf.detectAndCompute(gray, mask)
        return keypoints, descriptors

    def detect(self, image: np.ndarray, mask: np.ndarray | None = None) -> list[cv2.KeyPoint]:
        if len(image.shape) == 3:
            gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            gray = (image * 255).astype(np.uint8)
        keypoints = self.finder.detect(gray, mask)
        return keypoints  
    
    def compute(self, image: np.ndarray, keypoints: list[cv2.KeyPoint]) -> tuple[list, np.ndarray]:
        if len(image.shape) == 3:
            gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            gray = (image * 255).astype(np.uint8)
        
        _, descriptors = self.surf.compute(gray, keypoints)
        return descriptors


    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "SURF",
            "hessian_threshold": self.hessian_threshold,
            "n_octaves": self.n_octaves,
            "n_octave_layers": self.n_octave_layers,
            "extended": self.extended,
            "upright": self.upright
        }
        

class PCASIFTDescriptor(KeypointDescriptorMaker):
    def __init__(self, 
                finder: KeypointFinder = None,
                num_components: int = 128, # A more common value than 128
                n_features: int = 0, 
                n_octave_layers: int = 3, 
                contrast_threshold: float = 0.04, 
                edge_threshold: float = 10, 
                sigma: float = 1.6):
        self.finder = finder
        
        # SIFT parameters
        self.n_features = n_features
        self.n_octave_layers = n_octave_layers
        self.contrast_threshold = contrast_threshold
        self.edge_threshold = edge_threshold
        self.sigma = sigma
        
        # PCA parameters
        self.num_components = num_components
        
        # Create SIFT object
        # Handle different OpenCV versions (contrib vs. main)
        try:
            self.sift = cv2.SIFT_create(nfeatures=n_features, 
                                        nOctaveLayers=n_octave_layers, 
                                        contrastThreshold=contrast_threshold, 
                                        edgeThreshold=edge_threshold, 
                                        sigma=sigma)
        except AttributeError:
            self.sift = cv2.xfeatures2d.SIFT_create(nfeatures=n_features, 
                                                   nOctaveLayers=n_octave_layers, 
                                                   contrastThreshold=contrast_threshold, 
                                                   edgeThreshold=edge_threshold, 
                                                   sigma=sigma)
        

        self.pca = PCA(n_components=self.num_components)
        
    def detect_and_compute(self, image: np.ndarray, mask: np.ndarray | None = None) -> tuple[list, np.ndarray]:
        
        if len(image.shape) == 3:
            gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            gray = (image * 255).astype(np.uint8)
        
        # 1. Detect keypoints and compute SIFT descriptors
        keypoints, descriptors = self.sift.detectAndCompute(gray, mask)
        
        if descriptors is None or len(descriptors) == 0:
            return keypoints, np.array([])

        try:
            pca_descriptors = self.pca.fit_transform(descriptors)
        except ValueError as e:
            # This can happen if n_features < num_components
            print(f"PCA Error: {e}. Returning empty descriptors.")
            print("This often happens if the number of detected keypoints "
                  f"({len(descriptors)}) is less than num_components ({self.num_components}).")
            return keypoints, np.array([])


        # 3. Normalize the PCA-SIFT descriptors
        normalized_pca_descriptors = cv2.normalize(pca_descriptors, None)

        # 4. Compute RootSIFT descriptors
        #    We clip at 0 to avoid errors with tiny negative numbers
        normalized_pca_descriptors = np.maximum(normalized_pca_descriptors, 0)
        root_sift_descriptors = np.sqrt(normalized_pca_descriptors)

        # 5. Perform L2 normalization on RootSIFT descriptors
        #    Handle potential divide-by-zero if a descriptor vector is all zeros
        norms = np.linalg.norm(root_sift_descriptors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0 # Avoid division by zero
        final_descriptors = root_sift_descriptors / norms
        
        return keypoints, final_descriptors
    
    def detect(self, image: np.ndarray, mask: np.ndarray | None = None) -> list[cv2.KeyPoint]:
        if len(image.shape) == 3:
            gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            gray = (image * 255).astype(np.uint8)
        keypoints = self.finder.detect(gray, mask)
        return keypoints  
    
    def compute(self, image: np.ndarray, keypoints: list[cv2.KeyPoint]) -> tuple[list, np.ndarray]:
        
        if len(image.shape) == 3:
            gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            gray = (image * 255).astype(np.uint8)
        
        # 1. Detect keypoints and compute SIFT descriptors
        keypoints, descriptors = self.sift.compute(gray, keypoints)
        
        if descriptors is None or len(descriptors) == 0:
            return keypoints

        try:
            pca_descriptors = self.pca.fit_transform(descriptors)
        except ValueError as e:
            # This can happen if n_features < num_components
            print(f"PCA Error: {e}. Returning empty descriptors.")
            print("This often happens if the number of detected keypoints "
                  f"({len(descriptors)}) is less than num_components ({self.num_components}).")
            return keypoints


        # 3. Normalize the PCA-SIFT descriptors
        normalized_pca_descriptors = cv2.normalize(pca_descriptors, None)

        # 4. Compute RootSIFT descriptors
        #    We clip at 0 to avoid errors with tiny negative numbers
        normalized_pca_descriptors = np.maximum(normalized_pca_descriptors, 0)
        root_sift_descriptors = np.sqrt(normalized_pca_descriptors)

        # 5. Perform L2 normalization on RootSIFT descriptors
        #    Handle potential divide-by-zero if a descriptor vector is all zeros
        norms = np.linalg.norm(root_sift_descriptors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0 # Avoid division by zero
        final_descriptors = root_sift_descriptors / norms
        
        return keypoints, final_descriptors
    
    
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "PCA-SIFT",
            "n_features": self.n_features,
            "n_octave_layers": self.n_octave_layers,
            "contrast_threshold": self.contrast_threshold,
            "edge_threshold": self.edge_threshold,
            "sigma": self.sigma,
            "num_components": self.num_components
        }
    

class HOGDescriptor(KeypointDescriptorMaker):
    """
    NOTE: HOG is a descriptor, not a keypoint detector. This class
    uses a SIFT detector to find keypoints, and then computes HOG
    descriptors at those keypoint locations.
    """
    def __init__(self,
                # HOG descriptor parameters
                finder: KeypointFinder = None,
                win_size: tuple[int, int] = (32, 32),
                block_size: tuple[int, int] = (16, 16),
                block_stride: tuple[int, int] = (8, 8),
                cell_size: tuple[int, int] = (8, 8),
                nbins: int = 9):

        self.finder = finder
        # Store HOG parameters
        self.win_size = win_size
        self.block_size = block_size
        self.block_stride = block_stride
        self.cell_size = cell_size
        self.nbins = nbins


        # 1. Initialize the HOG Descriptor object
        self.hog = cv2.HOGDescriptor(win_size, 
                                    block_size, 
                                    block_stride, 
                                    cell_size, 
                                    nbins)
            

    def detect_and_compute(self, image: np.ndarray, mask: np.ndarray | None = None) -> tuple[list[cv2.KeyPoint], np.ndarray]:
        """
        Detects keypoints using SIFT, then computes HOG descriptors
        at those keypoint locations.
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            gray = (image * 255).astype(np.uint8)
        
        locations = self.hog.detect(gray)
                
        locations = [(round(kp.pt[0]), round(kp.pt[1])) for kp in locations]
        

        descriptors = self.hog.compute(gray, locations=locations)
        
        n_cells = (image.shape[0] // self.cell_size[0], image.shape[1] // self.cell_size[1])
        
        descriptors = descriptors.reshape(
            n_cells[1] - self.win_size[1] + 1,
            n_cells[0] - self.win_size[0] + 1,
            self.win_size[1] - self.block_size[1] + 1,
            self.win_size[0] - self.block_size[0] + 1,
            self.block_size[1],
            self.block_size[0],
            self.nbins)
        return locations, descriptors
    
    def detect(self, image: np.ndarray, mask: np.ndarray | None = None) -> list[cv2.KeyPoint]:
        if len(image.shape) == 3:
            gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            gray = (image * 255).astype(np.uint8)
        
        keypoints = self.finder.detect(gray)
        return keypoints
    
    def compute(self, image: np.ndarray, keypoints: list[cv2.KeyPoint]) -> np.ndarray:
        if len(image.shape) == 3:
            gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            gray = (image * 255).astype(np.uint8)
            
        locations = [(round(kp.pt[0]), round(kp.pt[1])) for kp in keypoints]

        descriptors_flat = self.hog.compute(gray, locations=locations)
        # n_cells = (image.shape[0] // self.cell_size[0], image.shape[1] // self.cell_size[1])
        
        # descriptors = descriptors.reshape(
        #     n_cells[1] - self.win_size[1] + 1,
        #     n_cells[0] - self.win_size[0] + 1,
        #     self.win_size[1] - self.block_size[1] + 1,
        #     self.win_size[0] - self.block_size[0] + 1,
        #     self.block_size[1],
        #     self.block_size[0],
        #     self.nbins)
        # Number of keypoints
        N = len(locations)

        # The dimensions for the *internal structure* of one descriptor
        # (These are the last 5 dimensions from your example)
        n_blocks_y = self.win_size[1] - self.block_size[1] + 1
        n_blocks_x = self.win_size[0] - self.block_size[0] + 1
        n_cells_y = self.block_size[1]
        n_cells_x = self.block_size[0]
        
        if descriptors_flat is not None:
            # El reshape este ha salido de un post de StackOverflow, no de la docu oficial de opencv xddddd
            # https://stackoverflow.com/questions/22373707/why-does-opencvs-hog-descriptor-return-so-many-values
            
            descriptors = descriptors_flat.reshape(N,self.hog.getDescriptorSize()
            )
        return descriptors
    
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "HOG",
            # HOG params
            "win_size": self.win_size,
            "block_size": self.block_size,
            "block_stride": self.block_stride,
            "cell_size": self.cell_size,
            "nbins": self.nbins,
            
            # SIFT detector params
            # "detector_type": "SIFT",
            # "n_features": self.n_features,
            # "n_octave_layers": self.n_octave_layers,
            # "contrast_threshold": self.contrast_threshold,
            # "edge_threshold": self.edge_threshold,
            # "sigma": self.sigma
        }

class GLOHDescriptor(KeypointDescriptorMaker):
    """
    GLOH Descriptor based on Medium article:
    "Exploring Gradient Location and Orientation Histogram (GLOH) for Image Recognition and Object Detection"
    by Vincent Chung
    (medium.com/@vincentchung_72457/exploring-gradient-location-orientation-histogram-gloh-for-image-recognition-and-object-detection-3e3c231a5b01)
    """
    def __init__(self,
                finder: KeypointFinder = None,
                nbins: int = 36,
                
                # SIFT detector parameters
                n_features: int = 0,
                n_octave_layers: int = 3,
                contrast_threshold: float = 0.04,
                edge_threshold: float = 10,
                sigma: float = 1.6):

        self.finder = finder

        # Store parameters
        self.nbins = nbins
        
        # Store SIFT detector parameters
        self.n_features = n_features
        self.n_octave_layers = n_octave_layers
        self.contrast_threshold = contrast_threshold
        self.edge_threshold = edge_threshold
        self.sigma = sigma
        
    def detect_and_compute(self, image: np.ndarray, mask: np.ndarray | None = None) -> tuple[list[cv2.KeyPoint], np.ndarray]:
        # BASED ON: medium.com/@vincentchung_72457/exploring-gradient-location-orientation-histogram-gloh-for-image-recognition-and-object-detection-3e3c231a5b01
        if len(image.shape) == 3:
            gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            gray = (image * 255).astype(np.uint8)
        
        
        # Compute gradient magnitude and orientation using Sobel operators
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        
        # Source: "mag, angle = cv2.cartToPolar(grad_x, grad_y, angleInDegrees=True)"
        _mag, angle = cv2.cartToPolar(grad_x, grad_y, angleInDegrees=True)

        # Compute keypoints using SIFT
        keypoints = self.finder.detect(gray, mask)

        if not keypoints:
            return [], np.array([])
            
        # Compute GLOH features for each keypoint
        gloh_features = []
        valid_keypoints = []
        h, w = gray.shape

        for kp in keypoints:
            x, y = int(kp.pt[0]), int(kp.pt[1])
            scale = int(kp.size / 2)
            
            # clip values at borders at image borders
            y_min = max(0, y - scale)
            y_max = min(h, y + scale)
            x_min = max(0, x - scale)
            x_max = min(w, x + scale)
            
            # Skip keypoints where the patch is empty
            if y_min >= y_max or x_min >= x_max:
                continue

            # Extract the patch of gradient angles
            angle_patch = angle[y_min:y_max, x_min:x_max]
            
            histogram = cv2.calcHist(
                [angle_patch], 
                channels=[0], 
                mask=None, 
                histSize=[self.nbins], 
                ranges=[0, 360]
            )
            
            # Add the computed histogram (descriptor)
            gloh_features.append(histogram)
            # Keep the keypoint that this descriptor belongs to
            valid_keypoints.append(kp)

        if not gloh_features:
            return [], np.array([])

        # Concatenate the GLOH features into a single feature vector
        descriptors = np.array(gloh_features).reshape(len(gloh_features), self.nbins)
        
        descriptors = cv2.normalize(descriptors, None, norm_type=cv2.NORM_L2)
        
        return valid_keypoints, descriptors
    
    def detect(self, image: np.ndarray, mask: np.ndarray | None = None) -> list[cv2.KeyPoint]:
        if len(image.shape) == 3:
            gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            gray = (image * 255).astype(np.uint8)
        
        keypoints = self.finder.detect(gray, mask)
        return keypoints
    
    def compute(self, image: np.ndarray, keypoints: list[cv2.KeyPoint]) -> np.ndarray:
        # BASED ON: medium.com/@vincentchung_72457/exploring-gradient-location-orientation-histogram-gloh-for-image-recognition-and-object-detection-3e3c231a5b01
        if len(image.shape) == 3:
            gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            gray = (image * 255).astype(np.uint8)
        
        
        # Compute gradient magnitude and orientation using Sobel operators
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        
        # Source: "mag, angle = cv2.cartToPolar(grad_x, grad_y, angleInDegrees=True)"
        _mag, angle = cv2.cartToPolar(grad_x, grad_y, angleInDegrees=True)
            
        # Compute GLOH features for each keypoint
        gloh_features = []
        valid_keypoints = []
        h, w = gray.shape

        for kp in keypoints:
            x, y = int(kp.pt[0]), int(kp.pt[1])
            scale = int(kp.size / 2)
            
            # clip values at borders at image borders
            y_min = max(0, y - scale)
            y_max = min(h, y + scale)
            x_min = max(0, x - scale)
            x_max = min(w, x + scale)
            
            # Skip keypoints where the patch is empty
            if y_min >= y_max or x_min >= x_max:
                continue

            # Extract the patch of gradient angles
            angle_patch = angle[y_min:y_max, x_min:x_max]
            
            histogram = cv2.calcHist(
                [angle_patch], 
                channels=[0], 
                mask=None, 
                histSize=[self.nbins], 
                ranges=[0, 360]
            )
            
            # Add the computed histogram (descriptor)
            gloh_features.append(histogram)
            # Keep the keypoint that this descriptor belongs to
            valid_keypoints.append(kp)

        if not gloh_features:
            return [], np.array([])

        # Concatenate the GLOH features into a single feature vector
        descriptors = np.array(gloh_features).reshape(len(gloh_features), self.nbins)
        
        # descriptors = cv2.normalize(descriptors, None, norm_type=cv2.NORM_L2)
        
        return descriptors
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "GLOH",
            "nbins": self.nbins,
            
            # SIFT detector params
            "detector_type": "SIFT",
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
    from pathlib import Path
    import traceback

    out_dir = Path("kp_results")
    out_dir.mkdir(exist_ok=True)

    # Try to load a real image; fallback to a generated dummy if it fails
    img_path = Path.home() / "MCV" / "C1" / "proyect" / "Team4" / "qsd1_w4" / "00002.jpg"
    if img_path.exists():
        img_bgr = cv2.imread(str(img_path))
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        print(f"Loaded image {img_path} with shape {img.shape}")
    else:
        print(f"Image {img_path} not found — using generated dummy image.")
        dummy = np.zeros((480, 640), dtype=np.float32)
        cv2.circle(dummy, (100, 100), 30, 0.8, -1)
        cv2.circle(dummy, (300, 250), 50, 0.6, -1)
        cv2.circle(dummy, (500, 400), 20, 0.9, -1)
        # convert to 3-channel RGB float in [0,1] to match expected input
        img = np.stack([dummy, dummy, dummy], axis=-1).astype(np.float32)

    # instantiate finders
    finders = [
        HarrisFinder(thresh=0.01),
        HarrisLaplacianFinder(scales=[1.0, 1.6, 2.0, 2.8], harris_thresh=0.01),
        SIFTFinder(n_features=500)
    ]

    # instantiate descriptors (do not attach a finder at construction; we'll set descriptor.finder = finder below)
    descriptors = []
    # core descriptors — some may raise if OpenCV lacks them; we catch exceptions later
    try:
        descriptors.append(ORBDescriptor())
    except Exception as e:
        print("ORBDescriptor unavailable:", e)
    try:
        descriptors.append(DaisyDescriptor())
    except Exception as e:
        print("DaisyDescriptor unavailable:", e)
    try:
        descriptors.append(SIFTDescriptor())
    except Exception as e:
        print("SIFTDescriptor unavailable:", e)
    try:
        descriptors.append(BRISKDescriptor())
    except Exception as e:
        print("BRISKDescriptor unavailable:", e)
    try:
        descriptors.append(AKAZEDescriptor())
    except Exception as e:
        print("AKAZEDescriptor unavailable:", e)
    try:
        descriptors.append(PCASIFTDescriptor(num_components=24, n_features=500))
    except Exception as e:
        print("PCASIFTDescriptor unavailable:", e)
    try:
        descriptors.append(HOGDescriptor(SIFTFinder(n_features=500)))  # HOG uses SIFT detector internally in your impl
    except Exception as e:
        print("HOGDescriptor unavailable:", e)
    try:
        descriptors.append(GLOHDescriptor(finder=SIFTFinder(n_features=500), nbins=36, n_features=500))
    except Exception as e:
        print("GLOHDescriptor unavailable:", e)
    # SURF often requires xfeatures; skip if not available
    try:
        descriptors.append(SURFDescriptor())
    except Exception as e:
        print("SURFDescriptor unavailable (likely missing xfeatures):", e)

    # Utility to draw and save keypoints
    def save_kp_image(rgb_image: np.ndarray, keypoints: list, outpath: Path, title: str = ""):
        # drawKeypoints expects uint8 BGR by default; convert
        vis = (rgb_image * 255.0).astype(np.uint8).copy()
        vis_bgr = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
        try:
            drawn = cv2.drawKeypoints(vis_bgr, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            cv2.imwrite(str(outpath), drawn)
        except Exception:
            # fallback: draw small circles at keypoint positions
            vis_rgb = rgb_image.copy()
            for kp in keypoints:
                x, y = int(round(kp.pt[0])), int(round(kp.pt[1]))
                cv2.circle(vis_bgr, (x, y), 3, (0, 0, 255), -1)
            cv2.imwrite(str(outpath), vis_bgr)

    # run through combinations
    summary = []
    for finder in finders:
        finder_name = finder.to_dict().get("type", finder.__class__.__name__)
        for desc in descriptors:
            desc_name = desc.to_dict().get("type", desc.__class__.__name__)
            combo_name = f"{finder_name}__{desc_name}"
            print(f"\n--- Running combo: {combo_name} ---")
            # attach finder to descriptor if supported
            try:
                if hasattr(desc, "finder"):
                    desc.finder = finder
            except Exception:
                pass

            try:
                # Step 1: detect keypoints using the finder
                kps = finder.detect(img, mask=None)
                n_kps = len(kps) if kps is not None else 0
                print(f"Finder detected {n_kps} keypoints.")

                # Step 2: compute descriptors
                descriptors_out = None
                # prefer compute(keypoints) if available
                if hasattr(desc, "compute"):
                    try:
                        descriptors_out = desc.compute(img, kps)
                    except TypeError:
                        # some compute signatures return (kps, desc) or require additional args
                        try:
                            descriptors_out = desc.compute(img, kps, None)
                        except Exception as e:
                            print("descriptor.compute() failed (TypeError branch):", e)
                            descriptors_out = None
                # fallback to detect_and_compute if compute not available or returned None
                if descriptors_out is None and hasattr(desc, "detect_and_compute"):
                    try:
                        kps2, descriptors_out = desc.detect_and_compute(img, mask=None)
                        # some detect_and_compute return keypoints as first value
                        if kps2 is not None and isinstance(kps2, list) and len(kps2) > 0 and isinstance(kps2[0], cv2.KeyPoint):
                            kps = kps2
                    except Exception as e:
                        print("descriptor.detect_and_compute() failed:", e)
                        descriptors_out = None

                # normalize descriptor output shape reporting
                if descriptors_out is None:
                    print(f"{combo_name}: No descriptors returned (None).")
                    summary.append((combo_name, n_kps, None, "no_desc"))
                else:
                    # descriptors could be a nested structure (HOG) — try to present shape
                    try:
                        dshape = np.asarray(descriptors_out).shape
                    except Exception:
                        dshape = ("unknown",)
                    print(f"{combo_name}: descriptors shape = {dshape}")
                    summary.append((combo_name, n_kps, dshape, "ok"))

                # save visualization of keypoints
                outpath = out_dir / f"{combo_name}.png"
                save_kp_image(img, kps if kps is not None else [], outpath)
                print(f"Saved keypoint image to {outpath}")

            except Exception as e:
                print(f"Combo {combo_name} FAILED with exception: {e}")
                traceback.print_exc()
                summary.append((combo_name, 0, None, f"error: {e}"))

    # Final summary print
    print("\n=== SUMMARY ===")
    for combo_name, n_kps, dshape, status in summary:
        print(f"{combo_name}: keypoints={n_kps}, descriptors_shape={dshape}, status={status}")

    print(f"\nAll images and visualizations saved under: {out_dir.resolve()}")
     