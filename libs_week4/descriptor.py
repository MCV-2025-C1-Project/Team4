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
from sklearn.decomposition import PCA
from cv2 import xfeatures2d

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
    if color_space == ColorSpace.BGR:
        return [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]
    
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



class DescriptorComputer(abc.ABC):
    #keypoint descriptor class
    
    @abc.abstractmethod
    def detect_and_compute(self, image: np.ndarray, mask: np.ndarray | None = None) -> tuple[list, np.ndarray]:
        
        #Detect keypoints and compute descriptors.
        #Returns: keypoints: List of cv2.KeyPoint objects, descriptors: Array of shape (n_keypoints, descriptor_dim)
        pass
    
    @abc.abstractmethod
    def to_dict(self) -> dict[str, Any]:
        pass


class ORBDescriptor(DescriptorComputer):
    def __init__(self, n_features: int = 500, scale_factor: float = 1.2, n_levels: int = 8,
                 wta_k: int = 2, score_type: int = cv2.ORB_HARRIS_SCORE, patch_size: int = 31):
        self.n_features = n_features
        self.scale_factor = scale_factor
        self.n_levels = n_levels
        self.wta_k = wta_k
        self.score_type = score_type
        self.patch_size = patch_size
        self.orb = cv2.ORB_create(
            nfeatures=n_features,
            scaleFactor=scale_factor,
            nlevels=n_levels,
            edgeThreshold=patch_size,  # Should be equal to patchSize
            firstLevel=0,
            WTA_K=wta_k,
            scoreType=score_type,
            patchSize=patch_size
        )
        
    def detect_and_compute(self, image: np.ndarray, mask: np.ndarray | None = None) -> tuple[list, np.ndarray]:
        if len(image.shape) == 3:
            gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
        else:
            gray = (image * 255).astype(np.uint8)
        
        keypoints, descriptors = self.orb.detectAndCompute(gray, mask)
        return keypoints, descriptors
    
    def to_dict(self) -> dict[str, Any]:
        score_type_name = "HARRIS" if self.score_type == cv2.ORB_HARRIS_SCORE else "FAST"
        return {
            "type": "ORB",
            "n_features": self.n_features,
            "scale_factor": self.scale_factor,
            "n_levels": self.n_levels,
            "wta_k": self.wta_k,
            "score_type": score_type_name,
            "patch_size": self.patch_size
        }

class DaisyDescriptor(DescriptorComputer):
    def __init__(self, step: int = 4, radius: int = 15, rings: int = 3, histograms: int = 8, orientations: int = 8):
        self.step = step
        self.radius = radius
        self.rings = rings
        self.histograms = histograms
        self.orientations = orientations
        
    def detect_and_compute(self, image: np.ndarray, mask: np.ndarray | None = None) -> tuple[list, np.ndarray]:
        if len(image.shape) == 3:
            gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
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
        
class SIFTDescriptor(DescriptorComputer):
    def __init__(self, n_features: int = 0, n_octave_layers: int = 3, contrast_threshold: float = 0.04, edge_threshold: float = 10, sigma: float = 1.6):
        self.n_features = n_features
        self.n_octave_layers = n_octave_layers
        self.contrast_threshold = contrast_threshold
        self.edge_threshold = edge_threshold
        self.sigma = sigma
        self.sift = cv2.SIFT_create(nfeatures=n_features, nOctaveLayers=n_octave_layers, contrastThreshold=contrast_threshold, edgeThreshold=edge_threshold, sigma=sigma)
        
    def detect_and_compute(self, image: np.ndarray, mask: np.ndarray | None = None) -> tuple[list, np.ndarray]:
        if len(image.shape) == 3:
            gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
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

class RootSIFTDescriptor(DescriptorComputer):
    def __init__(self, n_features: int = 0, n_octave_layers: int = 3, contrast_threshold: float = 0.04, edge_threshold: float = 10, sigma: float = 1.6):
        self.n_features = n_features
        self.n_octave_layers = n_octave_layers
        self.contrast_threshold = contrast_threshold
        self.edge_threshold = edge_threshold
        self.sigma = sigma
        self.sift = cv2.SIFT_create(nfeatures=n_features, nOctaveLayers=n_octave_layers, contrastThreshold=contrast_threshold, edgeThreshold=edge_threshold, sigma=sigma)

    def detect_and_compute(self, image: np.ndarray, mask: np.ndarray | None = None) -> tuple[list, np.ndarray]:
        if len(image.shape) == 3:
            gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
        else:
            gray = (image * 255).astype(np.uint8)

        keypoints, descriptors = self.sift.detectAndCompute(gray, mask)

        if descriptors is None or len(descriptors) == 0:
            return keypoints, descriptors

        # RootSIFT: Apply square root after L1 normalization
        # L1 normalize
        descriptors = descriptors / (descriptors.sum(axis=1, keepdims=True) + 1e-7)
        # Element-wise square root
        descriptors = np.sqrt(descriptors)

        return keypoints, descriptors

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "RootSIFT",
            "n_features": self.n_features,
            "n_octave_layers": self.n_octave_layers,
            "contrast_threshold": self.contrast_threshold,
            "edge_threshold": self.edge_threshold,
            "sigma": self.sigma
        }

class BRISKDescriptor(DescriptorComputer):
    def __init__(self, thresh: int = 30, octaves: int = 3, pattern_scale: float = 1.0):
        self.thresh = thresh
        self.octaves = octaves
        self.pattern_scale = pattern_scale
        self.brisk = cv2.BRISK_create(thresh=thresh, octaves=octaves, patternScale=pattern_scale)
        
    def detect_and_compute(self, image: np.ndarray, mask: np.ndarray | None = None) -> tuple[list, np.ndarray]:
        if len(image.shape) == 3:
            gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
        else:
            gray = (image * 255).astype(np.uint8)
        
        keypoints, descriptors = self.brisk.detectAndCompute(gray, mask)
        return keypoints, descriptors
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "BRISK",
            "thresh": self.thresh,
            "octaves": self.octaves,
            "pattern_scale": self.pattern_scale
        }
    
    
class KAZEDescriptor(DescriptorComputer):
    def __init__(self, extended: bool = False, upright: bool = False,
                 threshold: float = 0.001, n_octaves: int = 4,
                 n_octave_layers: int = 4, diffusivity: int = cv2.KAZE_DIFF_PM_G2):

        self.extended = extended
        self.upright = upright
        self.threshold = threshold
        self.n_octaves = n_octaves
        self.n_octave_layers = n_octave_layers
        self.diffusivity = diffusivity
        self.kaze = cv2.KAZE_create(
            extended=extended,
            upright=upright,
            threshold=threshold,
            nOctaves=n_octaves,
            nOctaveLayers=n_octave_layers,
            diffusivity=diffusivity
        )

    def detect_and_compute(self, image: np.ndarray, mask: np.ndarray | None = None) -> tuple[list, np.ndarray]:
        if len(image.shape) == 3:
            gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
        else:
            gray = (image * 255).astype(np.uint8)

        keypoints, descriptors = self.kaze.detectAndCompute(gray, mask)
        return keypoints, descriptors

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "KAZE",
            "extended": self.extended,
            "upright": self.upright,
            "threshold": self.threshold,
            "n_octaves": self.n_octaves,
            "n_octave_layers": self.n_octave_layers,
            "diffusivity": self.diffusivity
        }

class AKAZEDescriptor(DescriptorComputer):
    def __init__(self, descriptor_type: int = cv2.AKAZE_DESCRIPTOR_MLDB, 
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
            gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
        else:
            gray = (image * 255).astype(np.uint8)
        
        keypoints, descriptors = self.akaze.detectAndCompute(gray, mask)
        return keypoints, descriptors
    
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

class SURFDescriptor(DescriptorComputer):
    """
    Note: SURF is part of the opencv-contrib-python package. 
    Ensure you have it installed for this class to work.
    """
    def __init__(self, hessian_threshold: float = 100, n_octaves: int = 4, 
                 n_octave_layers: int = 3, extended: bool = False, upright: bool = False):
        self.hessian_threshold = hessian_threshold
        self.n_octaves = n_octaves
        self.n_octave_layers = n_octave_layers
        self.extended = extended
        self.upright = upright
        self.surf = cv2.xfeatures2d.SURF_create(
            hessianThreshold=hessian_threshold,
            nOctaves=n_octaves,
            nOctaveLayers=n_octave_layers,
            extended=extended,
            upright=upright
        )

    def detect_and_compute(self, image: np.ndarray, mask: np.ndarray | None = None) -> tuple[list, np.ndarray]:
        if len(image.shape) == 3:
            gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
        else:
            gray = (image * 255).astype(np.uint8)
            
        keypoints, descriptors = self.surf.detectAndCompute(gray, mask)
        return keypoints, descriptors

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "SURF",
            "hessian_threshold": self.hessian_threshold,
            "n_octaves": self.n_octaves,
            "n_octave_layers": self.n_octave_layers,
            "extended": self.extended,
            "upright": self.upright
        }
        

class PCASIFTDescriptor(DescriptorComputer):
    def __init__(self, 
                 num_components: int = 128, # A more common value than 128
                 n_features: int = 0, 
                 n_octave_layers: int = 3, 
                 contrast_threshold: float = 0.04, 
                 edge_threshold: float = 10, 
                 sigma: float = 1.6):
        
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
            gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
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
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "PCA-SIFT-Root (Flawed)",
            "n_features": self.n_features,
            "n_octave_layers": self.n_octave_layers,
            "contrast_threshold": self.contrast_threshold,
            "edge_threshold": self.edge_threshold,
            "sigma": self.sigma,
            "num_components": self.num_components
        }
    

class HOGDescriptor(DescriptorComputer):
    """
    NOTE: HOG is a descriptor, not a keypoint detector. This class
    uses a SIFT detector to find keypoints, and then computes HOG
    descriptors at those keypoint locations.
    """
    def __init__(self,
                 # HOG descriptor parameters
                 win_size: tuple[int, int] = (32, 32),
                 block_size: tuple[int, int] = (16, 16),
                 block_stride: tuple[int, int] = (8, 8),
                 cell_size: tuple[int, int] = (8, 8),
                 nbins: int = 9,
                 
                 # SIFT detector parameters (for finding keypoints)
                 n_features: int = 0,
                 n_octave_layers: int = 3,
                 contrast_threshold: float = 0.04,
                 edge_threshold: float = 10,
                 sigma: float = 1.6):

        # Store HOG parameters
        self.win_size = win_size
        self.block_size = block_size
        self.block_stride = block_stride
        self.cell_size = cell_size
        self.nbins = nbins
        
        # Store SIFT detector parameters
        self.n_features = n_features
        self.n_octave_layers = n_octave_layers
        self.contrast_threshold = contrast_threshold
        self.edge_threshold = edge_threshold
        self.sigma = sigma

        # 1. Initialize the HOG Descriptor object
        self.hog = cv2.HOGDescriptor(win_size, 
                                    block_size, 
                                    block_stride, 
                                    cell_size, 
                                    nbins)
        
        # 2. Initialize the SIFT Detector object
        try:
            self.sift_detector = cv2.SIFT_create(
                nfeatures=n_features, 
                nOctaveLayers=n_octave_layers, 
                contrastThreshold=contrast_threshold, 
                edgeThreshold=edge_threshold, 
                sigma=sigma
            )
        except AttributeError:
            self.sift_detector = cv2.xfeatures2d.SIFT_create(
                nfeatures=n_features, 
                nOctaveLayers=n_octave_layers, 
                contrastThreshold=contrast_threshold, 
                edgeThreshold=edge_threshold, 
                sigma=sigma
            )

    def detect_and_compute(self, image: np.ndarray, mask: np.ndarray | None = None) -> tuple[list[cv2.KeyPoint], np.ndarray]:
        """
        Detects keypoints using SIFT, then computes HOG descriptors
        at those keypoint locations.
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
        else:
            gray = (image * 255).astype(np.uint8)
        
        keypoints = self.sift_detector.detect(gray, mask)
        
        if not keypoints:
            return [], np.array([])
        
        locations = [(round(kp.pt[0]), round(kp.pt[1])) for kp in keypoints]
        

        descriptors = self.hog.compute(gray, locations=locations)
        
        if descriptors is None:
            return keypoints, np.array([])
            

        descriptors = descriptors.reshape((len(locations), self.hog.getDescriptorSize()))
        descriptors = cv2.normalize(descriptors, None, norm_type=cv2.NORM_L2)

        return keypoints, descriptors
    
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
            "detector_type": "SIFT",
            "n_features": self.n_features,
            "n_octave_layers": self.n_octave_layers,
            "contrast_threshold": self.contrast_threshold,
            "edge_threshold": self.edge_threshold,
            "sigma": self.sigma
        }

class GLOHDescriptor(DescriptorComputer):
    """
    GLOH Descriptor based on Medium article:
    "Exploring Gradient Location and Orientation Histogram (GLOH) for Image Recognition and Object Detection"
    by Vincent Chung
    (medium.com/@vincentchung_72457/exploring-gradient-location-orientation-histogram-gloh-for-image-recognition-and-object-detection-3e3c231a5b01)
    """
    def __init__(self,
                 # Histogram parameter from the article's code
                 nbins: int = 36,
                 
                 # SIFT detector parameters (used by the article)
                 n_features: int = 0,
                 n_octave_layers: int = 3,
                 contrast_threshold: float = 0.04,
                 edge_threshold: float = 10,
                 sigma: float = 1.6):

        # Store parameters
        self.nbins = nbins
        
        # Store SIFT detector parameters
        self.n_features = n_features
        self.n_octave_layers = n_octave_layers
        self.contrast_threshold = contrast_threshold
        self.edge_threshold = edge_threshold
        self.sigma = sigma
        
        # 1. Initialize the SIFT Detector object
        # (The article uses this to find keypoints)
        try:
            self.sift_detector = cv2.SIFT_create(
                nfeatures=n_features, 
                nOctaveLayers=n_octave_layers, 
                contrastThreshold=contrast_threshold, 
                edgeThreshold=edge_threshold, 
                sigma=sigma
            )
        except AttributeError:
            self.sift_detector = cv2.xfeatures2d.SIFT_create(
                nfeatures=n_features, 
                nOctaveLayers=n_octave_layers, 
                contrastThreshold=contrast_threshold, 
                edgeThreshold=edge_threshold, 
                sigma=sigma
            )

    def detect_and_compute(self, image: np.ndarray, mask: np.ndarray | None = None) -> tuple[list[cv2.KeyPoint], np.ndarray]:
        # BASED ON: medium.com/@vincentchung_72457/exploring-gradient-location-orientation-histogram-gloh-for-image-recognition-and-object-detection-3e3c231a5b01
        if len(image.shape) == 3:
            gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
        else:
            gray = (image * 255).astype(np.uint8)
        
        
        # Compute gradient magnitude and orientation using Sobel operators
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        
        # Source: "mag, angle = cv2.cartToPolar(grad_x, grad_y, angleInDegrees=True)"
        _mag, angle = cv2.cartToPolar(grad_x, grad_y, angleInDegrees=True)

        # Compute keypoints using SIFT
        keypoints = self.sift_detector.detect(gray, mask)

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
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "ArticleGLOH (Custom)",
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
    
    def match_keypoints_descriptors(
        self,
        keypoints1: list,
        descriptors1: np.ndarray,
        keypoints2: list,
        descriptors2: np.ndarray
    ) -> tuple[list, np.ndarray, list, np.ndarray]:
        if descriptors1 is None or descriptors2 is None or len(descriptors1) == 0 or len(descriptors2) == 0:
            return [], np.array([]), [], np.array([])

        matches = self.matcher.knnMatch(descriptors1, descriptors2, k=2)
        good_kp1, good_desc1, good_kp2, good_desc2 = [], [], [], []

        for m_n in matches:
            if len(m_n) != 2:
                continue
            m, n = m_n
            if m.distance < self.ratio_test_threshold * n.distance:
                good_kp1.append(keypoints1[m.queryIdx])
                good_kp2.append(keypoints2[m.trainIdx])
                good_desc1.append(descriptors1[m.queryIdx])
                good_desc2.append(descriptors2[m.trainIdx])

        return good_kp1, np.array(good_desc1), good_kp2, np.array(good_desc2)

    
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
            

class Scorer(abc.ABC):
    def __init__(self, matcher: DescriptorMatcher):
        super().__init__()
        self.matcher = matcher
    def score(self, query_image: np.ndarray, query_keypoints, query_descriptors, database_image: np.ndarray, database_keypoints, database_descriptors) -> tuple[bool, float, dict]:
        pass

    def to_dict(self) -> dict:
        return {
            'class': self.__class__.__name__,
            'matcher': self.matcher.to_dict(),
        }

# FIXME: this score can be improved a lot, like #inliers / sqrt(#keypoints_q * #keypoints_db) which is more symetrical
class HomographyScorer(Scorer):
    def __init__(self, matcher: DescriptorMatcher, 
                 ransac_thresh: float = 5.0, 
                 max_reproj_error: float = 5.0,
                 use_reproj_error_penalty: bool = True):
        
        super().__init__(matcher)
        self.ransac_thresh = ransac_thresh
        self.max_reproj_error = max_reproj_error
        self.use_reproj_error_penalty = use_reproj_error_penalty

    def score(self, query_image: np.ndarray, query_keypoints, query_descriptors, database_image: np.ndarray, database_keypoints, database_descriptors):

        src_kpts, src_desc, dst_kpts, dst_desc = self.matcher.match_keypoints_descriptors(query_keypoints, query_descriptors, database_keypoints, database_descriptors)

        # check if the homography can be computed in a stable manner, with less than 20 points it may find homographies that can be "whatever"
        if len(src_kpts) < 20: # TODO: this 20 can increase to 50 or so
            return False, 0.0, {"reason": "not_enough_points"}

        src_pts = np.float32([kpt.pt for kpt in src_kpts])
        dst_pts = np.float32([kpt.pt for kpt in dst_kpts])

        M, mask = cv2.findHomography(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=self.ransac_thresh)

        if M is None or mask is None:
            return False, 0.0, {"reason": "homography_failed"}

        inliers = mask.ravel().astype(bool)
        n_inliers = np.sum(inliers)
        inlier_ratio = n_inliers / len(src_pts)

        src_inliers = src_pts[inliers]
        dst_inliers = dst_pts[inliers]

        if len(src_inliers) == 0:
            return False, 0.0, {"reason": "no_inliers"}

        src_proj = cv2.perspectiveTransform(src_inliers.reshape(-1, 1, 2), M).reshape(-1, 2)
        reproj_error = np.sqrt(np.mean(np.sum((src_proj - dst_inliers) ** 2, axis=1)))

        det = np.linalg.det(M[:2, :2])
        if det <= 0.4 or det > 10:  # Negative det is a flip, det too large/small is bad
            valid = False
        elif reproj_error > self.max_reproj_error:
            valid = False
        else:
            valid = True

        # Calculate score based on configuration
        if self.use_reproj_error_penalty:
            score = inlier_ratio * np.exp(-0.5 * (reproj_error / self.max_reproj_error))
        else:
            score = inlier_ratio

        info = {
            "n_inliers": int(n_inliers),
            "total_matches": int(len(src_pts)),
            "inlier_ratio": float(inlier_ratio),
            "reproj_error": float(reproj_error),
            "det": float(det),
            "use_reproj_error_penalty": self.use_reproj_error_penalty
        }

        return valid, float(score), info

    def to_dict(self) -> dict:
        d = super().to_dict()
        d['ransac_thresh'] = self.ransac_thresh
        d['max_reproj_error'] = self.max_reproj_error
        d['use_reproj_error_penalty'] = self.use_reproj_error_penalty
        return d


class KeypointAndDescriptorMaker:
    def __init__(self, *, descriptor_computer: DescriptorComputer, color_conversion: ColorConversion, preprocess: ImagePreprocessStep | None = None):

        self.descriptor_computer = descriptor_computer
        self.color_conversion = color_conversion
        self.preprocess = preprocess


    def detect_and_compute(self, image: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
        if mask is None:
            mask = np.ones(image.shape[:2], dtype=np.uint8) * 255

        # image = image.astype(np.float32) / 255

        if self.preprocess is not None:
            preprocessed_image, preprocessed_mask = self.preprocess(image, mask)
        else:
            preprocessed_image, preprocessed_mask = image, mask

        colorspace_image, preprocessed_mask = self.color_conversion(preprocessed_image, preprocessed_mask)
        keypoints, descriptors = self.descriptor_computer.detect_and_compute(colorspace_image)
        # for part in descriptor_parts:
            # assert isclose(part.sum(), 1.0), f"The sum was {part.sum()}"
        return keypoints, descriptors
    
    def to_dict(self) -> dict:
        return {
            'descriptor_computer': self.descriptor_computer.to_dict(),
            'color_conversion': self.color_conversion.to_dict(),
            'preprocess': self.preprocess.to_dict(),
        }


if __name__ == "__main__":
    
    print("--- Creating a dummy image for testing ---")
    # Create a 480x640 float image [0,1]
    # The descriptor classes handle the conversion to uint8 [0,255]
    dummy_image = np.zeros((480, 640), dtype=np.float32)
    
    # Add some features for the detectors to find
    cv2.circle(dummy_image, (100, 100), 30, 0.8, -1)
    cv2.circle(dummy_image, (300, 250), 50, 0.6, -1)
    cv2.circle(dummy_image, (500, 400), 20, 0.9, -1)
    
    print(f"Dummy image created with shape: {dummy_image.shape}")

    # --- 1. Test PCASIFTDescriptor ---
    print("\n--- Testing PCASIFTDescriptor ---")
    try:
        # Note: This class uses the "flawed" PCA logic as implemented
        # We use n_features=500 to ensure we get enough keypoints
        # to be more than num_components (36)
        pca_sift = PCASIFTDescriptor(num_components=24, n_features=500)
        
        keypoints_pca, descriptors_pca = pca_sift.detect_and_compute(dummy_image)
        
        print(f"Detected {len(keypoints_pca)} keypoints.")
        if descriptors_pca is not None and descriptors_pca.shape[0] > 0:
            print(f"Computed PCA-SIFT descriptors with shape: {descriptors_pca.shape}")
            print(f"(Expected second dimension: {pca_sift.num_components})")
        else:
            print("No PCA-SIFT descriptors computed.")
        
        # print(f"Descriptor params: {pca_sift.to_dict()}")

    except Exception as e:
        print(f"PCASIFTDescriptor FAILED. Error: {e}")
        print("This might be because 'cv2.SIFT_create' is not available.")
        print("Please ensure you have a modern OpenCV or 'opencv-contrib-python'.")

    # --- 2. Test HOGDescriptor ---
    print("\n--- Testing HOGDescriptor ---")
    try:
        # HOG uses a SIFT detector internally per the class implementation
        hog_desc = HOGDescriptor(n_features=500)
        
        keypoints_hog, descriptors_hog = hog_desc.detect_and_compute(dummy_image)
        
        print(f"Detected {len(keypoints_hog)} keypoints (using SIFT detector).")
        if descriptors_hog is not None and descriptors_hog.shape[0] > 0:
            print(f"Computed HOG descriptors with shape: {descriptors_hog.shape}")
        else:
            print("No HOG descriptors computed.")
        
        # print(f"Descriptor params: {hog_desc.to_dict()}")

    except Exception as e:
        print(f"HOGDescriptor FAILED. Error: {e}")
        print("This might be because the internal SIFT detector failed.")

    # --- 3. Test ArticleGLOHDescriptor ---
    print("\n--- Testing ArticleGLOHDescriptor ---")
    try:
        # This also uses a SIFT detector internally
        gloh_desc = GLOHDescriptor(nbins=36, n_features=500)
        
        keypoints_gloh, descriptors_gloh = gloh_desc.detect_and_compute(dummy_image)
        
        print(f"Detected {len(keypoints_gloh)} keypoints (using SIFT detector).")
        if descriptors_gloh is not None and descriptors_gloh.shape[0] > 0:
            print(f"Computed 'ArticleGLOH' descriptors with shape: {descriptors_gloh.shape}")
            print(f"(Expected second dimension: {gloh_desc.nbins})")
        else:
            print("No 'GLOH' descriptors computed.")
        
        # print(f"Descriptor params: {gloh_desc.to_dict()}")

    except Exception as e:
        print(f"ArticleGLOHDescriptor FAILED. Error: {e}")
        print("This might be because the internal SIFT detector failed.")
