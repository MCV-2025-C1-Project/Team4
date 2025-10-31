import abc
import enum
from typing import Any, Literal, Protocol
import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern
import pywt

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






# if __name__ == "__main__":
    
    #Paths for the dataset
    # BBDD_DIR = "dataset/BBDD"
    # QSD1_DIR = "dataset/QSD1"
    # OUTPUT_DIR = "descriptors"
    
    # # dataset/qsd1_w1/00003.jpg
    # descr = ImageDescriptor(color_mapping='MAX_ABS_SCALE', color_space='RGB', bins_per_channel=64)
    # img = cv2.imread("qsd1_w1/00003.jpg")
    
    # hist = descr.compute_descriptor(img)
    
    # print(hist.shape)
    # print(hist.max())
    # print(hist.min())
    # print(hist.mean())
    # print(hist.std())
    # print(hist.sum())
if __name__ == "__main__":
    image_path = "/home/bernat/MCV/C1/proyect/Team4/plot_results/test_images/00010.jpg"
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError("Image not found.")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32) / 255.0

    # === Create descriptor maker ===
    idm = ImageDescriptorMaker(
        gamma_correction=1.0,
        blur_image=False,
        color_spaces=[ColorSpace.RGB],
        bins=8,
        keep_or_discard="K",
        weights=None,
        image_blocks=image_blocks_identity,
        color_channels=[0, 2]
    )

    # === 1D HISTOGRAM TEST ===
    hists_1d = idm.compute_1d_histogram(image)
    print("1D histograms:", [h.shape for h in hists_1d])

    # === 2D HISTOGRAM TEST ===
    hists_2d = idm.compute_2d_histogram(image)
    print("2D histograms:", [h.shape for h in hists_2d])

    # visualize one 2D hist
    plt.imshow(hists_2d[0], cmap='viridis')
    plt.title("2D Histogram (ch0 vs ch2)")
    plt.colorbar()
    plt.show()

    # === 3D HISTOGRAM TEST ===
    hists_3d = idm.compute_3d_histogram(image)
    print("3D histograms:", [h.shape for h in hists_3d])

    # visualize one 3D hist as slices
    if len(hists_3d) > 0:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        for i in range(3):
            axes[i].imshow(hists_3d[0][:, :, i], cmap='viridis')
            axes[i].set_title(f"3D hist slice {i}")
        plt.show()


        