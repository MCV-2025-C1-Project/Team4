import enum
from typing import Callable
import numpy as np
import cv2
from pathlib import Path
import pickle
import matplotlib.pyplot as plt


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


class ImageDescriptor:
    
    ''' Class to compute descriptors using 1D histograms'''

    def __init__(self, color_mapping=None, color_space='LAB', bins_per_channel = 16, normalize_histograms=False):
        self.color_mapping = color_mapping
        self.color_space = color_space.upper() # 'RGB', 'HSV', 'LAB', 'GRAY', 'YCrCb', 'Cielab'
        self.bins_per_channel = bins_per_channel
        self.normalize_histograms = normalize_histograms

    def compute_descriptor(self, image: np.ndarray):
        
        match self.color_space:
            case 'GRAY':
                converted_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                channels = [converted_img]
                ranges = [(0, 256)]
            case 'HSV':
                converted_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                channels = cv2.split(converted_img)
                ranges = [(0, 180), (0, 256), (0, 256)]
            case 'RGB':
                converted_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                channels = cv2.split(converted_img)
                ranges = [(0, 256), (0, 256), (0, 256)]
            case 'LAB':
                converted_img = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
                channels = cv2.split(converted_img)
                ranges = [(0, 256), (0, 256), (0, 256)]
            case 'YCRCB':
                converted_img = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
                channels = cv2.split(converted_img)
                ranges = [(0, 256), (0, 256), (0, 256)]
            case 'HLS':
                converted_img = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
                channels = cv2.split(converted_img)
                ranges = [(0, 180), (0, 256), (0, 256)]
            case 'CMYK':
                converted_img = cv2.cvtColor(image, cv2.COLOR_BGR2CMYK)
                channels = cv2.split(converted_img)
                ranges = [(0, 256), (0, 256), (0, 256), (0, 256)]
            case _:
                raise ValueError(f"Sorry, we do not have this color space yet: {self.color_space}")
        
        # Print info before normalization
        """
        print(f"\n=== Before normalization ===")
        for i, ch in enumerate(channels):
            print(f"Channel {i}: dtype={ch.dtype}, shape={ch.shape}, min={ch.min()}, max={ch.max()}, mean={ch.mean():.2f}, std={ch.std():.2f}")
        """
        # Apply normalization after color conversion
        match self.color_mapping:
            case 'MAX_ABS_SCALE':
                channels = [(ch - ch.min()) / (ch.max() - ch.min()) for ch in channels]
                # Update also the ranges
                ranges = [(0, 1) for _ in ranges]
            case 'STD_SCALER':
                channels = [(ch - ch.mean()) / ch.std() for ch in channels]
                # Update ranges based on actual min/max after standardization
                ranges = [(ch.min(), ch.max()) for ch in channels]
        
        # Print info after normalization
        if self.color_mapping:
            print(f"\n=== After normalization ({self.color_mapping}) ===")
            for i, ch in enumerate(channels):
                print(f"Channel {i}: dtype={ch.dtype}, shape={ch.shape}, min={ch.min():.4f}, max={ch.max():.4f}, mean={ch.mean():.4f}, std={ch.std():.4f}")
        
        """
        plt.figure()
        plt.title(f"Image")
        plt.imshow(image)
        plt.show()
        """
        histograms = []
        for i, channel in enumerate(channels):
            hist = np.histogram(channel, bins=self.bins_per_channel, range=ranges[i])[0]
            hist = hist.astype(np.float32)
            if self.normalize_histograms:
                hist = (hist * 1000) / (hist.sum() + 1e-8)  # Normalize histogram
            #Plot the histogram
            """
            print(f"Histogram Channel {i}: shape={hist.shape}, min={hist.min()}, max={hist.max()}, mean={hist.mean():.2f}, std={hist.std():.2f}")
            plt.figure()
            plt.title(f"Histogram Channel {i}")
            plt.plot(hist)
            plt.xlabel('Bin')
            plt.ylabel('Frequency')
            plt.show()
            """
            histograms.append(hist)
            
        descriptor = np.concatenate(histograms)
        return descriptor
                
    #TODO COMPUTE HISTO FOR WHOLE DATASET
                
def create_method1_descriptor(bins=32):
    "descriptor for hsv color space"
    return ImageDescriptor(color_space='HSV', bins_per_channel=bins)

def create_method2_descriptor(bins=256):
    "descriptor for gray color space"
    return ImageDescriptor(color_space='GRAY', bins_per_channel=bins)


def create_pyramid_weight(H: int, W: int):
    y = np.linspace(0, 1, H)
    x = np.linspace(0, 1, W)

    xs, ys = np.meshgrid(x, y)
    center_x, center_y = 0.5, 0.5
    
    dist_x = 1 - 2 * np.abs(xs - center_x)
    dist_y = 1 - 2 * np.abs(ys - center_y)
    
    pyramid_weight = np.minimum(dist_x, dist_y)
    return pyramid_weight


def create_cuadraticp_pyramid_weight(H: int, W: int):
    y = np.linspace(0, 1, H)
    x = np.linspace(0, 1, W)

    xs, ys = np.meshgrid(x, y)
    center_x, center_y = 0.5, 0.5
    
    dist_x = (1 - 2 * np.abs(xs - center_x)) ** 2
    dist_y = (1 - 2 * np.abs(ys - center_y)) ** 2
    
    pyramid_weight = np.minimum(dist_x, dist_y)
    return pyramid_weight


def create_cone_weight(H: int, W: int):
    y = np.linspace(-1, 1, H)
    x = np.linspace(-1, 1, W)
    
    xs, ys = np.meshgrid(x, y)
    
    radius = np.sqrt(xs**2 + ys**2)
    
    max_radius = np.sqrt(2)
    
    cone_weight = np.clip(1 - radius / max_radius, 0, 1)
    return cone_weight


def create_cuadratic_cone_weight(H: int, W: int):
    y = np.linspace(-1, 1, H)
    x = np.linspace(-1, 1, W)
    
    xs, ys = np.meshgrid(x, y)
    
    radius = np.sqrt(xs**2 + ys**2)
    
    max_radius = np.sqrt(2)
    
    cone_weight = np.clip(1 - radius / max_radius, 0, 1)
    cone_weight = cone_weight ** 2
    return cone_weight


def create_center_crop_weight(H, W, discard_borders=0.1):
    assert 0 <= discard_borders < 0.5
    
    border_h = int(H * discard_borders)
    border_w = int(W * discard_borders)
    
    center_crop_weight = np.zeros((H, W))
    center_crop_weight[border_h:H-border_h, border_w:W-border_w] = 1.0
    return center_crop_weight


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

class WeightStrategy(enum.Enum):
    PYRAMID = 'PYRAMID'
    # CUADRATIC_PYRAMID = 'CUADRATIC_PYRAMID'
    # CONE = 'CONE'
    # CUADRATIC_CONE = 'CUADRATIC_CONE'
    CENTER_CROP_05 = 'CENTER_CROP_05'
    CENTER_CROP_10 = 'CENTER_CROP_10'
    CENTER_CROP_15 = 'CENTER_CROP_15'
    # CENTER_CROP_20 = 'CENTER_CROP_20'

class ImageDescriptorMaker:
    def __init__(self, *, gamma_correction: float, blur_image: False | Callable[[np.ndarray], np.ndarray], color_spaces: list[ColorSpace], bins: int | list[int], keep_or_discard: None | str, weights: None | WeightStrategy):
        
        # assert keep_or_discard is None or len(color_spaces) == len(keep_or_discard)

        self.gamma_correction = gamma_correction
        self.blur_image = blur_image
        self.color_spaces = color_spaces
        self.keep_or_discard = keep_or_discard
        self.bins = bins
        self.weights = weights


    def make_descriptor(self, image: np.ndarray) -> np.ndarray:
        image = image.astype(np.float32) / 255
        image = image ** self.gamma_correction
        if self.blur_image:
            image = self.blur_image(image)

        descriptor_parts = []
        for color_space in self.color_spaces:
            match color_space:
                case ColorSpace.RGB:
                    descriptor_parts.append(self.compute_rgb_descriptor(image))
                # case ColorSpace.GRAY:
                    # descriptor_parts.append(self.compute_gray_descriptor(image))
                case ColorSpace.HSV:
                    descriptor_parts.append(self.compute_hsv_descriptor(image))
                case ColorSpace.LAB:
                    descriptor_parts.append(self.compute_lab_descriptor(image))
                case ColorSpace.YCRCB:
                    descriptor_parts.append(self.compute_ycrcb_descriptor(image))
                case ColorSpace.HLS:
                    descriptor_parts.append(self.compute_hls_descriptor(image))
                case ColorSpace.CMYK:
                    descriptor_parts.append(self.compute_cmyk_descriptor(image))
                case ColorSpace.LUV:
                    descriptor_parts.append(self.compute_luv_descriptor(image))
                case ColorSpace.XYZ:
                    descriptor_parts.append(self.compute_xyz_descriptor(image))
                case ColorSpace.YUV:
                    descriptor_parts.append(self.compute_yuv_descriptor(image))
                case _:
                    raise ValueError("Unknown color space.")

        descriptor_parts = flatten_list(descriptor_parts)

        descriptor_parts = [part for part, kod in zip(descriptor_parts, self.keep_or_discard) if kod == 'K']

        return np.concatenate(descriptor_parts)

    def compute_weights_image(self, image: np.ndarray) -> np.ndarray:
        match self.weights:
            case WeightStrategy.PYRAMID:
                return create_pyramid_weight(image.shape[0], image.shape[1])
            # case WeightStrategy.CUADRATIC_PYRAMID:
            #     return create_cuadraticp_pyramid_weight(image.shape[0], image.shape[1])
            # case WeightStrategy.CONE:
            #     return create_cone_weight(image.shape[0], image.shape[1])
            # case WeightStrategy.CUADRATIC_CONE:
            #     return create_cuadratic_cone_weight(image.shape[0], image.shape[1])
            case WeightStrategy.CENTER_CROP_05:
                return create_center_crop_weight(image.shape[0], image.shape[1], 0.05)
            case WeightStrategy.CENTER_CROP_10:
                return create_center_crop_weight(image.shape[0], image.shape[1], 0.1)
            case WeightStrategy.CENTER_CROP_15:
                return create_center_crop_weight(image.shape[0], image.shape[1], 0.15)
            # case WeightStrategy.CENTER_CROP_20:
            #     return create_center_crop_weight(image.shape[0], image.shape[1], 0.2)
            case _:
                raise ValueError("Unknown weight strategy.")


    def compute_histograms(self, image):
        if self.weights:
            weights = self.compute_weights_image(image)
            weights_sum = weights.sum()
        else:
            weights = None
            weights_sum = None
        
        if len(image.shape) == 2:
            image = np.expand_dims(image, 2)

        histograms = []
        for c in range(image.shape[2]):
            hist = np.histogram(image[:, :, c], bins=self.bins, weights=weights)[0]
            if weights is None:
                hist = hist / (image.shape[0] * image.shape[1])
            else:
                hist = hist / weights_sum
            histograms.append(hist)

        return histograms


    def compute_rgb_descriptor(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return self.compute_histograms(image)

    def compute_gray_descriptor(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return self.compute_histograms(image)

    def compute_hsv_descriptor(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        return self.compute_histograms(image)
    
    def compute_lab_descriptor(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
        return self.compute_histograms(image)
    
    def compute_ycrcb_descriptor(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        return self.compute_histograms(image)
    
    def compute_hls_descriptor(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
        return self.compute_histograms(image)

    def compute_cmyk_descriptor(self, image):
        image = bgr_to_cmyk(image)
        return self.compute_histograms(image)
    
    def compute_luv_descriptor(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2LUV)
        return self.compute_histograms(image)


    def compute_xyz_descriptor(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2XYZ)
        return self.compute_histograms(image)
    

    def compute_yuv_descriptor(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        return self.compute_histograms(image)
    

if __name__ == "__main__":
    
    #Paths for the dataset
    BBDD_DIR = "dataset/BBDD"
    QSD1_DIR = "dataset/QSD1"
    OUTPUT_DIR = "descriptors"
    
    # dataset/qsd1_w1/00003.jpg
    descr = ImageDescriptor(color_mapping='MAX_ABS_SCALE', color_space='RGB', bins_per_channel=64)
    img = cv2.imread("qsd1_w1/00003.jpg")
    
    hist = descr.compute_descriptor(img)
    
    print(hist.shape)
    print(hist.max())
    print(hist.min())
    print(hist.mean())
    print(hist.std())
    print(hist.sum())
