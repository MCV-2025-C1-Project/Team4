import abc
import enum
from typing import Any, Literal, Protocol
import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern
import pywt

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

def image_blocks_identity(image: np.ndarray) -> list[np.ndarray]:
    return [image]

def image_blocks_nm(image: np.ndarray, blocks_shape: list = (2,2)) -> list[np.ndarray]:
    blocks = []

    h, w, _ = image.shape
    block_h, block_w = h // blocks_shape[0], w // blocks_shape[1]

    for i in range(2):
        for j in range(2):
            block = image[i*block_h:(i+1)*block_h, j*block_w:(j+1)*block_w]
            blocks.append(block)

    return blocks


class ImageBlockSplitter(Protocol):
    def __call__(self, image: np.ndarray) -> list[np.ndarray]:
        ...
        
    def to_dict(self) -> dict[str, Any]:
        pass
    
    def num_blocks(self) -> int:
        pass


class IdentityImageBlockSplitter(ImageBlockSplitter):
    def __call__(self, image: np.ndarray) -> list[np.ndarray]:
        return [image]
    
    def to_dict(self):
        d = {'class': self.__class__.__name__}
        return d
    
    def num_blocks(self) -> int:
        return 1


class GridImageBlockSplitter(ImageBlockSplitter):
    def __init__(self, shape: tuple[int, int]):
        super().__init__()
        self.shape = shape

    def __call__(self, image: np.ndarray) -> list[np.ndarray]:
        blocks = []
        h, w, _ = image.shape
        block_h, block_w = h // self.shape[0], w // self.shape[1]
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                block = image[i*block_h:(i+1)*block_h, j*block_w:(j+1)*block_w]
                blocks.append(block)

        return blocks
    
    def to_dict(self):
        d = {
            'class': self.__class__.__name__,
            'shape': self.shape,
        }
        return d
    
    def num_blocks(self) -> int:
        return self.shape[0] * self.shape[1]


class PyramidImageBlockSplitter(ImageBlockSplitter):
    def __init__(self, shapes: list[tuple[int, int]]):
        super().__init__()
        self.shapes = shapes

    def __call__(self, image: np.ndarray) -> list[np.ndarray]:
        blocks = []
        for shape in self.shapes:
            grid_splitter = GridImageBlockSplitter(shape)
            sub_blocks = grid_splitter(image)
            blocks.extend(sub_blocks)
        return blocks
    
    def to_dict(self):
        d = {
            'class': self.__class__.__name__,
            'shapes': self.shapes,
        }
        return d
    
    def num_blocks(self) -> int:
        total = 0
        for shape in self.shapes:
            total += shape[0] * shape[1]
        return total


class HistogramComputer(abc.ABC):
    def __init__(self, weight_strategy: WeightStrategy, block_splitter: ImageBlockSplitter):
        super().__init__()
        self.weight_strategy = weight_strategy
        self.block_splitter = block_splitter
    
    @abc.abstractmethod
    def __call__(self, image: np.ndarray) -> list[np.ndarray]:
        ...
        
    def compute_weights_image(self, image: np.ndarray) -> np.ndarray:
        match self.weight_strategy:
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
            
    def to_dict(self) -> dict[str, Any]:
        d = {
            'class': self.__class__.__name__,
            'weight_strategy': self.weight_strategy.value if self.weight_strategy else None,
            'block_splitter': self.block_splitter.to_dict(),
        }
        return d

class Histogram1D(HistogramComputer):
    def __init__(self, channels: list[int], bins: int, weight_strategy: WeightStrategy | None, block_splitter: ImageBlockSplitter, range_: tuple[float, float] = (0, 1)):
        super().__init__(weight_strategy, block_splitter)
        self.bins = bins
        self.range_ = range_
        self.channels = channels

    def __call__(self, image: np.ndarray) -> list[np.ndarray]:
        image_blocks = self.block_splitter(image)
        if self.weight_strategy:
            weights = self.compute_weights_image(image)
            weight_blocks = self.block_splitter(weights)
        else:
            weights = None
            weight_blocks = [None for _ in image_blocks]
        
        histograms = []
        for block, weight_block in zip(image_blocks, weight_blocks):
            if len(block.shape) == 2:
                block = np.expand_dims(block, 2)
            
            for c in self.channels:
                hist = np.histogram(block[:, :, c], bins=self.bins, weights=weight_block, range=self.range_)[0].astype(np.float64)
                if weight_block is None:
                    hist = hist / np.float64(block.shape[0] * block.shape[1])
                else:
                    hist = hist / weight_block.sum()
                histograms.append(hist)

        return histograms
    
    def to_dict(self):
        d = super().to_dict()
        d['class'] = self.__class__.__name__
        d['bins'] = self.bins
        d['channels'] = self.channels
        return d


class Histogram2D(HistogramComputer):
    def __init__(self, channel_pairs: list[tuple[int, int]], bins: int, weight_strategy: WeightStrategy | None, block_splitter: ImageBlockSplitter, range_: tuple[float, float] = (0, 1)):
        super().__init__(weight_strategy, block_splitter)
        self.bins = bins
        self.range_ = range_
        self.channel_pairs = channel_pairs

    def __call__(self, image: np.ndarray) -> list[np.ndarray]:
        image_blocks = self.block_splitter(image)
        
        if self.weight_strategy:
            weights = self.compute_weights_image(image)
            weight_blocks = self.block_splitter(weights)
        else:
            weights = None
            weight_blocks = [None for _ in image_blocks]
        
        hist_matrices = []

        for channel_pair in self.channel_pairs:
            ch1_idx, ch2_idx = channel_pair

            for block, weight_block in zip(image_blocks, weight_blocks):
                ch1 = block[:, :, ch1_idx].ravel()
                ch2 = block[:, :, ch2_idx].ravel()

                hist_2d, _, _ = np.histogram2d(
                    ch1,
                    ch2,
                    bins=self.bins,
                    range=[self.range_, self.range_],
                    weights=weight_block.ravel() if weight_block is not None else None
                )

                if weight_block is None:
                    hist_2d = hist_2d / (block.shape[0] * block.shape[1])
                else:
                    hist_2d = hist_2d / weight_block.sum()

                hist_matrices.append(hist_2d.ravel()) # FIXME: hacer un ravel aqui es un poco cualquier cosa

        return hist_matrices
    
    def to_dict(self):
        d = super().to_dict()
        d['class'] = self.__class__.__name__
        d['bins'] = self.bins
        d['channel_pairs'] = self.channel_pairs
        return d


class Histogram3D(HistogramComputer):
    def __init__(self, channel_triplets: list[tuple[int, int, int]], bins: int, weight_strategy: WeightStrategy | None, block_splitter: ImageBlockSplitter, range_: tuple[float, float] = (0, 1)):
        super().__init__(weight_strategy, block_splitter)
        self.bins = bins
        self.range_ = range_
        self.channel_triplets = channel_triplets

    def __call__(self, image: np.ndarray) -> list[np.ndarray]:
        image_blocks = self.block_splitter(image)
        
        if self.weight_strategy:
            weights = self.compute_weights_image(image)
            weight_blocks = self.block_splitter(weights)
        else:
            weights = None
            weight_blocks = [None for _ in image_blocks]
        
        hist_matrices = []

        for channel_triplet in self.channel_triplets:
            ch1_idx, ch2_idx, ch3_idx = channel_triplet

            for block, weight_block in zip(image_blocks, weight_blocks):
                ch1 = block[:, :, ch1_idx].ravel()
                ch2 = block[:, :, ch2_idx].ravel()
                ch3 = block[:, :, ch3_idx].ravel()

                hist_3d, _ = np.histogramdd(
                    sample=(ch1, ch2, ch3),
                    bins=self.bins,
                    weights=weight_block.ravel() if weight_block is not None else None,
                    range=[self.range_, self.range_, self.range_]
                )

                if weight_block is None:
                    hist_3d = hist_3d / (block.shape[0] * block.shape[1])
                else:
                    hist_3d = hist_3d / weight_block.sum()

                hist_matrices.append(hist_3d.ravel()) # FIXME: hacer un ravel aqui es un poco cualquier cosa

        return hist_matrices

    def to_dict(self):
        d = super().to_dict()
        d['class'] = self.__class__.__name__
        d['bins'] = self.bins
        d['channel_triplets'] = self.channel_triplets
        return d


class LBPHistogramDescriptor(HistogramComputer):
    def __init__(self, channels: list[int], bins: int, n_points: int, radius: int, method: Literal['default', 'ror', 'uniform', 'nri_uniform', 'var'], block_splitter: ImageBlockSplitter):
        super().__init__(None, block_splitter)
        self.channels = channels
        self.bins = bins
        self.n_points = n_points
        self.radius = radius
        self.method = method


    def __call__(self, image: np.ndarray) -> list[np.ndarray]:
        descriptors = []
        for c in self.channels:
            lbps = local_binary_pattern(image[:, :, c], P=self.n_points, R=self.radius, method=self.method)
            hist = np.histogram(lbps, bins=self.bins, range=(0, 255))[0]
            hist = hist.astype(np.float32)
            hist = hist / hist.sum()
            descriptors.append(hist)

        return descriptors
    
    def to_dict(self):
        d = super().to_dict()
        d['channels'] = self.channels
        d['bins'] = self.bins
        d['n_points'] = self.n_points
        d['radius'] = self.radius
        d['method'] = self.method
        return d


class DCTDescriptor(HistogramComputer):
    """
    Computes a texture descriptor based on the Discrete Cosine Transform (DCT).

    This descriptor follows the process outlined in the assignment:
    1. The image is split into blocks.
    2. For each block and for each specified channel:
       a. A 2D DCT is applied.
       b. The resulting coefficients are scanned in a zig-zag pattern.
       c. The first 'n_coeffs' from the zig-zag scan are kept as the block's feature vector.
    3. The feature vectors from all blocks are concatenated to form the final descriptor.
    """
    def __init__(self, channels: list[int], block_splitter: ImageBlockSplitter, n_coeffs: int):
        """
        Initializes the DCTDescriptor.

        Args:
            channels (list[int]): List of channel indices to compute the descriptor on.
            block_splitter (ImageBlockSplitter): A strategy for splitting the image into blocks.
            n_coeffs (int): The number of DCT coefficients to keep from the start of the
                            zig-zag scan for each block.
        """
        # DCT descriptor as described doesn't use a weighting strategy.
        super().__init__(weight_strategy=None, block_splitter=block_splitter)
        self.channels = channels
        self.n_coeffs = n_coeffs

    def _zig_zag_scan(self, matrix: np.ndarray) -> np.ndarray:
        """
        Performs a zig-zag scan on a 2D matrix to linearize it.
        """
        rows, cols = matrix.shape
        result = np.empty(rows * cols, dtype=matrix.dtype)
        r, c = 0, 0
        moving_up = True

        for i in range(rows * cols):
            result[i] = matrix[r, c]

            if moving_up:
                if c == cols - 1:  # Hit right wall
                    r += 1
                    moving_up = False
                elif r == 0:  # Hit top wall
                    c += 1
                    moving_up = False
                else:  # Move diagonally up-right
                    r -= 1
                    c += 1
            else:  # Moving down
                if r == rows - 1:  # Hit bottom wall
                    c += 1
                    moving_up = True
                elif c == 0:  # Hit left wall
                    r += 1
                    moving_up = True
                else:  # Move diagonally down-left
                    r += 1
                    c -= 1
        return result

    def __call__(self, image: np.ndarray) -> list[np.ndarray]:
        """
        Computes the DCT descriptor for the given image.

        Args:
            image (np.ndarray): The input image (already converted to the desired colorspaces
                                and concatenated).

        Returns:
            list[np.ndarray]: A list containing the feature vectors for each block.
                              These will be concatenated later by the ImageDescriptorMaker.
        """
        image_blocks = self.block_splitter(image)
        all_block_descriptors = []

        for block in image_blocks:
            # Ensure block is 3D for consistent channel indexing
            if len(block.shape) == 2:
                block = np.expand_dims(block, axis=2)

            for c in self.channels:
                channel_block = block[:, :, c]

                # The input for cv2.dct must be float32
                if channel_block.dtype != np.float32:
                   channel_block = channel_block.astype(np.float32)

                # 1. Apply 2D DCT
                dct_block = cv2.dct(channel_block)

                # 2. Zig-zag scan
                zig_zag_coeffs = self._zig_zag_scan(dct_block)

                # 3. Keep first N coefficients
                # If block is smaller than N coeffs, pad with 0
                if len(zig_zag_coeffs) < self.n_coeffs:
                    pad_width = self.n_coeffs - len(zig_zag_coeffs)
                    block_feature_vector = np.pad(zig_zag_coeffs, (0, pad_width), 'constant')
                else:
                    block_feature_vector = zig_zag_coeffs[:self.n_coeffs]
                
                all_block_descriptors.append(block_feature_vector)
        
        return all_block_descriptors

    def to_dict(self) -> dict[str, Any]:
        """Serializes the descriptor's configuration."""
        d = super().to_dict()
        d['class'] = self.__class__.__name__
        d['channels'] = self.channels
        d['n_coeffs'] = self.n_coeffs
        return d

class WaveletDescriptor(HistogramComputer):
    """
    Computes a texture descriptor based on the Discrete Wavelet Transform (DWT).
    """
    def __init__(self, channels: list[int], block_splitter: ImageBlockSplitter, wavelet: str = 'haar', level: int = 2):
        super().__init__(weight_strategy=None, block_splitter=block_splitter)
        self.channels = channels
        if level < 1: raise ValueError("Decomposition level must be at least 1.")
        # try:
            # pywt.Wavelet(wavelet)
        # except ValueError:
            # raise ValueError(f"Wavelet '{wavelet}' not found.")
        self.wavelet = wavelet
        self.level = level

    def __call__(self, image: np.ndarray) -> list[np.ndarray]:
        image_blocks = self.block_splitter(image)
        all_block_descriptors = []
        for block in image_blocks:
            if len(block.shape) == 2: block = np.expand_dims(block, axis=2)
            for c in self.channels:
                channel_block = block[:, :, c]
                min_dim = 2**self.level
                if channel_block.shape[0] < min_dim or channel_block.shape[1] < min_dim:
                    channel_block = cv2.resize(channel_block, (min_dim, min_dim), interpolation=cv2.INTER_LINEAR)
                coeffs = pywt.wavedec2(channel_block, self.wavelet, level=self.level)
                channel_features = []
                coeffs_flat = [coeffs[0]]
                for detail_level in coeffs[1:]:
                    coeffs_flat.extend(detail_level)
                for band_matrix in coeffs_flat:
                    mean = np.mean(band_matrix)
                    std_dev = np.std(band_matrix)
                    channel_features.extend([mean, std_dev])
                all_block_descriptors.append(np.array(channel_features, dtype=np.float32))
        return all_block_descriptors

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d.update({'channels': self.channels, 'wavelet': self.wavelet, 'level': self.level})
        return d

class ImageDescriptorMaker:
    def __init__(self, *, histogram_computer: HistogramComputer, color_spaces: list[ColorSpace], preprocess: ImagePreprocessStep | None = None):

        self.histogram_computer = histogram_computer
        self.color_spaces = color_spaces
        self.preprocess = preprocess


    def generate_colorspaces_image(self, image: np.ndarray) -> np.ndarray:
        channel_images = []

        for color_space in self.color_spaces:
            match color_space:
                case ColorSpace.RGB:
                    converted = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                case ColorSpace.HSV:
                    converted = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                case ColorSpace.LAB:
                    converted = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
                    # L∈[0,100], a∈[-128,127], b∈[-128,127]
                    L, a, b = cv2.split(converted)
                    L = L / 100.0
                    a = (a + 128.0) / 255.0
                    b = (b + 128.0) / 255.0
                    converted = cv2.merge([L, a, b])
                case ColorSpace.YCRCB:
                    converted = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
                case ColorSpace.HLS:
                    converted = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
                case ColorSpace.CMYK:
                    converted = bgr_to_cmyk(image)
                case ColorSpace.LUV:
                    converted = cv2.cvtColor(image, cv2.COLOR_BGR2Luv)
                    # L∈[0,100], u∈[-134,220], v∈[-140,122] (approx)
                    L, u, v = cv2.split(converted)
                    L = L / 100.0
                    u = (u + 134.0) / (220.0 + 134.0)   # scale to [0,1]
                    v = (v + 140.0) / (122.0 + 140.0)   # scale to [0,1]
                    converted = cv2.merge([L, u, v])
                case ColorSpace.XYZ:
                    converted = cv2.cvtColor(image, cv2.COLOR_BGR2XYZ)
                case ColorSpace.YUV:
                    converted = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
                    # Y∈[0,1], U,V∈[-0.436,0.436],[-0.615,0.615]
                    Y, U, V = cv2.split(converted)
                    U = (U + 0.436) / (2 * 0.436)
                    V = (V + 0.615) / (2 * 0.615)
                    converted = cv2.merge([Y, U, V])
                case _:
                    raise ValueError(f"Unknown color space: {color_space}.")

            channel_images.append(converted)

        return np.concatenate(channel_images, axis=2)

    def make_descriptor(self, image: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
        if mask is None:
            mask = np.ones(image.shape[:2], dtype=np.uint8) * 255

        image = image.astype(np.float32) / 255

        if self.preprocess is not None:
            preprocessed_image, preprocessed_mask = self.preprocess(image, mask)
        else:
            preprocessed_image, preprocessed_mask = image, mask

        colorspace_image = self.generate_colorspaces_image(preprocessed_image)
        descriptor_parts = self.histogram_computer(colorspace_image)
        # for part in descriptor_parts:
            # assert isclose(part.sum(), 1.0), f"The sum was {part.sum()}"
        return np.concatenate(descriptor_parts)

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


        