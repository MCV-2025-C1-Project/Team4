import abc
import enum
from typing import Any, Callable, Protocol
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

# https://stackoverflow.com/questions/5595425/how-to-compare-floats-for-almost-equality-in-python
def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
    return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)



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


class IdentityImageBlockSplitter(ImageBlockSplitter):
    def __call__(self, image: np.ndarray) -> list[np.ndarray]:
        return [image]
    
    def to_dict(self):
        d = {'class': self.__class__.__name__}
        return d


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
                block = np.expand_dims(image, 2)
            
            for c in self.channels:
                hist = np.histogram(block[:, :, c], bins=self.bins, weights=weight_block, range=self.range_)[0]
                if weight_block is None:
                    hist = hist / (block.shape[0] * block.shape[1])
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
                    hist_2d = hist_2d / hist_2d.sum()
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
                    hist_3d = hist_3d / hist_3d.sum()
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


class ImageDescriptorMaker:
    def __init__(self, *, histogram_computer: HistogramComputer, gamma_correction: float, blur_image: False | Callable[[np.ndarray], np.ndarray], color_spaces: list[ColorSpace]):
        
        # assert keep_or_discard is None or len(color_spaces) == len(keep_or_discard)

        self.histogram_computer = histogram_computer
        self.gamma_correction = gamma_correction
        self.blur_image = blur_image
        self.color_spaces = color_spaces


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
                case ColorSpace.YCRCB:
                    converted = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
                case ColorSpace.HLS:
                    converted = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
                case ColorSpace.CMYK:
                    converted = bgr_to_cmyk(image)
                case ColorSpace.LUV:
                    converted = cv2.cvtColor(image, cv2.COLOR_BGR2LUV)
                case ColorSpace.XYZ:
                    converted = cv2.cvtColor(image, cv2.COLOR_BGR2XYZ)
                case ColorSpace.YUV:
                    converted = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
                case _:
                    raise ValueError(f"Unknown color space: {color_space}.")

            channel_images.append(converted)

        return np.concatenate(channel_images, axis=2)

    def make_descriptor(self, image: np.ndarray) -> np.ndarray:
        image = image.astype(np.float32) / 255
        image = image ** self.gamma_correction
        if self.blur_image:
            image = self.blur_image(image)

        colorspace_image = self.generate_colorspaces_image(image)
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