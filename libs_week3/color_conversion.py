import cv2
import numpy as np
from libs_week3.descriptor import ColorSpace
from libs_week3.preprocessing import ImagePreprocessStep


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



class ColorConversion(ImagePreprocessStep):
    def __init__(self, targets: list[ColorSpace], normalize: bool):
        super().__init__()
        self.targets = targets
        self.normalize = normalize

    def generate_colorspaces_image(self, image: np.ndarray) -> np.ndarray:
        assert np.issubdtype(image.dtype, np.unsignedinteger), "Image must be of unsigned integer type."
        
        channel_images = []
        for color_space in self.targets:
            match color_space:
                case ColorSpace.RGB:
                    converted = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    if self.normalize:
                        converted = converted.astype(np.float32) / 255.0
                case ColorSpace.HSV:
                    converted = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                    if self.normalize:
                        # H∈[0,179], S∈[0,255], V∈[0,255] in OpenCV
                        H, S, V = cv2.split(converted.astype(np.float32))
                        H = H / 179.0
                        S = S / 255.0
                        V = V / 255.0
                        converted = cv2.merge([H, S, V])
                case ColorSpace.LAB:
                    converted = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
                    if self.normalize:
                        # L∈[0,255], a∈[0,255], b∈[0,255] in OpenCV (scaled from L∈[0,100], a∈[-128,127], b∈[-128,127])
                        L, a, b = cv2.split(converted.astype(np.float32))
                        L = L / 255.0
                        a = a / 255.0
                        b = b / 255.0
                        converted = cv2.merge([L, a, b])
                case ColorSpace.YCRCB:
                    converted = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
                    if self.normalize:
                        # Y∈[0,255], Cr∈[0,255], Cb∈[0,255]
                        converted = converted.astype(np.float32) / 255.0
                case ColorSpace.HLS:
                    converted = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
                    if self.normalize:
                        # H∈[0,179], L∈[0,255], S∈[0,255] in OpenCV
                        H, L, S = cv2.split(converted.astype(np.float32))
                        H = H / 179.0
                        L = L / 255.0
                        S = S / 255.0
                        converted = cv2.merge([H, L, S])
                case ColorSpace.CMYK:
                    converted = bgr_to_cmyk(image.astype(np.float32) / 255.0)
                    # CMYK already in [0,1] range from bgr_to_cmyk
                    if not self.normalize:
                        # Scale back to [0,255] if not normalizing
                        converted = (converted * 255.0).astype(np.uint8)
                case ColorSpace.LUV:
                    converted = cv2.cvtColor(image, cv2.COLOR_BGR2Luv)
                    if self.normalize:
                        # L∈[0,255], u∈[0,255], v∈[0,255] in OpenCV
                        converted = converted.astype(np.float32) / 255.0
                case ColorSpace.XYZ:
                    converted = cv2.cvtColor(image, cv2.COLOR_BGR2XYZ)
                    if self.normalize:
                        # X∈[0,255], Y∈[0,255], Z∈[0,255] in OpenCV
                        converted = converted.astype(np.float32) / 255.0
                case ColorSpace.YUV:
                    converted = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
                    if self.normalize:
                        # Y∈[0,255], U∈[0,255], V∈[0,255] in OpenCV
                        converted = converted.astype(np.float32) / 255.0
                case _:
                    raise ValueError(f"Unknown color space: {color_space}.")

            channel_images.append(converted)

        return np.concatenate(channel_images, axis=2)

    def __call__(self, image, mask):
        return self.generate_colorspaces_image(image), mask

    def to_dict(self):
        d = super().to_dict()
        d['targets'] = [colorspace.value for colorspace in self.targets]
        d['normalize'] = self.normalize
        return d
