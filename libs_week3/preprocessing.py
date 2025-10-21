from typing import Protocol
import numpy as np


class ImagePreprocessStep(Protocol):
    def __call__(self, image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pass
    
    def to_dict(self) -> dict:
        pass

class Preprocess(ImagePreprocessStep):
    def __init__(self, steps: list[ImagePreprocessStep]):
        super().__init__()
        self.steps = steps

    def __call__(self, image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Apply a sequence of preprocessing steps to the image and mask.

        Args:
            image: Input image
            mask: Input mask

        Returns:
            Tuple of (processed_image, processed_mask)
        """
        for step in self.steps:
            image, mask = step(image, mask)
        return image, mask

    def to_dict(self) -> dict:
        return {
            'class': self.__class__.__name__,
            'steps': [step.to_dict() for step in self.steps]
        }


class ApplyGamma(ImagePreprocessStep):
    def __init__(self, gamma: float):
        super().__init__()
        self.gamma = gamma

    def __call__(self, image, mask):
        return image ** self.gamma, mask

    def to_dict(self) -> dict:
        return {
            'class': self.__class__.__name__,
            'gamma': self.gamma
        }

class OpenMask(ImagePreprocessStep):
    def __init__(self, remove_side_ratio: float):
        super().__init__()
        self.remove_side_ratio = remove_side_ratio

    def __call__(self, image, mask):
        if mask is None or self.remove_side_ratio <= 0:
            return image, mask

        # Ensure mask is binary (0 or 255)
        if mask.dtype != np.uint8:
            mask = (mask * 255).astype(np.uint8)

        # Make sure outside of image is 0
        # (This handles masks that might touch borders)
        opened_mask = mask.copy()

        # Find the bounding box of white pixels
        coords = np.argwhere(opened_mask > 127)

        if coords.size == 0:
            # Empty mask, return as is
            return image, mask

        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)

        # Calculate the amount to erode from each side
        height = y_max - y_min + 1
        width = x_max - x_min + 1

        erode_y = int(height * self.remove_side_ratio)
        erode_x = int(width * self.remove_side_ratio)

        # Create new mask with erosion applied
        opened_mask[:] = 0  # Start with all black

        # Calculate new bounds (eroded from all sides)
        new_y_min = y_min + erode_y
        new_y_max = y_max - erode_y
        new_x_min = x_min + erode_x
        new_x_max = x_max - erode_x

        # Only fill if there's still a valid region
        if new_y_min < new_y_max and new_x_min < new_x_max:
            opened_mask[new_y_min:new_y_max+1, new_x_min:new_x_max+1] = 255

        return image, opened_mask

    def to_dict(self) -> dict:
        return {
            'class': self.__class__.__name__,
            'remove_side_ratio': self.remove_side_ratio
        }


class CropToMask(ImagePreprocessStep):
    def __init__(self):
        super().__init__()

    def __call__(self, image, mask):
        """
        Crop the image and mask to the bounding box of white content in the mask.

        Args:
            image: Input image
            mask: Binary mask (usually a white rectangle)

        Returns:
            Tuple of (cropped_image, cropped_mask)
        """
        if mask is None:
            return image, mask

        # Ensure mask is binary
        if mask.dtype != np.uint8:
            mask_binary = (mask * 255).astype(np.uint8)
        else:
            mask_binary = mask

        # Find the bounding box of white pixels
        coords = np.argwhere(mask_binary > 127)

        if coords.size == 0:
            # Empty mask, return original
            return image, mask

        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)

        # Crop both image and mask to the bounding box
        cropped_image = image[y_min:y_max+1, x_min:x_max+1]
        cropped_mask = mask[y_min:y_max+1, x_min:x_max+1]

        return cropped_image, cropped_mask

    def to_dict(self) -> dict:
        return {
            'class': self.__class__.__name__
        }

class Crop(ImagePreprocessStep):
    def __init__(self, remove_side_ratio: float):
        assert 0 <= remove_side_ratio < 0.5, "remove_side_ratio must be in [0, 0.5)"
        self.remove_side_ratio = remove_side_ratio

    def __call__(self, image, mask):
        h, w = image.shape[:2]

        dh = int(h * self.remove_side_ratio)
        dw = int(w * self.remove_side_ratio)

        cropped_image = image[dh:h - dh, dw:w - dw, ...]
        cropped_mask = mask[dh:h - dh, dw:w - dw, ...] if mask is not None else None

        return cropped_image, cropped_mask
