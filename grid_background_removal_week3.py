from pathlib import Path
import pickle
import cv2
import numpy as np
from scipy import stats
import os
import matplotlib

import team2_segmentation

matplotlib.use('Agg')  # Use non-interactive backend to avoid Qt conflicts
import matplotlib.pyplot as plt
import itertools
from typing import Iterator, Protocol, List, Literal
import traceback
from enum import Enum


class SplitCase(Enum):
    """Enum representing the different painting split cases."""
    SINGLE = "single"
    HORIZONTAL = "horizontal"  # Left-to-right split
    VERTICAL = "vertical"      # Top-to-bottom split


class SplitCaseDetector(Protocol):
    """Protocol for detecting IF and WHAT TYPE of split exists in an image."""

    def detect(self, img: np.ndarray) -> SplitCase:
        """
        Detect which split case applies to the image.

        Args:
            img: BGR image

        Returns:
            SplitCase enum indicating SINGLE, HORIZONTAL, or VERTICAL
        """
        ...


class ImageSplitter(Protocol):
    """Protocol for determining WHERE to split an image for a given split case."""

    def split(self, img: np.ndarray, case: SplitCase, debug: bool = False) -> List[np.ndarray]:
        """
        Split the image according to the specified case.

        Args:
            img: BGR image
            case: The type of split to perform
            debug: If True, display debug visualizations

        Returns:
            List of sub-images in RGB format
        """
        ...


class GradientBasedCaseDetector:
    """
    Detects which type of split case exists in an image using gradient analysis.

    This class focuses solely on the DETECTION decision: determining IF an image
    contains two paintings and WHAT orientation they have (horizontal or vertical).
    """

    def __init__(self, grad_valley_thresh: float = 8.5, valley_width_frac: float = 0.05):
        """
        Args:
            grad_valley_thresh: Threshold for valley depth to consider as a split
            valley_width_frac: Fraction of image dimension to use as valley half-width
        """
        self.grad_valley_thresh = grad_valley_thresh
        self.valley_width_frac = valley_width_frac

    def _compute_gradient_profile(self, img: np.ndarray, axis: Literal['horizontal', 'vertical']) -> np.ndarray:
        """
        Compute gradient profile along specified axis.

        Args:
            img: RGB image
            axis: 'horizontal' for column-wise (vertical projection) or 'vertical' for row-wise (horizontal projection)

        Returns:
            Normalized gradient profile (0-100 scale)
        """
        # Convert to grayscale and normalize
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 1)

        # Compute Sobel gradients
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = cv2.magnitude(gx, gy)

        # Compute profile along specified axis
        if axis == 'horizontal':
            # Column-wise average (for detecting vertical splits between left/right paintings)
            profile = grad_mag.mean(axis=0)
            kernel_size = (1, 99)
        else:  # vertical
            # Row-wise average (for detecting horizontal splits between top/bottom paintings)
            profile = grad_mag.mean(axis=1)
            kernel_size = (99, 1)

        # Smooth the profile
        smooth_profile = cv2.GaussianBlur(profile.reshape(kernel_size), kernel_size, 0).flatten()

        # Normalize to 0-100 scale
        profile_norm = smooth_profile / (smooth_profile.max() + 1e-6) * 100

        return profile_norm

    def _has_valley(self, profile: np.ndarray, dimension_size: int) -> tuple[bool, float, float]:
        """
        Detect if there's a significant valley in the center of the profile.

        Args:
            profile: Normalized gradient profile
            dimension_size: Size of the dimension (width or height)

        Returns:
            Tuple of (has_split, min_value, mean_side_value)
        """
        # Look for valley in center region (35-65%)
        center_range = (int(dimension_size * 0.35), int(dimension_size * 0.65))
        center_vals = profile[center_range[0]:center_range[1]]

        if len(center_vals) == 0:
            return False, 0.0, 0.0

        min_idx_rel = np.argmin(center_vals)
        min_val = center_vals[min_idx_rel]
        split_pos = center_range[0] + min_idx_rel

        # Check if valley is deep enough
        valley_half_width = int(dimension_size * self.valley_width_frac)

        # Compute mean of sides (excluding valley region)
        left_mean = np.mean(profile[:split_pos - valley_half_width]) if split_pos - valley_half_width > 0 else min_val
        right_mean = np.mean(profile[split_pos + valley_half_width:]) if split_pos + valley_half_width < len(profile) else min_val
        mean_side = (left_mean + right_mean) / 2

        # Condition for split: valley must be deep enough
        has_split = (min_val < self.grad_valley_thresh) and (min_val < 0.5 * mean_side)

        return has_split, min_val, mean_side

    def detect(self, img: np.ndarray) -> SplitCase:
        """
        Detect which split case applies to the image.

        Strategy: Compute both horizontal and vertical gradient profiles,
        detect valleys in each, and determine which split is more prominent.

        Args:
            img: BGR image

        Returns:
            SplitCase enum (SINGLE, HORIZONTAL, or VERTICAL)
        """
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = img_rgb.shape

        # Compute gradient profiles for both axes
        horizontal_profile = self._compute_gradient_profile(img_rgb, 'horizontal')
        vertical_profile = self._compute_gradient_profile(img_rgb, 'vertical')

        # Detect valleys in both profiles
        h_has_split, h_min_val, h_mean_side = self._has_valley(horizontal_profile, w)
        v_has_split, v_min_val, v_mean_side = self._has_valley(vertical_profile, h)

        # Determine which split is more prominent based on valley depth ratio
        # Lower ratio = deeper valley = stronger signal
        h_valley_ratio = h_min_val / h_mean_side if h_mean_side > 0 else 1.0
        v_valley_ratio = v_min_val / v_mean_side if v_mean_side > 0 else 1.0

        # Decision logic
        if h_has_split and v_has_split:
            # Both detected - choose the one with deeper valley (lower ratio)
            if h_valley_ratio < v_valley_ratio:
                return SplitCase.HORIZONTAL
            else:
                return SplitCase.VERTICAL
        elif h_has_split:
            return SplitCase.HORIZONTAL
        elif v_has_split:
            return SplitCase.VERTICAL
        else:
            return SplitCase.SINGLE


class GradientBasedSplitter:
    """
    Determines WHERE to split an image using gradient analysis.

    This class focuses solely on finding the optimal split position for a given
    split case. It can use different criteria than detection (e.g., more lenient).
    """

    def __init__(self, grad_valley_thresh: float = 10.0, valley_width_frac: float = 0.05):
        """
        Args:
            grad_valley_thresh: Threshold for valley depth when finding split position
            valley_width_frac: Fraction of image dimension to use as valley half-width
        """
        self.grad_valley_thresh = grad_valley_thresh
        self.valley_width_frac = valley_width_frac

    def _compute_gradient_profile(self, img: np.ndarray, axis: Literal['horizontal', 'vertical']) -> np.ndarray:
        """
        Compute gradient profile along specified axis.

        Args:
            img: RGB image
            axis: 'horizontal' for column-wise (vertical projection) or 'vertical' for row-wise (horizontal projection)

        Returns:
            Normalized gradient profile (0-100 scale)
        """
        # Convert to grayscale and normalize
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 1)

        # Compute Sobel gradients
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = cv2.magnitude(gx, gy)

        # Compute profile along specified axis
        if axis == 'horizontal':
            # Column-wise average (for detecting vertical splits between left/right paintings)
            profile = grad_mag.mean(axis=0)
            kernel_size = (1, 99)
        else:  # vertical
            # Row-wise average (for detecting horizontal splits between top/bottom paintings)
            profile = grad_mag.mean(axis=1)
            kernel_size = (99, 1)

        # Smooth the profile
        smooth_profile = cv2.GaussianBlur(profile.reshape(kernel_size), kernel_size, 0).flatten()

        # Normalize to 0-100 scale
        profile_norm = smooth_profile / (smooth_profile.max() + 1e-6) * 100

        return profile_norm

    def _find_split_position(self, profile: np.ndarray, dimension_size: int) -> tuple[int, float, float]:
        """
        Find the best split position in the profile.

        Args:
            profile: Normalized gradient profile
            dimension_size: Size of the dimension (width or height)

        Returns:
            Tuple of (split_position, min_value, mean_side_value)
        """
        # Look for valley in center region (35-65%)
        center_range = (int(dimension_size * 0.35), int(dimension_size * 0.65))
        center_vals = profile[center_range[0]:center_range[1]]

        if len(center_vals) == 0:
            # Fallback: split in the middle
            return dimension_size // 2, 0.0, 0.0

        min_idx_rel = np.argmin(center_vals)
        min_val = center_vals[min_idx_rel]
        split_pos = center_range[0] + min_idx_rel

        # Compute mean of sides (for reporting)
        valley_half_width = int(dimension_size * self.valley_width_frac)
        left_mean = np.mean(profile[:split_pos - valley_half_width]) if split_pos - valley_half_width > 0 else min_val
        right_mean = np.mean(profile[split_pos + valley_half_width:]) if split_pos + valley_half_width < len(profile) else min_val
        mean_side = (left_mean + right_mean) / 2

        return split_pos, min_val, mean_side

    def split(self, img: np.ndarray, case: SplitCase, debug: bool = False) -> List[np.ndarray]:
        """
        Split the image according to the specified case.

        Args:
            img: BGR image
            case: The type of split to perform
            debug: If True, display debug visualizations

        Returns:
            List of sub-images (RGB format). Length is 1 for single painting, 2 for split cases.
        """
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = img_rgb.shape

        if case == SplitCase.SINGLE:
            if debug:
                print(f"Split case: SINGLE painting (no split)")
            return [img_rgb]

        elif case == SplitCase.HORIZONTAL:
            # Left-to-right split
            horizontal_profile = self._compute_gradient_profile(img_rgb, 'horizontal')
            split_col, min_val, mean_side = self._find_split_position(horizontal_profile, w)

            if debug:
                self._visualize_split(img_rgb, 'horizontal', split_col, horizontal_profile, min_val, mean_side)

            left_img = img_rgb[:, :split_col]
            right_img = img_rgb[:, split_col:]
            return [left_img, right_img]

        else:  # SplitCase.VERTICAL
            # Top-to-bottom split
            vertical_profile = self._compute_gradient_profile(img_rgb, 'vertical')
            split_row, min_val, mean_side = self._find_split_position(vertical_profile, h)

            if debug:
                self._visualize_split(img_rgb, 'vertical', split_row, vertical_profile, min_val, mean_side)

            top_img = img_rgb[:split_row, :]
            bottom_img = img_rgb[split_row:, :]
            return [top_img, bottom_img]

    def _visualize_split(self, img_rgb: np.ndarray, direction: str, split_pos: int,
                         profile: np.ndarray, min_val: float, mean_side: float):
        """Debug visualization of the detected split."""
        h, w, _ = img_rgb.shape

        # Show original image
        img_show = img_rgb.copy()
        cv2.imshow("Image", cv2.cvtColor(img_show, cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Plot gradient profile
        plt.figure(figsize=(10, 4))
        plt.plot(profile, label=f"Smoothed {direction} gradient profile")
        plt.axvline(split_pos, color='r', linestyle='--', label=f"Split @ {split_pos}")
        plt.title(f"{direction.upper()} split: Valley min={min_val:.2f}, side mean={mean_side:.2f}")
        plt.legend()
        plt.show()

        # Show split line on image
        img_show = img_rgb.copy()
        if direction == 'horizontal':
            cv2.line(img_show, (split_pos, 0), (split_pos, h), (255, 0, 0), 3)
        else:
            cv2.line(img_show, (0, split_pos), (w, split_pos), (255, 0, 0), 3)
        cv2.imshow(f"Detected {direction.upper()} Split", cv2.cvtColor(img_show, cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)
        cv2.destroyAllWindows()


class PaintingSplitPipeline:
    """
    Orchestrates detection and splitting of paintings in images.

    This class combines a SplitCaseDetector (determines IF/WHAT) and an
    ImageSplitter (determines WHERE), allowing flexible composition.
    """

    def __init__(self, detector: SplitCaseDetector, splitter: ImageSplitter):
        """
        Args:
            detector: Strategy for detecting split case
            splitter: Strategy for splitting the image
        """
        self.detector = detector
        self.splitter = splitter

    def process(self, img: np.ndarray, debug: bool = False) -> tuple[SplitCase, List[np.ndarray]]:
        """
        Detect and split an image containing one or two paintings.

        Args:
            img: BGR image
            debug: If True, display debug visualizations

        Returns:
            Tuple of (detected_case, list_of_sub_images)
        """
        case = self.detector.detect(img)
        sub_images = self.splitter.split(img, case, debug=debug)
        return case, sub_images


def split_if_two_paintings(img: np.ndarray, debug=False, grad_valley_thresh=8.5, valley_width_frac=0.05):
    """
    Detects if an image contains one or two paintings and splits accordingly.

    Handles three cases:
    - Single painting: returns [image]
    - Two paintings left-to-right: returns [left_image, right_image]
    - Two paintings top-to-bottom: returns [top_image, bottom_image]

    Args:
        img: BGR image
        debug: If True, display debug visualizations
        grad_valley_thresh: Threshold for valley depth (used for both detection and splitting)
        valley_width_frac: Fraction of dimension to use as valley half-width

    Returns:
        List of sub-images in RGB format
    """
    # Create detector and splitter with same parameters for backward compatibility
    detector = GradientBasedCaseDetector(grad_valley_thresh, valley_width_frac)
    splitter = GradientBasedSplitter(grad_valley_thresh, valley_width_frac)

    # Create pipeline and process
    pipeline = PaintingSplitPipeline(detector, splitter)
    _, sub_images = pipeline.process(img, debug=debug)

    return sub_images


import cv2
import numpy as np

def convert_to_colorspace(image: np.ndarray, colorspace: str) -> np.ndarray:
    """
    Convert BGR image to specified color space.
    Ensures output is normalized to [0, 1] using known channel ranges for each color space.
    """
    image_float = image.astype(np.float32) / 255.0

    if colorspace == 'BGR':
        converted = image_float
        ranges = [(0, 1)] * 3

    elif colorspace == 'RGB':
        converted = cv2.cvtColor(image_float, cv2.COLOR_BGR2RGB)
        ranges = [(0, 1)] * 3

    elif colorspace == 'GRAY':
        gray = cv2.cvtColor(image_float, cv2.COLOR_BGR2GRAY)
        converted = np.expand_dims(gray, axis=2)
        ranges = [(0, 1)]

    elif colorspace == 'HSV':
        # OpenCV HSV: H ∈ [0, 180], S,V ∈ [0, 1]
        converted = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
        ranges = [(0, 180), (0, 255), (0, 255)]

    elif colorspace == 'LAB':
        # L ∈ [0, 100], a,b ∈ [-128, 127]
        converted = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
        ranges = [(0, 100), (-128, 127), (-128, 127)]

    elif colorspace == 'LUV':
        # L ∈ [0, 100], u,v ∈ [-134, 220] (approx)
        converted = cv2.cvtColor(image, cv2.COLOR_BGR2Luv).astype(np.float32)
        ranges = [(0, 100), (-134, 220), (-134, 220)]

    elif colorspace == 'YCRCB':
        # Y,Cr,Cb ∈ [0, 255]
        converted = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb).astype(np.float32)
        ranges = [(0, 255)] * 3

    elif colorspace == 'HLS':
        # H ∈ [0,180], L,S ∈ [0,255]
        converted = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2HLS).astype(np.float32)
        ranges = [(0, 180), (0, 255), (0, 255)]

    elif colorspace == 'YUV':
        # Y,U,V ∈ [0,255]
        converted = cv2.cvtColor(image, cv2.COLOR_BGR2YUV).astype(np.float32)
        ranges = [(0, 255)] * 3

    elif colorspace == 'XYZ':
        # X,Y,Z ∈ [0,255] (OpenCV scaling)
        converted = cv2.cvtColor(image, cv2.COLOR_BGR2XYZ).astype(np.float32)
        ranges = [(0, 255)] * 3

    else:
        raise ValueError(f"Unknown colorspace: {colorspace}")

    # Normalize channels to [0, 1] using fixed known ranges
    for c, (lo, hi) in enumerate(ranges):
        converted[..., c] = np.clip((converted[..., c] - lo) / (hi - lo), 0, 1)

    return converted


def variance_background_removal(image: np.ndarray, channel_config: dict):
    """
    Remove background based on variance analysis.

    Args:
        image: BGR image (0-255)
        channel_config: dict with:
            - 'channels': list of (colorspace, channel_idx) tuples
            - 'threshold': variance threshold
    """
    channels_to_analyze = []

    # Extract specified channels from their color spaces
    for colorspace, channel_idx in channel_config['channels']:
        converted = convert_to_colorspace(image, colorspace)
        if channel_idx < converted.shape[2]:
            channels_to_analyze.append(converted[:, :, channel_idx])
        else:
            raise ValueError(f"Channel {channel_idx} doesn't exist in {colorspace}")

    # Stack channels into a single array
    if not channels_to_analyze:
        raise ValueError("No channels to analyze")

    height, width = channels_to_analyze[0].shape
    threshold = channel_config['threshold']

    # Store bounding boxes for each channel
    bboxes = []

    for channel in channels_to_analyze:
        # Compute variances along each axis
        variances_h = channel.var(axis=1)  # Variance per row
        variances_v = channel.var(axis=0)  # Variance per column

        # Find top edge: scan from top until variance exceeds threshold
        top = 0
        for i in range(10, height):
            if variances_h[i] >= threshold:
                top = i
                break

        # Find bottom edge: scan from bottom until variance exceeds threshold
        bottom = height - 1
        for i in range(height - 11, -1, -1):
            if variances_h[i] >= threshold:
                bottom = i
                break

        # Find left edge: scan from left until variance exceeds threshold
        left = 0
        for j in range(10, width):
            if variances_v[j] >= threshold:
                left = j
                break

        # Find right edge: scan from right until variance exceeds threshold
        right = width - 1
        for j in range(width - 11, -1, -1):
            if variances_v[j] >= threshold:
                right = j
                break

        bboxes.append((top, bottom, left, right))

    # Combine bboxes: take the intersection (most conservative)
    # This means taking the minimum foreground region across all channels
    final_top = max(bbox[0] for bbox in bboxes)
    final_bottom = min(bbox[1] for bbox in bboxes)
    final_left = max(bbox[2] for bbox in bboxes)
    final_right = min(bbox[3] for bbox in bboxes)

    # Create solid rectangular mask
    combined_mask = np.zeros((height, width), dtype=np.float32)
    if final_top <= final_bottom and final_left <= final_right:
        combined_mask[final_top:final_bottom+1, final_left:final_right+1] = 1.0

    return combined_mask


def generate_channel_configurations() -> Iterator[dict]:
    """
    Generate sensible channel configurations for background removal.
    Returns configurations that are likely to be useful.
    """
    # Define thresholds to test (for 0-1 normalized images)
    thresholds = [0.0001, 0.0005, 0.001, 0.002, 0.005, 0.01]

    # Define channel combinations to test
    channel_combinations = []

    # 1. Single color space - all channels
    channel_combinations.append({
        'name': 'RGB_all',
        'channels': [('RGB', 0), ('RGB', 1), ('RGB', 2)]
    })
    channel_combinations.append({
        'name': 'LAB_all',
        'channels': [('LAB', 0), ('LAB', 1), ('LAB', 2)]
    })
    channel_combinations.append({
        'name': 'HSV_all',
        'channels': [('HSV', 0), ('HSV', 1), ('HSV', 2)]
    })
    channel_combinations.append({
        'name': 'YCRCB_all',
        'channels': [('YCRCB', 0), ('YCRCB', 1), ('YCRCB', 2)]
    })
    # 2. Single channels (especially useful ones)
    channel_combinations.append({
        'name': 'GRAY',
        'channels': [('GRAY', 0)]
    })
    channel_combinations.append({
        'name': 'LAB_L',
        'channels': [('LAB', 0)]  # Lightness
    })
    channel_combinations.append({
        'name': 'HSV_V',
        'channels': [('HSV', 2)]  # Value
    })
    channel_combinations.append({
        'name': 'HSV_H',
        'channels': [('HSV', 0)]  # Hue
    })
    channel_combinations.append({
        'name': 'YCRCB_Y',
        'channels': [('YCRCB', 0)]  # Luma
    })
    # 3. Interesting cross-color-space combinations
    channel_combinations.append({
        'name': 'RGB+HSV_H',
        'channels': [('RGB', 0), ('RGB', 1), ('RGB', 2), ('HSV', 0)]
    })
    channel_combinations.append({
        'name': 'LAB_L+HSV_V',
        'channels': [('LAB', 0), ('HSV', 2)]
    })
    channel_combinations.append({
        'name': 'LAB_AB',
        'channels': [('LAB', 1), ('LAB', 2)]  # Color channels only
    })
    channel_combinations.append({
        'name': 'LAB_L+AB',
        'channels': [('LAB', 0), ('LAB', 1), ('LAB', 2)]
    })
    channel_combinations.append({
        'name': 'YCRCB_CrCb',
        'channels': [('YCRCB', 1), ('YCRCB', 2)]  # Chroma only
    })
    channel_combinations.append({
        'name': 'RGB+LAB_L',
        'channels': [('RGB', 0), ('RGB', 1), ('RGB', 2), ('LAB', 0)]
    })
    channel_combinations.append({
        'name': 'HSV_SV',
        'channels': [('HSV', 1), ('HSV', 2)]  # Saturation + Value
    })

    for thr in [10, 15, 20, 25, 30]:
        for pixel_border in [5, 10, 15, 20, 25]:
            for gradient_threshold in [0.05, 0.1, 0.15, 0.20, 0.25]:
                channel_combinations.append({
                    'name': 'TEAM2',
                    'channels':  [],
                    'thr': thr,
                    'pixel_border': pixel_border,
                    'gradient_threshold': gradient_threshold,
                })




    # Generate all combinations
    for combo in channel_combinations:
        if combo['name'] == 'TEAM2':
            yield {
                'name': combo['name'],
                'channels': combo['channels'],
                'threshold': combo['thr'],
                'thr': combo['thr'],
                'pixel_border': combo['pixel_border'],
                'gradient_threshold': combo['gradient_threshold'],
                'description': f"{combo['name']}, thr = {combo['thr']}, pixel_border = {combo['pixel_border']}, gradient_threshold = {combo['gradient_threshold']}"
            }
            continue

        for threshold in thresholds:
            yield {
                'name': combo['name'],
                'channels': combo['channels'],
                'threshold': threshold,
                'description': f"{combo['name']}_thresh_{threshold}"
            }


def visualize_masks(image: np.ndarray, predicted_mask: np.ndarray, gt_mask: np.ndarray, image_name: str = ""):
    """
    Visualize image, masks, masked images, and overlap analysis in a comprehensive figure.

    Args:
        image: Original image (H, W, 3) in BGR format
        predicted_mask: Predicted binary mask (H, W) with values 0 or 1
        gt_mask: Ground truth binary mask (H, W) with values 0 or 1
        image_name: Name of the image for the title
    """
    # Convert BGR to RGB for display
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Create masked images
    masked_gt = image_rgb.copy()
    masked_pred = image_rgb.copy()

    # Apply masks (set background to black)
    masked_gt[gt_mask < 0.5] = 0
    masked_pred[predicted_mask < 0.5] = 0

    # Create overlap visualization
    # True Positives (both masks agree - foreground): Green
    # True Negatives (both masks agree - background): Black
    # False Positives (predicted foreground, gt background): Red
    # False Negatives (predicted background, gt foreground): Blue

    overlap_viz = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)

    tp = (predicted_mask > 0.5) & (gt_mask > 0.5)  # True Positive - Green
    tn = (predicted_mask < 0.5) & (gt_mask < 0.5)  # True Negative - Black
    fp = (predicted_mask > 0.5) & (gt_mask < 0.5)  # False Positive - Red
    fn = (predicted_mask < 0.5) & (gt_mask > 0.5)  # False Negative - Blue

    overlap_viz[tp] = [0, 255, 0]    # Green
    overlap_viz[tn] = [0, 0, 0]      # Black
    overlap_viz[fp] = [255, 0, 0]    # Red
    overlap_viz[fn] = [0, 0, 255]    # Blue

    # Create figure with subplots
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle(f'Mask Analysis - {image_name}', fontsize=16, fontweight='bold')

    # Row 1: Original data
    axes[0, 0].imshow(image_rgb)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(gt_mask, cmap='gray')
    axes[0, 1].set_title('Ground Truth Mask')
    axes[0, 1].axis('off')

    axes[0, 2].imshow(predicted_mask, cmap='gray')
    axes[0, 2].set_title('Predicted Mask')
    axes[0, 2].axis('off')

    axes[0, 3].imshow(overlap_viz)
    axes[0, 3].set_title('Overlap Analysis\n(Green=TP, Red=FP, Blue=FN)')
    axes[0, 3].axis('off')

    # Row 2: Masked images and difference
    axes[1, 0].imshow(masked_gt)
    axes[1, 0].set_title('Image with GT Mask')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(masked_pred)
    axes[1, 1].set_title('Image with Predicted Mask')
    axes[1, 1].axis('off')

    # Intersection (both masks agree on foreground)
    intersection = (predicted_mask > 0.5) & (gt_mask > 0.5)
    axes[1, 2].imshow(intersection, cmap='gray')
    axes[1, 2].set_title('Intersection (TP)')
    axes[1, 2].axis('off')

    # Union minus intersection (parts that don't overlap)
    symmetric_diff = ((predicted_mask > 0.5) | (gt_mask > 0.5)) & ~intersection
    axes[1, 3].imshow(symmetric_diff, cmap='gray')
    axes[1, 3].set_title('Symmetric Difference (FP + FN)')
    axes[1, 3].axis('off')

    plt.tight_layout()

    # Save figure to file
    output_path = f"mask_analysis_{image_name.replace('.jpg', '')}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    return output_path


def compute_metrics(predicted_mask: np.ndarray, gt_mask: np.ndarray):
    """
    Compute precision, recall, F1-score, and mIoU for binary masks.

    Args:
        predicted_mask: Predicted binary mask (H, W) with values 0 or 1
        gt_mask: Ground truth binary mask (H, W) with values 0 or 1

    Returns:
        dict with precision, recall, f1_score, and miou
    """
    # Flatten masks
    pred_flat = predicted_mask.flatten().astype(bool)
    gt_flat = gt_mask.flatten().astype(bool)

    # True Positives, False Positives, False Negatives, True Negatives
    tp = np.sum(pred_flat & gt_flat)
    fp = np.sum(pred_flat & ~gt_flat)
    fn = np.sum(~pred_flat & gt_flat)
    tn = np.sum(~pred_flat & ~gt_flat)

    # Precision: TP / (TP + FP)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    # Recall: TP / (TP + FN)
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    # F1-score: 2 * (Precision * Recall) / (Precision + Recall)
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    # IoU for foreground class: TP / (TP + FP + FN)
    iou_foreground = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0

    # IoU for background class: TN / (TN + FP + FN)
    iou_background = tn / (tn + fp + fn) if (tn + fp + fn) > 0 else 0.0

    # mIoU: mean of foreground and background IoU
    miou = (iou_foreground + iou_background) / 2.0

    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'miou': miou,
        'iou_foreground': iou_foreground,
        'iou_background': iou_background
    }


# Load query images and ground truth from the provided queries_path.
def load_queries(queries_path: str):
    queries = []
    gt_path = os.path.join(queries_path, "gt_corresps.pkl")
    if os.path.exists(gt_path):
        gt = pickle.load(open(gt_path, 'rb'))
    else:
        gt = None
    
    for filename in sorted(os.listdir(queries_path)):
        if not filename.endswith(".jpg"):
            continue

        image_path = os.path.join(queries_path, filename)
        image = cv2.imread(image_path)
        
        gt_mask_path = Path(image_path).with_suffix(".png")
        gt_mask = cv2.imread(gt_mask_path)
        
        queries.append({
            'image': image,
            'name': filename,
            'id': int(Path(image_path).stem),
            'gt_mask': gt_mask
        })

    return queries, gt


if __name__ == '__main__':
    dataset_folder = "/media/arnau-marcos-almansa/Ubuntu Data/MCV/C1/qsd2_w3_denoised"
    test_dataset_folder = "/media/arnau-marcos-almansa/Ubuntu Data/MCV/C1/qst2_w3_denoised"
    # dataset_folder = "/media/arnau-marcos-almansa/Ubuntu Data/MCV/C1/qsd2_w2"

    print(f"running {len(list(generate_channel_configurations()))} tests.")

    queries, _ = load_queries(dataset_folder)

    visualize_best = True
    max_visualize = 3

    print("Starting grid search for background removal...")
    print(f"Total images: {len(queries)}")

    all_results = []

    for config in generate_channel_configurations():
        config_metrics = []

        for query in queries:
            image = query['image']          # BGR image loaded by cv2.imread
            gt_mask = query['gt_mask']
            image_name = query['name']

            try:
                # === NEW STEP 1: Split if two paintings ===
                detector = GradientBasedCaseDetector()
                splitter = GradientBasedSplitter()
                pipeline = PaintingSplitPipeline(detector, splitter)
                split_case, splitted_images = pipeline.process(image)

                # To store metrics of all parts of this query
                query_metrics_parts = []

                # === NEW STEP 2: Handle each (sub)painting independently ===
                for idx, sub_image in enumerate(splitted_images):
                    # Map sub-image to original GT mask based on split type
                    if gt_mask is None:
                        raise ValueError(f"No gt_mask for image {image_name}")

                    # Slice GT mask according to split type
                    if split_case == SplitCase.SINGLE:
                        gt_mask_sub = gt_mask
                    elif split_case == SplitCase.HORIZONTAL:
                        # Left-to-right split: slice by columns
                        widths = [si.shape[1] for si in splitted_images]
                        cum_widths = np.cumsum([0] + widths)
                        x_start = int(cum_widths[idx])
                        x_end = int(cum_widths[idx + 1])
                        gt_mask_sub = gt_mask[:, x_start:x_end]
                    elif split_case == SplitCase.VERTICAL:
                        # Top-to-bottom split: slice by rows
                        heights = [si.shape[0] for si in splitted_images]
                        cum_heights = np.cumsum([0] + heights)
                        y_start = int(cum_heights[idx])
                        y_end = int(cum_heights[idx + 1])
                        gt_mask_sub = gt_mask[y_start:y_end, :]
                    else:
                        gt_mask_sub = gt_mask

                    # Convert sub_image back to BGR because variance_background_removal expects BGR input
                    # split_if_two_paintings returns images in RGB (it converts internally), so convert back
                    sub_bgr = cv2.cvtColor(sub_image, cv2.COLOR_RGB2BGR)

                    if config['name'] == 'TEAM2':
                        predicted_mask = team2_segmentation.create_mask_from_gradient(image, thr=config['thr'], pixel_border=config['pixel_border'], gradient_threshold=config['gradient_threshold'])
                    else:
                        # Compute mask for this subimage
                        predicted_mask = variance_background_removal(sub_bgr, config)

                    # Convert GT to binary (handle 3-channel or single-channel GT)
                    if len(gt_mask_sub.shape) == 3:
                        gt_mask_binary = (gt_mask_sub[:, :, 0] > 127).astype(np.float32)
                    else:
                        gt_mask_binary = (gt_mask_sub > 127).astype(np.float32)

                    # If predicted_mask size does not match gt_mask_sub (should not happen), resize predicted_mask to match
                    if predicted_mask.shape != gt_mask_binary.shape:
                        predicted_mask = cv2.resize(predicted_mask, (gt_mask_binary.shape[1], gt_mask_binary.shape[0]),
                                                    interpolation=cv2.INTER_NEAREST)

                    # Compute metrics for this part
                    metrics = compute_metrics(predicted_mask, gt_mask_binary)
                    query_metrics_parts.append(metrics)

                # === NEW STEP 3: Average metrics across parts (if more than one part) ===
                avg_query_metrics = {
                    k: np.mean([m[k] for m in query_metrics_parts])
                    for k in query_metrics_parts[0].keys()
                }
                config_metrics.append(avg_query_metrics)

            except Exception as e:
                print(f"Error with config {config['description']} on {query['name']}: {e}")
                continue


        # === Existing code: Average across all queries for this config ===
        if config_metrics:
            avg_metrics = {
                'config': config['description'],
                'name': config['name'],
                'threshold': config['threshold'],
                'channels': str(config['channels']),
                'precision': np.mean([m['precision'] for m in config_metrics]),
                'recall': np.mean([m['recall'] for m in config_metrics]),
                'f1_score': np.mean([m['f1_score'] for m in config_metrics]),
                'miou': np.mean([m['miou'] for m in config_metrics]),
            }
            all_results.append(avg_metrics)

            print(f"{config['description']:40s} | mIoU: {avg_metrics['miou']:.4f} | "
                  f"F1: {avg_metrics['f1_score']:.4f} | "
                  f"Precision: {avg_metrics['precision']:.4f} | "
                  f"Recall: {avg_metrics['recall']:.4f}")


    # === Sort and summarize results ===
    all_results.sort(key=lambda x: x['miou'], reverse=True)

    print("\n" + "="*100)
    print("TOP 10 CONFIGURATIONS (sorted by mIoU):")
    print("="*100)
    for i, result in enumerate(all_results[:10], 1):
        print(f"{i:2d}. {result['config']:40s} | "
              f"mIoU: {result['miou']:.4f} | "
              f"F1: {result['f1_score']:.4f} | "
              f"P: {result['precision']:.4f} | "
              f"R: {result['recall']:.4f}")

    # === Save results to CSV ===
    import pandas as pd
    df = pd.DataFrame(all_results)
    csv_path = "background_removal_grid_search_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n✅ Results saved to: {csv_path}")

    # === Visualize best configuration ===
    if visualize_best and all_results:
        best_config = all_results[0]
        print(f"\n{'='*100}")
        print(f"Visualizing BEST configuration: {best_config['config']}")
        print(f"{'='*100}")

        # Reconstruct config dict
        best_config_dict = None
        for config in generate_channel_configurations():
            if config["description"] == best_config["config"]:
                best_config_dict = config
                break

        if best_config_dict:
            # queries, _ = load_queries(test_dataset_folder)

            for idx, query in enumerate(queries):
                image = query["image"]
                gt_mask = query["gt_mask"]
                image_name = query["name"]

                # --- Split if needed ---
                detector = GradientBasedCaseDetector()
                splitter = GradientBasedSplitter()
                pipeline = PaintingSplitPipeline(detector, splitter)
                split_case, splitted_images = pipeline.process(image)

                generated_mask_parts = []

                for part_idx, sub_image in enumerate(splitted_images):
                    # Compute corresponding GT region based on split type
                    if split_case == SplitCase.SINGLE:
                        gt_mask_sub = gt_mask
                    elif split_case == SplitCase.HORIZONTAL:
                        # Left-to-right split: slice by columns
                        widths = [si.shape[1] for si in splitted_images]
                        cum_widths = np.cumsum([0] + widths)
                        x_start = int(cum_widths[part_idx])
                        x_end = int(cum_widths[part_idx + 1])
                        gt_mask_sub = gt_mask[:, x_start:x_end]
                    elif split_case == SplitCase.VERTICAL:
                        # Top-to-bottom split: slice by rows
                        heights = [si.shape[0] for si in splitted_images]
                        cum_heights = np.cumsum([0] + heights)
                        y_start = int(cum_heights[part_idx])
                        y_end = int(cum_heights[part_idx + 1])
                        gt_mask_sub = gt_mask[y_start:y_end, :]
                    else:
                        gt_mask_sub = gt_mask

                    sub_bgr = cv2.cvtColor(sub_image, cv2.COLOR_RGB2BGR)

                    if best_config_dict['name'] == 'TEAM2':
                        predicted_mask = team2_segmentation.create_mask_from_gradient(image, thr=best_config_dict['thr'], pixel_border=best_config_dict['pixel_border'], gradient_threshold=best_config_dict['gradient_threshold'])
                    else:
                        # Compute mask for this subimage
                        predicted_mask = variance_background_removal(sub_bgr, best_config_dict)

                    generated_mask_parts.append(predicted_mask)

                    if len(gt_mask_sub.shape) == 3:
                        gt_mask_binary = (gt_mask_sub[:, :, 0] > 127).astype(np.float32)
                    else:
                        gt_mask_binary = (gt_mask_sub > 127).astype(np.float32)

                    # Resize predicted mask if needed
                    if predicted_mask.shape != gt_mask_binary.shape:
                        predicted_mask = cv2.resize(
                            predicted_mask,
                            (gt_mask_binary.shape[1], gt_mask_binary.shape[0]),
                            interpolation=cv2.INTER_NEAREST,
                        )

                    # Visualize each sub-painting separately
                    suffix = f"{image_name}_part{part_idx+1}_{best_config['name']}"
                    output_path = visualize_masks(sub_bgr, predicted_mask, gt_mask_binary, suffix)
                    print(f"  Saved: {output_path}")

                # Concatenate mask parts based on split type
                if split_case == SplitCase.HORIZONTAL:
                    mask = np.hstack(generated_mask_parts)
                elif split_case == SplitCase.VERTICAL:
                    mask = np.vstack(generated_mask_parts)
                else:  # SINGLE
                    mask = generated_mask_parts[0] if generated_mask_parts else np.zeros_like(gt_mask[:, :, 0])

                name = "generated_masks/" + image_name.split(".")[0] + ".png"
                cv2.imwrite(name, (mask * 255).astype(np.uint8))

        else:
            print("⚠️ Could not reconstruct best configuration — skipping visualization.")
