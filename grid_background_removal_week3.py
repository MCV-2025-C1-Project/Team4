from pathlib import Path
import pickle
import cv2
import numpy as np
from scipy import stats
import os
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid Qt conflicts
import matplotlib.pyplot as plt
import itertools
from typing import Iterator
import traceback


def split_if_two_paintings(img: np.ndarray, debug=False, grad_valley_thresh=8.5, valley_width_frac=0.05):
    """
    Detects if an image likely contains two paintings side by side
    by analyzing the column-wise gradient magnitude profile.
    Splits at the valley if found.
    """
    # --- Load and preprocess ---
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, _ = img.shape

    # Convert to grayscale and normalize
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 1)

    # --- Compute Sobel gradients ---
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = cv2.magnitude(gx, gy)

    # --- Column-wise average gradient (vertical projection) ---
    col_profile = grad_mag.mean(axis=0)

    # --- Smooth the profile ---
    smooth_profile = cv2.GaussianBlur(col_profile.reshape(1, -1), (1, 99), 0).flatten()

    # Normalize for plotting and relative thresholding
    profile_norm = smooth_profile / (smooth_profile.max() + 1e-6) * 100

    # --- Detect valley near the center ---
    center_range = (int(w * 0.35), int(w * 0.65))
    center_vals = profile_norm[center_range[0]:center_range[1]]
    min_idx_rel = np.argmin(center_vals)
    min_val = center_vals[min_idx_rel]
    split_col = center_range[0] + min_idx_rel

    # Check if this valley is deep enough
    # Compute local mean around it
    valley_half_width = int(w * valley_width_frac)
    left_mean = np.mean(profile_norm[:split_col - valley_half_width])
    right_mean = np.mean(profile_norm[split_col + valley_half_width:])
    mean_side = (left_mean + right_mean) / 2

    # Condition for two paintings
    is_two = (min_val < grad_valley_thresh) and (min_val < 0.5 * mean_side)

    if debug:
        import matplotlib.pyplot as plt
        img_show = img.copy()
        cv2.imshow("Image", cv2.cvtColor(img_show, cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        plt.figure(figsize=(10,4))
        plt.plot(profile_norm, label="Smoothed column gradient profile")
        plt.axvline(split_col, color='r', linestyle='--', label=f"Candidate split @ {split_col}")
        plt.title(f"Valley min={min_val:.2f}, side mean={mean_side:.2f}, Two paintings={is_two}")
        plt.legend()
        plt.waitforbuttonpress()
        plt.close()
        # plt.show()

        img_show = img.copy()
        cv2.line(img_show, (split_col, 0), (split_col, h), (255, 0, 0), 3)
        cv2.imshow("Detected Split", cv2.cvtColor(img_show, cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # --- Return results ---
    if not is_two:
        return [img]  # single painting

    # Split into two
    left_img = img[:, :split_col]
    right_img = img[:, split_col:]
    return [left_img, right_img]


def convert_to_colorspace(image: np.ndarray, colorspace: str) -> np.ndarray:
    """
    Convert BGR image to specified color space.
    Returns normalized image in [0, 1] range.
    """
    image_float = image.astype(np.float32) / 255.0

    if colorspace == 'BGR':
        return image_float
    elif colorspace == 'RGB':
        return cv2.cvtColor(image_float, cv2.COLOR_BGR2RGB)
    elif colorspace == 'GRAY':
        gray = cv2.cvtColor(image_float, cv2.COLOR_BGR2GRAY)
        return np.expand_dims(gray, axis=2)
    elif colorspace == 'HSV':
        return cv2.cvtColor(image_float, cv2.COLOR_BGR2HSV)
    elif colorspace == 'LAB':
        return cv2.cvtColor(image_float, cv2.COLOR_BGR2LAB)
    elif colorspace == 'LUV':
        return cv2.cvtColor(image_float, cv2.COLOR_BGR2LUV)
    elif colorspace == 'YCRCB':
        return cv2.cvtColor(image_float, cv2.COLOR_BGR2YCrCb)
    elif colorspace == 'HLS':
        return cv2.cvtColor(image_float, cv2.COLOR_BGR2HLS)
    elif colorspace == 'YUV':
        return cv2.cvtColor(image_float, cv2.COLOR_BGR2YUV)
    elif colorspace == 'XYZ':
        return cv2.cvtColor(image_float, cv2.COLOR_BGR2XYZ)
    else:
        raise ValueError(f"Unknown colorspace: {colorspace}")


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

    # Generate all combinations
    for combo in channel_combinations:
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
                splitted_images = split_if_two_paintings(image)

                # Compute widths and cumulative offsets of sub-images to map to GT
                widths = [si.shape[1] for si in splitted_images]  # each si is an image (RGB from split function)
                cum_widths = np.cumsum([0] + widths)  # start offsets: length = len(widths)+1

                # To store metrics of all parts of this query
                query_metrics_parts = []

                # === NEW STEP 2: Handle each (sub)painting independently ===
                for idx, sub_image in enumerate(splitted_images):
                    # Map sub-image columns back to original GT mask using cumulative widths
                    x_start = int(cum_widths[idx])
                    x_end = int(cum_widths[idx + 1])  # exclusive

                    # If GT mask is full-image, slice the correct columns; otherwise assume gt_mask already matches sub-image
                    if gt_mask is None:
                        raise ValueError(f"No gt_mask for image {image_name}")
                    # Ensure gt_mask width matches original image width
                    if gt_mask.shape[1] == image.shape[1]:
                        gt_mask_sub = gt_mask[:, x_start:x_end]
                    else:
                        # If gt_mask already corresponds to sub-image sizes (unlikely), fall back to using whole gt_mask
                        gt_mask_sub = gt_mask

                    # Convert sub_image back to BGR because variance_background_removal expects BGR input
                    # split_if_two_paintings returns images in RGB (it converts internally), so convert back
                    sub_bgr = cv2.cvtColor(sub_image, cv2.COLOR_RGB2BGR)

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
