from pathlib import Path
import pickle
import cv2
import numpy as np
from scipy import stats
import os
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid Qt conflicts
import matplotlib.pyplot as plt


def variance_background_removal(image: np.ndarray, thresholds: list[float]):
    assert image.shape[2] == len(thresholds), "One threshold per channel please"

    # Normalize image to [0, 1] range
    image_normalized = image.astype(np.float32) / 255.0

    height, width = image_normalized.shape[:2]

    # Store bounding boxes for each channel
    bboxes = []

    for c in range(image_normalized.shape[2]):
        threshold = thresholds[c]

        # Compute variances along each axis
        variances_h = image_normalized[:,:,c].var(axis=1)  # Variance per row
        variances_v = image_normalized[:,:,c].var(axis=0)  # Variance per column

        # Find top edge: scan from top until variance exceeds threshold
        top = 0
        for i in range(height):
            if variances_h[i] >= threshold:
                top = i
                break

        # Find bottom edge: scan from bottom until variance exceeds threshold
        bottom = height - 1
        for i in range(height - 1, -1, -1):
            if variances_h[i] >= threshold:
                bottom = i
                break

        # Find left edge: scan from left until variance exceeds threshold
        left = 0
        for j in range(width):
            if variances_v[j] >= threshold:
                left = j
                break

        # Find right edge: scan from right until variance exceeds threshold
        right = width - 1
        for j in range(width - 1, -1, -1):
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
    dataset_folder = "/media/arnau-marcos-almansa/Ubuntu Data/MCV/C1/qsd2_w2"

    queries, _ = load_queries(dataset_folder)

    # Define thresholds per channel (BGR format) - values are for [0, 1] normalized images
    # For reference: variance of 0-255 range ~= variance of 0-1 range * 255^2
    thresholds = [0.01, 0.01, 0.01]  # Adjust these values based on your needs

    # Set to True to visualize masks for each image
    visualize = True
    # Number of images to visualize (None = all)
    max_visualize = 5

    all_metrics = []

    for idx, query in enumerate(queries):
        image = query['image']
        gt_mask = query['gt_mask']

        # Generate predicted mask
        predicted_mask = variance_background_removal(image, thresholds)

        # Convert ground truth to binary (assuming it's 0 or 255)
        if len(gt_mask.shape) == 3:
            gt_mask_binary = (gt_mask[:, :, 0] > 127).astype(np.float32)
        else:
            gt_mask_binary = (gt_mask > 127).astype(np.float32)

        # Compute metrics
        metrics = compute_metrics(predicted_mask, gt_mask_binary)
        metrics['image_name'] = query['name']
        all_metrics.append(metrics)

        # Print metrics for this image
        # print(f"\n{query['name']}:")
        # print(f"  Precision: {metrics['precision']:.4f}")
        # print(f"  Recall:    {metrics['recall']:.4f}")
        # print(f"  F1-score:  {metrics['f1_score']:.4f}")
        # print(f"  mIoU:      {metrics['miou']:.4f}")

        # Visualize if requested
        if visualize:
            output_path = visualize_masks(image, predicted_mask, gt_mask_binary, query['name'])
            print(f"  Visualization saved to: {output_path}")

    # Compute average metrics
    print("\n" + "="*50)
    print("AVERAGE METRICS:")
    print("="*50)
    avg_precision = np.mean([m['precision'] for m in all_metrics])
    avg_recall = np.mean([m['recall'] for m in all_metrics])
    avg_f1 = np.mean([m['f1_score'] for m in all_metrics])
    avg_miou = np.mean([m['miou'] for m in all_metrics])

    print(f"Precision: {avg_precision:.4f}")
    print(f"Recall:    {avg_recall:.4f}")
    print(f"F1-score:  {avg_f1:.4f}")
    print(f"mIoU:      {avg_miou:.4f}")
    print("="*50)