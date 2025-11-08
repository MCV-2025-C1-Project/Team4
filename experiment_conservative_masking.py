#!/usr/bin/env python3
"""
Quick experiment to evaluate if conservative segmentation masking helps reduce
keypoint computation without losing painting information.

This script:
1. Loads a few query images
2. Tests keypoint detection with and without conservative masks
3. Reports potential speedup and visualizes results

Usage:
    python experiment_conservative_masking.py <database_path> <queries_path>
"""

import argparse
import os
import sys
from pathlib import Path
import cv2
import numpy as np
import pickle
from typing import List, Tuple

from libs_week3.preprocessing import CropToMask

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

from libs_week4.database import ImageDatabase
from libs_week4.descriptor import ORBDescriptor, RootSIFTDescriptor, KeypointAndDescriptorMaker
from libs_week3.color_conversion import ColorConversion, ColorSpace
import libs_week3.preprocessing as preprocessing
import grid_background_removal_week3


def get_conservative_painting_mask(image: np.ndarray, dilation_kernel_size: int = 31, iterations: int = 2) -> np.ndarray:
    """
    Create a conservative mask that captures 100% of the painting with aggressive dilation.

    Args:
        image: Input image (BGR, float32 [0,1])
        dilation_kernel_size: Size of dilation kernel (larger = more conservative)
        iterations: Number of dilation iterations

    Returns:
        Binary mask (uint8, 0 or 255)
    """
    # Use existing variance-based segmentation
    config = {'name': 'HSV_SV', 'channels': [('HSV', 1), ('HSV', 2)], 'threshold': 0.005}
    base_mask = grid_background_removal_week3.variance_background_removal(image, config).astype(np.uint8) * 255

    # Aggressive dilation to ensure NO painting pixels are lost
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_kernel_size, dilation_kernel_size))
    dilated_mask = cv2.dilate(base_mask, kernel, iterations=iterations)

    # Optional: Morphological closing to fill internal gaps
    closed_mask = cv2.morphologyEx(dilated_mask, cv2.MORPH_CLOSE, kernel)

    return closed_mask


def load_test_queries(queries_path: str, num_queries: int = 5) -> List[Tuple[np.ndarray, str]]:
    """Load a few query images for testing."""
    queries = []

    for i, filename in enumerate(sorted(os.listdir(queries_path))):
        if i >= num_queries:
            break

        if not filename.endswith(".jpg"):
            continue

        image_path = os.path.join(queries_path, filename)
        image = cv2.imread(image_path)

        if image is None:
            continue

        # # Convert to float [0, 1]
        # image = image.astype(np.float32) / 255.0

        # Split if multiple paintings
        imgs = grid_background_removal_week3.split_if_two_paintings(image)

        for idx, img in enumerate(imgs):
            queries.append((img, f"{filename}_part{idx}"))

    return queries


def check_keypoint_reduction(descriptor_maker: KeypointAndDescriptorMaker,
                            query_image: np.ndarray,
                            conservative_mask: np.ndarray) -> dict:
    """
    Test keypoint detection with and without mask.

    Returns:
        dict with statistics about keypoint reduction
    """
    # Without mask
    kp_no_mask, desc_no_mask = descriptor_maker.detect_and_compute(query_image, mask=None)


    crop_to_mask = CropToMask()

    cropped_query_image, cropped_mask = crop_to_mask(query_image, conservative_mask)

    # With conservative mask
    kp_with_mask, desc_with_mask = descriptor_maker.detect_and_compute(cropped_query_image, mask=cropped_mask)

    count_no_mask = len(kp_no_mask) if kp_no_mask else 0
    count_with_mask = len(kp_with_mask) if kp_with_mask else 0

    reduction_pct = 0.0 if count_no_mask == 0 else (1 - count_with_mask / count_no_mask) * 100
    speedup_factor = count_no_mask / count_with_mask if count_with_mask > 0 else 1.0

    return {
        'keypoints_no_mask': count_no_mask,
        'keypoints_with_mask': count_with_mask,
        'reduction_percent': reduction_pct,
        'speedup_factor': speedup_factor,
        'descriptors_no_mask': len(desc_no_mask) if desc_no_mask is not None else 0,
        'descriptors_with_mask': len(desc_with_mask) if desc_with_mask is not None else 0
    }


def visualize_mask_effect(image: np.ndarray, mask: np.ndarray,
                         kp_no_mask: List, kp_with_mask: List,
                         output_path: str):
    """Create visualization showing mask and keypoint reduction."""
    # Convert image back to uint8 for visualization
    img_vis = (image * 255).astype(np.uint8)

    # Create side-by-side comparison
    h, w = img_vis.shape[:2]

    # Panel 1: Original image with all keypoints
    panel1 = cv2.drawKeypoints(img_vis, kp_no_mask, None, color=(0, 255, 0))

    # Panel 2: Mask overlay
    mask_colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    panel2 = cv2.addWeighted(img_vis, 0.6, mask_colored, 0.4, 0)

    # Panel 3: Masked keypoints
    panel3 = cv2.drawKeypoints(img_vis, kp_with_mask, None, color=(0, 0, 255))

    # Stack panels
    top_row = np.hstack([panel1, panel2])
    bottom_row = np.hstack([panel3, np.zeros_like(panel3)])

    combined = np.vstack([top_row, bottom_row])

    # Add text labels
    cv2.putText(combined, f"No Mask: {len(kp_no_mask)} kpts", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(combined, "Conservative Mask", (w + 10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(combined, f"With Mask: {len(kp_with_mask)} kpts", (10, h + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imwrite(output_path, combined)


def main():
    parser = argparse.ArgumentParser(description="Conservative masking experiment")
    parser.add_argument("queries_path", type=str, help="Path to query images")
    parser.add_argument("--num_queries", type=int, default=5, help="Number of queries to test")
    parser.add_argument("--visualize", action="store_true", help="Save visualization images")
    parser.add_argument("--output_dir", type=str, default="masking_experiment_results",
                       help="Directory for visualization outputs")
    args = parser.parse_args()

    if args.visualize:
        os.makedirs(args.output_dir, exist_ok=True)

    print("="*60)
    print("CONSERVATIVE MASKING EXPERIMENT")
    print("="*60)
    print(f"\nLoading {args.num_queries} query images from {args.queries_path}...")

    queries = load_test_queries(args.queries_path, args.num_queries)

    if not queries:
        print("ERROR: No query images found!")
        return

    print(f"Loaded {len(queries)} query image(s)\n")

    # Test with best-performing descriptors from previous experiments
    descriptors_to_test = [
        ("ORB (n_features=3000)", ORBDescriptor(n_features=3000, scale_factor=1.2, n_levels=10)),
        ("RootSIFT (n_features=1000)", RootSIFTDescriptor(n_features=1000, contrast_threshold=0.04, sigma=2.0))
    ]

    for desc_name, descriptor in descriptors_to_test:
        print(f"\n{'='*60}")
        print(f"Testing: {desc_name}")
        print(f"{'='*60}\n")

        descriptor_maker = KeypointAndDescriptorMaker(
            descriptor_computer=descriptor,
            color_conversion=ColorConversion(targets=[ColorSpace.BGR], normalize=False),
            preprocess=preprocessing.Preprocess([])
        )

        all_results = []

        for idx, (query_img, query_name) in enumerate(queries):
            print(f"Query {idx+1}/{len(queries)}: {query_name}")

            # Generate conservative mask
            conservative_mask = get_conservative_painting_mask(query_img)

            # Test keypoint reduction
            result = check_keypoint_reduction(descriptor_maker, query_img, conservative_mask)
            all_results.append(result)

            print(f"  Keypoints: {result['keypoints_no_mask']:4d} (no mask) -> "
                  f"{result['keypoints_with_mask']:4d} (with mask)")
            print(f"  Reduction: {result['reduction_percent']:5.1f}%")
            print(f"  Potential speedup: {result['speedup_factor']:.2f}x\n")

            # Visualize if requested
            if args.visualize:
                kp_no_mask, _ = descriptor_maker.detect_and_compute(query_img, mask=None)
                kp_with_mask, _ = descriptor_maker.detect_and_compute(query_img, mask=conservative_mask)

                safe_name = query_name.replace('/', '_').replace('\\', '_')
                vis_path = os.path.join(args.output_dir,
                                       f"{desc_name.replace(' ', '_')}_{safe_name}.jpg")
                visualize_mask_effect(query_img, conservative_mask,
                                     kp_no_mask, kp_with_mask, vis_path)

        # Aggregate statistics
        avg_reduction = np.mean([r['reduction_percent'] for r in all_results])
        avg_speedup = np.mean([r['speedup_factor'] for r in all_results])

        print(f"\n{'-'*60}")
        print(f"SUMMARY for {desc_name}:")
        print(f"  Average keypoint reduction: {avg_reduction:.1f}%")
        print(f"  Average potential speedup: {avg_speedup:.2f}x")
        print(f"{'-'*60}")

        # Recommendation
        if avg_reduction < 10:
            print("  ⚠️  RECOMMENDATION: Reduction too small, not worth the effort")
        elif avg_reduction < 30:
            print("  ✓  RECOMMENDATION: Moderate reduction, worth exploring further")
        else:
            print("  ✓✓ RECOMMENDATION: Significant reduction, definitely implement!")

    print("\n" + "="*60)
    print("EXPERIMENT COMPLETE")
    print("="*60)

    if args.visualize:
        print(f"\nVisualizations saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
