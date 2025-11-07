import argparse
import os
from typing import Any

import grid_background_removal_week3
from libs_week3.average_precision import mapk
import cv2
import numpy as np

from matplotlib import pyplot as plt
from pathlib import Path
import pickle

# Week 4 imports - keypoint descriptors
from libs_week4.database import ImageDatabase
from libs_week4.descriptor import (
    ORBDescriptor,
    DescriptorMatcher,
    HomographyScorer,
    KeypointAndDescriptorMaker,
)
from libs_week3.color_conversion import ColorConversion, ColorSpace
from libs_week3.preprocessing import Preprocess


# Parse command-line arguments and return them.
def parse_arguments():
    parser = argparse.ArgumentParser(description="Week 4 Query-by-Sample using ORB keypoint descriptors")
    parser.add_argument("dataset_path", type=str)  # Dataset directory path.
    parser.add_argument("queries_path", type=str)  # Queries directory path.

    parser.add_argument("--k", type=int, default=10)  # Number of top results to retrieve.
    parser.add_argument("--pkl_output_path", type=str, default=None)  # Output path for pickled predictions.
    parser.add_argument("--generate_masks", default=False, action='store_true')  # Generate masks using variance method.
    parser.add_argument("--multiple_paintings", default=True, action='store_true')  # Handle multiple paintings per query.

    # Visualization options
    parser.add_argument("--visualize", default=False, action='store_true')  # Display visualizations interactively.
    parser.add_argument("--save_visualizations", type=str, default=None)  # Directory to save visualizations.

    return parser.parse_args()


# Load query images and ground truth from the provided queries_path.
def load_queries(queries_path: str, multiple_paintings=True, generate_masks=True) -> tuple[list[dict[str, Any]], list[list[int]]]:
    queries = []
    gt_path = os.path.join(queries_path, "gt_corresps.pkl")
    if os.path.exists(gt_path):
        with open(gt_path, 'rb') as f:
            gt = pickle.load(f)
    else:
        gt = None

    for filename in sorted(os.listdir(queries_path)):
        if not filename.endswith(".jpg"):
            continue

        image_path = os.path.join(queries_path, filename)
        image = cv2.imread(image_path)

        # Step 1: Split if multiple paintings detected
        if multiple_paintings:
            imgs = grid_background_removal_week3.split_if_two_paintings(image)
        else:
            imgs = [image]

        # Step 2: Generate masks using variance-based background removal
        if generate_masks:
            config = {
                'name': 'HSV_SV',
                'channels': [('HSV', 1), ('HSV', 2)],  # Saturation + Value
                'threshold': 0.005,
            }
            masks = [grid_background_removal_week3.variance_background_removal(img, config).astype(np.uint8) * 255 for img in imgs]
        else:
            # Use full white masks (no masking)
            masks = [np.ones(img.shape[:2], dtype=np.uint8) * 255 for img in imgs]

        # Add each query image with its metadata
        queries.append({
            'images': imgs,
            'masks': masks,
            'name': filename,
            'id': int(Path(image_path).stem)
        })

    return queries, gt


def prepare_gt_and_results_for_mapk(gt: list[list[int]], results: list[list[list[int]]]):
    new_gt = []
    new_results = []
    for gt_item, res_item in zip(gt, results):
        assert len(gt_item) == len(res_item)
        for id, topk in zip(gt_item, res_item):
            new_gt.append([id])
            new_results.append(topk)

    return new_gt, new_results


def create_red_x_image(height: int, width: int) -> np.ndarray:
    """
    Create a white image with a red X overlay.

    Args:
        height: Image height
        width: Image width

    Returns:
        RGB image with red X
    """
    img = np.ones((height, width, 3), dtype=np.uint8) * 255

    # Draw red X
    thickness = max(5, min(height, width) // 50)
    color = (255, 0, 0)  # Red in RGB

    # Diagonal from top-left to bottom-right
    cv2.line(img, (0, 0), (width - 1, height - 1), color, thickness)
    # Diagonal from top-right to bottom-left
    cv2.line(img, (width - 1, 0), (0, height - 1), color, thickness)

    return img


def resize_image_keep_aspect(image: np.ndarray, target_height: int) -> np.ndarray:
    """
    Resize image to target height while maintaining aspect ratio.

    Args:
        image: Input image
        target_height: Desired height in pixels

    Returns:
        Resized image
    """
    h, w = image.shape[:2]
    aspect_ratio = w / h
    new_width = int(target_height * aspect_ratio)
    return cv2.resize(image, (new_width, target_height), interpolation=cv2.INTER_AREA)


def visualize_query_results(
    query: dict,
    query_ground_truth: list[int],
    query_results: list[list[int]],
    database: 'ImageDatabase',
    display: bool = False,
    save_path: str = None
):
    """
    Create visualization of query results with query images, ground truth, and top-5 results.

    Args:
        query: Query dictionary containing 'images', 'masks', 'name', 'id'
        query_ground_truth: List of ground truth IDs (one per painting in query)
        query_results: List of top-k result IDs (one list per painting in query)
        database: ImageDatabase instance to fetch result images
        display: If True, display the visualization using matplotlib
        save_path: If provided, save visualization to this path (e.g., "output/00000.png")
    """
    # Skip if neither display nor save is requested
    if not display and save_path is None:
        return

    num_paintings = len(query['images'])

    # Each row: Query | GT | Rank1 | Rank2 | Rank3 | Rank4 | Rank5
    num_cols = 7

    # Create figure with subplots
    fig, axes = plt.subplots(num_paintings, num_cols, figsize=(20, 3 * num_paintings))

    # Handle single painting case (axes won't be 2D)
    if num_paintings == 1:
        axes = axes.reshape(1, -1)

    # Column titles
    col_titles = ['Query', 'Ground Truth', 'Rank 1', 'Rank 2', 'Rank 3', 'Rank 4', 'Rank 5']

    # Target height for resizing (to make images similar size)
    target_height = 300

    # Process each painting in the query
    for painting_idx in range(num_paintings):
        query_img = query['images'][painting_idx]
        gt_id = query_ground_truth[painting_idx]
        top_k_ids = query_results[painting_idx][:5]  # Top 5 results

        # Pad with -1 if fewer than 5 results
        while len(top_k_ids) < 5:
            top_k_ids.append(-1)

        # Column 0: Query image
        query_rgb = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)
        query_resized = resize_image_keep_aspect(query_rgb, target_height)
        axes[painting_idx, 0].imshow(query_resized)
        axes[painting_idx, 0].axis('off')
        if painting_idx == 0:
            axes[painting_idx, 0].set_title(col_titles[0], fontsize=12, fontweight='bold')

        # Column 1: Ground Truth
        if gt_id == -1:
            gt_img = create_red_x_image(target_height, target_height)
        else:
            # Find database image with matching ID
            db_img = next((img.image for img in database.images if img.id == gt_id), None)
            if db_img is not None:
                gt_rgb = cv2.cvtColor(db_img, cv2.COLOR_BGR2RGB)
                gt_img = resize_image_keep_aspect(gt_rgb, target_height)
            else:
                gt_img = create_red_x_image(target_height, target_height)

        axes[painting_idx, 1].imshow(gt_img)
        axes[painting_idx, 1].axis('off')
        if painting_idx == 0:
            axes[painting_idx, 1].set_title(col_titles[1], fontsize=12, fontweight='bold')

        # Columns 2-6: Top 5 results
        for rank_idx, result_id in enumerate(top_k_ids):
            col_idx = rank_idx + 2

            if result_id == -1:
                result_img = create_red_x_image(target_height, target_height)
            else:
                # Find database image with matching ID
                db_img = next((img.image for img in database.images if img.id == result_id), None)
                if db_img is not None:
                    result_rgb = cv2.cvtColor(db_img, cv2.COLOR_BGR2RGB)
                    result_img = resize_image_keep_aspect(result_rgb, target_height)
                else:
                    result_img = create_red_x_image(target_height, target_height)

            axes[painting_idx, col_idx].imshow(result_img)
            axes[painting_idx, col_idx].axis('off')
            if painting_idx == 0:
                axes[painting_idx, col_idx].set_title(col_titles[col_idx], fontsize=12, fontweight='bold')

    # Add main title with query ID
    fig.suptitle(f"Query {query['id']:05d} - {query['name']}", fontsize=14, fontweight='bold')
    plt.tight_layout()

    # Save if path provided
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    # Display if requested
    if display:
        plt.show()
    else:
        plt.close(fig)  # Close to free memory if not displaying


# Main function to execute the query-by-sample process.
def main():
    args = parse_arguments()  # Parse command-line arguments.

    print("="*60)
    print("Week 4 Query-by-Sample: ORB Keypoint Descriptor")
    print("="*60)

    print("\nConfiguration:")
    print("  Descriptor: ORB")
    print("    - n_features: 3000")
    print("    - scale_factor: 1.2")
    print("    - n_levels: 10")
    print("    - wta_k: 2")
    print("    - score_type: HARRIS")
    print("    - patch_size: 31")
    print("  Matcher: BruteForce")
    print("    - norm_type: HAMMING")
    print("    - cross_check: False")
    print("    - ratio_test_threshold: 0.65")
    print("  Scorer: HomographyScorer")
    print("    - ransac_thresh: 3.0")
    print("    - max_reproj_error: 3.0")
    print("    - use_reproj_error_penalty: False")
    print("    - min_points: 20")
    print("\nQuery Processing:")
    print(f"  Multiple paintings detection: {args.multiple_paintings}")
    print(f"  Mask generation (HSV S+V variance): {args.generate_masks}")
    print("\nVisualization:")
    print(f"  Display visualizations: {args.visualize}")
    print(f"  Save visualizations: {args.save_visualizations if args.save_visualizations else 'No'}")
    print()

    print("Loading database...")
    database = ImageDatabase.load(args.dataset_path)
    print(f"  Loaded {len(database.images)} images")

    print("\nLoading queries...")
    queries, ground_truth = load_queries(args.queries_path, args.multiple_paintings, args.generate_masks)

    # Count total query paintings (after splitting)
    total_paintings = sum(len(q['images']) for q in queries)
    print(f"  Loaded {len(queries)} query images")
    print(f"  Total paintings (after split detection): {total_paintings}")

    # Setup descriptor maker with specified configuration
    print("\nSetting up descriptor maker...")
    descriptor_computer = ORBDescriptor(
        n_features=3000,
        scale_factor=1.2,
        n_levels=10,
        wta_k=2,
        score_type=cv2.ORB_HARRIS_SCORE,
        patch_size=31
    )

    descriptor_maker = KeypointAndDescriptorMaker(
        descriptor_computer=descriptor_computer,
        color_conversion=ColorConversion(targets=[ColorSpace.BGR], normalize=True),
        preprocess=Preprocess([])
    )

    # Setup matcher and scorer
    matcher = DescriptorMatcher(
        matcher_type='BF',
        norm_type=cv2.NORM_HAMMING,
        cross_check=False,
        ratio_test_threshold=0.65
    )

    scorer = HomographyScorer(
        matcher=matcher,
        ransac_thresh=3.0,
        max_reproj_error=3.0,
        use_reproj_error_penalty=False,
        reproj_error_penalty_weight=0.5,
        min_points=20
    )

    print("Computing keypoints and descriptors for database...")
    database.reset_descriptors_distances_and_scores()
    database.compute_keypoints_and_descriptors(descriptor_maker)

    # Compute statistics
    stats = database.compute_keypoint_descriptor_statistics()
    print("\nDatabase Keypoint Statistics:")
    print(f"  Mean keypoints per image: {stats['keypoints']['mean']:.1f}")
    print(f"  Min/Max keypoints: {stats['keypoints']['min']}/{stats['keypoints']['max']}")
    print(f"  Total keypoints: {stats['keypoints']['total']}")
    print(f"  Descriptor dimensions: {stats['descriptors']['dimensions']}")

    print("\nQuerying...")
    results = []
    for query_idx, query in enumerate(queries):
        query_image_results = []
        for img, mask in zip(query['images'], query['masks']):
            # Compute keypoints and descriptors for query
            query_keypoints, query_descriptors = descriptor_maker.detect_and_compute(img, mask)

            # Query database
            result = database.query(img, query_keypoints, query_descriptors, scorer, k=args.k)

            # Extract top-k IDs
            top_k_ids = [res.id for res in result]

            # Pad with -1 if needed
            while len(top_k_ids) < args.k:
                top_k_ids.append(-1)

            query_image_results.append(top_k_ids)

        results.append(query_image_results)

        # Generate visualization if requested
        if args.visualize or args.save_visualizations:
            save_path = None
            if args.save_visualizations:
                save_path = os.path.join(args.save_visualizations, f"{query['id']:05d}.png")

            if ground_truth is not None:
                query_gt = ground_truth[query_idx]
            else:
                query_gt = [-1] * len(query_image_results)

            visualize_query_results(
                query=query,
                query_ground_truth=query_gt,
                query_results=query_image_results,
                database=database,
                display=args.visualize,
                save_path=save_path
            )

    if ground_truth is not None:
        print("\nEvaluating results...")

        # Reconcile results to match ground truth structure
        reconciled_results = []
        no_result_placeholder = [-1] * args.k
        for idx, gt_item in enumerate(ground_truth):
            res_item = results[idx]
            len_gt, len_res = len(gt_item), len(res_item)
            if len_gt == len_res:
                reconciled_results.append(res_item)
            elif len_res < len_gt:
                reconciled_results.append(res_item + [no_result_placeholder] * (len_gt - len_res))
            else:
                reconciled_results.append(res_item[:len_gt])

        map_gt, map_res = prepare_gt_and_results_for_mapk(ground_truth, reconciled_results)

        mapk1 = mapk(map_gt, map_res, k=1)
        print(f"  map@k=1: {mapk1:.5f}")

        if args.k >= 5:
            mapk5 = mapk(map_gt, map_res, k=5)
            print(f"  map@k=5: {mapk5:.5f}")

    # Generate a pickle file with the predictions if output path is provided.
    if args.pkl_output_path:
        print(f"\nSaving predictions to {args.pkl_output_path}...")
        with open(args.pkl_output_path, "wb") as f:
            pickle.dump(reconciled_results if ground_truth is not None else results, f)
        print("  Done!")

    # Summary of visualizations
    if args.save_visualizations:
        print(f"\nVisualizations saved to: {args.save_visualizations}")
        print(f"  Total visualizations: {len(queries)}")

    print("\n" + "="*60)
    print("Query-by-Sample Complete")
    print("="*60)


# Execute the main function when the script is run.
if __name__ == "__main__":
    main()
