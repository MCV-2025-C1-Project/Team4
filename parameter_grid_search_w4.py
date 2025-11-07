import argparse
import os
import time
from typing import Any, Dict
import cv2
import numpy as np
import pickle
import json
from pathlib import Path
import sys

# --- Fixes ModuleNotFoundError by adding project root to path ---
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

# --- Local Project Imports ---
from libs_week4.database import ImageDatabase
from libs_week3.average_precision import mapk
import grid_background_removal_week3

# --- Imports the NEW generator from Week 4 ---
from libs_week4.hyperparameter_combinations import descriptor_maker_grid_search, scorer_grid_search


def parse_arguments():
    # This function is unchanged from your original script.
    parser = argparse.ArgumentParser(description="Keypoint descriptor grid search for Week 4")
    parser.add_argument("database_path", type=str)
    parser.add_argument("queries_path", type=str)
    parser.add_argument("--from_iter", type=int, default=0)
    parser.add_argument("--every", type=int, default=1)
    parser.add_argument("--results_folder", type=str, required=True)
    return parser.parse_args()


def load_queries(queries_path: str, multiple_paintings=True, generate_masks=False) -> tuple[list[dict[str, Any]], list[list[int]]]:
    # This function is unchanged from your original script.
    queries = []
    with open(os.path.join(queries_path, "gt_corresps.pkl"), 'rb') as f:
        gt = pickle.load(f)

    for filename in sorted(os.listdir(queries_path)):
        if not filename.endswith(".jpg"):
            continue
        image_path = os.path.join(queries_path, filename)
        image = cv2.imread(image_path)

        if multiple_paintings:
            imgs = grid_background_removal_week3.split_if_two_paintings(image)
        else:
            imgs = [image]

        if generate_masks:
            config = {'name': 'HSV_SV', 'channels': [('HSV', 1), ('HSV', 2)], 'threshold': 0.005}
            masks = [grid_background_removal_week3.variance_background_removal(img, config).astype(np.uint8) * 255 for img in imgs]
        else:
            masks = [np.ones(img.shape[:2], dtype=np.uint8) * 255 for img in imgs]

        queries.append({'images': imgs, 'masks': masks, 'name': filename, 'gt': int(Path(image_path).stem)})
    return queries, gt


def save_results_for_config(folder: str, iteration: int, results: Dict):
    # Simplified save function for the new workflow.
    os.makedirs(folder, exist_ok=True)
    filename = f"{iteration:05d}.json"
    filepath = os.path.join(folder, filename)
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=4)


def prepare_gt_and_results_for_mapk(gt: list[list[int]], results: list[list[list[int]]]):
    # This is the EXACT function from your trusted Week 3 script.
    # It correctly flattens the data for evaluation.
    new_gt = []
    new_results = []
    for gt_item, res_item in zip(gt, results):
        assert len(gt_item) == len(res_item)
        for id_val, topk in zip(gt_item, res_item):
            new_gt.append([id_val])
            new_results.append(topk)
    return new_gt, new_results


def main():
    args = parse_arguments()
    os.makedirs(args.results_folder, exist_ok=True)

    print("Loading database...")
    database = ImageDatabase.load(args.database_path)

    print("Loading queries...")
    queries, ground_truth = load_queries(args.queries_path)

    # OUTER LOOP: Iterate over descriptor makers
    for desc_idx, descriptor_maker in enumerate(descriptor_maker_grid_search()):

        # Check if we should skip this descriptor based on command line args
        if desc_idx < args.from_iter or (desc_idx - args.from_iter) % args.every != 0:
            continue

        print(f"\n{'='*60}")
        print(f"DESCRIPTOR MAKER {desc_idx}")
        print(f"{'='*60}")

        # Pretty print the descriptor maker configuration
        descriptor_dict = descriptor_maker.to_dict()
        print("\nDescriptor Configuration:")
        print(json.dumps(descriptor_dict, indent=2))

        # COMPUTE DESCRIPTORS ONCE for this descriptor maker
        print("\nComputing descriptors for entire database...")
        start_time_descriptors = time.time()
        database.reset_descriptors_distances_and_scores()
        database.compute_keypoints_and_descriptors(descriptor_maker)
        descriptor_time = time.time() - start_time_descriptors
        print(f"Descriptor computation time: {descriptor_time:.2f}s")

        # Compute and print statistics
        stats = database.compute_keypoint_descriptor_statistics()
        print("\nKeypoint & Descriptor Statistics:")
        print(json.dumps(stats, indent=2))

        # Store results for all scorer configurations
        all_results = []

        # INNER LOOP: Iterate over scorer configurations
        for scorer_idx, scorer_config in enumerate(scorer_grid_search(descriptor_maker)):

            matcher = scorer_config['matcher']
            scorer = scorer_config['scorer']

            print(f"\n--- Scorer {scorer_idx}: "
                  f"ratio={matcher.ratio_test_threshold:.2f}, "
                  f"ransac={scorer.ransac_thresh:.1f}, min_pts={scorer.min_points} ---")

            # Query and evaluate WITHOUT recomputing descriptors
            start_time_query = time.time()

            results_top_5 = []
            for query in queries:
                query_image_results = []
                for img, mask in zip(query['images'], query['masks']):
                    query_keypoints, query_descriptors = descriptor_maker.detect_and_compute(img, mask)

                    result = database.query(img, query_keypoints, query_descriptors, scorer, k=10)

                    top_5_ids = [res.id for res in result[:5]]
                    while len(top_5_ids) < 5: top_5_ids.append(-1)
                    query_image_results.append(top_5_ids)
                results_top_5.append(query_image_results)

            query_time = time.time() - start_time_query
            print(f"  Query processing time: {query_time:.2f}s")

            # Reconciliation block to handle detection mismatches robustly
            reconciled_results = []
            no_result_placeholder = [-1] * 5
            for idx, gt_item in enumerate(ground_truth):
                res_item = results_top_5[idx]
                len_gt, len_res = len(gt_item), len(res_item)
                if len_gt == len_res:
                    reconciled_results.append(res_item)
                elif len_res < len_gt:
                    reconciled_results.append(res_item + [no_result_placeholder] * (len_gt - len_res))
                else:
                    reconciled_results.append(res_item[:len_gt])

            # --- CORRECT EVALUATION ---
            # 1. Prepare the data using your trusted flattening function.
            # 2. Call mapk with the correct argument order.
            map_gt, map_top_5 = prepare_gt_and_results_for_mapk(ground_truth, reconciled_results)

            map5 = mapk(map_gt, map_top_5, k=5)
            map1 = mapk(map_gt, map_top_5, k=1)

            print(f"  --> Results: map@k1={map1:.4f}, map@k5={map5:.4f}")

            # Append results with BOTH descriptor maker and scorer parameters
            all_results.append({
                'params': {
                    'keypoint_and_descriptor_maker': descriptor_maker.to_dict(),
                    'matcher': matcher.to_dict(),
                    'scorer': scorer.to_dict()
                },
                'metrics': {'map@k1': map1, 'map@k5': map5},
                'timing': {
                    'descriptor_computation_time': descriptor_time,
                    'query_time': query_time
                },
                'indices': {'desc_idx': desc_idx, 'scorer_idx': scorer_idx},
                'statistics': stats,
                'predictions': {
                    'ground_truth': ground_truth,
                    'reconciled_results': reconciled_results
                }
            })

        # Save all results for this descriptor maker in one file
        save_results_for_config(args.results_folder, desc_idx, all_results)

        print(f"\nCompleted all {len(all_results)} scorer configs for descriptor maker {desc_idx}")
        print(f"Results saved to {args.results_folder}/{desc_idx:05d}.json")

    print(f"\n{'='*60}")
    print(f"GRID SEARCH COMPLETE")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()