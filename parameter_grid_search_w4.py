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

# --- ADD THIS AT THE TOP to fix ModuleNotFoundError ---
# This ensures the project root (e.g., 'Team4') is on the Python path
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))
# ---

# Now, these imports will work correctly
from libs_week3.database import ImageDatabase
from libs_week3.average_precision import mapk
import grid_background_removal_week3

# IMPORTANT: Import the NEW grid search generator from week 4
from libs_week4.hyperparameter_combinations import keypoint_hyperparameter_grid_search


def parse_arguments():
    parser = argparse.ArgumentParser(description="Keypoint descriptor grid search for Week 4")
    parser.add_argument("database_path", type=str)
    parser.add_argument("queries_path", type=str)
    parser.add_argument("--from_iter", type=int, default=0)
    parser.add_argument("--every", type=int, default=1)
    parser.add_argument("--results_folder", type=str, required=True)
    return parser.parse_args()


def load_queries(queries_path: str, multiple_paintings=True, generate_masks=True) -> tuple[list[dict[str, Any]], list[list[int]]]:
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
            config = {
                'name': 'HSV_SV',
                'channels': [('HSV', 1), ('HSV', 2)],
                'threshold': 0.005,
            }
            masks = [grid_background_removal_week3.variance_background_removal(img, config).astype(np.uint8) * 255 for img in imgs]
        else:
            masks = [np.ones(img.shape[:2], dtype=np.uint8) * 255 for img in imgs]

        queries.append({
            'images': imgs,
            'masks': masks,
            'name': filename,
            'gt': int(Path(image_path).stem)
        })
    return queries, gt


def save_results_for_config(folder: str, iteration: int, results: Dict):
    """Saves a single JSON file for the current hyperparameter configuration."""
    os.makedirs(folder, exist_ok=True)
    filename = f"{iteration:05d}.json"
    filepath = os.path.join(folder, filename)
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=4)


def prepare_gt_and_results_for_mapk(gt: list[list[int]], results: list[list[list[int]]]):
    """This function is unchanged and works as expected."""
    new_gt, new_results = [], []
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

    # Main grid search loop using the NEW generator
    for i, params in enumerate(keypoint_hyperparameter_grid_search()):
        if i < args.from_iter or (i - args.from_iter) % args.every != 0:
            continue

        # Unpack parameters for the current iteration
        descriptor_maker = params['keypoint_descriptor']
        matcher = params['matcher']
        preprocess = params['preprocess']
        color_conversion = params['color_conversion']

        print(f"\n--- Iteration {i:04d}: Descriptor: {descriptor_maker.to_dict()['type']}, Matcher: {matcher.to_dict()['matcher_type']} ---")

        # 1. Compute descriptors for the entire database
        start_time = time.time()
        db_descriptors_cache = []
        for db_image in database.images:
            img = db_image.image
            mask = np.ones(img.shape[:2], dtype=np.uint8) * 255
            
            if preprocess:
                img, mask = preprocess(img, mask)

            # --- FIX 1: Unpack the tuple returned by color_conversion ---
            processed_img, processed_mask = color_conversion(img, mask)
            
            # --- FIX 2: Pass the unpacked variables to detect_and_compute ---
            _, descs = descriptor_maker.detect_and_compute(processed_img, processed_mask)
            db_descriptors_cache.append({'id': db_image.id, 'descriptors': descs})
        db_desc_time = time.time() - start_time
        print(f"  Database descriptor computation time: {db_desc_time:.2f}s")

        # 2. Compute descriptors for all queries
        start_time = time.time()
        for query in queries:
            query['descriptors_list'] = []
            for img, mask in zip(query['images'], query['masks']):
                if preprocess:
                    img, mask = preprocess(img, mask)
                
                # --- FIX 3: Unpack the tuple here as well ---
                processed_img, processed_mask = color_conversion(img, mask)

                # --- FIX 4: Pass the unpacked variables here ---
                _, descs = descriptor_maker.detect_and_compute(processed_img, processed_mask)
                query['descriptors_list'].append(descs)
        query_desc_time = time.time() - start_time
        print(f"  Query descriptor computation time: {query_desc_time:.2f}s")
        
        # 3. Match each query against the database and rank by matches
        start_time = time.time()
        results_top_5 = []
        for query in queries:
            query_results = []
            for query_descs in query['descriptors_list']:
                if query_descs is None or len(query_descs) == 0:
                    query_results.append([-1] * 5)
                    continue

                match_counts = []
                for db_entry in db_descriptors_cache:
                    db_descs = db_entry['descriptors']
                    num_matches = 0
                    if db_descs is not None and len(db_descs) > 0:
                        good_matches = matcher.match(query_descs, db_descs)
                        num_matches = len(good_matches)
                    match_counts.append({'id': db_entry['id'], 'matches': num_matches})
                
                sorted_results = sorted(match_counts, key=lambda x: x['matches'], reverse=True)
                top_5_ids = [res['id'] for res in sorted_results[:5]]
                
                while len(top_5_ids) < 5:
                    top_5_ids.append(-1)
                
                query_results.append(top_5_ids)

            results_top_5.append(query_results)
        matching_time = time.time() - start_time
        print(f"  Total matching time: {matching_time:.2f}s")

        # 4. Evaluate results using MAPK
        map_gt, map_top_5 = prepare_gt_and_results_for_mapk(ground_truth, results_top_5)
        map5 = mapk(map_top_5, map_gt, k=5)
        map1 = mapk(map_top_5, map_gt, k=1)
        print(f"  --> Results: map@k1={map1:.4f}, map@k5={map5:.4f}")

        # 5. Save results for this specific configuration
        results_data = {
            'params': {
                'keypoint_descriptor': descriptor_maker.to_dict(),
                'matcher': matcher.to_dict(),
                'preprocess': preprocess.to_dict() if preprocess else None,
                'color_conversion': color_conversion.to_dict()
            },
            'metrics': {
                'map@k1': map1,
                'map@k5': map5
            },
            'timing': {
                'database_descriptors': db_desc_time,
                'queries_descriptors': query_desc_time,
                'matching': matching_time,
                'total': db_desc_time + query_desc_time + matching_time
            }
        }
        save_results_for_config(args.results_folder, i, results_data)
if __name__ == "__main__":
    main()