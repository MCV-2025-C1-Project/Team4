import argparse
import os
from typing import Any
import cv2
import numpy as np
from libs_week2.average_precision import mapk
from libs_week2.database import ImageDatabase
from libs_week2.descriptor import ImageDescriptorMaker
import libs_week2.distances as distances
from matplotlib import pyplot as plt
from pathlib import Path
import pickle
import json

from libs_week2.hyperparameter_combinations import hyperparameter_grid_search


def parse_arguments():
    parser = argparse.ArgumentParser()
    # Define required command-line arguments
    parser.add_argument("database_path", type=str)
    parser.add_argument("queries_path", type=str)
    parser.add_argument("--from_iter", type=int, default=0)
    parser.add_argument("--every", type=int, default=1)
    parser.add_argument("--results_folder", type=str, required=True)
    return parser.parse_args()


def load_queries(queries_path: str):
    queries = []
    # Load ground truth correspondences
    gt = pickle.load(open(os.path.join(queries_path, "gt_corresps.pkl"), 'rb'))
    for filename in sorted(os.listdir(queries_path)):
        if not filename.endswith(".jpg"):
            continue
        image_path = os.path.join(queries_path, filename)
        image = cv2.imread(image_path)

        # Load mask if available, otherwise create full white mask
        mask_path = Path(image_path).with_suffix('.png')
        if mask_path.exists():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        else:
            # Use full white mask for queries without mask
            mask = np.ones(image.shape[:2], dtype=np.uint8) * 255

        # Add each query image with its filename, ground truth id, and mask
        queries.append({
            'image': image,
            'mask': mask,
            'name': filename,
            'gt': int(Path(image_path).stem)
        })
    return queries, gt


def show_results(query, results):
    plt.figure()
    plt.title('Query')
    # Display the query image (converted to RGB)
    plt.imshow(cv2.cvtColor(query['image'], cv2.COLOR_BGR2RGB))
    plt.show()

    # Display each result image with its distance
    for i, entry in enumerate(results, start=1):
        plt.figure()
        plt.title(f'Top {i}, distance = {entry["distance"]:.5f}')
        plt.imshow(cv2.cvtColor(entry['image'], cv2.COLOR_BGR2RGB))
        plt.show()


def save_results_for_descriptor(folder: str, iteration: int, results: list[dict]):
    os.makedirs(folder, exist_ok=True)
    filename = f"{iteration:05d}.json"
    filepath = os.path.join(folder, filename)
    # Save the results in JSON format with indentation
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=4)


def main():
    args = parse_arguments()
    from_iter = args.from_iter
    every = args.every
    results_folder = args.results_folder

    print("Loading database..")
    database = ImageDatabase.load(args.database_path)

    print("Loading queries..")
    queries, ground_truth = load_queries(args.queries_path)

    # Iterate over hyperparameter settings
    for i, params in enumerate(hyperparameter_grid_search()):
        if i < from_iter:
            continue
        if (i - from_iter) % every != 0:
            continue

        # Initialize descriptor maker using current hyperparameters
        descriptor_maker = ImageDescriptorMaker(
            color_spaces=params['color_spaces'],
            histogram_computer=params['histogram_computer'],
            preprocess=params['preprocess'],
        )
        print("Computing descriptors...")
        # Reset and compute descriptors for the current iteration
        database.reset_descriptors_and_distances()
        database.compute_descriptors(descriptor_maker)

        results_for_descriptor = []
        # Evaluate using simple distance functions
        for distance_name, distance in distances.iter_simple_distances():
            print("Querying...", params, distance_name)
            results_top_5 = []
            for query in queries:
                query_descriptor = descriptor_maker.make_descriptor(query['image'], query['mask'])
                top_5 = database.query(query_descriptor, distance, k=5)
                results_top_5.append([im.id for im in top_5])

            # Compute evaluation metrics
            map5 = mapk(ground_truth, results_top_5, k=5)
            map1 = mapk(ground_truth, results_top_5, k=1)
            print(params, distance_name, map1, map5)

            # Get preprocess description
            preprocess_dict = None
            if params['preprocess'] is not None:
                preprocess_dict = params['preprocess'].to_dict()

            results_for_descriptor.append({
                'color_spaces': [space.value for space in params['color_spaces']],
                'histogram_computer': params['histogram_computer'].to_dict(),
                'preprocess': preprocess_dict,
                'distance': distance_name,
                'map@k1': map1,
                'map@k5': map5,
            })


        # FIXME: this should work
        """
        # Special case: EMD distance
        print("Querying...", params, "emd_distance")
        results_top_5 = []
        for image in queries:
            query_descriptor = descriptor_maker.make_descriptor(image['image'])
            top_5 = database.query(query_descriptor, lambda h1, h2: distances.emd_multichannel(h1, h2, num_channels=int(query_descriptor.shape[0] / params['bins']), bins_per_channel=params['bins']), k=5)
            results_top_5.append([im.id for im in top_5])

        map5 = mapk(ground_truth, results_top_5, k=5)
        map1 = mapk(ground_truth, results_top_5, k=1)
        print(params, "emd_distance", map1, map5)

        results_for_descriptor.append({
            'gamma_correction': params['gamma_correction'],
            'blur_image': False,
            'color_spaces': [space.value for space in params['color_spaces']],
            'bins': params['bins'],
            'distance': "emd_distance",
            'map@k1': map1,
            'map@k5': map5,
        })

        # Special case: Multichannel Quadratic Form distance
        print("Querying...", params, "multichannel_quadratic_form_distance")
        results_top_5 = []
        for image in queries:
            query_descriptor = descriptor_maker.make_descriptor(image['image'])
            top_5 = database.query(query_descriptor, lambda h1, h2: distances.multichannel_quadratic_form_distance(h1, h2, num_channels=int(query_descriptor.shape[0] / params['bins']), bins_per_channel=params['bins']), k=5)
            results_top_5.append([im.id for im in top_5])

        map5 = mapk(ground_truth, results_top_5, k=5)
        map1 = mapk(ground_truth, results_top_5, k=1)
        print(params, "multichannel_quadratic_form_distance", map1, map5)

        results_for_descriptor.append({
            'gamma_correction': params['gamma_correction'],
            'blur_image': False,
            'color_spaces': [space.value for space in params['color_spaces']],
            'bins': params['bins'],
            'distance': "multichannel_quadratic_form_distance",
            'map@k1': map1,
            'map@k5': map5,
        })
        """

        # Save results for the current hyperparameter configuration
        save_results_for_descriptor(results_folder, i, results_for_descriptor)


if __name__ == "__main__":
    main()
