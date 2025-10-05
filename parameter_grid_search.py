import argparse
import os
from typing import Any
import cv2
from libs_week1.average_precision import mapk
from libs_week1.database import ImageDatabase
from libs_week1.descriptor import ImageDescriptor, ImageDescriptorMaker
import libs_week1.distances
from matplotlib import pyplot as plt
from pathlib import Path
import pickle
import json

from libs_week1.hyperparameter_combinations import hyperparameter_grid_search

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("database_path", type=str)
    parser.add_argument("queries_path", type=str)
    parser.add_argument("--from_iter", type=int, default=0)
    parser.add_argument("--every", type=int, default=1)
    parser.add_argument("--results_folder", type=str, required=True)

    return parser.parse_args()


def load_queries(queries_path: str):
    queries = []
    gt = pickle.load(open(os.path.join(queries_path, "gt_corresps.pkl"), 'rb'))
    for filename in sorted(os.listdir(queries_path)):
        if not filename.endswith(".jpg"):
            continue

        image_path = os.path.join(queries_path, filename)
        image = cv2.imread(image_path)
        
        queries.append({
            'image': image,
            'name': filename,
            'gt': int(Path(image_path).stem)
        })

    return queries, gt



def add_descriptors_to_dataset(dataset: list[dict[str, Any]]):
    descriptor_maker = ImageDescriptor(normalize_histograms=True)
    for entry in dataset:
        entry['descriptor'] = descriptor_maker.compute_descriptor(entry['image'])


# El nombre es una mierda
def split_query_and_dataset(dataset: list[dict[str, Any]], query_name: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    query = None
    new_dataset = []
    for entry in dataset:
        if entry['name'] == query_name:
            query = entry
        else:
            new_dataset.append(entry)
    
    assert query is not None
    assert len(dataset) == (len(new_dataset) + 1)
    return query, new_dataset


def find_k_closests(query: dict[str, Any], dataset: list[dict[str, Any]], k=2):
    for entry in dataset:
        distance = distances.l1_distance(entry['descriptor'], query['descriptor'])
        entry['distance'] = distance

    return list(sorted(dataset, key=lambda e: e['distance']))[:k]


def show_results(query, results):
    plt.figure()
    plt.title('Query')
    plt.imshow(cv2.cvtColor(query['image'], cv2.COLOR_BGR2RGB))
    plt.show()

    for i, entry in enumerate(results, start=1):
        plt.figure()
        plt.title(f'Top {i}, distance = {entry["distance"]:.5f}')
        plt.imshow(cv2.cvtColor(entry['image'], cv2.COLOR_BGR2RGB))
        plt.show()


def save_results_for_descriptor(folder: str, iteration: int, results: list[dict]):
    os.makedirs(folder, exist_ok=True)

    filename = f"{iteration:05d}.json"
    filepath = os.path.join(folder, filename)

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

    for i, params in enumerate(hyperparameter_grid_search()):
        if i < from_iter:
            continue

        if (i - from_iter) % every != 0:
            continue

        descriptor_maker = ImageDescriptorMaker(
            gamma_correction=params['gamma_correction'],
            blur_image=False,
            color_spaces=params['color_spaces'],
            bins=params['bins'],
            keep_or_discard=params['keep_or_discard'],
            weights=params['weight'],
        )
        print("Computing descriptors...")
        database.reset_descriptors_and_distances()
        database.compute_descriptors(descriptor_maker)

        results_for_descriptor = []
        for distance_name, distance in distances.iter_simple_distances():
            print("Querying...", params, distance_name)
            results_top_5 = []
            for image in queries:
                query_descriptor = descriptor_maker.make_descriptor(image['image'])
                top_5 = database.query(query_descriptor, distance, k=5)
                results_top_5.append([im.id for im in top_5])

            map5 = mapk(ground_truth, results_top_5, k=5)
            map1 = mapk(ground_truth, results_top_5, k=1)
            
            print(params, distance_name, map1, map5)

            results_for_descriptor.append({
                'gamma_correction': params['gamma_correction'],
                'blur_image': False,
                'color_spaces': [space.value for space in params['color_spaces']],
                'bins': params['bins'],
                'keep_or_discard': params['keep_or_discard'],
                'weights': params['weight'].value if params['weight'] is not None else None,
                'distance': distance_name,
                'map@k1': map1,
                'map@k5': map5,
            })

        # special case
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
            'keep_or_discard': params['keep_or_discard'],
            'weights': params['weight'].value if params['weight'] is not None else None,
            'distance': "emd_distance",
            'map@k1': map1,
            'map@k5': map5,
        })

        # special case
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
            'keep_or_discard': params['keep_or_discard'],
            'weights': params['weight'].value if params['weight'] is not None else None,
            'distance': "multichannel_quadratic_form_distance",
            'map@k1': map1,
            'map@k5': map5,
        })

        save_results_for_descriptor(results_folder, i, results_for_descriptor)


if __name__ == "__main__":
    main()
