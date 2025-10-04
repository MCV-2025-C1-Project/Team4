import argparse
import os
from typing import Any
import cv2
from database import ImageDatabase
from descriptor import ImageDescriptor, ImageDescriptorMaker
import distances
from matplotlib import pyplot as plt
from pathlib import Path
import pickle

from hyperparameter_combinations import hyperparameter_grid_search

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("database_path", type=str)
    parser.add_argument("queries_path", type=str)

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



def main():
    args = parse_arguments()

    print("Loading dataset..")
    database = ImageDatabase.load(args.database_path)
    queries, ground_truth = load_queries(args.queries_path)

    for params in hyperparameter_grid_search():
        descriptor_maker = ImageDescriptorMaker(params['gamma_correction'])
        database.reset_descriptors_and_distances()
        database.compute_descriptors(descriptor_maker)

        for distance in distances.iter_simple_distances():
            results_top_5 = []
            for image in query:
                query_descriptor = descriptor_maker.make_descriptor(image)
                top_5 = database.query(query_descriptor, distance, k=5)
                results_top_5.append(top_5)

            




        print("Computing descriptors..")
        add_descriptors_to_dataset(database)
        add_descriptors_to_dataset(queries)

        for query in queries:
            print("Querying...")
            closest_k = find_k_closests(query, database, k=1)
            print("Showing...")
            show_results(query, closest_k)




if __name__ == "__main__":
    main()
