import argparse
import os
from typing import Any
import cv2
from descriptor import ImageDescriptor
import distances
from matplotlib import pyplot as plt
from pathlib import Path
import pickle

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_path", type=str)
    parser.add_argument("queries_path", type=str)

    return parser.parse_args()


def load_database(database_path: str):
    database: list[dict[str, Any]] = []
    for filename in sorted(os.listdir(database_path)):
        if not filename.endswith(".jpg"):
            continue

        image_path = os.path.join(database_path, filename)
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image {filename}.")
        
        painting_name_path = Path(image_path).with_suffix('.txt')
        try:
            painting_name = painting_name_path.read_text(encoding="utf-8")
        except Exception as e:
            print(f"Image {filename} doesn't have associated .txt file.")
            painting_name = None

        database.append({
            'name': filename,
            'image': image,
            'painting_name': painting_name
        })

    return database


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

    return queries



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
    database = load_database(args.dataset_path)
    queries = load_queries(args.queries_path)
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
