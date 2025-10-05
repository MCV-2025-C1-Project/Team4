import argparse
import os
from typing import Any
import cv2
from libs_week1.descriptor import ImageDescriptor
from libs_week1.distances import euclidean_distance
from matplotlib import pyplot as plt

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_path", type=str)
    parser.add_argument("query_name", type=str)

    return parser.parse_args()


def load_dataset(dataset_path: str):
    dataset: list[dict[str, Any]] = []
    for filename in os.listdir(dataset_path):
        if not filename.endswith(".jpg"):
            continue

        image_path = os.path.join(dataset_path, filename)
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image {filename}.")
        
        dataset.append({
            'name': filename,
            'image': image,
        })

    return dataset


def add_descriptors_to_dataset(dataset: list[dict[str, Any]]):
    descriptor_maker = ImageDescriptor()
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
        distance = euclidean_distance(entry['descriptor'], query['descriptor'])
        entry['distance'] = distance

    return list(sorted(dataset, key=lambda e: e['distance']))[:k]


def show_results(query, results):
    plt.figure()
    plt.title('Query')
    plt.imshow(cv2.cvtColor(query['image'], cv2.COLOR_BGR2RGB))
    plt.show()

    for i, entry in enumerate(results, start=1):
        plt.figure()
        plt.title(f'Top {i}')
        plt.imshow(cv2.cvtColor(entry['image'], cv2.COLOR_BGR2RGB))
        plt.show()


def main():
    args = parse_arguments()

    print("Loading dataset..")
    dataset = load_dataset(args.dataset_path)
    print("Computing descriptors..")
    add_descriptors_to_dataset(dataset)
    print("Fetching query...")
    query, dataset = split_query_and_dataset(dataset, args.query_name)
    print("Querying...")
    closest_k = find_k_closests(query, dataset)
    print("Showing...")
    show_results(query, closest_k)


if __name__ == "__main__":
    main()
