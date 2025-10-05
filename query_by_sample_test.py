import argparse
import os
from typing import Any
import cv2
from libs_week1.descriptor import ImageDescriptor
from libs_week1.distances import euclidean_distance
from matplotlib import pyplot as plt


def parse_arguments():
    # Parse command line arguments for dataset path and query image name.
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_path", type=str)
    parser.add_argument("query_name", type=str)
    # Return parsed command line arguments.
    return parser.parse_args()


def load_dataset(dataset_path: str):
    # Load JPG images from the dataset directory.
    dataset: list[dict[str, Any]] = []
    for filename in os.listdir(dataset_path):
        if not filename.endswith(".jpg"):
            continue  # Skip non-JPG files.
        image_path = os.path.join(dataset_path, filename)
        image = cv2.imread(image_path)
        if image is None:
            # Raise an error if the image can't be read.
            raise ValueError(f"Could not read image {filename}.")
        dataset.append({
            'name': filename,
            'image': image,
        })
    # Return the dataset list.
    return dataset


def add_descriptors_to_dataset(dataset: list[dict[str, Any]]):
    # Compute and add descriptors for each image in the dataset.
    descriptor_maker = ImageDescriptor()
    for entry in dataset:
        entry['descriptor'] = descriptor_maker.compute_descriptor(entry['image'])

# El nombre es una mierda
# Split the query image from the rest of the dataset.
def split_query_and_dataset(dataset: list[dict[str, Any]], query_name: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    query = None
    new_dataset = []
    for entry in dataset:
        if entry['name'] == query_name:
            query = entry  # Identify the query image.
        else:
            new_dataset.append(entry)
    # Ensure the query image was found.
    assert query is not None
    # Validate dataset integrity.
    assert len(dataset) == (len(new_dataset) + 1)
    return query, new_dataset


def find_k_closests(query: dict[str, Any], dataset: list[dict[str, Any]], k=2):
    # Calculate distances between the query and all other images.
    for entry in dataset:
        distance = euclidean_distance(entry['descriptor'], query['descriptor'])
        entry['distance'] = distance
    # Return the top k images with the smallest distances.
    return list(sorted(dataset, key=lambda e: e['distance']))[:k]


def show_results(query, results):
    # Display the query image.
    plt.figure()
    plt.title('Query')
    plt.imshow(cv2.cvtColor(query['image'], cv2.COLOR_BGR2RGB))
    plt.show()
    
    # Display each of the k closest matching images.
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
    # Execute main when running as a script.
    main()
