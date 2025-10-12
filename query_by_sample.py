import argparse
import os
from typing import Any
from libs_week2.average_precision import mapk
import cv2
import numpy as np
from libs_week2.database import ImageDatabase
from libs_week2.descriptor import (
    ColorSpace, ImageDescriptorMaker,
    Histogram1D, IdentityImageBlockSplitter, GridImageBlockSplitter,
    Preprocess, OpenMask, CropToMask, ApplyGamma
)
import libs_week2.distances as distances
from matplotlib import pyplot as plt
from pathlib import Path
import pickle


def variance_background_removal(image: np.ndarray, ):
    # Extract specified channels from their color spaces
    channels_to_analyze = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))

    # Stack channels into a single array
    if not channels_to_analyze:
        raise ValueError("No channels to analyze")

    height, width = channels_to_analyze[0].shape
    threshold = 0.005

    # Store bounding boxes for each channel
    bboxes = []

    for channel in channels_to_analyze:
        # Compute variances along each axis
        variances_h = channel.var(axis=1)  # Variance per row
        variances_v = channel.var(axis=0)  # Variance per column

        # Find top edge: scan from top until variance exceeds threshold
        top = 0
        for i in range(height):
            if variances_h[i] >= threshold:
                top = i
                break

        # Find bottom edge: scan from bottom until variance exceeds threshold
        bottom = height - 1
        for i in range(height - 1, -1, -1):
            if variances_h[i] >= threshold:
                bottom = i
                break

        # Find left edge: scan from left until variance exceeds threshold
        left = 0
        for j in range(width):
            if variances_v[j] >= threshold:
                left = j
                break

        # Find right edge: scan from right until variance exceeds threshold
        right = width - 1
        for j in range(width - 1, -1, -1):
            if variances_v[j] >= threshold:
                right = j
                break

        bboxes.append((top, bottom, left, right))

    # Combine bboxes: take the intersection (most conservative)
    # This means taking the minimum foreground region across all channels
    final_top = max(bbox[0] for bbox in bboxes)
    final_bottom = min(bbox[1] for bbox in bboxes)
    final_left = max(bbox[2] for bbox in bboxes)
    final_right = min(bbox[3] for bbox in bboxes)

    # Create solid rectangular mask
    combined_mask = np.zeros((height, width), dtype=np.float32)
    if final_top <= final_bottom and final_left <= final_right:
        combined_mask[final_top:final_bottom+1, final_left:final_right+1] = 1.0

    return combined_mask



def generate_mask(image: np.ndarray) -> np.ndarray:
    return variance_background_removal(image).astype(np.uint8) * 255

# Parse the provided color space string and convert it to a ColorSpace enum.
def parse_colorspace(string: str):
    try:
        return ColorSpace(string.upper())
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid color space: {string}.")

# Parse a distance string by iterating over available simple distances.
def parse_distance(string: str):
    for name, distance in distances.iter_simple_distances():
        if name == string:
            return distance
    raise argparse.ArgumentTypeError(f"Invalid distance: {string}.")

# Parse command-line arguments and return them.
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_path", type=str)  # Dataset directory path.
    parser.add_argument("queries_path", type=str)  # Queries directory path.

    parser.add_argument("--gamma", type=float, default=0.8)  # Gamma correction factor.
    parser.add_argument("--color_spaces", type=parse_colorspace, nargs='+', default=[ColorSpace.LAB])
    parser.add_argument("--bins", type=int, default=32)  # Number of bins in histogram.
    parser.add_argument("--distance", type=parse_distance, default=distances.l1_distance)
    parser.add_argument("--remove_side_ratio", type=float, default=0.0)  # Preprocessing: remove edges (0.15 = 15%)
    parser.add_argument("--grid_split", type=int, nargs=2, default=[4, 4])  # Grid split (e.g., 2 2 for 2x2)
    parser.add_argument("--k", type=int, default=10)  # Number of top results to retrieve.
    parser.add_argument("--pkl_output_path", type=str, default=None)  # Output path for pickled predictions.
    parser.add_argument("--generate_masks", default=False, action='store_true')  # Output path for pickled predictions.

    return parser.parse_args()

# Load query images and ground truth from the provided queries_path.
def load_queries(queries_path: str, gen_mask=False):
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

        # Load mask if available, otherwise create full white mask
        # if gen_mask:
        #     mask = generate_mask(image)
        # else:
        #     mask_path = Path(image_path).with_suffix('.png')
        #     if mask_path.exists():
        #         mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        #     else:
        #         mask = np.ones(image.shape[:2], dtype=np.uint8) * 255

        mask_path = Path(image_path).with_suffix('.png')
        if mask_path.exists():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        else:
            mask = np.ones(image.shape[:2], dtype=np.uint8) * 255

        queries.append({
            'image': image,
            'mask': mask,
            'name': filename,
            'id': int(Path(image_path).stem)
        })

    return queries, gt

# Display the query image and its corresponding results.
def show_results(query, results):
    plt.figure()  # Create a new figure for the query image.
    plt.title('Query')
    plt.imshow(cv2.cvtColor(query['image'], cv2.COLOR_BGR2RGB))
    plt.show()

    for i, entry in enumerate(results, start=1):
        plt.figure()  # Create a new figure for each result image.
        plt.title(f'Top {i}')
        plt.imshow(cv2.cvtColor(entry['image'], cv2.COLOR_BGR2RGB))
        plt.show()

# Main function to execute the query-by-sample process.
def main():
    args = parse_arguments()  # Parse command-line arguments.

    print("Loading database...")
    database = ImageDatabase.load(args.dataset_path)  # Load the image database.
    print("Loading queries...")
    queries, ground_truth = load_queries(args.queries_path, args.generate_masks)  # Load query images and ground truth.

    print("Setting up descriptor maker...")
    print(f"  Bins: {args.bins}")
    print(f"  Color Spaces: {[cs.value for cs in args.color_spaces]}")
    print(f"  Preprocessing: CropToMask")
    print(f"  Distance: {args.distance.__name__ if hasattr(args.distance, '__name__') else 'custom'}")
    print(f"  Gamma: {args.gamma}")

    # Count total channels
    total_channels = sum(4 if cs == ColorSpace.CMYK else 3 for cs in args.color_spaces)
    all_channels = list(range(total_channels))

    # Setup block splitter
    if args.grid_split:
        block_splitter = GridImageBlockSplitter(tuple(args.grid_split))
        print(f"  Block splitter: Grid {tuple(args.grid_split)}")
    else:
        block_splitter = IdentityImageBlockSplitter()
        print(f"  Block splitter: Identity (no splitting)")

    # Setup histogram computer (Histogram1D)
    histogram_computer = Histogram1D(
        channels=all_channels,
        bins=args.bins,
        weight_strategy=None,
        block_splitter=block_splitter,
        range_=None
    )

    # Setup preprocessing: Gamma -> OpenMask -> CropToMask
    preprocess = Preprocess([
        CropToMask(),
        ApplyGamma(gamma=args.gamma),
        # OpenMask(remove_side_ratio=0.15),
    ])

    # Create descriptor maker
    descriptor_maker = ImageDescriptorMaker(
        color_spaces=args.color_spaces,
        histogram_computer=histogram_computer,
        preprocess=preprocess
    )

    print("Computing descriptors...")
    database.reset_descriptors_and_distances()  # Reset any existing descriptors and distances.
    database.compute_descriptors(descriptor_maker)  # Compute image descriptors for the database.

    descriptors = [image.descriptor for image in database.images]

    print("Querying...")
    results = []
    for query in queries:
        query_descriptor = descriptor_maker.make_descriptor(query['image'], query['mask'])
        descriptors.append(query_descriptor)
        top_k = database.query(query_descriptor, args.distance, k=args.k)  # Retrieve top-k nearest images.
        results.append(top_k)

    if ground_truth is not None:
        print("Ground truth is present: evaluating...")
        clean_results = [[image.id for image in top_k] for top_k in results]
        mapk1 = mapk(ground_truth, clean_results, k=1)  # Compute map@1 metric.
        print(f"map@k=1 is {mapk1:.5f}")
        if args.k >= 5:
            mapk5 = mapk(ground_truth, clean_results, k=5)  # Compute map@5 metric.
            print(f"map@k=5 is {mapk5:.5f}")

    # Generate a pickle file with the predictions if output path is provided.
    if args.pkl_output_path:
        print("Dumping predictions pkl...")
        clean_results = [[image.id for image in top_k] for top_k in results]
        queries_indexes = [[query['id']] for query in queries]
        pkl_content = [queries_indexes, clean_results]

        # Dump the cleaned results into the provided pickle file path.
        with open(args.pkl_output_path, "wb") as f:
            pickle.dump(clean_results, f)

    # for descriptor in descriptors:
        # print(descriptor)

# Execute the main function when the script is run.
if __name__ == "__main__":
    main()
