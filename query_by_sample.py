import argparse
import os
from typing import Any
from libs_week1.average_precision import mapk
import cv2
from libs_week1.database import ImageDatabase
from libs_week1.descriptor import ColorSpace, ImageDescriptor, ImageDescriptorMaker, WeightStrategy
import libs_week1.distances
from matplotlib import pyplot as plt
from pathlib import Path
import pickle

def parse_colorspace(string: str):
    try:
        return ColorSpace(string.upper())
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid color space: {string}.")

def parse_weightstrategy(string: str):
    try:
        return WeightStrategy(string.upper())
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid color space: {string}.")

def parse_distance(string: str):
    for name, distance in distances.iter_simple_distances():
        if name == string:
            return distance
        
    raise argparse.ArgumentTypeError(f"Invalid distance: {string}.")
    

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_path", type=str)
    parser.add_argument("queries_path", type=str)
    
    parser.add_argument("--gamma", type=float, default=0.8)
    parser.add_argument("--color_spaces", type=parse_colorspace, nargs='+', default=[ColorSpace.LAB])
    parser.add_argument("--keep_or_discard", type=str, default='KKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKK') # so it uses all channels by default
    parser.add_argument("--weight_strategy", type=parse_weightstrategy, default=WeightStrategy.CENTER_CROP_15)
    parser.add_argument("--bins", type=int, default=32)
    parser.add_argument("--distance", type=parse_distance, default=distances.canberra_distance)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--pkl_output_path", type=str, default=None)

    return parser.parse_args()


def load_queries(queries_path: str):
    queries = []
    gt_path = os.path.join(queries_path, "gt_corresps.pkl")
    if gt_path:
        gt = pickle.load(open(gt_path, 'rb'))
    else:
        gt = None
    for filename in sorted(os.listdir(queries_path)):
        if not filename.endswith(".jpg"):
            continue

        image_path = os.path.join(queries_path, filename)
        image = cv2.imread(image_path)
        
        queries.append({
            'image': image,
            'name': filename,
            'id': int(Path(image_path).stem)
        })

    return queries, gt


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

    print("Loading database...")
    database = ImageDatabase.load(args.dataset_path)
    print("Loading queries...")
    queries, ground_truth = load_queries(args.queries_path)
    print("Computing descriptors...")
    descriptor_maker = ImageDescriptorMaker(
        blur_image=False,
        gamma_correction=args.gamma,
        color_spaces=args.color_spaces,
        keep_or_discard=args.keep_or_discard,
        weights=args.weight_strategy,
        bins=args.bins,
    )
    database.reset_descriptors_and_distances()
    database.compute_descriptors(descriptor_maker)

    print("Querying...")
    results = []
    for query in queries:
        query_descriptor = descriptor_maker.make_descriptor(query['image'])
        top_k = database.query(query_descriptor, args.distance, k=args.k)
        results.append(top_k)
        
    if ground_truth is not None:
        print("Ground truth is present: evaluating...")
        clean_results = [[image.id for image in top_k] for top_k in results]
        mapk1 = mapk(ground_truth, clean_results, k=1)
        print(f"map@k=1 is {mapk1:.5f}")
        if args.k >= 5:
            mapk5 = mapk(ground_truth, clean_results, k=5)
            print(f"map@k=5 is {mapk5:.5f}")
    
    # generate pkl with predictions
    if args.pkl_output_path:
        print("Dumping predictions pkl...")
        clean_results = [[image.id for image in top_k] for top_k in results]
        queries_indexes = [[query['id']] for query in queries]
        pkl_content = [queries_indexes, clean_results]
        
        # commented because we only have to give the results
        # pickle.dump(pkl_content, open(args.pkl_output_path, "wb"))
        pickle.dump(clean_results, open(args.pkl_output_path, "wb"))


if __name__ == "__main__":
    main()
