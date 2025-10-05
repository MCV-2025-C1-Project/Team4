import argparse
import ot
import os
from typing import Any, Callable, Dict, List
import numpy as np
import pickle
from functools import partial
import csv
from libs_week1.descriptor import ImageDescriptor
import libs_week1.distances as dist
from query_by_sample import load_database, load_queries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def add_descriptors_to_dataset(dataset: list[dict[str, Any]], descriptor_maker: ImageDescriptor):
    for entry in dataset:
        entry['descriptor'] = descriptor_maker.compute_descriptor(entry['image'])

def find_closest(query: Dict[str, Any], dataset: List[Dict[str, Any]], distance_func: Callable, k: int):
    for entry in dataset:
        distance = distance_func(entry['descriptor'], query['descriptor'])
        entry['distance'] = distance

    similarity_functions = ['hist_intersection', 'hellinger_similarity']

    if isinstance(distance_func, partial):
        func_name = distance_func.func.__name__
    else:
        func_name = distance_func.__name__

    reverse_sort = func_name in similarity_functions

    return list(sorted(dataset, key=lambda e: e['distance'], reverse=reverse_sort))[:k]

def calculate_average_precision(ground_truth_indices: List[int], predicted_indices: List[int]) -> float:
    precision_at_k = []
    num_correct = 0
    for i, p in enumerate(predicted_indices):
        if p in ground_truth_indices:
            num_correct += 1
            precision_at_k.append(num_correct / (i + 1))

    if not precision_at_k:
        return 0.0

    return np.mean(precision_at_k)

def run_evaluation(database, queries, ground_truth, descriptor_maker, top_k, csv_writer):
    add_descriptors_to_dataset(database, descriptor_maker)
    add_descriptors_to_dataset(queries, descriptor_maker)

    distance_functions_to_test = {
        "Euclidean Distance": dist.euclidean_distance,
        "L1 Distance": dist.l1_distance,
        "X^2 Distance": dist.x2_distance,
        "Histogram Intersection": dist.hist_intersection,
        "Hellinger Similarity": dist.hellinger_similarity,
        "KL Divergence": dist.kl_divergence,
        "Jensen Shannon Divergence": dist.jensen_shannon_divergence
    }

    color_space_channels = {'GRAY': 1, 'HSV': 3, 'RGB': 3, 'LAB': 3, 'YCRCB': 3, 'HLS': 3}
    num_channels = color_space_channels[descriptor_maker.color_space.upper()]
    bins_per_channel = descriptor_maker.bins_per_channel

    distance_functions_to_test["Earth Mover's Distance"] = partial(
        dist.emd_multichannel,
        num_channels=num_channels,
        bins_per_channel=bins_per_channel
    )

    print(f"\nEvaluating with bin_per_channel={bins_per_channel} and color_space={descriptor_maker.color_space}...")
    for dist_name, dist_func in distance_functions_to_test.items():
        average_precisions = []
        for i, query in enumerate(queries):
            closest_k = find_closest(query, database, dist_func, k=top_k)
            predicted_indices = [int(os.path.splitext(entry['name'])[0].split('_')[-1]) for entry in closest_k]

            gt_indices = ground_truth[i]
            if not isinstance(gt_indices, list):
                gt_indices = [gt_indices]

            ap = calculate_average_precision(gt_indices, predicted_indices)
            average_precisions.append(ap)

        mean_ap = np.mean(average_precisions)
        print(f"  Result for {dist_name}:")
        print(f"    Mean Average Precision (mAP) @{top_k}: {mean_ap:.4f}")
        
        csv_writer.writerow([descriptor_maker.color_space, bins_per_channel, dist_name, f"{mean_ap:.4f}"])

def create_plots(results_file, top_k):

    print("\nGenerating plots from results...")
    df = pd.read_csv(results_file)
    map_column = f"mAP@{top_k}"

    df_sorted = df.sort_values(by=map_column, ascending=False).head(15)
    df_sorted['Combination'] = df_sorted['Color Space'] + " " + \
                               df_sorted['Bins per Channel'].astype(str) + " " + \
                               df_sorted['Distance Function']
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x=map_column, y='Combination', data=df_sorted, palette='viridis')
    plt.title('Top 15 Performing Combinations')
    plt.xlabel(f'Mean Average Precision (mAP @{top_k})')
    plt.ylabel('Configuration')
    plt.tight_layout()
    plt.savefig('performance_summary.png')
    print("Saved performance_summary.png")

    fig, axes = plt.subplots(2, 1, figsize=(12, 12))
    
    avg_dist = df.groupby('Distance Function')[map_column].mean().sort_values(ascending=False).reset_index()
    sns.barplot(ax=axes[0], x=map_column, y='Distance Function', data=avg_dist, palette='plasma')
    axes[0].set_title('Average Performance by Distance Function')
    
    avg_color = df.groupby('Color Space')[map_column].mean().sort_values(ascending=False).reset_index()
    sns.barplot(ax=axes[1], x=map_column, y='Color Space', data=avg_color, palette='magma')
    axes[1].set_title('Average Performance by Color Space')
    
    plt.tight_layout()
    plt.savefig('average_performance.png')
    print("Saved average_performance.png")

    distance_functions = df['Distance Function'].unique()
    num_dist = len(distance_functions)
    fig, axes = plt.subplots(int(np.ceil(num_dist / 2)), 2, figsize=(15, 4 * np.ceil(num_dist / 2)))
    axes = axes.flatten()

    for i, dist_func in enumerate(distance_functions):
        subset = df[df['Distance Function'] == dist_func]
        pivot_table = subset.pivot_table(values=map_column, index='Color Space', columns='Bins per Channel')
        sns.heatmap(pivot_table, ax=axes[i], annot=True, fmt=".3f", cmap="YlGnBu")
        axes[i].set_title(dist_func)

    for i in range(num_dist, len(axes)):
        fig.delaxes(axes[i])
        
    plt.tight_layout()
    plt.savefig('heatmap_by_distance_function.png')
    print("Saved heatmap_by_distance_function.png")
    plt.close('all')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_path", type=str)
    parser.add_argument("queries_path", type=str)
    parser.add_argument("-k", "--top_k", type=int, default=1)
    parser.add_argument("--bins_per_channel", type=int, default=32)
    parser.add_argument("--color_space", type=str, default="HSV")
    parser.add_argument("--test", action="store_true", help="Run with a range of bins and color spaces")
    parser.add_argument("--output_file", type=str, default="results1.csv", help="Path to save the results CSV file")
    args = parser.parse_args()

    database = load_database(args.dataset_path)
    queries = load_queries(args.queries_path)

    gt_path = os.path.join(args.queries_path, "gt_corresps.pkl")
    try:
        with open(gt_path, 'rb') as f:
            ground_truth = pickle.load(f)
    except FileNotFoundError:
        print(f"Error: Ground truth file not found at {gt_path}")
        return

    with open(args.output_file, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["Color Space", "Bins per Channel", "Distance Function", f"mAP@{args.top_k}"])

        if args.test:
            bins_per_channel_to_test = [16, 32, 64, 128, 255]
            color_spaces_to_test = ["GRAY", "HSV", "RGB", "LAB", "YCRCB", "HLS"]

            for bins in bins_per_channel_to_test:
                for cs in color_spaces_to_test:
                    descriptor_maker = ImageDescriptor(color_space=cs, bins_per_channel=bins, normalize_histograms=True)
                    run_evaluation(database, queries, ground_truth, descriptor_maker, args.top_k, csv_writer)
        else:
            descriptor_maker = ImageDescriptor(color_space=args.color_space, bins_per_channel=args.bins_per_channel, normalize_histograms=True)
            run_evaluation(database, queries, ground_truth, descriptor_maker, args.top_k, csv_writer)

    print(f"\nResults saved to {args.output_file}")
    
    create_plots(args.output_file, args.top_k)

if __name__ == "__main__":
    main()