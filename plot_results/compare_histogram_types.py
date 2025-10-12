import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_json_data(filepath):
    """Load JSON data and filter out incomplete entries."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    # Filter out entries without map@k5
    filtered_data = [item for item in data if 'map@k5' in item and item.get('map@k5') is not None]
    return filtered_data

def extract_performance_by_distance(data, histogram_type):
    """Extract average performance per distance metric."""
    distance_performance = {}
    
    for item in data:
        if 'distance' not in item or 'map@k5' not in item:
            continue
            
        distance = item['distance']
        map_k5 = item['map@k5']
        map_k1 = item.get('map@k1', 0)
        
        if distance not in distance_performance:
            distance_performance[distance] = {
                'map@k5': [],
                'map@k1': [],
                'time': []
            }
        
        distance_performance[distance]['map@k5'].append(map_k5)
        distance_performance[distance]['map@k1'].append(map_k1)
        distance_performance[distance]['time'].append(item.get('time_total', 0))
    
    # Calculate averages
    avg_performance = {}
    for distance, metrics in distance_performance.items():
        avg_performance[distance] = {
            'map@k5': np.mean(metrics['map@k5']) if metrics['map@k5'] else 0,
            'map@k1': np.mean(metrics['map@k1']) if metrics['map@k1'] else 0,
            'time': np.mean(metrics['time']) if metrics['time'] else 0,
            'histogram_type': histogram_type
        }
    
    return avg_performance

def get_top_configs(data, n=5):
    """Get top N configurations by map@k5."""
    valid_data = [item for item in data if 'map@k5' in item and item.get('map@k5') is not None]
    sorted_data = sorted(valid_data, key=lambda x: x['map@k5'], reverse=True)
    return sorted_data[:n]

def main():
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import json

    # Function to create a descriptive label for each block splitter configuration
    def get_splitter_config(row):
        splitter_class = row.get('histogram_computer.block_splitter.class')
        if splitter_class == 'GridImageBlockSplitter':
            shape = row.get('histogram_computer.block_splitter.shape')
            return f'Block {shape[0]}x{shape[1]}'
        elif splitter_class == 'PyramidImageBlockSplitter':
            level = len(row.get('histogram_computer.block_splitter.shapes'))
            return 'Pyramid L' + str(level)
        elif splitter_class == 'IdentityImageBlockSplitter':
            return 'Identity'
        return 'Unknown'

    # Load the JSON file
    with open('aggregated1d2d3d.json', 'r') as f:
        data = json.load(f)

    # Create a DataFrame from the JSON data
    df = pd.json_normalize(data)

    # Create the new descriptive column for splitter configurations
    df['splitter_config'] = df.apply(get_splitter_config, axis=1)

    # Aggregate the data by the new configuration and calculate the mean of the metrics
    aggregated_data = df.groupby('splitter_config').agg(
    mean_map_k1=('map@k1', 'mean'),
    max_map_k1=('map@k1', 'max')
).reset_index()

    
    # Set up the plot with a green palette
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle('Performance Comparison of Block Splitter Configurations (map@k1)', fontsize=20)

    # Plot for Mean of map@k1
    ax1 = axes[0]
    sns.barplot(ax=ax1, data=aggregated_data, x='splitter_config', y='mean_map_k1', palette='Greens_d')
    ax1.set_title('Average map@k1', fontsize=14)
    ax1.set_xlabel('Block Splitting Configuration', fontsize=12)
    ax1.set_ylabel('Mean map@k1', fontsize=12)
    # do the same range for both y axes
    ax1.set_ylim(0, 0.95)
    ax1.tick_params(axis='x', rotation=10)
    for container in ax1.containers:
        ax1.bar_label(container, fmt='%.3f')

    # Plot for Max of map@k1
    ax2 = axes[1]
    sns.barplot(ax=ax2, data=aggregated_data, x='splitter_config', y='max_map_k1', palette='Greens_d')
    ax2.set_title('Maximum map@k1', fontsize=14)
    ax2.set_xlabel('Block Splitting Configuration', fontsize=12)
    ax2.set_ylabel('Max map@k1', fontsize=12)
    ax2.set_ylim(0, 0.95)
    ax2.tick_params(axis='x', rotation=10)
    for container in ax2.containers:
        ax2.bar_label(container, fmt='%.3f')

    # Adjust layout and save the figure
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('block_splitter_mapk1_comparison.png')

    print("Plot saved as block_splitter_mapk1_comparison.png")
    print("\nAggregated Data:")
    print(aggregated_data)
if __name__ == "__main__":
    main()