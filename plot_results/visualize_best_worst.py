"""
Script to visualize the best and worst configurations from aggregated.json results.
Shows top 10 and bottom 10 configurations based on mAP@k5.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pandas as pd


def load_results(json_path='aggregated.json'):
    """Load results from JSON file and filter out zero scores and duplicates."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Filter out configurations where both mAP@k1 and mAP@k5 are 0
    filtered_data = [
        config for config in data 
        if not (config.get('map@k1', 0) == 0 and config.get('map@k5', 0) == 0)
    ]
    
    print(f"Total configurations: {len(data)}")
    print(f"Filtered out {len(data) - len(filtered_data)} configurations with mAP@k1=0 and mAP@k5=0")
    
    # Remove duplicates by keeping the first occurrence of each unique configuration
    # Create a unique key for each configuration based on all parameters
    seen_configs = {}
    unique_data = []
    
    for config in filtered_data:
        # Create a unique key from the configuration parameters
        key = (
            config['gamma_correction'],
            config['blur_image'],
            tuple(sorted(config['color_spaces'])),  # Sort to handle different orderings
            config['bins'],
            config['keep_or_discard'],
            config['weights'],
            config['distance']
        )
        
        # If we haven't seen this configuration, add it
        if key not in seen_configs:
            seen_configs[key] = config
            unique_data.append(config)
    
    print(f"Filtered out {len(filtered_data) - len(unique_data)} duplicate configurations")
    print(f"Remaining unique configurations: {len(unique_data)}")
    
    return unique_data


def create_config_label(config):
    """Create a readable label for a configuration."""
    # Format color spaces
    color_spaces = '+'.join(config['color_spaces'])
    
    # Shorten weight names
    weight = config['weights'].replace('CENTER_CROP_', 'CC')
    
    # Shorten distance names
    distance = config['distance'].replace('_distance', '').replace('_', '')
    
    label = f"{color_spaces}|B{config['bins']}|{weight}|{distance}"
    
    return label


def visualize_top_bottom(results, metric='map@k5', output_dir='plot_results'):
    """Visualize top 10 and bottom 10 configurations."""
    Path(output_dir).mkdir(exist_ok=True)
    
    # Sort by metric
    sorted_results = sorted(results, key=lambda x: x[metric], reverse=True)
    
    # Get top 10 and bottom 10
    top_10 = sorted_results[:10]
    bottom_10 = sorted_results[-10:]
    
    # Create labels and values
    top_labels = [create_config_label(config) for config in top_10]
    top_values = [config[metric] for config in top_10]
    
    bottom_labels = [create_config_label(config) for config in bottom_10]
    bottom_values = [config[metric] for config in bottom_10]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
    
    # Top 10 configurations
    colors_top = plt.cm.Greens(np.linspace(0.5, 0.9, 10))
    bars1 = ax1.barh(range(10), top_values, color=colors_top, edgecolor='black', linewidth=1.5)
    ax1.set_yticks(range(10))
    ax1.set_yticklabels(top_labels, fontsize=9)
    ax1.set_xlabel(f'{metric.upper()}', fontsize=12, fontweight='bold')
    ax1.set_title(f'Top 10 Configurations by {metric.upper()}', fontsize=14, fontweight='bold', color='green')
    ax1.grid(axis='x', alpha=0.3, linestyle='--')
    ax1.invert_yaxis()  # Best on top
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars1, top_values)):
        ax1.text(val + 0.005, i, f'{val:.4f}', va='center', fontweight='bold', fontsize=10)
    
    # Add rank numbers
    for i in range(10):
        ax1.text(-0.02, i, f'#{i+1}', va='center', ha='right', fontweight='bold', 
                fontsize=11, color='green')
    
    # Bottom 10 configurations
    colors_bottom = plt.cm.Reds(np.linspace(0.5, 0.9, 10))
    bars2 = ax2.barh(range(10), bottom_values, color=colors_bottom, edgecolor='black', linewidth=1.5)
    ax2.set_yticks(range(10))
    ax2.set_yticklabels(bottom_labels, fontsize=9)
    ax2.set_xlabel(f'{metric.upper()}', fontsize=12, fontweight='bold')
    ax2.set_title(f'Bottom 10 Configurations by {metric.upper()}', fontsize=14, fontweight='bold', color='red')
    ax2.grid(axis='x', alpha=0.3, linestyle='--')
    ax2.invert_yaxis()  # Worst on top
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars2, bottom_values)):
        ax2.text(val + 0.005, i, f'{val:.4f}', va='center', fontweight='bold', fontsize=10)
    
    # Add rank numbers (from bottom)
    total = len(results)
    for i in range(10):
        rank = total - 9 + i
        ax2.text(-0.02, i, f'#{rank}', va='center', ha='right', fontweight='bold', 
                fontsize=11, color='red')
    
    # Overall title
    fig.suptitle(f'Best and Worst Configurations (Total: {len(results)} configs)', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    # Save
    output_path = Path(output_dir) / 'top_bottom_10_configurations.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved visualization: {output_path}")
    
    plt.show()
    
    return top_10, bottom_10


def print_detailed_results(top_10, bottom_10, metric='map@k5', output_file=None):
    """Print detailed information about top and bottom configurations."""
    lines = []
    
    lines.append("=" * 100)
    lines.append(f"TOP 10 CONFIGURATIONS (sorted by {metric.upper()})")
    lines.append("=" * 100)
    
    for i, config in enumerate(top_10, 1):
        lines.append(f"\n#{i} - {metric.upper()}: {config[metric]:.4f}")
        lines.append(f"   Color Spaces: {', '.join(config['color_spaces'])}")
        lines.append(f"   Bins: {config['bins']}")
        lines.append(f"   Weight Strategy: {config['weights']}")
        lines.append(f"   Distance: {config['distance']}")
        lines.append(f"   Gamma: {config['gamma_correction']}, Blur: {config['blur_image']}")
        lines.append(f"   Keep/Discard: {config['keep_or_discard']}")
        if 'map@k1' in config:
            lines.append(f"   mAP@k1: {config['map@k1']:.4f}, mAP@k5: {config['map@k5']:.4f}")
    
    lines.append("\n" + "=" * 100)
    lines.append(f"BOTTOM 10 CONFIGURATIONS (sorted by {metric.upper()})")
    lines.append("=" * 100)
    
    for i, config in enumerate(bottom_10, 1):
        lines.append(f"\n#{i} (from bottom) - {metric.upper()}: {config[metric]:.4f}")
        lines.append(f"   Color Spaces: {', '.join(config['color_spaces'])}")
        lines.append(f"   Bins: {config['bins']}")
        lines.append(f"   Weight Strategy: {config['weights']}")
        lines.append(f"   Distance: {config['distance']}")
        lines.append(f"   Gamma: {config['gamma_correction']}, Blur: {config['blur_image']}")
        lines.append(f"   Keep/Discard: {config['keep_or_discard']}")
        if 'map@k1' in config:
            lines.append(f"   mAP@k1: {config['map@k1']:.4f}, mAP@k5: {config['map@k5']:.4f}")
    
    lines.append("\n" + "=" * 100)
    
    # Print to console
    for line in lines:
        print(line)
    
    # Save to file if specified
    if output_file:
        with open(output_file, 'w') as f:
            f.write('\n'.join(lines))
        print(f"\nSaved detailed results to: {output_file}")


def analyze_best_patterns(top_10, output_file=None):
    """Analyze patterns in the best configurations."""
    lines = []
    
    lines.append("=" * 100)
    lines.append("ANALYSIS OF TOP 10 CONFIGURATIONS")
    lines.append("=" * 100)
    
    # Color spaces
    color_spaces_count = {}
    for config in top_10:
        cs = '+'.join(sorted(config['color_spaces']))
        color_spaces_count[cs] = color_spaces_count.get(cs, 0) + 1
    
    lines.append("\nMost common color space combinations:")
    for cs, count in sorted(color_spaces_count.items(), key=lambda x: x[1], reverse=True):
        lines.append(f"  {cs}: {count} times")
    
    # Bins
    bins_count = {}
    for config in top_10:
        bins = config['bins']
        bins_count[bins] = bins_count.get(bins, 0) + 1
    
    lines.append("\nMost common bin sizes:")
    for bins, count in sorted(bins_count.items(), key=lambda x: x[1], reverse=True):
        lines.append(f"  {bins} bins: {count} times")
    
    # Weights
    weights_count = {}
    for config in top_10:
        weight = config['weights']
        weights_count[weight] = weights_count.get(weight, 0) + 1
    
    lines.append("\nMost common weight strategies:")
    for weight, count in sorted(weights_count.items(), key=lambda x: x[1], reverse=True):
        lines.append(f"  {weight}: {count} times")
    
    # Distances
    distance_count = {}
    for config in top_10:
        dist = config['distance']
        distance_count[dist] = distance_count.get(dist, 0) + 1
    
    lines.append("\nMost common distance functions:")
    for dist, count in sorted(distance_count.items(), key=lambda x: x[1], reverse=True):
        lines.append(f"  {dist}: {count} times")
    
    lines.append("\n" + "=" * 100)
    
    # Print to console
    for line in lines:
        print(line)
    
    # Append to file if specified
    if output_file:
        with open(output_file, 'a') as f:
            f.write('\n\n' + '\n'.join(lines))


def main():
    """Main function."""
    json_path = 'aggregated.json'
    output_dir = '/'
    results_file = f'{output_dir}/top_bottom_10_results.txt'
    
    print(f"Loading results from: {json_path}")
    results = load_results(json_path)
    print(f"Loaded {len(results)} configurations")
    
    # Visualize top and bottom 10
    print("\nCreating visualization...")
    top_10, bottom_10 = visualize_top_bottom(results, metric='map@k5', output_dir=output_dir)
    
    # Print detailed results and save to file
    print("\nGenerating detailed results...")
    print_detailed_results(top_10, bottom_10, metric='map@k5', output_file=results_file)
    
    # Analyze patterns and append to file
    print("\nAnalyzing patterns...")
    analyze_best_patterns(top_10, output_file=results_file)
    
    print(f"\nResults saved to: {results_file}")
    print("\nDone!")


if __name__ == "__main__":
    main()
