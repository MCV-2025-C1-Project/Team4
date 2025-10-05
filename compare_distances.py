import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import defaultdict

def load_results(json_path='plot_results/aggregated.json'):
    """Load results from JSON file and filter out zero scores and duplicates."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Filter out configurations where both mAP@k1 and mAP@k5 are 0
    filtered_data = [
        config for config in data 
        if not (config.get('map@k1', 0) == 0 and config.get('map@k5', 0) == 0)
    ]
    
    # Remove duplicates
    seen_configs = {}
    unique_data = []
    
    for config in filtered_data:
        key = (
            config['gamma_correction'],
            config['blur_image'],
            tuple(sorted(config['color_spaces'])),
            config['bins'],
            config['keep_or_discard'],
            config['weights'],
            config['distance']
        )
        
        if key not in seen_configs:
            seen_configs[key] = config
            unique_data.append(config)
    
    print(f"Loaded {len(unique_data)} unique configurations")
    return unique_data

def analyze_by_distance(data):
    """Analyze performance grouped by distance function."""
    distance_stats = defaultdict(lambda: {'map@k1': [], 'map@k5': [], 'configs': []})
    
    for config in data:
        distance = config['distance']
        distance_stats[distance]['map@k1'].append(config['map@k1'])
        distance_stats[distance]['map@k5'].append(config['map@k5'])
        distance_stats[distance]['configs'].append(config)
    
    # Calculate statistics for each distance function
    results = {}
    for distance, stats in distance_stats.items():
        results[distance] = {
            'mean_k1': np.mean(stats['map@k1']),
            'mean_k5': np.mean(stats['map@k5']),
            'median_k1': np.median(stats['map@k1']),
            'median_k5': np.median(stats['map@k5']),
            'max_k1': np.max(stats['map@k1']),
            'max_k5': np.max(stats['map@k5']),
            'min_k1': np.min(stats['map@k1']),
            'min_k5': np.min(stats['map@k5']),
            'std_k1': np.std(stats['map@k1']),
            'std_k5': np.std(stats['map@k5']),
            'count': len(stats['map@k1']),
            'all_k1': stats['map@k1'],
            'all_k5': stats['map@k5']
        }
    
    return results

def create_comparison_plots(distance_results):
    """Create comparison plots for distance functions."""
    
    # Sort distances by mean mAP@k5
    sorted_distances = sorted(distance_results.items(), 
                              key=lambda x: x[1]['mean_k5'], 
                              reverse=True)
    distances = [d[0] for d in sorted_distances]
    
    # Create abbreviations for distance names
    abbreviations = {
        'canberra_distance': 'Canberra',
        'emd_distance': 'EMD',
        'l1_distance': 'L1',
        'x2_distance': 'χ²',
        'jensen_shannon_divergence': 'JS-Div',
        'kl_divergence': 'KL-Div',
        'hellinger_similarity': 'Hellinger',
        'multichannel_quadratic_form_distance': 'Multi-QF',
        'simple_quadratic_form_distance': 'Simple-QF',
        'euclidean_distance': 'Euclidean'
    }
    
    distance_labels = [abbreviations.get(d, d) for d in distances]
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    x = np.arange(len(distances))
    width = 0.35
    
    # 1. Mean performance
    mean_k1 = [distance_results[d]['mean_k1'] for d in distances]
    mean_k5 = [distance_results[d]['mean_k5'] for d in distances]
    
    bars1 = ax1.bar(x - width/2, mean_k1, width, label='mAP@k1', alpha=0.9, color='#3498db', edgecolor='white', linewidth=1.5)
    bars2 = ax1.bar(x + width/2, mean_k5, width, label='mAP@k5', alpha=0.9, color='#e74c3c', edgecolor='white', linewidth=1.5)
    
    ax1.set_xlabel('Distance Function', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Mean mAP', fontsize=13, fontweight='bold')
    ax1.set_title('Mean Performance by Distance Function', fontsize=15, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(distance_labels, rotation=45, ha='right', fontsize=11)
    ax1.legend(fontsize=11, frameon=True, shadow=True)
    ax1.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.7)
    ax1.set_facecolor('#f8f9fa')
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=9)
    
    # 2. Best performance
    max_k1 = [distance_results[d]['max_k1'] for d in distances]
    max_k5 = [distance_results[d]['max_k5'] for d in distances]
    
    bars1 = ax2.bar(x - width/2, max_k1, width, label='mAP@k1', alpha=0.9, color='#2ecc71', edgecolor='white', linewidth=1.5)
    bars2 = ax2.bar(x + width/2, max_k5, width, label='mAP@k5', alpha=0.9, color='#9b59b6', edgecolor='white', linewidth=1.5)
    
    ax2.set_xlabel('Distance Function', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Max mAP', fontsize=13, fontweight='bold')
    ax2.set_title('Best Performance by Distance Function', fontsize=15, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(distance_labels, rotation=45, ha='right', fontsize=11)
    ax2.legend(fontsize=11, frameon=True, shadow=True)
    ax2.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.7)
    ax2.set_facecolor('#f8f9fa')
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    output_path = 'plot_results/distance_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved comparison plot: {output_path}")
    
    return fig

def print_detailed_comparison(distance_results, output_file=None):
    """Print detailed comparison of distance functions."""
    
    # Sort by mean mAP@k5
    sorted_distances = sorted(distance_results.items(), 
                              key=lambda x: x[1]['mean_k5'], 
                              reverse=True)
    
    lines = []
    lines.append("=" * 100)
    lines.append("DETAILED COMPARISON OF DISTANCE FUNCTIONS")
    lines.append("=" * 100)
    lines.append("")
    
    for rank, (distance, stats) in enumerate(sorted_distances, 1):
        lines.append(f"#{rank} - {distance}")
        lines.append("-" * 100)
        lines.append(f"  Configurations tested: {stats['count']}")
        lines.append(f"  mAP@k1 - Mean: {stats['mean_k1']:.4f}, Median: {stats['median_k1']:.4f}, "
                    f"Max: {stats['max_k1']:.4f}, Min: {stats['min_k1']:.4f}, Std: {stats['std_k1']:.4f}")
        lines.append(f"  mAP@k5 - Mean: {stats['mean_k5']:.4f}, Median: {stats['median_k5']:.4f}, "
                    f"Max: {stats['max_k5']:.4f}, Min: {stats['min_k5']:.4f}, Std: {stats['std_k5']:.4f}")
        lines.append("")
    
    lines.append("=" * 100)
    
    # Print to console
    for line in lines:
        print(line)
    
    # Save to file if specified
    if output_file:
        with open(output_file, 'w') as f:
            f.write('\n'.join(lines))
        print(f"\nSaved detailed comparison to: {output_file}")

def main():
    print("Loading results from: plot_results/aggregated.json")
    data = load_results()
    
    print("\nAnalyzing performance by distance function...")
    distance_results = analyze_by_distance(data)
    
    print(f"\nFound {len(distance_results)} distance functions")
    
    print("\nCreating comparison plots...")
    create_comparison_plots(distance_results)
    
    print("\nGenerating detailed comparison...")
    results_file = 'plot_results/distance_comparison_results.txt'
    print_detailed_comparison(distance_results, output_file=results_file)
    
    print("\nDone!")

if __name__ == "__main__":
    main()
