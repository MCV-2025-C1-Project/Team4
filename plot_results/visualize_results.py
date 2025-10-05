"""
Comprehensive visualization script for image retrieval results.
Analyzes and visualizes performance metrics across different configurations.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import argparse

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

def load_and_prepare_data(csv_file):
    """Load CSV and prepare data for analysis."""
    df = pd.read_csv(csv_file)
    
    # Extract mAP column name
    map_col = [col for col in df.columns if 'mAP' in col][0]
    
    # Convert mAP to numeric
    df[map_col] = pd.to_numeric(df[map_col], errors='coerce')
    
    return df, map_col

def print_summary_statistics(df, map_col):
    """Print comprehensive summary statistics."""
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    # Overall statistics
    print(f"\nOverall Performance:")
    print(f"  Mean {map_col}: {df[map_col].mean():.4f}")
    print(f"  Median {map_col}: {df[map_col].median():.4f}")
    print(f"  Std Dev: {df[map_col].std():.4f}")
    print(f"  Min: {df[map_col].min():.4f}")
    print(f"  Max: {df[map_col].max():.4f}")
    
    # Top 10 configurations
    print(f"\n{'='*80}")
    print("TOP 10 BEST CONFIGURATIONS")
    print("="*80)
    top_10 = df.nlargest(10, map_col)
    for idx, row in top_10.iterrows():
        print(f"{row[map_col]:.4f} | {row['Color Space']:6s} | Bins: {row['Bins per Channel']:3d} | {row['Distance Function']}")
    
    # Bottom 10 configurations
    print(f"\n{'='*80}")
    print("TOP 10 WORST CONFIGURATIONS")
    print("="*80)
    bottom_10 = df.nsmallest(10, map_col)
    for idx, row in bottom_10.iterrows():
        print(f"{row[map_col]:.4f} | {row['Color Space']:6s} | Bins: {row['Bins per Channel']:3d} | {row['Distance Function']}")
    
    # Best per color space
    print(f"\n{'='*80}")
    print("BEST CONFIGURATION PER COLOR SPACE")
    print("="*80)
    for cs in df['Color Space'].unique():
        best = df[df['Color Space'] == cs].nlargest(1, map_col).iloc[0]
        print(f"{cs:6s}: {best[map_col]:.4f} | Bins: {best['Bins per Channel']:3d} | {best['Distance Function']}")
    
    # Best per distance function
    print(f"\n{'='*80}")
    print("BEST CONFIGURATION PER DISTANCE FUNCTION")
    print("="*80)
    for dist in df['Distance Function'].unique():
        best = df[df['Distance Function'] == dist].nlargest(1, map_col).iloc[0]
        print(f"{dist:28s}: {best[map_col]:.4f} | {best['Color Space']:6s} | Bins: {best['Bins per Channel']:3d}")
    
    # Best per bins
    print(f"\n{'='*80}")
    print("BEST CONFIGURATION PER BIN SIZE")
    print("="*80)
    for bins in sorted(df['Bins per Channel'].unique()):
        best = df[df['Bins per Channel'] == bins].nlargest(1, map_col).iloc[0]
        print(f"Bins {bins:3d}: {best[map_col]:.4f} | {best['Color Space']:6s} | {best['Distance Function']}")

def create_visualizations(df, map_col, output_dir='results_analysis'):
    """Create comprehensive visualizations."""
    Path(output_dir).mkdir(exist_ok=True)
    
    # 1. Top 20 configurations bar plot
    print("\nGenerating Top 20 Configurations plot...")
    plt.figure(figsize=(14, 10))
    top_20 = df.nlargest(20, map_col).copy()
    top_20['Config'] = (top_20['Color Space'] + ' ' + 
                        top_20['Bins per Channel'].astype(str) + 'bins ' + 
                        top_20['Distance Function'])
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(top_20)))
    bars = plt.barh(range(len(top_20)), top_20[map_col], color=colors)
    plt.yticks(range(len(top_20)), top_20['Config'], fontsize=9)
    plt.xlabel(map_col, fontsize=12, fontweight='bold')
    plt.title('Top 20 Best Performing Configurations', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(f'{output_dir}/01_top20_configurations.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Performance by Color Space (boxplot)
    print("Generating Color Space comparison...")
    plt.figure(figsize=(12, 6))
    color_order = df.groupby('Color Space')[map_col].median().sort_values(ascending=False).index
    sns.boxplot(data=df, x='Color Space', y=map_col, order=color_order, palette='Set2')
    plt.title('Performance Distribution by Color Space', fontsize=14, fontweight='bold')
    plt.ylabel(map_col, fontsize=12)
    plt.xlabel('Color Space', fontsize=12)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/02_colorspace_boxplot.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Performance by Distance Function (boxplot)
    print("Generating Distance Function comparison...")
    plt.figure(figsize=(14, 6))
    dist_order = df.groupby('Distance Function')[map_col].median().sort_values(ascending=False).index
    sns.boxplot(data=df, x='Distance Function', y=map_col, order=dist_order, palette='Set3')
    plt.title('Performance Distribution by Distance Function', fontsize=14, fontweight='bold')
    plt.ylabel(map_col, fontsize=12)
    plt.xlabel('Distance Function', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/03_distance_boxplot.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Performance by Bins per Channel (line plot)
    print("Generating Bins per Channel analysis...")
    plt.figure(figsize=(12, 6))
    avg_by_bins = df.groupby('Bins per Channel')[map_col].agg(['mean', 'std', 'min', 'max'])
    plt.plot(avg_by_bins.index, avg_by_bins['mean'], 'o-', linewidth=2, markersize=8, label='Mean')
    plt.fill_between(avg_by_bins.index, 
                     avg_by_bins['mean'] - avg_by_bins['std'], 
                     avg_by_bins['mean'] + avg_by_bins['std'], 
                     alpha=0.3, label='±1 Std Dev')
    plt.plot(avg_by_bins.index, avg_by_bins['max'], 's--', label='Max', alpha=0.6)
    plt.plot(avg_by_bins.index, avg_by_bins['min'], 's--', label='Min', alpha=0.6)
    plt.xlabel('Bins per Channel', fontsize=12, fontweight='bold')
    plt.ylabel(map_col, fontsize=12, fontweight='bold')
    plt.title('Performance vs Number of Bins', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/04_bins_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Heatmap: Color Space vs Distance Function (average across bins)
    print("Generating Color Space vs Distance Function heatmap...")
    plt.figure(figsize=(14, 8))
    pivot = df.pivot_table(values=map_col, index='Distance Function', columns='Color Space', aggfunc='mean')
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='YlOrRd', cbar_kws={'label': map_col})
    plt.title('Average Performance: Color Space vs Distance Function', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/05_heatmap_colorspace_distance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 6. Heatmap: Bins vs Color Space (average across distance functions)
    print("Generating Bins vs Color Space heatmap...")
    plt.figure(figsize=(10, 8))
    pivot = df.pivot_table(values=map_col, index='Color Space', columns='Bins per Channel', aggfunc='mean')
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='viridis', cbar_kws={'label': map_col})
    plt.title('Average Performance: Color Space vs Bins per Channel', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/06_heatmap_colorspace_bins.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 7. Individual heatmaps for each distance function
    print("Generating individual distance function heatmaps...")
    distance_functions = df['Distance Function'].unique()
    n_dist = len(distance_functions)
    n_cols = 2
    n_rows = int(np.ceil(n_dist / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 5 * n_rows))
    axes = axes.flatten()
    
    for idx, dist_func in enumerate(distance_functions):
        subset = df[df['Distance Function'] == dist_func]
        pivot = subset.pivot_table(values=map_col, index='Color Space', columns='Bins per Channel')
        
        if pivot.empty:
            axes[idx].axis('off')
            continue
            
        sns.heatmap(pivot, ax=axes[idx], annot=True, fmt='.3f', cmap='RdYlGn', 
                   vmin=df[map_col].min(), vmax=df[map_col].max(),
                   cbar_kws={'label': map_col})
        axes[idx].set_title(f'{dist_func}', fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('Bins per Channel', fontsize=10)
        axes[idx].set_ylabel('Color Space', fontsize=10)
    
    # Hide unused subplots
    for idx in range(n_dist, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/07_heatmaps_by_distance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 8. Average performance comparison (grouped bar chart)
    print("Generating average performance comparison...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # By Color Space
    avg_color = df.groupby('Color Space')[map_col].mean().sort_values(ascending=False)
    axes[0].bar(range(len(avg_color)), avg_color.values, color=plt.cm.Set2(range(len(avg_color))))
    axes[0].set_xticks(range(len(avg_color)))
    axes[0].set_xticklabels(avg_color.index, rotation=0)
    axes[0].set_ylabel(map_col, fontweight='bold')
    axes[0].set_title('Average Performance by Color Space', fontweight='bold')
    axes[0].grid(axis='y', alpha=0.3)
    
    # By Distance Function
    avg_dist = df.groupby('Distance Function')[map_col].mean().sort_values(ascending=False)
    axes[1].bar(range(len(avg_dist)), avg_dist.values, color=plt.cm.Set3(range(len(avg_dist))))
    axes[1].set_xticks(range(len(avg_dist)))
    axes[1].set_xticklabels(avg_dist.index, rotation=45, ha='right', fontsize=9)
    axes[1].set_ylabel(map_col, fontweight='bold')
    axes[1].set_title('Average Performance by Distance Function', fontweight='bold')
    axes[1].grid(axis='y', alpha=0.3)
    
    # By Bins
    avg_bins = df.groupby('Bins per Channel')[map_col].mean().sort_values(ascending=False)
    axes[2].bar(range(len(avg_bins)), avg_bins.values, color=plt.cm.viridis(np.linspace(0, 1, len(avg_bins))))
    axes[2].set_xticks(range(len(avg_bins)))
    axes[2].set_xticklabels(avg_bins.index, rotation=0)
    axes[2].set_ylabel(map_col, fontweight='bold')
    axes[2].set_title('Average Performance by Bins per Channel', fontweight='bold')
    axes[2].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/08_average_performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 9. Performance distribution histogram
    print("Generating performance distribution...")
    plt.figure(figsize=(12, 6))
    plt.hist(df[map_col], bins=30, edgecolor='black', alpha=0.7, color='skyblue')
    plt.axvline(df[map_col].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {df[map_col].mean():.4f}')
    plt.axvline(df[map_col].median(), color='green', linestyle='--', linewidth=2, label=f'Median: {df[map_col].median():.4f}')
    plt.xlabel(map_col, fontsize=12, fontweight='bold')
    plt.ylabel('Frequency', fontsize=12, fontweight='bold')
    plt.title('Distribution of Performance Scores', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/09_performance_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nAll visualizations saved to '{output_dir}/' directory!")

def save_summary_report(df, map_col, output_dir='results_analysis'):
    """Save a text summary report."""
    Path(output_dir).mkdir(exist_ok=True)
    
    report_path = f'{output_dir}/summary_report.txt'
    
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("IMAGE RETRIEVAL PERFORMANCE ANALYSIS REPORT\n")
        f.write("="*80 + "\n\n")
        
        # Overall statistics
        f.write("OVERALL STATISTICS\n")
        f.write("-"*80 + "\n")
        f.write(f"Total configurations tested: {len(df)}\n")
        f.write(f"Mean {map_col}: {df[map_col].mean():.4f}\n")
        f.write(f"Median {map_col}: {df[map_col].median():.4f}\n")
        f.write(f"Std Dev: {df[map_col].std():.4f}\n")
        f.write(f"Min: {df[map_col].min():.4f}\n")
        f.write(f"Max: {df[map_col].max():.4f}\n\n")
        
        # Best overall
        best = df.nlargest(1, map_col).iloc[0]
        f.write("BEST OVERALL CONFIGURATION\n")
        f.write("-"*80 + "\n")
        f.write(f"Color Space: {best['Color Space']}\n")
        f.write(f"Bins per Channel: {best['Bins per Channel']}\n")
        f.write(f"Distance Function: {best['Distance Function']}\n")
        f.write(f"{map_col}: {best[map_col]:.4f}\n\n")
        
        # Top 10
        f.write("TOP 10 CONFIGURATIONS\n")
        f.write("-"*80 + "\n")
        for idx, row in df.nlargest(10, map_col).iterrows():
            f.write(f"{row[map_col]:.4f} | {row['Color Space']:6s} | Bins: {row['Bins per Channel']:3d} | {row['Distance Function']}\n")
        
        f.write("\n" + "="*80 + "\n")
    
    print(f"Summary report saved to '{report_path}'")

def main():
    parser = argparse.ArgumentParser(description='Visualize image retrieval experiment results')
    parser.add_argument('csv_file', type=str, default='results1.csv', nargs='?',
                       help='Path to the results CSV file (default: results.csv)')
    parser.add_argument('--output_dir', type=str, default='results_analysis',
                       help='Directory to save visualizations (default: results_analysis)')
    args = parser.parse_args()
    
    print(f"\nLoading results from: {args.csv_file}")
    df, map_col = load_and_prepare_data(args.csv_file)
    
    print(f"Loaded {len(df)} configurations")
    
    # Print summary statistics to console
    print_summary_statistics(df, map_col)
    
    # Create visualizations
    print(f"\n{'='*80}")
    print("GENERATING VISUALIZATIONS")
    print("="*80)
    create_visualizations(df, map_col, args.output_dir)
    
    # Save text report
    save_summary_report(df, map_col, args.output_dir)
    
    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\nCheck the '{args.output_dir}' directory for all outputs.")

if __name__ == "__main__":
    main()
