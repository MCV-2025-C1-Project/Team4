"""
Simple script to compare histograms between two images.
Shows full image histograms vs center-cropped histograms.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os


def extract_center(image, crop_percent=0.15):
    """Extract center region by removing borders."""
    H, W = image.shape[:2]
    border_h = int(H * crop_percent)
    border_w = int(W * crop_percent)
    center = image[border_h:H-border_h, border_w:W-border_w]
    return center


def compute_histogram(image, bins=16):
    """Compute RGB histogram for an image."""
    histograms = []
    colors = ['red', 'green', 'blue']
    
    for i, color in enumerate(colors):
        hist = cv2.calcHist([image], [i], None, [bins], [0, 256])
        hist = hist.flatten()
        hist = hist / (hist.sum() + 1e-10)  # Normalize
        histograms.append(hist)
    
    return histograms, colors


def canberra_distance(hist1, hist2):
    """Compute Canberra distance between two histograms."""
    epsilon = 1e-10
    distance = np.sum(np.abs(hist1 - hist2) / (np.abs(hist1) + np.abs(hist2) + epsilon))
    return distance


def visualize_comparison(image_path1, image_path2, crop_percent=0.15, bins=16, output_dir='results_analysis'):
    """Compare histograms between two images (full vs center-cropped)."""
    Path(output_dir).mkdir(exist_ok=True)
    
    # Read images
    img1 = cv2.imread(image_path1)
    img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    
    img2 = cv2.imread(image_path2)
    img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    
    # Extract centers
    center1 = extract_center(img1_rgb, crop_percent)
    center2 = extract_center(img2_rgb, crop_percent)
    
    # Compute histograms
    hist_full1, colors = compute_histogram(img1_rgb, bins)
    hist_full2, _ = compute_histogram(img2_rgb, bins)
    hist_center1, _ = compute_histogram(center1, bins)
    hist_center2, _ = compute_histogram(center2, bins)
    
    # Compute Canberra distances for each channel
    canberra_full_r = canberra_distance(hist_full1[0], hist_full2[0])
    canberra_full_g = canberra_distance(hist_full1[1], hist_full2[1])
    canberra_full_b = canberra_distance(hist_full1[2], hist_full2[2])
    canberra_full_avg = (canberra_full_r + canberra_full_g + canberra_full_b) / 3
    
    canberra_center_r = canberra_distance(hist_center1[0], hist_center2[0])
    canberra_center_g = canberra_distance(hist_center1[1], hist_center2[1])
    canberra_center_b = canberra_distance(hist_center1[2], hist_center2[2])
    canberra_center_avg = (canberra_center_r + canberra_center_g + canberra_center_b) / 3
    
    # Create figure
    fig = plt.figure(figsize=(20, 12))
    
    # === Row 1: Full Images ===
    ax1 = plt.subplot(3, 4, 1)
    ax1.imshow(img1_rgb)
    ax1.set_title(f'Image 1 - Full\n{Path(image_path1).name}', fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    ax2 = plt.subplot(3, 4, 2)
    ax2.imshow(img2_rgb)
    ax2.set_title(f'Image 2 - Full\n{Path(image_path2).name}', fontsize=12, fontweight='bold')
    ax2.axis('off')
    
    # === Row 2: Center Regions ===
    ax3 = plt.subplot(3, 4, 5)
    ax3.imshow(center1)
    ax3.set_title(f'Image 1 - Center ({crop_percent*100:.0f}% cropped)', fontsize=12, fontweight='bold', color='green')
    ax3.axis('off')
    
    ax4 = plt.subplot(3, 4, 6)
    ax4.imshow(center2)
    ax4.set_title(f'Image 2 - Center ({crop_percent*100:.0f}% cropped)', fontsize=12, fontweight='bold', color='green')
    ax4.axis('off')
    
    # === Row 3: Histogram Comparisons ===
    x = np.arange(bins)
    channel_names = ['Red', 'Green', 'Blue']
    canberra_full_distances = [canberra_full_r, canberra_full_g, canberra_full_b]
    canberra_center_distances = [canberra_center_r, canberra_center_g, canberra_center_b]
    
    # Store axes for y-axis synchronization
    axes_pairs = []
    
    # Full image histograms and center-cropped histograms side by side
    for i in range(3):
        # Full image histogram
        ax_full = plt.subplot(3, 4, 3 + i*4)
        ax_full.plot(x, hist_full1[i], color='blue', linewidth=2.5, alpha=0.7, label='Image 1', linestyle='-', marker='o', markersize=4)
        ax_full.plot(x, hist_full2[i], color='red', linewidth=2.5, alpha=0.7, label='Image 2', linestyle='-', marker='s', markersize=4)
        ax_full.set_title(f'{channel_names[i]} - Full Image\nCanberra Distance: {canberra_full_distances[i]:.4f}', 
                     fontsize=11, fontweight='bold')
        ax_full.set_xlabel('Bin Index', fontsize=10)
        ax_full.set_ylabel('Frequency', fontsize=10)
        ax_full.legend(loc='upper right', fontsize=9)
        ax_full.grid(alpha=0.3)
        ax_full.set_xlim([0, bins-1])
        
        # Center-cropped histogram
        ax_center = plt.subplot(3, 4, 4 + i*4)
        ax_center.plot(x, hist_center1[i], color='blue', linewidth=2.5, alpha=0.7, label='Image 1', linestyle='-', marker='o', markersize=4)
        ax_center.plot(x, hist_center2[i], color='red', linewidth=2.5, alpha=0.7, label='Image 2', linestyle='-', marker='s', markersize=4)
        ax_center.set_title(f'{channel_names[i]} - Center Only\nCanberra Distance: {canberra_center_distances[i]:.4f}', 
                     fontsize=11, fontweight='bold', color='green')
        ax_center.set_xlabel('Bin Index', fontsize=10)
        ax_center.set_ylabel('Frequency', fontsize=10)
        ax_center.legend(loc='upper right', fontsize=9)
        ax_center.grid(alpha=0.3)
        ax_center.set_xlim([0, bins-1])
        
        # Store the pair for y-axis synchronization
        axes_pairs.append((ax_full, ax_center))
    
    # Synchronize y-axis for each channel pair
    for ax_full, ax_center in axes_pairs:
        # Get the maximum y value from both plots
        y_max_full = max([line.get_ydata().max() for line in ax_full.get_lines()])
        y_max_center = max([line.get_ydata().max() for line in ax_center.get_lines()])
        y_max = max(y_max_full, y_max_center) * 1.1  # Add 10% padding
        
        # Set the same y-axis limits for both
        ax_full.set_ylim([0, y_max])
        ax_center.set_ylim([0, y_max])
    
    # Add overall title
    fig.suptitle(f'Histogram Comparison: Full Image vs Center-Cropped ({crop_percent*100:.0f}% border removal)',
                 fontsize=16, fontweight='bold', y=0.98)
    # Add overall title
    fig.suptitle(f'Histogram Comparison (Bins={bins}): Full Image vs Center-Cropped ({crop_percent*100:.0f}% border removal)\n'
                 f'Average Canberra Distance: Full Image = {canberra_full_avg:.4f}, Center Only = {canberra_center_avg:.4f}',
                 fontsize=15, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save
    output_path = os.path.join(output_dir, 'two_images_histogram_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved: {output_path}")
    
    # Print statistics
    print(f"\n{'='*80}")
    print(f"Comparing: {Path(image_path1).name} vs {Path(image_path2).name}")
    print(f"{'='*80}")
    print(f"Histogram bins: {bins}")
    print(f"Border removal: {crop_percent*100:.0f}%")
    print(f"\nCanberra Distance (lower = more similar):")
    print(f"  Full Image:")
    print(f"    Red:   {canberra_full_r:.4f}")
    print(f"    Green: {canberra_full_g:.4f}")
    print(f"    Blue:  {canberra_full_b:.4f}")
    print(f"    Average: {canberra_full_avg:.4f}")
    print(f"\n  Center Only:")
    print(f"    Red:   {canberra_center_r:.4f}")
    print(f"    Green: {canberra_center_g:.4f}")
    print(f"    Blue:  {canberra_center_b:.4f}")
    print(f"    Average: {canberra_center_avg:.4f}")
    print(f"\n  Difference: {abs(canberra_full_avg - canberra_center_avg):.4f}")
    print(f"{'='*80}\n")
    
    plt.show()


def main():
    """Compare two images from test_images folder."""
    test_images_dir = 'test_images'
    
    if not os.path.exists(test_images_dir):
        print(f"Error: Directory '{test_images_dir}' not found!")
        return
    
    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(Path(test_images_dir).glob(f'*{ext}'))
        image_files.extend(Path(test_images_dir).glob(f'*{ext.upper()}'))
    
    # Remove duplicates and sort
    image_files = sorted(list(set([str(f) for f in image_files])))
    
    if len(image_files) < 2:
        print(f"Need at least 2 images to compare. Found {len(image_files)}")
        return
    
    print(f"Found {len(image_files)} images")
    print(f"Comparing: {Path(image_files[0]).name} vs {Path(image_files[1]).name}")
    print("="*80)
    
    # Compare first two images with 16 bins
    visualize_comparison(image_files[0], image_files[1], crop_percent=0.15, bins=16)


if __name__ == "__main__":
    main()
