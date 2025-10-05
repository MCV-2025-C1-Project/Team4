"""
Visualization script to show the effect of center cropping techniques on images.
This helps understand how much of the border is removed or weighted differently.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os

def create_center_crop_weight(H, W, discard_borders=0.1):
    """Create center crop weight matrix (binary mask)."""
    assert 0 <= discard_borders < 0.5
    
    border_h = int(H * discard_borders)
    border_w = int(W * discard_borders)
    
    center_crop_weight = np.zeros((H, W))
    center_crop_weight[border_h:H-border_h, border_w:W-border_w] = 1.0
    return center_crop_weight


def create_pyramid_weight(H, W):
    """Create pyramid weight matrix (linear falloff from center)."""
    y = np.linspace(0, 1, H)
    x = np.linspace(0, 1, W)

    xs, ys = np.meshgrid(x, y)
    center_x, center_y = 0.5, 0.5
    
    dist_x = 1 - 2 * np.abs(xs - center_x)
    dist_y = 1 - 2 * np.abs(ys - center_y)
    
    pyramid_weight = np.minimum(dist_x, dist_y)
    return pyramid_weight


def create_cone_weight(H, W):
    """Create cone weight matrix (radial falloff from center)."""
    y = np.linspace(-1, 1, H)
    x = np.linspace(-1, 1, W)
    
    xs, ys = np.meshgrid(x, y)
    
    radius = np.sqrt(xs**2 + ys**2)
    
    max_radius = np.sqrt(2)
    
    cone_weight = np.clip(1 - radius / max_radius, 0, 1)
    return cone_weight


def apply_weight_visualization(image, weight_matrix):
    """Apply weight matrix to image for visualization."""
    # Expand weight matrix to match image channels
    weight_3d = np.stack([weight_matrix] * 3, axis=2)
    
    # Apply weights
    weighted_image = (image * weight_3d).astype(np.uint8)
    
    return weighted_image


def visualize_cropping(image_path, output_dir='cropping_visualizations'):
    """Visualize different cropping/weighting techniques on an image."""
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    # Read image
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    H, W = image.shape[:2]
    
    # Create different weight matrices
    center_crop_5 = create_center_crop_weight(H, W, discard_borders=0.05)
    center_crop_10 = create_center_crop_weight(H, W, discard_borders=0.1)
    center_crop_15 = create_center_crop_weight(H, W, discard_borders=0.15)
    pyramid_weight = create_pyramid_weight(H, W)
    cone_weight = create_cone_weight(H, W)
    
    # Apply weights to create visualizations
    img_center_5 = apply_weight_visualization(image_rgb, center_crop_5)
    img_center_10 = apply_weight_visualization(image_rgb, center_crop_10)
    img_center_15 = apply_weight_visualization(image_rgb, center_crop_15)
    img_pyramid = apply_weight_visualization(image_rgb, pyramid_weight)
    img_cone = apply_weight_visualization(image_rgb, cone_weight)
    
    # Create visualization
    fig = plt.figure(figsize=(20, 12))
    
    # Original image
    ax1 = plt.subplot(3, 3, 1)
    ax1.imshow(image_rgb)
    ax1.set_title('Original Image', fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    # Center Crop 5%
    ax2 = plt.subplot(3, 3, 2)
    ax2.imshow(img_center_5)
    ax2.set_title('Center Crop 5%\n(Removes 5% from each edge)', fontsize=14, fontweight='bold')
    ax2.axis('off')
    # Add border indicators
    border_h = int(H * 0.05)
    border_w = int(W * 0.05)
    rect = plt.Rectangle((border_w, border_h), W - 2*border_w, H - 2*border_h, 
                         fill=False, edgecolor='red', linewidth=3)
    ax2.add_patch(rect)
    
    # Center Crop 10%
    ax3 = plt.subplot(3, 3, 3)
    ax3.imshow(img_center_10)
    ax3.set_title('Center Crop 10%\n(Removes 10% from each edge)', fontsize=14, fontweight='bold')
    ax3.axis('off')
    border_h = int(H * 0.1)
    border_w = int(W * 0.1)
    rect = plt.Rectangle((border_w, border_h), W - 2*border_w, H - 2*border_h, 
                         fill=False, edgecolor='red', linewidth=3)
    ax3.add_patch(rect)
    
    # Center Crop 15%
    ax4 = plt.subplot(3, 3, 4)
    ax4.imshow(img_center_15)
    ax4.set_title('Center Crop 15%\n(Removes 15% from each edge)', fontsize=14, fontweight='bold')
    ax4.axis('off')
    border_h = int(H * 0.15)
    border_w = int(W * 0.15)
    rect = plt.Rectangle((border_w, border_h), W - 2*border_w, H - 2*border_h, 
                         fill=False, edgecolor='red', linewidth=3)
    ax4.add_patch(rect)
    ax4.add_patch(rect)
    
    # Pyramid Weight
    ax5 = plt.subplot(3, 3, 5)
    ax5.imshow(img_pyramid)
    ax5.set_title('Pyramid Weight\n(Linear falloff from center)', fontsize=14, fontweight='bold')
    ax5.axis('off')
    
    # Cone Weight
    ax6 = plt.subplot(3, 3, 6)
    ax6.imshow(img_cone)
    ax6.set_title('Cone Weight\n(Radial falloff from center)', fontsize=14, fontweight='bold')
    ax6.axis('off')
    
    # Weight matrices visualization
    ax7 = plt.subplot(3, 3, 7)
    im1 = ax7.imshow(center_crop_5, cmap='hot', vmin=0, vmax=1)
    ax7.set_title('Center Crop 5% - Weight Matrix', fontsize=12)
    ax7.axis('off')
    plt.colorbar(im1, ax=ax7, fraction=0.046)
    
    ax8 = plt.subplot(3, 3, 8)
    im2 = ax8.imshow(center_crop_10, cmap='hot', vmin=0, vmax=1)
    ax8.set_title('Center Crop 10% - Weight Matrix', fontsize=12)
    ax8.axis('off')
    plt.colorbar(im2, ax=ax8, fraction=0.046)
    
    ax9 = plt.subplot(3, 3, 9)
    im3 = ax9.imshow(center_crop_15, cmap='hot', vmin=0, vmax=1)
    ax9.set_title('Center Crop 15% - Weight Matrix', fontsize=12)
    ax9.axis('off')
    plt.colorbar(im3, ax=ax9, fraction=0.046)
    
    # Calculate pixel statistics
    total_pixels = H * W
    
    # Center Crop 5%
    kept_pixels_5 = (W - 2*int(W*0.05)) * (H - 2*int(H*0.05))
    discarded_pixels_5 = total_pixels - kept_pixels_5
    discarded_percent_5 = (discarded_pixels_5 / total_pixels) * 100
    
    # Center Crop 10%
    kept_pixels_10 = (W - 2*int(W*0.1)) * (H - 2*int(H*0.1))
    discarded_pixels_10 = total_pixels - kept_pixels_10
    discarded_percent_10 = (discarded_pixels_10 / total_pixels) * 100
    
    # Center Crop 15%
    kept_pixels_15 = (W - 2*int(W*0.15)) * (H - 2*int(H*0.15))
    discarded_pixels_15 = total_pixels - kept_pixels_15
    discarded_percent_15 = (discarded_pixels_15 / total_pixels) * 100
    
    # Add information text
    info_text = f"""
    Image Dimensions: {W} x {H} pixels | Total Pixels: {total_pixels:,}
    
    Center Crop 5%:  Keeps {kept_pixels_5:,} pixels | Discards {discarded_pixels_5:,} pixels ({discarded_percent_5:.2f}%)
    Center Crop 10%: Keeps {kept_pixels_10:,} pixels | Discards {discarded_pixels_10:,} pixels ({discarded_percent_10:.2f}%)
    Center Crop 15%: Keeps {kept_pixels_15:,} pixels | Discards {discarded_pixels_15:,} pixels ({discarded_percent_15:.2f}%)
    
    Binary Masks (0 or 1): Center crop methods completely ignore border pixels
    Gradient Weights (0 to 1): Pyramid and Cone give gradually less weight to edge pixels
    """
    
    fig.text(0.5, 0.02, info_text, ha='center', fontsize=11, 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    
    # Print statistics to console
    print(f"\n{'='*70}")
    print(f"Image Dimensions: {W} x {H} pixels | Total Pixels: {total_pixels:,}")
    print(f"{'='*70}")
    print(f"Center Crop 5%:")
    print(f"  - Keeps:    {kept_pixels_5:,} pixels ({100-discarded_percent_5:.2f}%)")
    print(f"  - Discards: {discarded_pixels_5:,} pixels ({discarded_percent_5:.2f}%)")
    print(f"\nCenter Crop 10%:")
    print(f"  - Keeps:    {kept_pixels_10:,} pixels ({100-discarded_percent_10:.2f}%)")
    print(f"  - Discards: {discarded_pixels_10:,} pixels ({discarded_percent_10:.2f}%)")
    print(f"\nCenter Crop 15%:")
    print(f"  - Keeps:    {kept_pixels_15:,} pixels ({100-discarded_percent_15:.2f}%)")
    print(f"  - Discards: {discarded_pixels_15:,} pixels ({discarded_percent_15:.2f}%)")
    print(f"{'='*70}\n")
    
    # Save figure
    image_name = Path(image_path).stem
    output_path = os.path.join(output_dir, f'{image_name}_cropping_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved visualization: {output_path}")
    
    # Also create detailed comparison showing all three crop levels
    fig2, axes = plt.subplots(2, 2, figsize=(18, 16))
    
    # Original with overlay showing 5% crop region
    axes[0, 0].imshow(image_rgb)
    border_h = int(H * 0.05)
    border_w = int(W * 0.05)
    rect = plt.Rectangle((border_w, border_h), W - 2*border_w, H - 2*border_h, 
                         fill=False, edgecolor='lime', linewidth=4, label='5% crop')
    axes[0, 0].add_patch(rect)
    border_h = int(H * 0.1)
    border_w = int(W * 0.1)
    rect = plt.Rectangle((border_w, border_h), W - 2*border_w, H - 2*border_h, 
                         fill=False, edgecolor='yellow', linewidth=4, label='10% crop')
    axes[0, 0].add_patch(rect)
    border_h = int(H * 0.15)
    border_w = int(W * 0.15)
    rect = plt.Rectangle((border_w, border_h), W - 2*border_w, H - 2*border_h, 
                         fill=False, edgecolor='red', linewidth=4, label='15% crop')
    axes[0, 0].add_patch(rect)
    axes[0, 0].set_title('Original Image\n(Lime=5%, Yellow=10%, Red=15% crop regions)', 
                         fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')
    axes[0, 0].legend(loc='upper right', fontsize=12)
    
    # Center Crop 5%
    axes[0, 1].imshow(img_center_5)
    axes[0, 1].set_title('After Center Crop 5%\n(Keeps 90% x 90% center)', 
                         fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')
    
    # Center Crop 10%
    axes[1, 0].imshow(img_center_10)
    axes[1, 0].set_title('After Center Crop 10%\n(Keeps 80% x 80% center)', 
                         fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')
    
    # Center Crop 15%
    axes[1, 1].imshow(img_center_15)
    axes[1, 1].set_title('After Center Crop 15%\n(Keeps 70% x 70% center)', 
                         fontsize=14, fontweight='bold')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    output_path2 = os.path.join(output_dir, f'{image_name}_crop_comparison_detail.png')
    plt.savefig(output_path2, dpi=300, bbox_inches='tight')
    print(f"Saved detail view: {output_path2}")
    
    plt.show()


def main():
    """Process all images in test_images folder."""
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
    
    if not image_files:
        print(f"No images found in '{test_images_dir}'")
        return
    
    print(f"Found {len(image_files)} images to process")
    print("="*80)
    
    for image_path in image_files:
        print(f"\nProcessing: {image_path}")
        visualize_cropping(str(image_path))
    
    print("\n" + "="*80)
    print("All visualizations saved to 'cropping_visualizations/' directory")
    print("="*80)


if __name__ == "__main__":
    main()
