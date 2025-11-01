import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from libs_week4.descriptor import ORBDescriptor, DaisyDescriptor, SIFTDescriptor

def load_image_from_bbdd(image_id: int, bbdd_folder: str = "BBDD") -> tuple[np.ndarray, str]:
    """
    Load an image based on its ID from the BBDD folder.
    
    Args:
        image_id: ID of the image (e.g., 0 for bbdd_00000.jpg)
        bbdd_folder: Path to BBDD folder
        
    Returns:
        Tuple of (image as numpy array (RGB, float32, normalized to [0,1]), image_path)
    """
    # Construct the image path directly
    image_path = Path(bbdd_folder) / f"bbdd_{image_id:05d}.jpg"
    
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    # Load the image
    image = cv2.imread(str(image_path))
    
    if image is None:
        raise FileNotFoundError(f"Failed to load image: {image_path}")
    
    # Convert BGR to RGB and normalize
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_norm = image_rgb.astype(np.float32) / 255.0
    
    # Optional: Read metadata from text file if it exists
    txt_path = Path(bbdd_folder) / f"bbdd_{image_id:05d}.txt"
    metadata = None
    if txt_path.exists():
        with open(txt_path, 'r') as f:
            metadata = f.read().strip()
    
    return image_norm, str(image_path)

def visualize_keypoints(image: np.ndarray, keypoints: list, title: str = "Keypoints"):
    """
    Visualize detected keypoints on the image.
    
    Args:
        image: Input image (RGB, [0,1])
        keypoints: List of cv2.KeyPoint objects
        title: Title for the plot
    """
    # Convert back to uint8 for drawing
    img_uint8 = (image * 255).astype(np.uint8)
    
    # Draw keypoints
    img_with_kp = cv2.drawKeypoints(
        img_uint8, 
        keypoints, 
        None, 
        color=(0, 255, 0),
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
    
    # Display
    plt.figure(figsize=(12, 8))
    plt.imshow(img_with_kp)
    plt.title(f"{title}\nNumber of keypoints: {len(keypoints)}", fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def compare_descriptors(image: np.ndarray, mask: np.ndarray | None = None):
    """
    Compare all three descriptor types on the same image.
    
    Args:
        image: Input image (RGB, [0,1])
        mask: Optional mask for keypoint detection
    """
    # Initialize descriptors
    orb = ORBDescriptor(n_features=500, scale_factor=1.2, n_levels=8)
    
    # FIXED: Increase step size to reduce keypoint density
    # step=4 creates ~50,000+ keypoints, step=16 creates ~3,000 keypoints
    daisy = DaisyDescriptor(step=16, radius=15, rings=3, histograms=8, orientations=8)
    
    sift = SIFTDescriptor(n_features=0)  # 0 = detect all features
    
    # Compute descriptors
    print("Computing ORB descriptors...")
    orb_kp, orb_desc = orb.detect_and_compute(image, mask)
    print(f"  → Found {len(orb_kp)} keypoints, descriptor shape: {orb_desc.shape if orb_desc is not None else 'None'}")
    
    print("Computing DAISY descriptors...")
    daisy_kp, daisy_desc = daisy.detect_and_compute(image, mask)
    print(f"  → Found {len(daisy_kp)} keypoints, descriptor shape: {daisy_desc.shape if daisy_desc is not None else 'None'}")
    
    print("Computing SIFT descriptors...")
    sift_kp, sift_desc = sift.detect_and_compute(image, mask)
    print(f"  → Found {len(sift_kp)} keypoints, descriptor shape: {sift_desc.shape if sift_desc is not None else 'None'}")
    
    # Visualize side by side
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    
    descriptors_data = [
        (orb_kp, "ORB", orb_desc),
        (daisy_kp, "DAISY", daisy_desc),
        (sift_kp, "SIFT", sift_desc)
    ]
    
    for idx, (kp, name, desc) in enumerate(descriptors_data):
        img_uint8 = (image * 255).astype(np.uint8)
        
        # FIXED: Use simpler visualization for DAISY (too many keypoints)
        if name == "DAISY" and len(kp) > 5000:
            # Draw only circles without orientation lines
            img_with_kp = cv2.drawKeypoints(
                img_uint8,
                kp,
                None,
                color=(0, 255, 0),
                flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT  # Simple circles
            )
        else:
            img_with_kp = cv2.drawKeypoints(
                img_uint8,
                kp,
                None,
                color=(0, 255, 0),
                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
            )
        
        axes[idx].imshow(img_with_kp)
        desc_info = f"Desc: {desc.shape}" if desc is not None else "Desc: None"
        axes[idx].set_title(f"{name}\nKeypoints: {len(kp)}\n{desc_info}", 
                           fontsize=12, fontweight='bold')
        axes[idx].axis('off')
    
    plt.suptitle("Keypoint Descriptor Comparison", fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.show()
    
    return {
        'orb': (orb_kp, orb_desc),
        'daisy': (daisy_kp, daisy_desc),
        'sift': (sift_kp, sift_desc)
    }

def analyze_descriptor_statistics(descriptors: np.ndarray, name: str):
    """
    Analyze and print statistics about the descriptor vectors.
    
    Args:
        descriptors: Descriptor array of shape (n_keypoints, descriptor_dim)
        name: Name of the descriptor type
    """
    if descriptors is None:
        print(f"\n{name} - No descriptors computed")
        return
    
    print(f"\n{'='*60}")
    print(f"{name} DESCRIPTOR STATISTICS")
    print('='*60)
    print(f"Number of keypoints: {descriptors.shape[0]}")
    print(f"Descriptor dimension: {descriptors.shape[1]}")
    print(f"Descriptor dtype: {descriptors.dtype}")
    print(f"\nValue Statistics:")
    print(f"  Min:    {descriptors.min():.4f}")
    print(f"  Max:    {descriptors.max():.4f}")
    print(f"  Mean:   {descriptors.mean():.4f}")
    print(f"  Median: {np.median(descriptors):.4f}")
    print(f"  Std:    {descriptors.std():.4f}")
    print(f"\nSparsity:")
    zero_count = np.sum(descriptors == 0)
    total_count = descriptors.size
    sparsity = (zero_count / total_count) * 100
    print(f"  Zero values: {zero_count}/{total_count} ({sparsity:.2f}%)")

def visualize_descriptor_distribution(results: dict):
    """
    Visualize the distribution of descriptor values for each method.
    
    Args:
        results: Dictionary with descriptor results from compare_descriptors
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, (name, (kp, desc)) in enumerate(results.items()):
        if desc is None:
            axes[idx].text(0.5, 0.5, f"No descriptors\nfor {name.upper()}", 
                          ha='center', va='center', fontsize=14)
            axes[idx].axis('off')
            continue
        
        # Flatten all descriptor values
        values = desc.flatten()
        
        # Plot histogram
        axes[idx].hist(values, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
        axes[idx].set_title(f"{name.upper()} Descriptor Values\n"
                           f"Shape: {desc.shape}", 
                           fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('Descriptor Value')
        axes[idx].set_ylabel('Frequency')
        axes[idx].grid(axis='y', alpha=0.3)
        
        # Add statistics text
        stats_text = f"Mean: {values.mean():.2f}\nStd: {values.std():.2f}"
        axes[idx].text(0.02, 0.98, stats_text, 
                      transform=axes[idx].transAxes,
                      fontsize=10, verticalalignment='top',
                      bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle("Descriptor Value Distributions", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

def test_descriptor_on_image(image_id: int = 0, bbdd_folder: str = "BBDD"):
    """
    Complete test and visualization pipeline for descriptors.
    
    Args:
        image_id: ID of the image to test (0-indexed)
        bbdd_folder: Path to BBDD folder
    """
    print("="*80)
    print(f"TESTING KEYPOINT DESCRIPTORS ON IMAGE {image_id:05d}")
    print("="*80)
    
    # Load image
    print(f"\n1. Loading image from BBDD...")
    image, image_path = load_image_from_bbdd(image_id, bbdd_folder)
    print(f"   → Loaded: {image_path}")
    print(f"   → Shape: {image.shape}")
    
    # Display original image
    print("\n2. Displaying original image...")
    plt.figure(figsize=(10, 8))
    plt.imshow(image)
    plt.title(f"Original Image (ID: {image_id:05d})", fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    # Compare descriptors
    print("\n3. Computing and comparing descriptors...")
    results = compare_descriptors(image)
    
    # Analyze statistics
    print("\n4. Analyzing descriptor statistics...")
    for name, (kp, desc) in results.items():
        analyze_descriptor_statistics(desc, name.upper())
    
    # Visualize distributions
    print("\n5. Visualizing descriptor value distributions...")
    visualize_descriptor_distribution(results)
    
    print("\n" + "="*80)
    print("TEST COMPLETE!")
    print("="*80)
    
    return results

def test_multiple_images(image_ids: list[int] = [0, 10, 20, 30], 
                         descriptor_type: str = 'orb',
                         bbdd_folder: str = "BBDD"):
    """
    Test the same descriptor on multiple images.
    
    Args:
        image_ids: List of image IDs to test
        descriptor_type: 'orb', 'daisy', or 'sift'
        bbdd_folder: Path to BBDD folder
    """
    # Initialize descriptor
    if descriptor_type.lower() == 'orb':
        descriptor = ORBDescriptor(n_features=500, scale_factor=1.2, n_levels=8)
    elif descriptor_type.lower() == 'daisy':
        descriptor = DaisyDescriptor(step=4, radius=15, rings=3, histograms=8, orientations=8)
    elif descriptor_type.lower() == 'sift':
        descriptor = SIFTDescriptor(n_features=0)
    else:
        raise ValueError(f"Unknown descriptor type: {descriptor_type}")
    
    print(f"Testing {descriptor_type.upper()} on {len(image_ids)} images...")
    
    fig, axes = plt.subplots(2, len(image_ids), figsize=(5*len(image_ids), 10))
    if len(image_ids) == 1:
        axes = axes.reshape(-1, 1)
    
    for idx, img_id in enumerate(image_ids):
        # Load image
        image, _ = load_image_from_bbdd(img_id, bbdd_folder)
        
        # Compute descriptors
        kp, desc = descriptor.detect_and_compute(image)
        
        # Original image
        axes[0, idx].imshow(image)
        axes[0, idx].set_title(f"Image {img_id:05d}", fontsize=12, fontweight='bold')
        axes[0, idx].axis('off')
        
        # Image with keypoints
        img_uint8 = (image * 255).astype(np.uint8)
        img_with_kp = cv2.drawKeypoints(
            img_uint8,
            kp,
            None,
            color=(0, 255, 0),
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )
        axes[1, idx].imshow(img_with_kp)
        axes[1, idx].set_title(f"{len(kp)} keypoints\nDesc: {desc.shape if desc is not None else 'None'}", 
                              fontsize=10)
        axes[1, idx].axis('off')
        
        print(f"  Image {img_id:05d}: {len(kp)} keypoints, desc shape: {desc.shape if desc is not None else 'None'}")
    
    plt.suptitle(f"{descriptor_type.upper()} Descriptor - Multiple Images", 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # OPTION 1: Test single image with all descriptors
    print("\n" + "="*80)
    print("OPTION 1: Testing all descriptors on a single image")
    print("="*80)
    results = test_descriptor_on_image(image_id=7, bbdd_folder="BBDD")
    
    # OPTION 2: Test multiple images with one descriptor
    # print("\n" + "="*80)
    # print("OPTION 2: Testing ORB on multiple images")
    # print("="*80)
    # test_multiple_images(image_ids=[0, 10, 20, 30, 40], descriptor_type='orb', bbdd_folder="BBDD")
    
    # OPTION 3: Test specific descriptor on single image
    # print("\n" + "="*80)
    # print("OPTION 3: Testing SIFT on a single image")
    # print("="*80)
    # image, _ = load_image_from_bbdd(15, "BBDD")
    # sift = SIFTDescriptor()
    # kp, desc = sift.detect_and_compute(image)
    # visualize_keypoints(image, kp, "SIFT Keypoints")
    # analyze_descriptor_statistics(desc, "SIFT")