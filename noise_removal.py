import cv2
import numpy as np
import argparse
import os
from scipy import stats
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import mean_squared_error as mse
import json




#SOME GPT FUNCTIONS USED FOR TESTING ---------------------------------------------------


def apply_bilateral_filter(image, d=9, sigma_color=75, sigma_space=75):
    """
    Bilateral filter - preserves edges while removing noise.
    Good for salt & pepper with edge preservation.
    
    Args:
        image: Input image
        d: Diameter of pixel neighborhood
        sigma_color: Filter sigma in color space
        sigma_space: Filter sigma in coordinate space
        
    Returns:
        Denoised image
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    # Apply bilateral filter to L channel
    lab[:, :, 0] = cv2.bilateralFilter(lab[:, :, 0], d, sigma_color, sigma_space)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

def apply_nlm_denoising(image, h=10, template_window_size=7, search_window_size=21):
    """
    Non-Local Means denoising - very effective for salt & pepper.
    Uses similar patches across the image.
    
    Args:
        image: Input image
        h: Filter strength (higher = more smoothing)
        template_window_size: Size of template patch
        search_window_size: Size of search area
        
    Returns:
        Denoised image
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    # Apply NLM to L channel
    lab[:, :, 0] = cv2.fastNlMeansDenoising(
        lab[:, :, 0], 
        None, 
        h=h, 
        templateWindowSize=template_window_size,
        searchWindowSize=search_window_size
    )
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

def apply_morphological_filter(image, kernel_size=3):
    """
    Morphological opening to remove salt & pepper noise.
    Opening = Erosion followed by Dilation.
    
    Args:
        image: Input image
        kernel_size: Size of structuring element
        
    Returns:
        Denoised image
    """
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    # Morphological opening (removes white noise - salt)
    opened = cv2.morphologyEx(lab[:, :, 0], cv2.MORPH_OPEN, kernel)
    # Morphological closing (removes black noise - pepper)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    
    lab[:, :, 0] = closed
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

def apply_cascaded_filter(image, kernel_size=5):
    """
    Cascaded filtering: Multiple techniques in sequence.
    Best overall results for heavy salt & pepper noise.
    
    Args:
        image: Input image
        kernel_size: Base kernel size
        
    Returns:
        Denoised image
    """
    # Step 1: Morphological filter to remove isolated noise
    denoised = apply_morphological_filter(image, kernel_size=3)
    
    # Step 2: Adaptive median filter for remaining noise
    denoised = apply_adaptive_median_filter(denoised, max_window_size=7)
    
    # Step 3: Light bilateral filter to smooth while preserving edges
    denoised = apply_bilateral_filter(denoised, d=5, sigma_color=50, sigma_space=50)
    
    return denoised


def grid_search_methods(noisy_folder="qsd1_w3", clean_folder="qsd1_w3/non_augmented", 
                       num_images=30):
    """
    Grid search to find the best denoising method for each noise type.
    
    Args:
        noisy_folder: Path to noisy images
        clean_folder: Path to clean reference images
        num_images: Number of images to process
        
    Returns:
        Dictionary with results for each method
    """
    methods = ['median', 'adaptive', 'bilateral', 'nlm', 'morphological', 'cascaded']
    
    print("="*80)
    print("GRID SEARCH FOR BEST DENOISING METHOD")
    print("="*80)
    
    all_results = {}
    
    for method in methods:
        print(f"\nTesting method: {method.upper()}")
        print("-"*80)
        
        results = []
        
        for i in range(num_images):
            noisy_path = f"{noisy_folder}/{i:05d}.jpg"
            clean_path = f"{clean_folder}/{i:05d}.jpg"
            
            if not os.path.exists(noisy_path) or not os.path.exists(clean_path):
                continue
                
            noisy = cv2.imread(noisy_path)
            clean = cv2.imread(clean_path)
            
            if noisy is None or clean is None:
                continue
            
            # Detect noise
            noise_info = detect_noise(noisy)
            
            # Apply denoising with current method
            denoised = remove_noise(noisy, noise_info['noise_type'], method=method)
            
            # Evaluate
            eval_result = evaluate_denoising(clean, noisy, denoised)
            eval_result['image_id'] = i
            eval_result['noise_type'] = noise_info['noise_type']
            results.append(eval_result)
        
        # Calculate aggregate statistics
        if results:
            avg_psnr_gain = np.mean([r['improvement']['psnr_gain'] for r in results])
            avg_ssim_gain = np.mean([r['improvement']['ssim_gain'] for r in results])
            avg_mse_reduction = np.mean([r['improvement']['mse_reduction_percent'] for r in results])
            
            avg_psnr_denoised = np.mean([r['denoised_metrics']['psnr'] for r in results])
            avg_ssim_denoised = np.mean([r['denoised_metrics']['ssim'] for r in results])
            
            all_results[method] = {
                'method': method,
                'avg_psnr_gain': avg_psnr_gain,
                'avg_ssim_gain': avg_ssim_gain,
                'avg_mse_reduction': avg_mse_reduction,
                'avg_psnr_denoised': avg_psnr_denoised,
                'avg_ssim_denoised': avg_ssim_denoised,
                'num_images': len(results)
            }
            
            print(f"  Avg PSNR: {avg_psnr_denoised:.2f} dB (Gain: {avg_psnr_gain:+.2f})")
            print(f"  Avg SSIM: {avg_ssim_denoised:.4f} (Gain: {avg_ssim_gain:+.4f})")
            print(f"  MSE Reduction: {avg_mse_reduction:.1f}%")
    
    # Find best methods
    print("\n" + "="*80)
    print("COMPARISON TABLE - ALL METHODS")
    print("="*80)
    print(f"{'Method':>15} | {'PSNR (dB)':>10} | {'SSIM':>8} | {'PSNR Gain':>12} | {'SSIM Gain':>12} | {'MSE Reduc.':>12}")
    print("-"*80)
    
    for method in methods:
        if method in all_results:
            r = all_results[method]
            print(f"{method:>15} | {r['avg_psnr_denoised']:>10.2f} | {r['avg_ssim_denoised']:>8.4f} | "
                  f"{r['avg_psnr_gain']:>+12.2f} | {r['avg_ssim_gain']:>+12.4f} | {r['avg_mse_reduction']:>11.1f}%")
    
    best_by_psnr = max(all_results.items(), key=lambda x: x[1]['avg_psnr_denoised'])
    best_by_ssim = max(all_results.items(), key=lambda x: x[1]['avg_ssim_denoised'])
    
    print("\n" + "="*80)
    print("BEST METHOD BY METRIC:")
    print("="*80)
    print(f"Best by PSNR: {best_by_psnr[0].upper()} (PSNR: {best_by_psnr[1]['avg_psnr_denoised']:.2f} dB)")
    print(f"Best by SSIM: {best_by_ssim[0].upper()} (SSIM: {best_by_ssim[1]['avg_ssim_denoised']:.4f})")
    print("="*80)
    
    return all_results

def grid_search_comprehensive(noisy_folder="qsd1_w3", clean_folder="qsd1_w3/non_augmented", 
                              num_images=30):
    """
    Comprehensive grid search: test all methods with different parameters.
    
    Returns:
        Dictionary with results organized by noise type and method
    """
    print("="*80)
    print("COMPREHENSIVE GRID SEARCH - METHODS & PARAMETERS")
    print("="*80)
    
    # Define parameter combinations for each method
    test_configs = {
        'median': [
            {'kernel_size': 3},
            {'kernel_size': 5},
            {'kernel_size': 7},
            {'kernel_size': 9}
        ],
        'adaptive': [
            {'max_window_size': 5},
            {'max_window_size': 7},
            {'max_window_size': 9},
            {'max_window_size': 11}
        ],
        'bilateral': [
            {'d': 5, 'sigma_color': 50, 'sigma_space': 50},
            {'d': 9, 'sigma_color': 75, 'sigma_space': 75},
            {'d': 11, 'sigma_color': 100, 'sigma_space': 100}
        ],
        'nlm': [
            {'h': 5, 'template_window_size': 7, 'search_window_size': 21},
            {'h': 10, 'template_window_size': 7, 'search_window_size': 21},
            {'h': 15, 'template_window_size': 7, 'search_window_size': 21}
        ],
        'morphological': [
            {'kernel_size': 3},
            {'kernel_size': 5}
        ],
        'cascaded': [
            {'kernel_size': 3},
            {'kernel_size': 5}
        ]
    }
    
    results_by_noise_type = {
        'salt_and_pepper': {},
        'gaussian': {},
        'uniform': {},
        'none': {}
    }
    
    for method, configs in test_configs.items():
        print(f"\n{'='*80}")
        print(f"Testing method: {method.upper()}")
        print('='*80)
        
        for config in configs:
            print(f"\nParameters: {config}")
            print("-"*80)
            
            method_results = {noise_type: [] for noise_type in results_by_noise_type.keys()}
            
            for i in range(num_images):
                noisy_path = f"{noisy_folder}/{i:05d}.jpg"
                clean_path = f"{clean_folder}/{i:05d}.jpg"
                
                if not os.path.exists(noisy_path) or not os.path.exists(clean_path):
                    continue
                    
                noisy = cv2.imread(noisy_path)
                clean = cv2.imread(clean_path)
                
                if noisy is None or clean is None:
                    continue
                
                # Detect noise
                noise_info = detect_noise(noisy)
                noise_type = noise_info['noise_type']
                
                # Apply denoising based on method and config
                if method == 'median' and noise_type == 'salt_and_pepper':
                    denoised = apply_median_filter(noisy, **config)
                elif method == 'adaptive' and noise_type == 'salt_and_pepper':
                    denoised = apply_adaptive_median_filter(noisy, **config)
                elif method == 'bilateral' and noise_type == 'salt_and_pepper':
                    denoised = apply_bilateral_filter(noisy, **config)
                elif method == 'nlm' and noise_type == 'salt_and_pepper':
                    denoised = apply_nlm_denoising(noisy, **config)
                elif method == 'morphological' and noise_type == 'salt_and_pepper':
                    denoised = apply_morphological_filter(noisy, **config)
                elif method == 'cascaded' and noise_type == 'salt_and_pepper':
                    denoised = apply_cascaded_filter(noisy, **config)
                else:
                    continue
                
                # Evaluate
                psnr_val = psnr(clean, denoised)
                ssim_val = ssim(clean, denoised, channel_axis=2, data_range=255)
                mse_val = mse(clean, denoised)
                
                method_results[noise_type].append({
                    'psnr': psnr_val,
                    'ssim': ssim_val,
                    'mse': mse_val
                })
            
            # Calculate statistics for this method-config combination
            for noise_type, noise_results in method_results.items():
                if noise_results:
                    avg_psnr = np.mean([r['psnr'] for r in noise_results])
                    avg_ssim = np.mean([r['ssim'] for r in noise_results])
                    avg_mse = np.mean([r['mse'] for r in noise_results])
                    
                    config_key = f"{method}_{str(config)}"
                    
                    if config_key not in results_by_noise_type[noise_type]:
                        results_by_noise_type[noise_type][config_key] = {
                            'method': method,
                            'config': config,
                            'avg_psnr': avg_psnr,
                            'avg_ssim': avg_ssim,
                            'avg_mse': avg_mse,
                            'num_images': len(noise_results)
                        }
                    
                    if noise_type == 'salt_and_pepper':  # Only print for relevant noise
                        print(f"  {noise_type}: PSNR={avg_psnr:.2f} dB, SSIM={avg_ssim:.4f} ({len(noise_results)} images)")
    
    # Print best results for each noise type
    print("\n" + "="*80)
    print("BEST CONFIGURATION FOR EACH NOISE TYPE")
    print("="*80)
    
    for noise_type, methods_dict in results_by_noise_type.items():
        if methods_dict:
            best = max(methods_dict.items(), key=lambda x: x[1]['avg_ssim'])
            print(f"\n{noise_type.upper()}:")
            print(f"  Method: {best[1]['method']}")
            print(f"  Config: {best[1]['config']}")
            print(f"  PSNR: {best[1]['avg_psnr']:.2f} dB")
            print(f"  SSIM: {best[1]['avg_ssim']:.4f}")
            print(f"  Images: {best[1]['num_images']}")
    
    return results_by_noise_type

def save_best_denoised_images(noisy_folder="qsd1_w3", clean_folder="qsd1_w3/non_augmented",
                               output_folder="output_best", num_images=30):
    """
    Apply best denoising method for each image and save results.
    Automatically selects the best method based on noise type.
    Also saves metrics to a JSON file for comparison.
    
    Args:
        noisy_folder: Path to noisy images
        clean_folder: Path to clean reference images (for comparison)
        output_folder: Path to save denoised images
        num_images: Number of images to process
    """
    os.makedirs(output_folder, exist_ok=True)
    
    print("="*80)
    print("APPLYING BEST DENOISING METHOD PER IMAGE")
    print("="*80)
    
    methods_to_test = ['median', 'adaptive', 'bilateral', 'nlm', 'morphological', 'cascaded']
    
    # Store metrics for each image
    metrics_data = []
    
    for i in range(num_images):
        noisy_path = f"{noisy_folder}/{i:05d}.jpg"
        clean_path = f"{clean_folder}/{i:05d}.jpg"
        output_path = f"{output_folder}/{i:05d}.jpg"
        
        if not os.path.exists(noisy_path):
            continue
        
        noisy = cv2.imread(noisy_path)
        clean = cv2.imread(clean_path) if os.path.exists(clean_path) else None
        
        if noisy is None:
            continue
        
        # Detect noise
        noise_info = detect_noise(noisy)
        
        if noise_info['noise_type'] == 'none':
            # No noise, save original
            cv2.imwrite(output_path, noisy)
            print(f"Image {i:05d}: No noise detected, saved original")
            
            # Still calculate metrics if clean image exists
            if clean is not None:
                psnr_noisy = psnr(clean, noisy)
                ssim_noisy = ssim(clean, noisy, channel_axis=2, data_range=255)
                
                metrics_data.append({
                    'image_id': i,
                    'noise_type': 'none',
                    'best_method': 'none',
                    'original_psnr': float(psnr_noisy),
                    'original_ssim': float(ssim_noisy),
                    'denoised_psnr': float(psnr_noisy),
                    'denoised_ssim': float(ssim_noisy),
                    'psnr_gain': 0.0,
                    'ssim_gain': 0.0
                })
            continue
        
        # Test all methods and find the best one
        best_method = None
        best_ssim = -1
        best_denoised = None
        
        for method in methods_to_test:
            denoised = remove_noise(noisy, noise_info['noise_type'], method=method)
            
            if clean is not None:
                ssim_val = ssim(clean, denoised, channel_axis=2, data_range=255)
                
                if ssim_val > best_ssim:
                    best_ssim = ssim_val
                    best_method = method
                    best_denoised = denoised
        
        # Save best result and calculate metrics
        if best_denoised is not None and clean is not None:
            cv2.imwrite(output_path, best_denoised)
            
            # Calculate metrics for original noisy image
            psnr_noisy = psnr(clean, noisy)
            ssim_noisy = ssim(clean, noisy, channel_axis=2, data_range=255)
            
            # Calculate metrics for denoised image
            psnr_denoised = psnr(clean, best_denoised)
            ssim_denoised = ssim(clean, best_denoised, channel_axis=2, data_range=255)
            
            # Calculate improvements
            psnr_gain = psnr_denoised - psnr_noisy
            ssim_gain = ssim_denoised - ssim_noisy
            
            metrics_data.append({
                'image_id': i,
                'noise_type': noise_info['noise_type'],
                'best_method': best_method,
                'original_psnr': float(psnr_noisy),
                'original_ssim': float(ssim_noisy),
                'denoised_psnr': float(psnr_denoised),
                'denoised_ssim': float(ssim_denoised),
                'psnr_gain': float(psnr_gain),
                'ssim_gain': float(ssim_gain)
            })
            
            print(f"Image {i:05d}: Noise={noise_info['noise_type']:<15} Best={best_method:<15} "
                  f"PSNR: {psnr_noisy:.2f}→{psnr_denoised:.2f} (Δ{psnr_gain:+.2f}) "
                  f"SSIM: {ssim_noisy:.4f}→{ssim_denoised:.4f} (Δ{ssim_gain:+.4f})")
        else:
            cv2.imwrite(output_path, noisy)
            print(f"Image {i:05d}: Failed to denoise, saved original")
    
    # Save metrics to JSON file
    metrics_file = f"{output_folder}/denoising_metrics.json"
    with open(metrics_file, 'w') as f:
        json.dump(metrics_data, f, indent=2)
    
    # Calculate and print aggregate statistics
    if metrics_data:
        denoised_images = [m for m in metrics_data if m['noise_type'] != 'none']
        if denoised_images:
            avg_psnr_gain = np.mean([m['psnr_gain'] for m in denoised_images])
            avg_ssim_gain = np.mean([m['ssim_gain'] for m in denoised_images])
            avg_psnr_original = np.mean([m['original_psnr'] for m in denoised_images])
            avg_psnr_denoised = np.mean([m['denoised_psnr'] for m in denoised_images])
            avg_ssim_original = np.mean([m['original_ssim'] for m in denoised_images])
            avg_ssim_denoised = np.mean([m['denoised_ssim'] for m in denoised_images])
            
            print("\n" + "="*80)
            print("AGGREGATE STATISTICS:")
            print("="*80)
            print(f"Images with noise detected: {len(denoised_images)}")
            print(f"Average Original PSNR: {avg_psnr_original:.2f} dB")
            print(f"Average Denoised PSNR: {avg_psnr_denoised:.2f} dB")
            print(f"Average PSNR Gain: {avg_psnr_gain:+.2f} dB")
            print(f"Average Original SSIM: {avg_ssim_original:.4f}")
            print(f"Average Denoised SSIM: {avg_ssim_denoised:.4f}")
            print(f"Average SSIM Gain: {avg_ssim_gain:+.4f}")
            print("="*80)
    
    print(f"\nBest denoised images saved to: {output_folder}/")
    print(f"Metrics saved to: {metrics_file}")
    
    



def visualize_single_denoising(image_id, noisy_folder="qsd1_w3", clean_folder="qsd1_w3/non_augmented", 
                                method='median', show_metrics=True):
    """
    Apply denoising to a single image and display noisy vs denoised side by side.
    
    Args:
        image_id: ID of the image to process (e.g., 5 for 00005.jpg)
        noisy_folder: Path to noisy images
        clean_folder: Path to clean reference images (optional, for metrics)
        method: Denoising method to use ('median', 'adaptive', 'bilateral', 'nlm', 'morphological', 'cascaded')
        show_metrics: Whether to display metrics on the image
    """
    import matplotlib.pyplot as plt
    
    # Load noisy image
    noisy_path = f"{noisy_folder}/{image_id:05d}.jpg"
    clean_path = f"{clean_folder}/{image_id:05d}.jpg"
    
    if not os.path.exists(noisy_path):
        print(f"Error: Image {noisy_path} not found!")
        return
    
    noisy = cv2.imread(noisy_path)
    clean = cv2.imread(clean_path) if os.path.exists(clean_path) else None
    
    # Detect noise
    noise_info = detect_noise(noisy)
    
    # Apply denoising
    denoised = remove_noise(noisy, noise_info['noise_type'], method=method)
    
    # Convert BGR to RGB for matplotlib
    noisy_rgb = cv2.cvtColor(noisy, cv2.COLOR_BGR2RGB)
    denoised_rgb = cv2.cvtColor(denoised, cv2.COLOR_BGR2RGB)
    
    # Calculate metrics if clean image is available
    metrics_text = ""
    if clean is not None:
        eval_result = evaluate_denoising(clean, noisy, denoised)
        metrics_text = (
            f"PSNR: {eval_result['noisy_metrics']['psnr']:.2f} → {eval_result['denoised_metrics']['psnr']:.2f} dB "
            f"(Δ{eval_result['improvement']['psnr_gain']:+.2f})\n"
            f"SSIM: {eval_result['noisy_metrics']['ssim']:.4f} → {eval_result['denoised_metrics']['ssim']:.4f} "
            f"(Δ{eval_result['improvement']['ssim_gain']:+.4f})\n"
            f"MSE Reduction: {eval_result['improvement']['mse_reduction_percent']:.1f}%"
        )
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    
    # Noisy image
    axes[0].imshow(noisy_rgb)
    axes[0].set_title(f'Noisy Image {image_id:05d}\nDetected: {noise_info["noise_type"]}', 
                     fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # Denoised image
    axes[1].imshow(denoised_rgb)
    axes[1].set_title(f'Denoised Image (Method: {method})', 
                     fontsize=12, fontweight='bold')
    axes[1].axis('off')
    
    # Add metrics text if available
    if show_metrics and metrics_text:
        fig.text(0.5, 0.02, metrics_text, ha='center', fontsize=10, 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout(rect=[0, 0.08, 1, 1] if show_metrics else [0, 0, 1, 1])
    plt.show()
    
    return denoised

def compare_methods_single_image(image_id, noisy_folder="qsd1_w3", clean_folder="qsd1_w3/non_augmented"):
    """
    Compare all denoising methods on a single image.
    
    Args:
        image_id: ID of the image to process
        noisy_folder: Path to noisy images
        clean_folder: Path to clean reference images (for metrics)
    """
    import matplotlib.pyplot as plt
    
    methods = ['median', 'adaptive', 'bilateral', 'nlm', 'morphological', 'cascaded']
    
    # Load noisy image
    noisy_path = f"{noisy_folder}/{image_id:05d}.jpg"
    clean_path = f"{clean_folder}/{image_id:05d}.jpg"
    
    if not os.path.exists(noisy_path):
        print(f"Error: Image {noisy_path} not found!")
        return
    
    noisy = cv2.imread(noisy_path)
    clean = cv2.imread(clean_path) if os.path.exists(clean_path) else None
    
    # Detect noise
    noise_info = detect_noise(noisy)
    
    # Create figure with all methods
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    # Show original noisy image
    noisy_rgb = cv2.cvtColor(noisy, cv2.COLOR_BGR2RGB)
    axes[0].imshow(noisy_rgb)
    axes[0].set_title(f'Noisy (Original)\n{noise_info["noise_type"]}', fontweight='bold')
    axes[0].axis('off')
    
    # Apply each method and display
    for idx, method in enumerate(methods, start=1):
        denoised = remove_noise(noisy, noise_info['noise_type'], method=method)
        denoised_rgb = cv2.cvtColor(denoised, cv2.COLOR_BGR2RGB);
        
        axes[idx].imshow(denoised_rgb)
        
        # Add metrics if clean image available
        if clean is not None:
            psnr_val = psnr(clean, denoised)
            ssim_val = ssim(clean, denoised, channel_axis=2, data_range=255)
            axes[idx].set_title(f'{method.upper()}\nPSNR: {psnr_val:.2f} | SSIM: {ssim_val:.4f}', 
                              fontweight='bold')
        else:
            axes[idx].set_title(method.upper(), fontweight='bold')
        
        axes[idx].axis('off')
    
    # Hide unused subplot
    axes[7].axis('off')
    
    plt.suptitle(f'Denoising Method Comparison - Image {image_id:05d}', 
                fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.show()


def denoise_specific_images(input_folder="qst1_w3", output_folder="qst1_w3_denoisedv2", 
                           image_ids=[0,3,6,7,8,9,10,14,15,16,17,18,19,20,22,23,24,30,32,33,34,35,36,37,39,40,41,43,48]):
    """
    Denoise specific images from a folder and save results.
    Applies median filter to all images.
    
    Args:
        input_folder: Path to input images
        output_folder: Path to save denoised images
        image_ids: List of image IDs to process
    
    Returns:
        Dictionary with processing results for each image
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    print("="*80)
    print(f"DENOISING SPECIFIC IMAGES FROM {input_folder}")
    print("="*80)
    
    processing_results = []
    
    for img_id in image_ids:
        input_path = f"{input_folder}/{img_id:05d}.jpg"
        output_path = f"{output_folder}/{img_id:05d}.jpg"
        
        if not os.path.exists(input_path):
            print(f"Warning: Image {input_path} not found, skipping...")
            continue
        
        # Load image
        image = cv2.imread(input_path)
        
        if image is None:
            print(f"Warning: Failed to load {input_path}, skipping...")
            continue
        
        # Detect noise (for information only)
        noise_info = detect_noise(image)
        
        # Apply median filter to all images
        denoised = apply_median_filter(image, kernel_size=3)
        # denoised = apply_cascaded_filter(image, kernel_size=5)
        method_used = 'median'

        print(f"Image {img_id:05d}: Applied median filter")
        print(f"  → Detected noise type: {noise_info['noise_type']}")
        
        # Save denoised image
        cv2.imwrite(output_path, denoised)
        
        # Store processing info
        processing_results.append({
            'image_id': img_id,
            'noise_type': noise_info['noise_type'],
            'noise_level': noise_info['noise_level'],
            'method_used': method_used,
            'kurtosis': float(noise_info['kurtosis']),
            'noise_std': float(noise_info['noise_std']),
            'snr': float(noise_info['snr']),
            'confidence': float(noise_info['confidence'])
        })
        
        print(f"  → Saved to {output_path}")
        print(f"  → Noise STD: {noise_info['noise_std']:.4f}, SNR: {noise_info['snr']:.2f} dB")
    
    # Save processing info to JSON
    info_file = f"{output_folder}/processing_info.json"
    with open(info_file, 'w') as f:
        json.dump(processing_results, f, indent=2)
    
    print("\n" + "="*80)
    print(f"PROCESSING COMPLETE")
    print(f"Denoised images saved to: {output_folder}/")
    print(f"Processing info saved to: {info_file}")
    print("="*80)
    
    # Print summary statistics
    noise_types = {}
    for result in processing_results:
        noise_type = result['noise_type']
        noise_types[noise_type] = noise_types.get(noise_type, 0) + 1
    
    print("\nSUMMARY:")
    print(f"Total images processed: {len(processing_results)}")
    print(f"Method applied: Median filter (kernel size 3) to ALL images")
    print("Noise types detected (for information):")
    for noise_type, count in noise_types.items():
        print(f"  {noise_type}: {count} images")
    
    return processing_results

#-----------------------------------------------------------------------------------------------GPT-END

def apply_gaussian_filter(image, kernel_size=5, sigma=1.0):
    if kernel_size % 2 == 0:
        kernel_size += 1
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    # Denoise only the L (lightness) channel
    l_channel = cv2.GaussianBlur(lab[:, :, 0], (kernel_size, kernel_size), sigma)
    lab[:, :, 0] = l_channel
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

def apply_median_filter(image, kernel_size=5):
    if kernel_size % 2 == 0:
        kernel_size += 1
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    # Denoise only the L channel
    lab[:, :, 0] = cv2.medianBlur(lab[:, :, 0], kernel_size)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

def apply_mean_filter(image, kernel_size=5):
    if kernel_size % 2 == 0:
        kernel_size += 1
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    # Denoise only the L channel
    lab[:, :, 0] = cv2.blur(lab[:, :, 0], (kernel_size, kernel_size))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

def apply_adaptive_median_filter(image, max_window_size=7):
    #THIS ONE IS GPT GEN BECAUSE I WANTED TO FIND A BETTER MEDIAN FILTER
    if max_window_size % 2 == 0:
        max_window_size += 1
    
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0].astype(np.float32)
    
    h, w = l_channel.shape
    output = l_channel.copy()
    
    def adaptive_process(img, i, j, max_size):
        """Process single pixel with adaptive window"""
        window_size = 3
        while window_size <= max_size:
            half = window_size // 2
            i_min = max(0, i - half)
            i_max = min(h, i + half + 1)
            j_min = max(0, j - half)
            j_max = min(w, j + half + 1)
            
            window = img[i_min:i_max, j_min:j_max]
            z_min = np.min(window)
            z_max = np.max(window)
            z_med = np.median(window)
            z_xy = img[i, j]
            
            # Level A: Check if median is impulse
            if z_min < z_med < z_max:
                # Level B: Check if pixel is impulse
                if z_min < z_xy < z_max:
                    return z_xy  # Not impulse, keep original
                else:
                    return z_med  # Impulse, replace with median
            else:
                # Increase window size
                window_size += 2
        
        return z_med  # Return median if max size reached
    
    # Apply adaptive median to each pixel
    for i in range(h):
        for j in range(w):
            output[i, j] = adaptive_process(l_channel, i, j, max_window_size)
    
    lab[:, :, 0] = output.astype(np.uint8)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def detect_noise(image, noise_threshold=0.045):
    
    if len(image.shape) == 3:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        gray = lab[:, :, 0]  # L channel only
    else:
        gray = image.copy()
        
    gray_norm = gray.astype(np.float32) / 255.0
    
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    noise_var = laplacian.var()
    
    blurred = cv2.GaussianBlur(gray_norm, (5,5), 1.0)
    
    noise = gray_norm - blurred
    
    noise_flat = noise.flatten()
    kurtosis_val = stats.kurtosis(noise_flat, fisher=True)
    
    noise_std = np.std(noise_flat)
    
    # Calculate signal-to-noise ratio for better detection
    signal_std = np.std(gray_norm)
    snr = signal_std / noise_std if noise_std > 0 else float('inf')
    
    hist, bins = np.histogram(gray.flatten(), bins=256, range=[0, 256])
    
    gray_float = gray.astype(np.float32)
    median = cv2.medianBlur(gray_float, 3)
    diff = np.abs(gray_float - median)
    std_local = np.std(gray_float)

    # Pixels much higher/lower than local median
    salt_mask = diff > (3 * std_local)
    impulse_ratio = np.sum(salt_mask) / gray.size

    # Detect salt vs pepper
    pepper_mask = (gray_float < median - 3 * std_local)
    salt_mask = (gray_float > median + 3 * std_local)
    salt_ratio = np.sum(salt_mask) / gray.size
    pepper_ratio = np.sum(pepper_mask) / gray.size

    has_salt = salt_ratio > 0.005
    has_pepper = pepper_ratio > 0.005
    has_salt_pepper = (salt_ratio + pepper_ratio) > 0.02
    
    # print(f"Has noise: {has_salt_pepper}, Impulse ratio: {impulse_ratio:.4f}")
  
    # More conservative noise detection
    has_noise = noise_std >= noise_threshold and snr < 15
    
    noise_type = ""
    confidence = 0.0
    
    # print(f"Kurtosis: {kurtosis_val:.2f}, Impulse ratio: {impulse_ratio:.4f}, SNR: {snr:.2f}")
    if not has_noise:
        noise_type = "none"
        confidence = 1.0
        noise_level = "none"
        
    #QSD1-W3 ADJUSTED LOGIC
    elif impulse_ratio != 0.0:
        if ((kurtosis_val > 5.0 or (has_salt and has_pepper))) and snr < 4.0:
            noise_type = "salt_and_pepper"
            confidence = min(1.0, (kurtosis_val / 10.0) if kurtosis_val > 0 else 0.5)
        elif -0.5 <= kurtosis_val < 5.0:
            noise_type = "gaussian"
            confidence = 1.0 - abs(kurtosis_val) / 3.0
        # elif kurtosis_val < -0.5:
        #     noise_type = "uniform"
        #     confidence = min(1.0, abs(kurtosis_val) / 2.0)
        # else:
        #     noise_type = "mixed"
        #     confidence = 0.5
        else:
            noise_type = "none"
    
    # QSD2-W3 ADJUSTED LOGIC
    # elif impulse_ratio != 0.0:
    #     if ((kurtosis_val > 5.0 or (has_salt and has_pepper))) and snr < 4.5:
    #         noise_type = "salt_and_pepper"
    #         confidence = min(1.0, (kurtosis_val / 10.0) if kurtosis_val > 0 else 0.5)
    #     elif -0.5 <= kurtosis_val < 5.0:
    #         noise_type = "none"
    #         confidence = 1.0 - abs(kurtosis_val) / 3.0
    #     # elif kurtosis_val < -0.5:
    #     #     noise_type = "uniform"
    #     #     confidence = min(1.0, abs(kurtosis_val) / 2.0)
    #     # else:
    #     #     noise_type = "mixed"
    #     #     confidence = 0.5
    #     else:
    #         noise_type = "none"
    # Adjusted thresholds for noise levels
    if noise_std < 0.025:
        noise_level = "very_low"
    elif noise_std < 0.06:
        noise_level = "low"
    elif noise_std < 0.12:
        noise_level = "medium"
    elif noise_std < 0.25:
        noise_level = "high"
    else:
        noise_level = "very_high"
        
    return {
        'noise_type': noise_type,
        'kurtosis': kurtosis_val,
        'noise_std': noise_std,
        'noise_level': noise_level,
        'laplacian_variance': noise_var,
        'has_salt': has_salt,
        'has_pepper': has_pepper,
        'confidence': confidence,
        'snr': snr
    }
    




def remove_noise(image, noise_type, method='median'):
    
    if noise_type == "salt_and_pepper":
        if method == 'adaptive':
            return apply_adaptive_median_filter(image, max_window_size=3)
        elif method == 'bilateral':
            return apply_bilateral_filter(image, d=9, sigma_color=75, sigma_space=75)
        elif method == 'nlm':
            return apply_nlm_denoising(image, h=10)
        elif method == 'morphological':
            return apply_morphological_filter(image, kernel_size=3)
        elif method == 'cascaded':
            return apply_cascaded_filter(image)
        elif method == 'median':
            return apply_median_filter(image, kernel_size=3) #best method for impulse noise, kernel size 3
        elif method == 'gaussian':
            return apply_gaussian_filter(image, kernel_size=3, sigma=1.0)
    elif noise_type == "gaussian":
        return apply_adaptive_median_filter(image, max_window_size=5) #works the best on image 6 at this moment, needs a better solution
    
    return image        
    

def evaluate_denoising(original, noisy, denoised):
    
    # Calculate metrics for noisy vs original
    psnr_noisy = psnr(original, noisy)
    ssim_noisy = ssim(original, noisy, channel_axis=2, data_range=255)
    mse_noisy = mse(original, noisy)
    
    # Calculate metrics for denoised vs original
    psnr_denoised = psnr(original, denoised)
    ssim_denoised = ssim(original, denoised, channel_axis=2, data_range=255)
    mse_denoised = mse(original, denoised)
    
    # Calculate improvement
    psnr_improvement = psnr_denoised - psnr_noisy
    ssim_improvement = ssim_denoised - ssim_noisy
    mse_improvement = ((mse_noisy - mse_denoised) / mse_noisy * 100) if mse_noisy > 0 else 0
    
    return {
        'noisy_metrics': {
            'psnr': psnr_noisy,
            'ssim': ssim_noisy,
            'mse': mse_noisy
        },
        'denoised_metrics': {
            'psnr': psnr_denoised,
            'ssim': ssim_denoised,
            'mse': mse_denoised
        },
        'improvement': {
            'psnr_gain': psnr_improvement,
            'ssim_gain': ssim_improvement,
            'mse_reduction_percent': mse_improvement
        }
    }

def evaluate_dataset(noisy_folder="qsd1_w3", clean_folder="qsd1_w3/non_augmented", num_images=30):

    results = []
    
    for i in range(num_images):
        # Load images
        noisy_path = f"{noisy_folder}/{i:05d}.jpg"
        clean_path = f"{clean_folder}/{i:05d}.jpg"
        
        if not os.path.exists(noisy_path) or not os.path.exists(clean_path):
            print(f"Warning: Missing image {i:05d}")
            continue
            
        noisy = cv2.imread(noisy_path)
        clean = cv2.imread(clean_path)
        
        if noisy is None or clean is None:
            print(f"Warning: Failed to load image {i:05d}")
            continue
        
        # Detect and remove noise
        noise_info = detect_noise(noisy)
        denoised = remove_noise(noisy, noise_info['noise_type'], method='median')
        
        # Evaluate
        eval_result = evaluate_denoising(clean, noisy, denoised)
        eval_result['image_id'] = i
        eval_result['noise_detected'] = noise_info['noise_type']
        results.append(eval_result)
        
        # Print individual results
        print(f"Image {i:05d} - Noise: {noise_info['noise_type']}")
        print(f"  PSNR: {eval_result['noisy_metrics']['psnr']:.2f} → {eval_result['denoised_metrics']['psnr']:.2f} dB (Δ{eval_result['improvement']['psnr_gain']:+.2f})")
        print(f"  SSIM: {eval_result['noisy_metrics']['ssim']:.4f} → {eval_result['denoised_metrics']['ssim']:.4f} (Δ{eval_result['improvement']['ssim_gain']:+.4f})")
        print(f"  MSE Reduction: {eval_result['improvement']['mse_reduction_percent']:.1f}%")
        print()
    
    # Calculate aggregate statistics
    if results:
        avg_psnr_gain = np.mean([r['improvement']['psnr_gain'] for r in results])
        avg_ssim_gain = np.mean([r['improvement']['ssim_gain'] for r in results])
        avg_mse_reduction = np.mean([r['improvement']['mse_reduction_percent'] for r in results])
        
        print("="*60)
        print("AGGREGATE RESULTS:")
        print(f"Average PSNR Improvement: {avg_psnr_gain:+.2f} dB")
        print(f"Average SSIM Improvement: {avg_ssim_gain:+.4f}")
        print(f"Average MSE Reduction: {avg_mse_reduction:.1f}%")
        print("="*60)
    
    return results




if __name__ == "__main__":
    
    # # STEP 1: Test noise detection on sample images - Run it to see what images have noise and some infos
     for i in range(0, 30):
        image = cv2.imread(f"qsd2_w3/{i:05d}.jpg")
       
        result = detect_noise(image)
        print(f"Image {i:05d}: {result['noise_type']} - {result['snr']:.2f} dB - noise std: {result['noise_std']:.4f} kurtosis: {result['kurtosis']:.2f} ")
    
    
    
    # # STEP 2: Evaluate denoising on the whole dataset, and see the result for each image in the json file
    # results = evaluate_dataset()
    # # Optional: Save results to file
    # with open('denoising_evaluationTASK1.json', 'w') as f:
    #     json.dump(results, f, indent=2)
    
    
    
    # STEP 3: Grid search to find the best parameters
    # Method 1: Simple method comparison
    # print("\n=== METHOD COMPARISON ===")
    # method_results = grid_search_methods()
    
    # with open('grid_search_methods.json', 'w') as f:
    #     json.dump({k: v for k, v in method_results.items()}, f, indent=2)
    
    # # Method 2: Comprehensive parameter search
    # print("\n\n=== COMPREHENSIVE PARAMETER SEARCH ===")
    # comprehensive_results = grid_search_comprehensive()
    
    # with open('grid_search_comprehensive.json', 'w') as f:
    #     json.dump(comprehensive_results, f, indent=2, default=str)
    
    # Method 3: Save images with best method
    # print("\n\n=== SAVING BEST DENOISED IMAGES ===")
    # save_best_denoised_images(output_folder="output_best_method")
    
    # print("\n" + "="*80)
    # print("ALL GRID SEARCH OPERATIONS COMPLETED")
    # print("="*80)
    
     # Method 4: View single image denoising
    # visualize_single_denoising(image_id=6, method='median')
    
    # Method 5: Compare all methods on a single image
    # compare_methods_single_image(image_id=6)
    # print("\n=== DENOISING SPECIFIC IMAGES ===")
    # results = denoise_specific_images(
    #     input_folder="qst1_w3",
    #     output_folder="qst1_w3_denoised",
    #     image_ids=[0,3,6,7,8,9,10,14,15,16,17,18,19,20,22,23,24,30,32,33,34,35,36,37,39,40,41,43,48]
    # )

