import cv2
import numpy as np
import matplotlib.pyplot as plt

def visualize_variance_background_removal(image: np.ndarray, channel_config: dict):
    """
    Visualize the background removal process based on variance analysis.
    Uses the same logic as `variance_background_removal` but adds visual outputs.
    """
    def convert_to_colorspace(img, colorspace):
        if colorspace == 'RGB':
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif colorspace == 'HSV':
            return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        elif colorspace == 'LAB':
            return cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        elif colorspace == 'YCrCb':
            return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        else:
            raise ValueError(f"Unsupported color space: {colorspace}")

    # --- replicate your function logic ---
    channels_to_analyze = []
    for colorspace, channel_idx in channel_config['channels']:
        converted = convert_to_colorspace(image, colorspace)
        if channel_idx < converted.shape[2]:
            channels_to_analyze.append(converted[:, :, channel_idx].astype(np.float32))
        else:
            raise ValueError(f"Channel {channel_idx} doesn't exist in {colorspace}")

    if not channels_to_analyze:
        raise ValueError("No channels to analyze")

    height, width = channels_to_analyze[0].shape
    threshold = channel_config['threshold']
    bboxes = []

    # --- visualization for each channel ---
    for (colorspace, channel_idx), channel in zip(channel_config['channels'], channels_to_analyze):
        variances_h = channel.var(axis=1)
        variances_v = channel.var(axis=0)
        # normalize variances for better visualization
        # variances_h = cv2.normalize(variances_h, None, 0, 255, cv2.NORM_MINMAX)
        # variances_v = cv2.normalize(variances_v, None, 0, 255, cv2.NORM_MINMAX)
        # threshold = np.percentile(np.concatenate([variances_h, variances_v]), 30)

        # find borders
        top = next((i for i in range(height) if variances_h[i] >= threshold), 0)
        bottom = next((i for i in range(height - 1, -1, -1) if variances_h[i] >= threshold), height - 1)
        left = next((j for j in range(width) if variances_v[j] >= threshold), 0)
        right = next((j for j in range(width - 1, -1, -1) if variances_v[j] >= threshold), width - 1)
        bboxes.append((top, bottom, left, right))

        # --- local variance heatmap (for visualization only) ---
        k = 15  # window size for local variance
        kernel = np.ones((k, k), np.float32) / (k * k)
        mean = cv2.filter2D(channel, -1, kernel)
        mean_sq = cv2.filter2D(channel ** 2, -1, kernel)
        var_map = mean_sq - mean ** 2
        var_map = np.clip(var_map, 0, None)
        var_map = cv2.normalize(var_map, None, 0, 1, cv2.NORM_MINMAX)

        # --- plot ---
        fig, axs = plt.subplots(2, 2, figsize=(10, 8))
        fig.suptitle(f'Variance Visualization – {colorspace} channel {channel_idx} with {threshold} threshold', fontsize=14, weight='bold')

        axs[0, 0].imshow(channel, cmap='gray')
        axs[0, 0].set_title('Selected Channel')
        axs[0, 0].axis('off')

        axs[0, 1].imshow(var_map, cmap='inferno')
        axs[0, 1].set_title(f'Local Variance Map (window={k})')
        axs[0, 1].axis('off')

        axs[1, 0].plot(variances_h)
        axs[1, 0].axhline(threshold, color='r', linestyle='--')
        axs[1, 0].set_title('Variance per Row')
        axs[1, 0].set_xlabel('Row index')

        axs[1, 1].plot(variances_v)
        axs[1, 1].axhline(threshold, color='r', linestyle='--')
        axs[1, 1].set_title('Variance per Column')
        axs[1, 1].set_xlabel('Column index')

        plt.tight_layout()
        plt.show()

    # --- Combine bounding boxes across channels (intersection) ---
    final_top = max(b[0] for b in bboxes)
    final_bottom = min(b[1] for b in bboxes)
    final_left = max(b[2] for b in bboxes)
    final_right = min(b[3] for b in bboxes)

    # --- Create mask and overlay bounding box ---
    mask = np.zeros((height, width), dtype=np.float32)
    if final_top <= final_bottom and final_left <= final_right:
        mask[final_top:final_bottom+1, final_left:final_right+1] = 1.0

    image_box = image.copy()
    cv2.rectangle(image_box, (final_left, final_top), (final_right, final_bottom), (0, 255, 0), 2)

    # --- Show final output ---
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axs[0].set_title('Original Image')
    axs[0].axis('off')

    axs[1].imshow(cv2.cvtColor(image_box, cv2.COLOR_BGR2RGB))
    axs[1].set_title('Detected Bounding Box')
    axs[1].axis('off')

    axs[2].imshow(mask, cmap='gray')
    axs[2].set_title('Final Binary Mask')
    axs[2].axis('off')

    plt.tight_layout()
    plt.show()

    print(f"Detected region: top={final_top}, bottom={final_bottom}, left={final_left}, right={final_right}")
    return mask

if __name__ == "__main__":
    for i in range(14,15):
        image = cv2.imread(f"qsd2_w1/000{i}.jpg")
        image_float = image.astype(np.float32) / 255.0
        # normalize image
        # image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
        config = {
            "channels": [('HSV', 0), ('HSV', 1), ('HSV', 2)],  # Example: L and V channels
            "threshold": 0.005           
        }

        mask = visualize_variance_background_removal(image_float, config)
