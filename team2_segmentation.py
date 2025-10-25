import cv2
import numpy as np
from scipy import ndimage
from scipy.ndimage import binary_opening, morphological_gradient
import os


# --- Helper functions from the second script ---

def convert_to_representation(image):
    """Convert RGB image to LAB and return all channels as float arrays."""
    # Assuming the input image is already in RGB format and scaled 0-255
    im_lab = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2LAB)
    L, a, b = cv2.split(im_lab)
    return L.astype(float), a.astype(float), b.astype(float)


def compute_edge_mask(im_lab, gradient_threshold=0.15):
    """Compute morphological gradient on LAB channels and combine."""
    L, a, b = im_lab

    # Compute morphological gradients per channel
    grad_L = morphological_gradient(L, structure=np.ones((5, 5)))
    grad_a = morphological_gradient(a, structure=np.ones((5, 5)))
    grad_b = morphological_gradient(b, structure=np.ones((5, 5)))

    # Combine gradients using Euclidean magnitude
    grad_combined = np.sqrt((grad_L ** 2) + (grad_a ** 2) + (grad_b ** 2))

    # Threshold to highlight edges
    threshold = np.max(grad_combined) * gradient_threshold
    grad_bin = grad_combined > threshold

    # Apply opening to clean noise
    mask = binary_opening(grad_bin, structure=np.ones((3, 3)))

    return mask


def create_border_suppressed_mask(mask_bool, pixel_border, h, w):
    """Create a mask with border pixels suppressed for extreme point computation."""
    if not pixel_border or int(pixel_border) <= 0:
        return mask_bool

    mask_for_extremes = mask_bool.copy()
    pb = int(pixel_border)

    # Avoid border larger than dimensions
    pb_h = min(pb, h // 2)
    pb_w = min(pb, w // 2)

    # Suppress border pixels
    if pb_w > 0:
        mask_for_extremes[:, :pb_w] = False
        mask_for_extremes[:, w - pb_w:] = False
    if pb_h > 0:
        mask_for_extremes[:pb_h, :] = False
        mask_for_extremes[h - pb_h:, :] = False

    return mask_for_extremes


def compute_extreme_points(mask_for_extremes, h, w):
    """Compute leftmost, rightmost, topmost, and bottommost white pixels."""
    # Row-wise extremes (leftmost and rightmost white pixel per row)
    rows_any = mask_for_extremes.any(axis=1)
    left_all = mask_for_extremes.argmax(axis=1)
    right_all = w - 1 - np.argmax(mask_for_extremes[:, ::-1], axis=1)
    left = np.where(rows_any, left_all, -1)
    right = np.where(rows_any, right_all, -1)

    # Column-wise extremes (topmost and bottommost white pixel per column)
    cols_any = mask_for_extremes.any(axis=0)
    top_all = mask_for_extremes.argmax(axis=0)
    bottom_all = h - 1 - np.argmax(mask_for_extremes[::-1, :], axis=0)
    top = np.where(cols_any, top_all, -1)
    bottom = np.where(cols_any, bottom_all, -1)

    return left, right, top, bottom, rows_any, cols_any


def filter_extremes_by_median(extremes, threshold):
    """Filter extreme points by median deviation threshold."""
    filtered = extremes.copy()
    valid = filtered >= 0

    if valid.any():
        median_val = int(np.median(filtered[valid]))
        filtered[valid] = np.where(
            np.abs(filtered[valid] - median_val) <= threshold,
            filtered[valid],
            -1
        )

    return filtered


def create_border_mask(left, right, top, bottom, rows_any, cols_any, shape):
    """Create a border mask by marking filtered extreme points."""
    border_mask = np.zeros(shape, dtype=bool)

    # Mark left and right extremes for each row
    row_inds = np.where(rows_any)[0]
    for r in row_inds:
        if left[r] >= 0:
            border_mask[r, left[r]] = True
        if right[r] >= 0:
            border_mask[r, right[r]] = True

    # Mark top and bottom extremes for each column
    col_inds = np.where(cols_any)[0]
    for c in col_inds:
        if top[c] >= 0:
            border_mask[top[c], c] = True
        if bottom[c] >= 0:
            border_mask[bottom[c], c] = True

    return border_mask


def fit_x_of_y(y_indices, x_values):
    """Fit line x = a*y + b for vertical-like sides."""
    if len(y_indices) >= 2:
        a, b = np.polyfit(y_indices, x_values, 1)
    elif len(y_indices) == 1:
        a, b = 0.0, float(x_values[0])
    else:
        a, b = None, None
    return a, b


def fit_y_of_x(x_indices, y_values):
    """Fit line y = c*x + d for horizontal-like sides."""
    if len(x_indices) >= 2:
        c, d = np.polyfit(x_indices, y_values, 1)
    elif len(x_indices) == 1:
        c, d = 0.0, float(y_values[0])
    else:
        c, d = None, None
    return c, d


def intersect_lines(a, b, c, d):
    """Find intersection between lines x = ay + b and y = cx + d."""
    if a is None or c is None:
        return None

    denom = 1.0 - a * c
    if abs(denom) < 1e-6:
        return None

    x = (a * d + b) / denom
    y = c * x + d
    return (x, y)


def compute_polygon_corners(left, right, top, bottom, h, w):
    """Compute the four corners of the document polygon by fitting lines and finding intersections."""
    rows_left = np.where(left >= 0)[0]
    xs_left = left[rows_left] if rows_left.size > 0 else np.array([])

    rows_right = np.where(right >= 0)[0]
    xs_right = right[rows_right] if rows_right.size > 0 else np.array([])

    cols_top = np.where(top >= 0)[0]
    ys_top = top[cols_top] if cols_top.size > 0 else np.array([])

    cols_bottom = np.where(bottom >= 0)[0]
    ys_bottom = bottom[cols_bottom] if cols_bottom.size > 0 else np.array([])

    a_left, b_left = fit_x_of_y(rows_left, xs_left)
    a_right, b_right = fit_x_of_y(rows_right, xs_right)
    c_top, d_top = fit_y_of_x(cols_top, ys_top)
    c_bottom, d_bottom = fit_y_of_x(cols_bottom, ys_bottom)

    corners = [
        intersect_lines(a_left, b_left, c_top, d_top),
        intersect_lines(a_right, b_right, c_top, d_top),
        intersect_lines(a_right, b_right, c_bottom, d_bottom),
        intersect_lines(a_left, b_left, c_bottom, d_bottom)
    ]

    final_corners = []
    for i, corner in enumerate(corners):
        if corner is None:
            if i == 0:  # left-top
                x = np.median(xs_left) if xs_left.size > 0 else 0
                y = np.median(ys_top) if ys_top.size > 0 else 0
            elif i == 1:  # right-top
                x = np.median(xs_right) if xs_right.size > 0 else (w - 1)
                y = np.median(ys_top) if ys_top.size > 0 else 0
            elif i == 2:  # right-bottom
                x = np.median(xs_right) if xs_right.size > 0 else (w - 1)
                y = np.median(ys_bottom) if ys_bottom.size > 0 else (h - 1)
            else:  # left-bottom
                x = np.median(xs_left) if xs_left.size > 0 else 0
                y = np.median(ys_bottom) if ys_bottom.size > 0 else (h - 1)
            corner = (float(x), float(y))

        x_cl = int(np.clip(round(corner[0]), 0, w - 1))
        y_cl = int(np.clip(round(corner[1]), 0, h - 1))
        final_corners.append((x_cl, y_cl))

    return final_corners


def create_polygon_mask(corners, shape, fallback_mask):
    """Create a filled polygon mask from corner points."""
    h, w = shape
    polygon = np.array(corners, dtype=np.int32)
    poly_mask = np.zeros((h, w), dtype=np.uint8)

    try:
        cv2.fillPoly(poly_mask, [polygon], 1)
    except Exception:
        poly_mask = fallback_mask.astype(np.uint8)

    return poly_mask


# --- Main function using the new method ---

def create_mask_from_gradient(bgr_image, thr=20, pixel_border=15, gradient_threshold=0.15):
    """
    Creates a binary mask for the main object in an image using the morphological gradient method.
    """
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

    # Preprocess
    im_lab = convert_to_representation(rgb_image)
    mask = compute_edge_mask(im_lab, gradient_threshold)

    # Setup for border pixel processing
    mask_bool = mask.astype(bool)
    h, w = mask_bool.shape

    # Create mask with borders suppressed
    mask_for_extremes = create_border_suppressed_mask(mask_bool, pixel_border, h, w)

    # Compute extreme points
    left, right, top, bottom, rows_any, cols_any = compute_extreme_points(
        mask_for_extremes, h, w
    )

    # Filter extremes
    left_filtered = filter_extremes_by_median(left, thr)
    right_filtered = filter_extremes_by_median(right, thr)
    top_filtered = filter_extremes_by_median(top, thr)
    bottom_filtered = filter_extremes_by_median(bottom, thr)

    # Create a fallback mask from filtered extremes
    border_mask = create_border_mask(
        left_filtered, right_filtered, top_filtered, bottom_filtered,
        rows_any, cols_any, mask_bool.shape
    )

    # Fit lines and compute polygon corners
    final_corners = compute_polygon_corners(
        left_filtered, right_filtered, top_filtered, bottom_filtered, h, w
    )

    # Create final polygon mask
    poly_mask = create_polygon_mask(final_corners, (h, w), border_mask)

    # Convert mask from 0/1 to 0/255 for saving
    final_mask = (poly_mask * 255).astype(np.uint8)

    return final_mask


# --- Main execution logic ---

def process_and_save_masks(folder_path, num_images_to_test, output_folder='masks'):
    """
    Processes images in a folder, generates masks using the gradient method,
    and saves them to an output folder.
    """
    image_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.jpg')])

    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    print(f"Outputting masks to '{output_folder}/' folder.")

    for i, img_filename in enumerate(image_files[:num_images_to_test]):
        base_name = os.path.splitext(img_filename)[0]
        image_path = os.path.join(folder_path, img_filename)

        # Generate the mask using the new method
        predicted_mask = create_mask_from_gradient(image_path)

        # Save the predicted mask to the output folder
        output_path = os.path.join(output_folder, f"{base_name}.png")
        cv2.imwrite(output_path, predicted_mask)
        print(f"({i + 1}/{num_images_to_test}) Saved mask for {img_filename} to: {output_path}")


if __name__ == '__main__':
    dataset_folder = 'qsd2_w3'
    images_to_process = 29

    # Check if the dataset folder exists
    if not os.path.isdir(dataset_folder):
        print(f"Error: Dataset folder '{dataset_folder}' not found.")
        print("Please ensure the images are in the correct directory.")
    else:
        process_and_save_masks(dataset_folder, images_to_process, output_folder='masks')