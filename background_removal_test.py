import cv2
import numpy as np
from scipy import stats
import os

def create_solid_mask(image_path, k=3, debug=False):
    image = cv2.imread(image_path)
    pixel_vals = image.reshape((-1, 3))
    pixel_vals = np.float32(pixel_vals)
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85)
    retval, labels, centers = cv2.kmeans(pixel_vals, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    labels_reshaped = labels.reshape(image.shape[0], image.shape[1])
    
    border_pixels = np.concatenate([labels_reshaped[0, :], labels_reshaped[-1, :], labels_reshaped[:, 0], labels_reshaped[:, -1]])
    background_cluster_id = stats.mode(border_pixels, keepdims=False).mode
    
    initial_mask = np.zeros(labels_reshaped.shape, dtype=np.uint8)
    initial_mask[labels_reshaped != background_cluster_id] = 255
    
    if debug:
        cv2.imshow('Mask', initial_mask)

    kernel_open = np.ones((10, 10), np.uint8)
    mask_after_opening = cv2.morphologyEx(initial_mask, cv2.MORPH_OPEN, kernel_open)
    
    if debug:
        cv2.imshow('2 - Mask opening', mask_after_opening)

    kernel_close = np.ones((30, 30), np.uint8)
    final_mask = cv2.morphologyEx(mask_after_opening, cv2.MORPH_CLOSE, kernel_close)

    if debug:
        cv2.imshow('Mask closing', final_mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    # After your final_mask is computed, add this:
    # ys, xs = np.where(final_mask == 255)
    # if len(xs) > 0 and len(ys) > 0:
    #     xmin, xmax = xs.min(), xs.max()
    #     ymin, ymax = ys.min(), ys.max()
    #     final_mask_rect = np.zeros_like(final_mask)
    #     final_mask_rect[ymin:ymax+1, xmin:xmax+1] = 255
    # else:
    #     final_mask_rect = final_mask.copy()  # fallback


    return final_mask

#Averages over 29 images: Precision=0.842, Recall=0.737, F1=0.775
# Precision=0.730, Recall=0.998, F1=0.830 cu fill
def resize_with_aspect_ratio(image, new_height):
    h, w = image.shape[:2]
    aspect_ratio = w / h
    new_width = int(new_height * aspect_ratio)
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return resized_image


def evaluate_and_visualize(folder_path, num_images_to_test, display_height=400, output_folder='output'):
    image_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.jpg')])
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    for i, img_filename in enumerate(image_files[:num_images_to_test]):
        base_name = os.path.splitext(img_filename)[0]
        image_path = os.path.join(folder_path, img_filename)
        gt_path = os.path.join(folder_path, f"{base_name}.png")

        original_img = cv2.imread(image_path)
        ground_truth_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)

        predicted_mask = create_solid_mask(image_path, k=2, debug=False)
        
        # Save the predicted mask to output folder
        output_path = os.path.join(output_folder, f"{base_name}.png")
        cv2.imwrite(output_path, predicted_mask)
        print(f"Saved mask to: {output_path}")

    #     resized_original = resize_with_aspect_ratio(original_img, display_height)
    #     resized_gt = resize_with_aspect_ratio(ground_truth_mask, display_height)
    #     resized_predicted = resize_with_aspect_ratio(predicted_mask, display_height)

    #     gt_bgr = cv2.cvtColor(resized_gt, cv2.COLOR_GRAY2BGR)
    #     predicted_bgr = cv2.cvtColor(resized_predicted, cv2.COLOR_GRAY2BGR)

    #     cv2.putText(resized_original, 'Original Image', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    #     cv2.putText(gt_bgr, 'Ground Truth', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    #     cv2.putText(predicted_bgr, 'Predicted Mask', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    #     comparison_image = np.hstack([resized_original, gt_bgr, predicted_bgr])

    #     window_title = f"Comparison {i+1}/{num_images_to_test}: {img_filename}"
    #     cv2.imshow(window_title, comparison_image)
        
    #     key = cv2.waitKey(0)
        
    #     # Safely close window - cv2.destroyAllWindows handles all cases
    #     if key == 27:  # ESC key to exit early
    #         break

    # cv2.destroyAllWindows()


if __name__ == '__main__':
    dataset_folder = 'qsd2_w1'
    images_to_process = 29

    evaluate_and_visualize(dataset_folder, images_to_process, display_height=400)
    
    # create_solid_mask('dataset/qsd2_w2/00002.jpg', k=3, debug=True)