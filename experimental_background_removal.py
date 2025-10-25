import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

from scipy import signal

def segment_paintings(image_path, debug=False):
    # -----------------------------
    # Step 1: Load + Illumination correction
    # -----------------------------
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, _ = img.shape

    if debug:
        cv2.imshow("Original image", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)

    # Convert to LAB for luminance processing
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    L, A, B = cv2.split(lab)

    # Retinex-like illumination removal
    L_blur = cv2.GaussianBlur(L, (50 * 6 + 1, 50 * 6 + 1), 50)
    L_corr = cv2.normalize(L.astype(np.float32) - L_blur.astype(np.float32),
                           None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Replace L channel and convert back
    lab_corr = cv2.merge([L_corr, A, B])
    img_corr = cv2.cvtColor(lab_corr, cv2.COLOR_LAB2RGB)

    if debug:
        cv2.imshow("Illumination corrected", cv2.cvtColor(img_corr, cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)

    # -----------------------------
    # Step 2: Gradient map (Sobel)
    # -----------------------------
    gray = cv2.cvtColor(img_corr, cv2.COLOR_RGB2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(gx, gy)
    grad_mask = (grad >= 40).astype(np.float32)
    grad = grad * grad_mask
    grad = cv2.normalize(grad, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    grad_blur = cv2.GaussianBlur(grad, (5,5), 1)

    if debug:
        cv2.imshow("Gradient magnitude", grad_blur)
        cv2.waitKey(0)

    # -----------------------------
    # Step 3: Vertical & Horizontal Projections
    # -----------------------------
    col_sum = np.mean(grad_blur, axis=0)
    row_sum = np.mean(grad_blur, axis=1)

    # Smooth projections
    col_sum_smooth = cv2.blur(col_sum.reshape(1,-1).astype(np.float32), (51,1)).flatten()
    row_sum_smooth = cv2.blur(row_sum.reshape(-1,1).astype(np.float32), (1,51)).flatten()

    # col_sum_smooth[:15] = 0.0
    # col_sum_smooth[-15:] = 0.0
    # row_sum_smooth[:15] = 0.0
    # row_sum_smooth[-15:] = 0.0

    # Threshold to detect likely painting borders
    # col_thresh = np.mean(col_sum_smooth) * 2.5
    # row_thresh = np.mean(row_sum_smooth) * 2.0


    col_peaks = signal.find_peaks(col_sum_smooth, distance=5, prominence=2)
    row_peaks = signal.find_peaks(row_sum_smooth, distance=5, prominence=2)
    
    col_valleys = signal.find_peaks(-col_sum_smooth, distance=5, prominence=2)
    row_valleys = signal.find_peaks(-row_sum_smooth, distance=5, prominence=2)


    col_thresh = (np.mean(col_sum_smooth[:15]) + np.mean(col_sum_smooth[-15:])) / 2 * 1.5
    row_thresh = (np.mean(row_sum_smooth[:15]) + np.mean(row_sum_smooth[-15:])) / 2 * 1.5

    col_edges = np.where(col_sum_smooth > col_thresh)[0]
    row_edges = np.where(row_sum_smooth > row_thresh)[0]

    if debug:
        plt.figure()
        plt.title("Column sum")
        plt.hlines(y=[col_thresh], xmin=0, xmax=len(col_sum_smooth), colors=['r'], linestyles=['-'])
        plt.vlines(x=col_peaks[0], ymin=col_sum_smooth.min(), ymax=col_sum_smooth.max(), colors=['g' for _ in col_peaks[0]])
        plt.vlines(x=col_valleys[0], ymin=col_sum_smooth.min(), ymax=col_sum_smooth.max(), colors=['r' for _ in col_valleys[0]])
        plt.plot(col_sum_smooth)
        plt.show()

        plt.figure()
        plt.title("Row sum")
        plt.hlines(y=[row_thresh], xmin=0, xmax=len(row_sum_smooth), colors=['r'], linestyles=['-'])
        plt.vlines(x=row_peaks[0], ymin=row_sum_smooth.min(), ymax=row_sum_smooth.max(), colors=['g' for _ in row_peaks[0]])
        plt.vlines(x=row_valleys[0], ymin=row_sum_smooth.min(), ymax=row_sum_smooth.max(), colors=['r' for _ in row_valleys[0]])
        plt.plot(row_sum_smooth)
        plt.show()

        plt.figure()
        plt.imshow(img)
        plt.vlines(x=col_peaks[0], ymin=0, ymax=img.shape[0], colors=['g' for _ in col_peaks[0]])
        plt.vlines(x=col_valleys[0], ymin=0, ymax=img.shape[0], colors=['r' for _ in col_valleys[0]])
        plt.hlines(y=row_peaks[0], xmin=0, xmax=img.shape[1], colors=['g' for _ in row_peaks[0]])
        plt.hlines(y=row_valleys[0], xmin=0, xmax=img.shape[1], colors=['r' for _ in row_valleys[0]])
        plt.show()



    # Group consecutive columns into edge clusters
    def cluster_edges(edges, min_gap=5):
        if len(edges) == 0:
            return []
        clusters = [[edges[0]]]
        for e in edges[1:]:
            if e - clusters[-1][-1] > min_gap:
                clusters.append([e])
            else:
                clusters[-1].append(e)
        return [int(np.mean(c)) for c in clusters]


    row_mask = np.zeros(img.shape[:2], dtype=np.float32)
    row_mask[row_edges] = 1.0
    row_mask[:15] = 0.0
    row_mask[-15:] = 0.0
    col_mask = np.zeros(img.shape[:2], dtype=np.float32)
    col_mask[:, col_edges] = 1.0
    col_mask[:15] = 0.0
    col_mask[-15:] = 0.0

    mask = row_mask * col_mask


    vertical_bounds = cluster_edges(col_edges)
    horizontal_bounds = cluster_edges(row_edges)

    if debug:
        plt.figure()
        plt.title("Preliminary mask")
        plt.imshow(mask)
        plt.show()

        plt.figure()
        plt.title("Preliminary masked image")
        plt.imshow(img * np.expand_dims(mask.astype(np.uint8), axis=2))
        plt.show()

        img_proj = img.copy()
        for x in vertical_bounds:
            cv2.line(img_proj, (x,0), (x,h), (255,0,0), 2)
        for y in horizontal_bounds:
            cv2.line(img_proj, (0,y), (w,y), (0,255,0), 2)
        cv2.imshow("Projection edges", cv2.cvtColor(img_proj, cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)

    # -----------------------------
    # Step 4: Region extraction by variance
    # -----------------------------
    subregions = []
    if len(vertical_bounds) < 2:
        vertical_bounds = [0, w]  # fallback: one painting
    for i in range(len(vertical_bounds)-1):
        x1, x2 = vertical_bounds[i], vertical_bounds[i+1]
        sub_img = gray[:, x1:x2]
        var_map = cv2.blur((sub_img - cv2.blur(sub_img,(15,15)))**2, (15,15))
        var_sum = np.sum(var_map, axis=1)
        var_smooth = cv2.blur(var_sum.reshape(-1,1), (1,51)).flatten()
        row_thresh_local = np.mean(var_smooth) * 2.0
        y_candidates = np.where(var_smooth > row_thresh_local)[0]
        if len(y_candidates) == 0:
            y1, y2 = 0, h
        else:
            y1, y2 = y_candidates[0], y_candidates[-1]
        subregions.append((x1, y1, x2, y2))

    # -----------------------------
    # Step 5: (Optional) Perspective correction
    # -----------------------------
    masks = []
    for (x1, y1, x2, y2) in subregions:
        roi = gray[y1:y2, x1:x2]
        edges = cv2.Canny(roi, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=80, minLineLength=50, maxLineGap=20)

        # For simplicity, skip full rectification here (you can estimate corners if needed)
        # Just keep rectangular region for now
        mask = np.zeros((h,w), np.uint8)
        mask[y1:y2, x1:x2] = 255
        masks.append(mask)

    # -----------------------------
    # Step 6: Refinement and mask cleaning
    # -----------------------------
    final_mask = np.zeros((h,w), np.uint8)
    for m in masks:
        final_mask = cv2.bitwise_or(final_mask, m)

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15,15))
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel)
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel)

    if debug:
        cv2.imshow("Final mask", final_mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return final_mask


import cv2
import numpy as np

def split_if_two_paintings(image_path, debug=False, grad_valley_thresh=8.5, valley_width_frac=0.05):
    """
    Detects if an image likely contains two paintings side by side
    by analyzing the column-wise gradient magnitude profile.
    Splits at the valley if found.
    """
    # --- Load and preprocess ---
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, _ = img.shape

    # Convert to grayscale and normalize
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 1)

    # --- Compute Sobel gradients ---
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = cv2.magnitude(gx, gy)

    # --- Column-wise average gradient (vertical projection) ---
    col_profile = grad_mag.mean(axis=0)

    # --- Smooth the profile ---
    smooth_profile = cv2.GaussianBlur(col_profile.reshape(1, -1), (1, 99), 0).flatten()

    # Normalize for plotting and relative thresholding
    profile_norm = smooth_profile / (smooth_profile.max() + 1e-6) * 100

    # --- Detect valley near the center ---
    center_range = (int(w * 0.35), int(w * 0.65))
    center_vals = profile_norm[center_range[0]:center_range[1]]
    min_idx_rel = np.argmin(center_vals)
    min_val = center_vals[min_idx_rel]
    split_col = center_range[0] + min_idx_rel

    # Check if this valley is deep enough
    # Compute local mean around it
    valley_half_width = int(w * valley_width_frac)
    left_mean = np.mean(profile_norm[:split_col - valley_half_width])
    right_mean = np.mean(profile_norm[split_col + valley_half_width:])
    mean_side = (left_mean + right_mean) / 2

    # Condition for two paintings
    is_two = (min_val < grad_valley_thresh) and (min_val < 0.5 * mean_side)

    if debug:
        import matplotlib.pyplot as plt
        img_show = img.copy()
        cv2.imshow("Image", cv2.cvtColor(img_show, cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        plt.figure(figsize=(10,4))
        plt.plot(profile_norm, label="Smoothed column gradient profile")
        plt.axvline(split_col, color='r', linestyle='--', label=f"Candidate split @ {split_col}")
        plt.title(f"Valley min={min_val:.2f}, side mean={mean_side:.2f}, Two paintings={is_two}")
        plt.legend()
        plt.waitforbuttonpress()
        plt.close()
        # plt.show()

        img_show = img.copy()
        cv2.line(img_show, (split_col, 0), (split_col, h), (255, 0, 0), 3)
        cv2.imshow("Detected Split", cv2.cvtColor(img_show, cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # --- Return results ---
    if not is_two:
        return [img]  # single painting

    # Split into two
    left_img = img[:, :split_col]
    right_img = img[:, split_col:]
    return [left_img, right_img]



if __name__ == '__main__':
    folder = "/media/arnau-marcos-almansa/Ubuntu Data/MCV/C1/qsd2_w3_denoised"
    folder = "/media/arnau-marcos-almansa/Ubuntu Data/MCV/C1/qst2_w3_denoised"
    with open(os.path.join(folder, "paintings_per_image.txt"), "rt") as file:
        gt = [int(line.strip()) for line in file]

    i = 0
    for filename in sorted(os.listdir(folder)):
        if not filename.endswith('.jpg'):
            continue
        path = os.path.join(folder, filename)
        # segment_paintings(path, debug=True)
        parts = split_if_two_paintings(path, debug=False) # gt[i] == 2)
        report = "OK" if len(parts) == gt[i] else "BAD"
        print(f"Painting {filename} has {len(parts)} parts ({report}).")
        i += 1
