import cv2
import numpy as np
import os

def binarize_mask(mask):
    """Convert mask to binary (0, 1). Assumes FG is 255."""
    return (mask > 127).astype(np.uint8)

def compute_metrics(gt_mask, pred_mask):
    """Compute TP, FP, FN, TN and derive precision, recall, F1."""
    TP = np.sum((gt_mask == 1) & (pred_mask == 1))
    FP = np.sum((gt_mask == 0) & (pred_mask == 1))
    FN = np.sum((gt_mask == 1) & (pred_mask == 0))
    TN = np.sum((gt_mask == 0) & (pred_mask == 0))
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1, TP, FP, FN, TN

def evaluate_masks(gt_folder, pred_folder):
    gt_files = sorted([f for f in os.listdir(gt_folder) if f.endswith('.png')])
    pred_files = sorted([f for f in os.listdir(pred_folder) if f.endswith('.png')])
    metrics_all = []

    for fname in gt_files:
        gt_path = os.path.join(gt_folder, fname)
        pred_path = os.path.join(pred_folder, fname)

        if not os.path.exists(pred_path):
            print(f"Predicted mask not found for {fname}. Skipping.")
            continue

        gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        pred_mask = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)

        gt_bin = binarize_mask(gt_mask)
        pred_bin = binarize_mask(pred_mask)
        precision, recall, f1, TP, FP, FN, TN = compute_metrics(gt_bin, pred_bin)
        metrics_all.append((fname, precision, recall, f1, TP, FP, FN, TN))

        print(f"{fname}: Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}, TP={TP}, FP={FP}, FN={FN}, TN={TN}")

    # Optionally, compute and print average metrics
    if metrics_all:
        avg_prec = np.mean([m[1] for m in metrics_all])
        avg_rec = np.mean([m[2] for m in metrics_all])
        avg_f1 = np.mean([m[3] for m in metrics_all])
        print(f"\nAverages over {len(metrics_all)} images:")
        print(f"Precision={avg_prec:.3f}, Recall={avg_rec:.3f}, F1={avg_f1:.3f}")

if __name__ == "__main__":
    GT_FOLDER = "qsd2_w1"
    OUTPUT_FOLDER = "output4"
    evaluate_masks(GT_FOLDER, OUTPUT_FOLDER)
