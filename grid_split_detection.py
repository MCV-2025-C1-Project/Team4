"""
Grid search script for painting split detection and splitting.

This script focuses exclusively on evaluating split detection accuracy
without background removal, allowing you to find configurations that
achieve 100% split detection accuracy.
"""

from pathlib import Path
import cv2
import numpy as np
import os
import traceback
from typing import Iterator, List
from enum import Enum

# Import all the split-related classes from the main script
from grid_background_removal_week3 import (
    SplitCase,
    GradientBasedCaseDetector,
    AspectRatioBasedCaseDetector,
    HybridCaseDetector,
    GradientBasedSplitter,
    HistogramBasedSplitter,
    EdgeBasedSplitter,
    PaintingSplitPipeline,
    load_split_ground_truth,
    evaluate_split_detection,
)


def generate_split_pipeline_configurations_extended() -> Iterator[dict]:
    """
    Generate comprehensive configurations for painting split detection and splitting.

    This includes MORE parameter variations than the original to find the best configuration.
    """

    # === 1. GRADIENT-BASED DETECTOR + GRADIENT SPLITTER (EXPANDED) ===
    # More granular search
    detection_thresholds = [5.0, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 11.0, 12.0]
    splitting_thresholds = [6.0, 7.0, 8.0, 8.5, 9.0, 10.0, 11.0, 12.0, 13.0]
    valley_width_fracs = [0.03, 0.04, 0.05, 0.06, 0.07, 0.08]

    for detect_thresh in detection_thresholds:
        for detect_valley_frac in valley_width_fracs:
            for split_thresh in splitting_thresholds:
                for split_valley_frac in valley_width_fracs:
                    detector = GradientBasedCaseDetector(
                        grad_valley_thresh=detect_thresh,
                        valley_width_frac=detect_valley_frac
                    )
                    splitter = GradientBasedSplitter(
                        grad_valley_thresh=split_thresh,
                        valley_width_frac=split_valley_frac
                    )
                    pipeline = PaintingSplitPipeline(detector, splitter)

                    yield {
                        'pipeline': pipeline,
                        'detector': detector,
                        'splitter': splitter,
                        'detector_type': 'Gradient',
                        'splitter_type': 'Gradient',
                        'description': (
                            f"GradDet(th={detect_thresh:.1f},vf={detect_valley_frac:.2f})+"
                            f"GradSplit(th={split_thresh:.1f},vf={split_valley_frac:.2f})"
                        ),
                        'detect_thresh': detect_thresh,
                        'detect_valley_frac': detect_valley_frac,
                        'split_thresh': split_thresh,
                        'split_valley_frac': split_valley_frac,
                    }
    """
    # === 2. ASPECT RATIO DETECTOR + VARIOUS SPLITTERS (EXPANDED) ===
    aspect_h_ratios = [1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
    aspect_v_ratios = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]

    for aspect_h in aspect_h_ratios:
        for aspect_v in aspect_v_ratios:
            detector = AspectRatioBasedCaseDetector(
                horizontal_ratio_thresh=aspect_h,
                vertical_ratio_thresh=aspect_v
            )

            # Try with Gradient splitter
            for split_thresh in [7.0, 8.0, 8.5, 9.0, 10.0, 11.0]:
                for split_valley_frac in [0.04, 0.05, 0.06, 0.07]:
                    splitter = GradientBasedSplitter(split_thresh, split_valley_frac)
                    pipeline = PaintingSplitPipeline(detector, splitter)

                    yield {
                        'pipeline': pipeline,
                        'detector': detector,
                        'splitter': splitter,
                        'detector_type': 'AspectRatio',
                        'splitter_type': 'Gradient',
                        'description': (
                            f"AspectDet(h={aspect_h:.2f},v={aspect_v:.2f})+"
                            f"GradSplit(th={split_thresh:.1f},vf={split_valley_frac:.2f})"
                        ),
                        'aspect_h': aspect_h,
                        'aspect_v': aspect_v,
                        'split_thresh': split_thresh,
                        'split_valley_frac': split_valley_frac,
                    }

            # Try with Histogram splitter
            for bins in [8, 12, 16, 20]:
                splitter = HistogramBasedSplitter(bins=bins)
                pipeline = PaintingSplitPipeline(detector, splitter)

                yield {
                    'pipeline': pipeline,
                    'detector': detector,
                    'splitter': splitter,
                    'detector_type': 'AspectRatio',
                    'splitter_type': 'Histogram',
                    'description': (
                        f"AspectDet(h={aspect_h:.2f},v={aspect_v:.2f})+"
                        f"HistoSplit(bins={bins})"
                    ),
                    'aspect_h': aspect_h,
                    'aspect_v': aspect_v,
                    'histo_bins': bins,
                }

            # Try with Edge splitter
            for canny_low, canny_high in [(20, 80), (30, 100), (40, 120), (50, 150)]:
                splitter = EdgeBasedSplitter(canny_low=canny_low, canny_high=canny_high)
                pipeline = PaintingSplitPipeline(detector, splitter)

                yield {
                    'pipeline': pipeline,
                    'detector': detector,
                    'splitter': splitter,
                    'detector_type': 'AspectRatio',
                    'splitter_type': 'Edge',
                    'description': (
                        f"AspectDet(h={aspect_h:.2f},v={aspect_v:.2f})+"
                        f"EdgeSplit(low={canny_low},high={canny_high})"
                    ),
                    'aspect_h': aspect_h,
                    'aspect_v': aspect_v,
                    'canny_low': canny_low,
                    'canny_high': canny_high,
                }

    # === 3. HYBRID DETECTOR + VARIOUS SPLITTERS (EXPANDED) ===
    for grad_thresh in [6.0, 7.0, 8.0, 8.5, 9.0, 10.0, 11.0]:
        for gradient_weight in [0.5, 0.6, 0.7, 0.8, 0.9]:
            for aspect_h_thresh in [1.4, 1.5, 1.6]:
                for aspect_v_thresh in [0.60, 0.67, 0.75]:
                    detector = HybridCaseDetector(
                        grad_valley_thresh=grad_thresh,
                        valley_width_frac=0.05,
                        aspect_h_thresh=aspect_h_thresh,
                        aspect_v_thresh=aspect_v_thresh,
                        gradient_weight=gradient_weight
                    )

                    # Try with Gradient splitter
                    splitter = GradientBasedSplitter(grad_thresh, 0.05)
                    pipeline = PaintingSplitPipeline(detector, splitter)

                    yield {
                        'pipeline': pipeline,
                        'detector': detector,
                        'splitter': splitter,
                        'detector_type': 'Hybrid',
                        'splitter_type': 'Gradient',
                        'description': (
                            f"HybridDet(gth={grad_thresh:.1f},gw={gradient_weight:.1f},"
                            f"ah={aspect_h_thresh:.2f},av={aspect_v_thresh:.2f})+"
                            f"GradSplit(th={grad_thresh:.1f})"
                        ),
                        'grad_thresh': grad_thresh,
                        'gradient_weight': gradient_weight,
                        'aspect_h_thresh': aspect_h_thresh,
                        'aspect_v_thresh': aspect_v_thresh,
                    }

                    # Try with Histogram splitter
                    splitter = HistogramBasedSplitter(bins=16)
                    pipeline = PaintingSplitPipeline(detector, splitter)

                    yield {
                        'pipeline': pipeline,
                        'detector': detector,
                        'splitter': splitter,
                        'detector_type': 'Hybrid',
                        'splitter_type': 'Histogram',
                        'description': (
                            f"HybridDet(gth={grad_thresh:.1f},gw={gradient_weight:.1f},"
                            f"ah={aspect_h_thresh:.2f},av={aspect_v_thresh:.2f})+"
                            f"HistoSplit(bins=16)"
                        ),
                        'grad_thresh': grad_thresh,
                        'gradient_weight': gradient_weight,
                        'aspect_h_thresh': aspect_h_thresh,
                        'aspect_v_thresh': aspect_v_thresh,
                    }

    # === 4. GRADIENT DETECTOR + ALTERNATIVE SPLITTERS (EXPANDED) ===
    for detect_thresh in [6.0, 7.0, 8.0, 8.5, 9.0, 10.0, 11.0]:
        detector = GradientBasedCaseDetector(detect_thresh, 0.05)

        # Histogram splitter
        for bins in [8, 12, 16, 20]:
            splitter = HistogramBasedSplitter(bins=bins)
            pipeline = PaintingSplitPipeline(detector, splitter)

            yield {
                'pipeline': pipeline,
                'detector': detector,
                'splitter': splitter,
                'detector_type': 'Gradient',
                'splitter_type': 'Histogram',
                'description': (
                    f"GradDet(th={detect_thresh:.1f})+"
                    f"HistoSplit(bins={bins})"
                ),
                'detect_thresh': detect_thresh,
                'histo_bins': bins,
            }

        # Edge splitter
        for canny_low, canny_high in [(30, 100), (40, 120), (50, 150), (60, 180)]:
            splitter = EdgeBasedSplitter(canny_low=canny_low, canny_high=canny_high)
            pipeline = PaintingSplitPipeline(detector, splitter)

            yield {
                'pipeline': pipeline,
                'detector': detector,
                'splitter': splitter,
                'detector_type': 'Gradient',
                'splitter_type': 'Edge',
                'description': (
                    f"GradDet(th={detect_thresh:.1f})+"
                    f"EdgeSplit(low={canny_low},high={canny_high})"
                ),
                'detect_thresh': detect_thresh,
                'canny_low': canny_low,
                'canny_high': canny_high,
            }
    """


def quick_split_detection_check(split_predictions: dict[str, SplitCase],
                                split_ground_truth: dict[str, SplitCase]) -> dict:
    """
    Quick evaluation of split detection for a single configuration.

    Args:
        split_predictions: Dictionary mapping image names to predicted SplitCase
        split_ground_truth: Dictionary mapping image names to ground truth SplitCase

    Returns:
        Dictionary with 'num_correct', 'num_total', 'num_failures', 'failures' list
    """
    if not split_ground_truth or not split_predictions:
        return {'num_correct': 0, 'num_total': 0, 'num_failures': 0, 'failures': []}

    # Count matches and failures
    num_correct = 0
    num_total = 0
    failures = []

    for img_name, predicted_case in split_predictions.items():
        if img_name in split_ground_truth:
            num_total += 1
            gt_case = split_ground_truth[img_name]
            if predicted_case == gt_case:
                num_correct += 1
            else:
                failures.append(f"{img_name}(pred:{predicted_case.value},gt:{gt_case.value})")

    num_failures = num_total - num_correct

    return {
        'num_correct': num_correct,
        'num_total': num_total,
        'num_failures': num_failures,
        'failures': failures
    }


def load_queries_simple(queries_path: str) -> list:
    """Load query images without masks (only need images for split detection)."""
    queries = []

    for filename in sorted(os.listdir(queries_path)):
        if not filename.endswith(".jpg"):
            continue

        image_path = os.path.join(queries_path, filename)
        image = cv2.imread(image_path)

        if image is None:
            print(f"Warning: Could not load {image_path}")
            continue

        queries.append({
            'image': image,
            'name': filename,
            'id': int(Path(image_path).stem),
        })

    return queries


if __name__ == '__main__':
    dataset_folder = "/home/arnau-marcos-almansa/workspace/Team4/qsd1_w4"

    # Load split detection ground truth
    split_gt_path = os.path.join(dataset_folder, "split_ground_truth.txt")
    split_ground_truth = load_split_ground_truth(split_gt_path)

    if not split_ground_truth:
        print("ERROR: No split ground truth found! Cannot evaluate.")
        print(f"Please ensure {split_gt_path} exists.")
        exit(1)

    # Load images
    queries = load_queries_simple(dataset_folder)

    # Count total configurations
    num_configs = len(list(generate_split_pipeline_configurations_extended()))

    print(f"="*100)
    print(f"SPLIT DETECTION GRID SEARCH")
    print(f"="*100)
    print(f"Dataset: {dataset_folder}")
    print(f"Total images: {len(queries)}")
    print(f"Ground truth entries: {len(split_ground_truth)}")
    print(f"Total configurations to test: {num_configs}")
    print(f"="*100)
    print()

    all_results = []
    perfect_configs = []  # Track configs with 100% accuracy

    # Test each configuration
    for config_idx, split_config in enumerate(generate_split_pipeline_configurations_extended(), 1):
        pipeline = split_config['pipeline']
        split_desc = split_config['description']

        # Store predictions for this configuration
        split_predictions = {}

        # Process all images
        for query in queries:
            image = query['image']
            image_name = query['name']

            try:
                # Detect split case
                split_case, _ = pipeline.process(image)

                # Store detection (use image name without extension)
                img_name_no_ext = Path(image_name).stem
                split_predictions[img_name_no_ext] = split_case

            except Exception as e:
                print(f"ERROR processing {image_name} with {split_desc[:50]}: {e}")
                traceback.print_exc()
                continue

        # Evaluate split detection
        split_eval = quick_split_detection_check(split_predictions, split_ground_truth)

        # Store results
        result = {
            'config': split_desc,
            'detector_type': split_config.get('detector_type', 'Unknown'),
            'splitter_type': split_config.get('splitter_type', 'Unknown'),
            'num_correct': split_eval['num_correct'],
            'num_total': split_eval['num_total'],
            'num_failures': split_eval['num_failures'],
            'accuracy': split_eval['num_correct'] / split_eval['num_total'] if split_eval['num_total'] > 0 else 0.0,
            'failures': split_eval['failures'],
        }

        # Add all config parameters for CSV export
        for key, value in split_config.items():
            if key not in ['pipeline', 'detector', 'splitter', 'description']:
                result[key] = value

        all_results.append(result)

        # Track perfect configurations
        if split_eval['num_failures'] == 0 and split_eval['num_total'] > 0:
            perfect_configs.append(result)

        # Print progress
        if split_eval['num_failures'] == 0:
            status = f"✓ OK ({split_eval['num_correct']}/{split_eval['num_total']})"
        else:
            status = f"✗ {split_eval['num_failures']} fail"

        # Truncate description for display
        display_desc = split_desc if len(split_desc) <= 70 else split_desc[:67] + "..."

        print(f"[{config_idx:4d}/{num_configs}] {display_desc:70s} | {status}")

        # If there are failures, show details
        if split_eval['num_failures'] > 0 and split_eval['num_failures'] <= 5:
            failures_str = ', '.join(split_eval['failures'])
            print(f"           └─ Failures: {failures_str}")

    # === RESULTS SUMMARY ===
    print("\n" + "="*100)
    print("SUMMARY")
    print("="*100)

    # Sort by accuracy (descending), then by num_failures (ascending)
    all_results.sort(key=lambda x: (-x['accuracy'], x['num_failures']))

    # Show perfect configurations
    print(f"\n✓ PERFECT CONFIGURATIONS (100% accuracy): {len(perfect_configs)}")
    if perfect_configs:
        print("-"*100)
        for i, result in enumerate(perfect_configs[:20], 1):  # Show first 20
            config_short = result['config'][:70] if len(result['config']) > 70 else result['config']
            print(f"{i:3d}. {config_short:70s} | {result['num_correct']}/{result['num_total']}")

        if len(perfect_configs) > 20:
            print(f"     ... and {len(perfect_configs) - 20} more perfect configurations")

    # Show top 20 overall (including imperfect ones)
    print(f"\nTOP 20 CONFIGURATIONS (sorted by accuracy):")
    print("-"*100)
    print(f"{'Rank':<5} {'Configuration':<60} {'Accuracy':>10} {'Correct/Total':>15} {'Failures':>10}")
    print("-"*100)
    for i, result in enumerate(all_results[:20], 1):
        config_short = result['config'][:60] if len(result['config']) > 60 else result['config']
        correct_total = f"{result['num_correct']}/{result['num_total']}"
        print(f"{i:<5} {config_short:<60} {result['accuracy']:>10.4f} {correct_total:>15} {result['num_failures']:>10}")

    # === SAVE RESULTS TO CSV ===
    import pandas as pd
    df = pd.DataFrame(all_results)
    csv_path = "split_detection_grid_search_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n✅ Full results saved to: {csv_path}")

    # Save perfect configurations to separate file
    if perfect_configs:
        perfect_df = pd.DataFrame(perfect_configs)
        perfect_csv_path = "split_detection_perfect_configs.csv"
        perfect_df.to_csv(perfect_csv_path, index=False)
        print(f"✅ Perfect configurations saved to: {perfect_csv_path}")

    print(f"\n{'='*100}")
    print(f"Grid search complete!")
    print(f"Total configurations tested: {len(all_results)}")
    print(f"Perfect configurations found: {len(perfect_configs)}")
    print(f"{'='*100}")
