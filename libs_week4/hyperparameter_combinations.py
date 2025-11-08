# libs_week4/hyperparameter_combinations.py

import sys
from pathlib import Path
import itertools
from typing import Iterator, Dict, Any
import cv2

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from libs_week4.descriptor import (
    DescriptorComputer, HomographyScorer, MatchRatioScorer, SymmetricMatchRatioScorer,
    HomographyDistanceScorer, MultiFactorScorer, ORBDescriptor, DaisyDescriptor, SIFTDescriptor,
    RootSIFTDescriptor, BRISKDescriptor, KAZEDescriptor, AKAZEDescriptor, PCASIFTDescriptor,
    HOGDescriptor, GLOHDescriptor, DescriptorMatcher, KeypointAndDescriptorMaker, SURFDescriptor,
    DescriptorValueType
)
from libs_week3.color_conversion import ColorConversion, ColorSpace
import libs_week3.preprocessing as preprocessing


def generate_orb_configs() -> Iterator[Dict[str, Any]]:
    # OPTIMIZED GRID (9 configs): Based on ranking results - ORB is the clear winner!
    # mAP@k1 = 0.74-0.77 (best overall), fast descriptor computation (8-12s)
    #
    # Key findings from experiments:
    # - n_features: 2000-3000 are optimal (top 10 results)
    # - scale_factor: Both 1.2 and 1.5 work well
    # - wta_k=2 DOMINATES: wta_k=3 has 15-20% worse performance + 10x slower queries
    # - Add n_features=2500 to explore the sweet spot between 2000 and 3000
    param_grid = {
        'n_features': [2000, 2500, 3000],
        'scale_factor': [1.2, 1.5],
        'n_levels': [10],
        'wta_k': [2],  # Removed wta_k=3: much worse performance and query time
        'score_type': [cv2.ORB_HARRIS_SCORE],
        'patch_size': [31]
    }
    # Total: 3 × 2 × 1 = 6 configs
    keys, values = zip(*param_grid.items())
    for v in itertools.product(*values):
        yield dict(zip(keys, v))
        

def generate_surf_configs() -> Iterator[Dict[str, Any]]:
    param_grid = {
        'hessian_threshold': [100],
        'n_octaves': [4],
        'n_octave_layers': [3],
        'extended': [False],
        'upright': [False],
    }
    keys, values = zip(*param_grid.items())
    for v in itertools.product(*values):
        yield dict(zip(keys, v))


def generate_sift_configs() -> Iterator[Dict[str, Any]]:
    # DEPRECATED: Regular SIFT consistently underperforms RootSIFT (0.59-0.64 vs 0.69-0.77)
    # RootSIFT has same computational cost but better normalization
    # Keeping this function for reference but it's not used in the grid search
    param_grid = {
        'n_features': [1000],
        'n_octave_layers': [4],
        'contrast_threshold': [0.03],
        'sigma': [1.6],
        'edge_threshold': [15]
    }
    # Total: 1 config (not used)
    keys, values = zip(*param_grid.items())
    for v in itertools.product(*values):
        yield dict(zip(keys, v))

def generate_rootsift_configs() -> Iterator[Dict[str, Any]]:
    # EXPERIMENTAL: RISKY CONFIGS ONLY (for textureless images)
    # This is a focused experiment to test if lower thresholds help textureless images
    # without re-running the safe configs we already tested.
    #
    # RISKY PARAMETERS (for textureless images):
    # - contrast_threshold: 0.01, 0.02 (REMOVED safe 0.03, 0.04)
    # - edge_threshold: 10 (REMOVED safe 15)
    # - Keeping n_features and sigma variations as in original
    #
    # WARNING: May degrade performance on textured images or add noise
    param_grid = {
        'n_features': [1000, 1500, 2000],
        'n_octave_layers': [4],
        'contrast_threshold': [0.01, 0.02],  # RISKY ONLY: Low thresholds for textureless images
        'sigma': [1.6, 2.0],
        'edge_threshold': [10]  # RISKY ONLY: Low edge threshold
    }
    # Total: 3 × 2 × 2 × 1 = 12 configs (focused on risky params)
    keys, values = zip(*param_grid.items())
    for v in itertools.product(*values):
        yield dict(zip(keys, v))

def generate_brisk_configs() -> Iterator[Dict[str, Any]]:
    # ORIGINAL FULL GRID (3 configs):
    # param_grid = {
    #     'thresh': [50, 70, 100],
    #     'octaves': [3],
    #     'pattern_scale': [1.0]
    # }

    # REDUCED GRID (3 configs): Already minimal - BRISK was too slow (too many keypoints)
    # Keep existing config to see if higher thresholds help
    param_grid = {
        'thresh': [70, 100, 120],  # Increased thresholds to reduce keypoint count
        'octaves': [3],
        'pattern_scale': [1.0]
    }
    keys, values = zip(*param_grid.items())
    for v in itertools.product(*values):
        yield dict(zip(keys, v))

def generate_kaze_configs() -> Iterator[Dict[str, Any]]:
    # ORIGINAL FULL GRID (6 configs):
    # param_grid = {
    #     'extended': [False],
    #     'upright': [False],
    #     'threshold': [0.0001, 0.001, 0.003],
    #     'n_octaves': [4],
    #     'n_octave_layers': [4, 5]
    # }

    # REDUCED GRID (4 configs): Increased threshold to reduce excessive keypoints
    # threshold=0.0001 generates 24k+ keypoints (way too many, very slow)
    # Increasing to [0.001, 0.003] for reasonable keypoint counts
    param_grid = {
        'extended': [False],
        'upright': [False],
        'threshold': [0.001, 0.003],  # Removed 0.0001 - too many keypoints
        'n_octaves': [4],
        'n_octave_layers': [4, 5]
    }
    keys, values = zip(*param_grid.items())
    for v in itertools.product(*values):
        yield dict(zip(keys, v))

def generate_akaze_configs() -> Iterator[Dict[str, Any]]:
    # ORIGINAL FULL GRID (12 configs):
    # param_grid = {
    #     'threshold': [0.003, 0.005, 0.007],
    #     'n_octaves': [4, 5],
    #     'n_octave_layers': [4, 5]
    # }

    # REDUCED GRID (6 configs): AKAZE mAP@k1 ~0.45 (underperformer)
    # Key insights: One config fails (no keypoints), performs poorly overall
    # Reduce to most stable configurations
    param_grid = {
        'threshold': [0.003, 0.005],
        'n_octaves': [4],
        'n_octave_layers': [4, 5, 6]
    }
    keys, values = zip(*param_grid.items())
    for v in itertools.product(*values):
        yield dict(zip(keys, v))

def generate_daisy_configs() -> Iterator[Dict[str, Any]]:
    # ORIGINAL FULL GRID (3 configs):
    # param_grid = {
    #     'step': [16, 32, 64],
    #     'radius': [15],
    #     'rings': [3],
    #     'histograms': [8],
    #     'orientations': [8]
    # }

    # REDUCED GRID (3 configs): DAISY was too slow (24k keypoints at step=16)
    # Key insights: Larger step = fewer keypoints = faster
    # Keep only larger steps for speed
    param_grid = {
        'step': [32, 48, 64],  # Increased minimum step to reduce keypoints
        'radius': [15],
        'rings': [3],
        'histograms': [8],
        'orientations': [8]
    }
    keys, values = zip(*param_grid.items())
    for v in itertools.product(*values):
        yield dict(zip(keys, v))

def generate_hog_configs() -> Iterator[Dict[str, Any]]:
    param_grid = {
        'win_size': [(32, 32), (48, 48)],
        'block_size': [(16, 16)],
        'block_stride': [(8, 8)],
        'cell_size': [(8, 8)],
        'nbins': [9, 12],
        'n_features': [500, 1000, 2000] # SIFT param
    }
    keys, values = zip(*param_grid.items())
    for v in itertools.product(*values):
        yield dict(zip(keys, v))

def generate_gloh_configs() -> Iterator[Dict[str, Any]]:
    param_grid = {
        'nbins': [36, 48],
        'n_features': [500, 1000, 2000] # SIFT param
    }
    keys, values = zip(*param_grid.items())
    for v in itertools.product(*values):
        yield dict(zip(keys, v))
        
def generate_pcasift_configs() -> Iterator[Dict[str, Any]]:
    # EXPLORATORY GRID (6 configs): Now properly implemented with database-wide PCA fitting
    # Previous results were invalid (fit PCA per-image instead of per-database)
    #
    # Testing reduced dimensionality to see if it helps:
    # - num_components: Focus on smaller dimensions (24, 36) for speed
    # - Use RootSIFT's best parameters as base
    param_grid = {
        'num_components': [24, 36],
        'n_features': [1000, 1500],
        'n_octave_layers': [4],
        'contrast_threshold': [0.03],  # Best from RootSIFT
        'sigma': [1.6],  # Standard value
        'edge_threshold': [15]
    }
    # Total: 2 × 2 = 4 configs
    keys, values = zip(*param_grid.items())
    for v in itertools.product(*values):
        yield dict(zip(keys, v))


def generate_keypoint_descriptors() -> Iterator[DescriptorComputer]:
    """
    Generates instances of different keypoint descriptors by iterating
    through all their specified hyperparameter configurations.

    EXPERIMENTAL BRANCH: RISKY CONFIGS ONLY
    Testing ONLY risky RootSIFT configs to see if they help textureless images.
    - Removed ORB entirely (already tested, 94.8% baseline)
    - RootSIFT: 12 RISKY configs with low thresholds:
      - contrast_threshold: 0.01, 0.02 (vs safe 0.03)
      - edge_threshold: 10 (vs safe 15)
    Total: 12 descriptor configs × 5 scorer configs = 60 experiments
    """
    # === EXPERIMENTAL BRANCH: RISKY CONFIGS ONLY ===
    # Removed ORB to avoid repeating experiments
    # Keeping only risky RootSIFT configs to test textureless image support

    # RootSIFT: RISKY configs with low thresholds
    for config in generate_rootsift_configs():
        yield RootSIFTDescriptor(**config)

    # === DISCARDED DESCRIPTORS (from ranking analysis) ===

    # PCA-SIFT: Consistently poor performance (mAP@k1 ~0.51-0.72, bottom 20%)
    # Slow fitting + poor results make it not worth exploring
    # for config in generate_pcasift_configs():
    #     yield PCASIFTDescriptor(**config)

    # SIFT: Consistently worse than RootSIFT (0.59-0.64 vs 0.69-0.77)
    # for config in generate_sift_configs():
    #     yield SIFTDescriptor(**config)

    # KAZE: Too slow (455-544s) + poor performance (0.46-0.54) + too many keypoints (5k-24k)
    # for config in generate_kaze_configs():
    #     yield KAZEDescriptor(**config)

    # SURF: Not available by default in opencv
    # for config in generate_surf_configs():
    #     yield SURFDescriptor(**config)

    # BRISK: Too many keypoints, slow
    # for config in generate_brisk_configs():
    #     yield BRISKDescriptor(**config)

    # AKAZE: Poor performance (mAP@k1 ~0.45), some configs fail
    # for config in generate_akaze_configs():
    #     yield AKAZEDescriptor(**config)

    # DAISY: Too many keypoints (24k at step=16), very slow
    # for config in generate_daisy_configs():
    #     yield DaisyDescriptor(**config)

    # HOG: Very poor performance
    # for config in generate_hog_configs():
    #     yield HOGDescriptor(**config)

    # GLOH: Very poor performance
    # for config in generate_gloh_configs():
    #     yield GLOHDescriptor(**config)


def generate_color_space_combinations() -> list[list[ColorSpace]]:
    return [[ColorSpace.BGR]]

def generate_preprocess_strategies() -> list[preprocessing.ImagePreprocessStep | None]:
    return [
        preprocessing.Preprocess([
            # preprocessing.CropToMask(),
        ])
    ]

def descriptor_maker_grid_search() -> Iterator[KeypointAndDescriptorMaker]:
    """
    Generator for descriptor maker configurations.
    Yields KeypointAndDescriptorMaker instances that can be used to compute descriptors.

    This should be used in the outer loop - compute descriptors once per config,
    then iterate through scorers without recomputing.
    """
    for color_spaces in generate_color_space_combinations():
        for preprocess in generate_preprocess_strategies():
            for descriptor in generate_keypoint_descriptors():
                yield KeypointAndDescriptorMaker(
                    descriptor_computer=descriptor,
                    color_conversion=ColorConversion(targets=color_spaces, normalize=True),
                    preprocess=preprocess
                )

def generate_alternative_scorers(descriptor_maker: KeypointAndDescriptorMaker) -> Iterator[Dict[str, Any]]:
    """
    Generate alternative scorer configurations to compare against HomographyScorer.

    Tests:
    1. MatchRatioScorer - Simple baseline (no geometric verification)
    2. SymmetricMatchRatioScorer - Normalized baseline
    3. HomographyDistanceScorer - Homography + distance consistency

    Args:
        descriptor_maker: The KeypointAndDescriptorMaker to generate scorers for

    Yields:
        Dictionary containing 'matcher' and 'scorer' keys
    """
    # Determine the appropriate norm type
    descriptor_value_type = descriptor_maker.descriptor_computer.get_value_type()

    if descriptor_value_type == DescriptorValueType.FLOAT:
        norm_type = cv2.NORM_L2
    else:  # BINARY
        norm_type = cv2.NORM_HAMMING

    # Use the best ratio from previous experiments
    ratio_threshold = 0.7

    matcher = DescriptorMatcher(
        matcher_type='BF',
        norm_type=norm_type,
        ratio_test_threshold=ratio_threshold,
        cross_check=False
    )

    # Config 1: Simple Match Ratio (baseline - no homography)
    yield {
        'matcher': matcher,
        'scorer': MatchRatioScorer(matcher, min_matches=10)
    }

    # Config 2: Symmetric Match Ratio (normalized baseline)
    yield {
        'matcher': matcher,
        'scorer': SymmetricMatchRatioScorer(matcher, min_matches=10)
    }

    # Config 3: Homography + Distance Consistency (light weight)
    yield {
        'matcher': matcher,
        'scorer': HomographyDistanceScorer(
            matcher,
            ransac_thresh=3.0,
            max_reproj_error=3.0,
            min_points=20,
            distance_weight=0.2  # Light emphasis on distance consistency
        )
    }

    # Config 4: Homography + Distance Consistency (moderate weight)
    yield {
        'matcher': matcher,
        'scorer': HomographyDistanceScorer(
            matcher,
            ransac_thresh=3.0,
            max_reproj_error=3.0,
            min_points=20,
            distance_weight=0.4  # More emphasis on distance consistency
        )
    }

    # NOTE: MultiFactorScorer is implemented but commented out for later exploration
    # It requires more extensive tuning of weights (inlier, reproj, distance)
    # Uncomment and add grid search if time permits:
    #
    # yield {
    #     'matcher': matcher,
    #     'scorer': MultiFactorScorer(
    #         matcher,
    #         ransac_thresh=3.0,
    #         max_reproj_error=3.0,
    #         min_points=20,
    #         inlier_weight=0.5,
    #         reproj_weight=0.3,
    #         distance_weight=0.2
    #     )
    # }


def scorer_grid_search(descriptor_maker: KeypointAndDescriptorMaker, include_alternative_scorers: bool = False) -> Iterator[Dict[str, Any]]:
    """
    Generator for scorer configurations for a given descriptor maker.

    EXPERIMENTAL BRANCH: RISKY CONFIGS ONLY (min_points=10)
    Testing if very low min_points (10) can help match textureless images
    that have few keypoints detected with risky descriptor thresholds.

    Normal optimal values (from 94.8% mAP@k1 baseline):
    - ratio_threshold=0.65 (kept: 0.65, 0.7)
    - ransac_thresh=3.0 (kept)
    - max_reproj_error=3.0 (kept)
    - min_points=20 (REPLACED with 10 for this experiment)
    - reproj_error_penalty_weight=0.1 or 0.2 (kept)

    WARNING: min_points=10 is VERY LOW and may cause false positives.
    This is intentional to see if we can salvage textureless images.

    Args:
        descriptor_maker: The KeypointAndDescriptorMaker to generate scorers for
        include_alternative_scorers: If True, also yield alternative scorer types
                                     (MatchRatio, SymmetricMatchRatio, HomographyDistance)

    Yields:
        Dictionary containing 'matcher' and 'scorer' keys
    """

    # First, yield alternative scorers if requested
    if include_alternative_scorers:
        for scorer_config in generate_alternative_scorers(descriptor_maker):
            yield scorer_config
    # Determine the appropriate norm type based on descriptor value type
    descriptor_value_type = descriptor_maker.descriptor_computer.get_value_type()

    if descriptor_value_type == DescriptorValueType.FLOAT:
        norm_type = cv2.NORM_L2
    else:  # BINARY
        norm_type = cv2.NORM_HAMMING

    # EXPERIMENTAL: RISKY CONFIGS ONLY (min_points=10 for textureless images)
    # Removed all safe configs (min_points 15, 20, 25) to focus experiment
    # Testing ONLY min_points=10 to see if it helps textureless images
    #
    # RISKY PARAMETER:
    # - min_points: 10 (very low threshold to allow matching with few keypoints)
    # WARNING: May allow false positives on textured images
    scorer_configs = [
        # Config 1: RISKY - min_points=10, ratio=0.65, no penalty
        {
            'ratio_threshold': 0.65,
            'ransac_thresh': 3.0,
            'max_reproj_error': 3.0,
            'use_reproj_error_penalty': False,
            'reproj_error_penalty_weight': 0.1,
            'min_points': 10  # RISKY: Very low threshold for textureless images
        },
        # Config 2: RISKY - min_points=10, ratio=0.65, gentle penalty
        {
            'ratio_threshold': 0.65,
            'ransac_thresh': 3.0,
            'max_reproj_error': 3.0,
            'use_reproj_error_penalty': True,
            'reproj_error_penalty_weight': 0.1,
            'min_points': 10  # RISKY: Very low threshold for textureless images
        },
        # Config 3: RISKY - min_points=10, ratio=0.65, moderate penalty
        {
            'ratio_threshold': 0.65,
            'ransac_thresh': 3.0,
            'max_reproj_error': 3.0,
            'use_reproj_error_penalty': True,
            'reproj_error_penalty_weight': 0.2,
            'min_points': 10  # RISKY: Very low threshold for textureless images
        },
        # Config 4: RISKY - min_points=10, ratio=0.7, no penalty
        {
            'ratio_threshold': 0.7,
            'ransac_thresh': 3.0,
            'max_reproj_error': 3.0,
            'use_reproj_error_penalty': False,
            'reproj_error_penalty_weight': 0.1,
            'min_points': 10  # RISKY: Very low threshold for textureless images
        },
        # Config 5: RISKY - min_points=10, ratio=0.7, gentle penalty
        {
            'ratio_threshold': 0.7,
            'ransac_thresh': 3.0,
            'max_reproj_error': 3.0,
            'use_reproj_error_penalty': True,
            'reproj_error_penalty_weight': 0.1,
            'min_points': 10  # RISKY: Very low threshold for textureless images
        },
    ]

    # === DISCARDED SCORER CONFIGS (from ranking analysis) ===
    # - ratio=0.75, 0.8: Consistently worse than 0.65
    # - ransac=5.0, 8.0: Too loose, worse precision (removed)
    # - max_reproj_error=5.0, 8.0: Too loose, worse precision (removed)
    # - reproj_error_penalty_weight=0.5: Too harsh, degrades performance (removed)
    # - Strong reproj_error_penalty (weight=0.5): Previous experiments showed it hurts performance
    #   → Now testing gentle penalties (0.1, 0.2) to handle scale differences
    # - Lenient configs in general perform poorly

    for config in scorer_configs:
        ratio_threshold = config['ratio_threshold']

        matcher = DescriptorMatcher(
            matcher_type='BF',
            norm_type=norm_type,
            ratio_test_threshold=ratio_threshold,
            cross_check=False
        )

        scorer = HomographyScorer(
            matcher,
            ransac_thresh=config['ransac_thresh'],
            max_reproj_error=config['max_reproj_error'],
            use_reproj_error_penalty=config['use_reproj_error_penalty'],
            reproj_error_penalty_weight=config['reproj_error_penalty_weight'],
            min_points=config['min_points']
        )

        yield {
            'matcher': matcher,
            'scorer': scorer
        }

def generate_homography_scorer_configs() -> Iterator[Dict[str, Any]]:
    """
    Generate configurations for HomographyScorer parameters.
    """
    param_grid = {
        'ransac_thresh': [3.0, 5.0, 8.0],
        'max_reproj_error': [3.0, 5.0, 8.0],
        'use_reproj_error_penalty': [True, False],
        'min_points': [15, 20, 30]
    }
    keys, values = zip(*param_grid.items())
    for v in itertools.product(*values):
        yield dict(zip(keys, v))

if __name__ == '__main__':
    print("="*50)
    print("ESTIMATING GRID SEARCH SIZE (NEW SPLIT APPROACH)")
    print("="*50)

    # Count descriptor maker configs
    descriptor_makers = list(descriptor_maker_grid_search())
    num_descriptor_makers = len(descriptor_makers)

    print(f"\nDESCRIPTOR MAKER CONFIGURATIONS: {num_descriptor_makers}")

    # Count descriptor types
    desc_counts = {}
    for maker in descriptor_makers:
        desc_type = maker.descriptor_computer.to_dict()['type']
        desc_counts[desc_type] = desc_counts.get(desc_type, 0) + 1

    print("\nBreakdown by descriptor type:")
    for desc_type, count in sorted(desc_counts.items()):
        print(f"  - {desc_type:<25}: {count} configurations")

    # Count scorer configs for a sample descriptor maker
    if descriptor_makers:
        sample_maker = descriptor_makers[0]
        scorer_configs = list(scorer_grid_search(sample_maker))
        num_scorer_configs = len(scorer_configs)

        print(f"\nSCORER CONFIGURATIONS PER DESCRIPTOR MAKER: {num_scorer_configs}")
        print(f"  - Ratio test thresholds: 3 (0.7, 0.75, 0.8)")
        print(f"  - Homography scorer params: {len(list(generate_homography_scorer_configs()))}")

        print(f"\nTOTAL COMBINATIONS: {num_descriptor_makers} × {num_scorer_configs} = {num_descriptor_makers * num_scorer_configs}")
        print("\nNOTE: With the new split approach:")
        print(f"  - Compute descriptors {num_descriptor_makers} times (once per descriptor maker)")
        print(f"  - Evaluate {num_descriptor_makers * num_scorer_configs} times (all scorer combinations)")
        print(f"  - This avoids recomputing descriptors {num_scorer_configs - 1} times per descriptor maker!")