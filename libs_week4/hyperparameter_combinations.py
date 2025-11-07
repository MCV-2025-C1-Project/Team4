# libs_week4/hyperparameter_combinations.py

import sys
from pathlib import Path
import itertools
from typing import Iterator, Dict, Any
import cv2

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from libs_week4.descriptor import (
    DescriptorComputer, HomographyScorer, ORBDescriptor, DaisyDescriptor, SIFTDescriptor, RootSIFTDescriptor, BRISKDescriptor,
    KAZEDescriptor, AKAZEDescriptor, PCASIFTDescriptor, HOGDescriptor, GLOHDescriptor,
    DescriptorMatcher, KeypointAndDescriptorMaker, SURFDescriptor, DescriptorValueType
)
from libs_week3.color_conversion import ColorConversion, ColorSpace
import libs_week3.preprocessing as preprocessing


def generate_orb_configs() -> Iterator[Dict[str, Any]]:
    # ORIGINAL FULL GRID (135 configs):
    # param_grid = {
    #     'n_features': [500, 1000, 2000, 3000, 5000],
    #     'scale_factor': [1.1, 1.2, 1.5],
    #     'n_levels': [8, 10, 12],
    #     'wta_k': [2, 3, 4],
    #     'score_type': [cv2.ORB_HARRIS_SCORE],
    #     'patch_size': [31]
    # }

    # REDUCED GRID (12 configs): Focus on best performer (ORB mAP@k1 > 0.7)
    # Key insights: ORB is crushing it, so focus on variations that matter
    # - n_features: 2000, 3000 (sweet spot for quality)
    # - scale_factor: 1.2 (standard), 1.5 (faster pyramids)
    # - n_levels: 10 (good compromise)
    # - wta_k: 2 (default, best tested), 3 (better discrimination)
    param_grid = {
        'n_features': [2000, 3000],
        'scale_factor': [1.2, 1.5],
        'n_levels': [10],
        'wta_k': [2, 3],
        'score_type': [cv2.ORB_HARRIS_SCORE],
        'patch_size': [31]
    }
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
    # ORIGINAL FULL GRID (24 configs):
    # param_grid = {
    #     'n_features': [500, 1000, 2000, 0],
    #     'n_octave_layers': [3, 4],
    #     'contrast_threshold': [0.03, 0.04, 0.06],
    #     'sigma': [1.6, 2.0],
    #     'edge_threshold': [15]
    # }

    # REDUCED GRID (12 configs): SIFT mAP@k1 = 0.5-0.6 (decent baseline)
    # Key insights: Classic descriptor, focus on quality vs quantity trade-off
    # - n_features: 1000, 2000, 0 (unlimited - let it find what it needs)
    # - contrast_threshold: 0.03 (more keypoints), 0.04 (standard)
    # - sigma: 1.6 (standard), 2.0 (smoother)
    param_grid = {
        'n_features': [1000, 2000, 0],
        'n_octave_layers': [4],
        'contrast_threshold': [0.03, 0.04],
        'sigma': [1.6, 2.0],
        'edge_threshold': [15]
    }
    keys, values = zip(*param_grid.items())
    for v in itertools.product(*values):
        yield dict(zip(keys, v))

def generate_rootsift_configs() -> Iterator[Dict[str, Any]]:
    # ORIGINAL FULL GRID (24 configs):
    # param_grid = {
    #     'n_features': [500, 1000, 2000, 0],
    #     'n_octave_layers': [4],
    #     'contrast_threshold': [0.03, 0.04, 0.06],
    #     'sigma': [1.6, 2.0],
    #     'edge_threshold': [15]
    # }

    # REDUCED GRID (12 configs): RootSIFT should outperform SIFT
    # Key insights: Improved normalization should give better results than SIFT
    # Use same grid as SIFT for fair comparison
    param_grid = {
        'n_features': [1000, 2000, 0],
        'n_octave_layers': [4],
        'contrast_threshold': [0.03, 0.04],
        'sigma': [1.6, 2.0],
        'edge_threshold': [15]
    }
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

    # REDUCED GRID (6 configs): Keep full grid - KAZE is float version of AKAZE, might be better
    # Already at target size (6 configs)
    param_grid = {
        'extended': [False],
        'upright': [False],
        'threshold': [0.0001, 0.001, 0.003],
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
    param_grid = {
        'num_components': [24, 36, 48],
        'n_features': [500, 1000, 1500, 2000] # SIFT param
    }
    keys, values = zip(*param_grid.items())
    for v in itertools.product(*values):
        yield dict(zip(keys, v))


def generate_keypoint_descriptors() -> Iterator[DescriptorComputer]:
    """
    Generates instances of different keypoint descriptors by iterating
    through all their specified hyperparameter configurations.
    """
    for config in generate_sift_configs(): # good
        yield SIFTDescriptor(**config) # good

    for config in generate_orb_configs(): # good
        yield ORBDescriptor(**config) # good

    for config in generate_rootsift_configs(): # improved version of SIFT
        yield RootSIFTDescriptor(**config) # improved version of SIFT

    # for config in generate_surf_configs(): # not available by default in opencv
        # yield SURFDescriptor(**config) # not available by default in opencv

    for config in generate_brisk_configs(): # bad too many keypoints
        yield BRISKDescriptor(**config) # bad too many keypoints

    for config in generate_kaze_configs(): # float version of AKAZE, might be better
        yield KAZEDescriptor(**config) # float version of AKAZE, might be better

    # one akaza config fails (no keypoints detected maybe)
    for config in generate_akaze_configs(): # bad performs pretty bad
        yield AKAZEDescriptor(**config) # bad performs pretty bad
        
    for config in generate_daisy_configs(): # bad too many keypoints
        yield DaisyDescriptor(**config) # bad too many keypoints
    
    # for config in generate_hog_configs(): # bad performs like shit
        # yield HOGDescriptor(**config) # bad performs like shit

    # for config in generate_gloh_configs(): # bad pefrorms like shit
        # yield GLOHDescriptor(**config) # bad pefrorms like shit
    
    # FIXME: the PCA should be run only once for the whole database end then we have to transform all images when computing the descriptor
    # for config in generate_pcasift_configs(): # bad performs pretty bad
        # yield PCASIFTDescriptor(**config) # bad performs pretty bad


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

def scorer_grid_search(descriptor_maker: KeypointAndDescriptorMaker) -> Iterator[Dict[str, Any]]:
    """
    Generator for scorer configurations for a given descriptor maker.

    Args:
        descriptor_maker: The KeypointAndDescriptorMaker to generate scorers for

    Yields:
        Dictionary containing 'matcher' and 'scorer' keys
    """
    # ORIGINAL FULL GRID (162 configs):
    # - ratio_threshold: [0.7, 0.75, 0.8]
    # - cross_check: [False]
    # - ransac_thresh: [3.0, 5.0, 8.0]
    # - max_reproj_error: [3.0, 5.0, 8.0]
    # - use_reproj_error_penalty: [True, False]
    # - min_points: [15, 20, 30]
    # Total: 3 × 1 × 54 = 162

    # REDUCED GRID (4 configs): Minimal set for initial descriptor testing
    # After finding best descriptors, we can expand scorer grid
    # Key combinations:
    #   1. Standard tight config (ratio=0.75, ransac=5.0, reproj=5.0, penalty=True, min_pts=20)
    #   2. Lenient config (ratio=0.8, ransac=8.0, reproj=8.0, penalty=True, min_pts=15)
    #   3. Tight without penalty (ratio=0.7, ransac=3.0, reproj=3.0, penalty=False, min_pts=20)
    #   4. Balanced config (ratio=0.75, ransac=5.0, reproj=5.0, penalty=False, min_pts=20)

    # Determine the appropriate norm type based on descriptor value type
    descriptor_value_type = descriptor_maker.descriptor_computer.get_value_type()

    if descriptor_value_type == DescriptorValueType.FLOAT:
        norm_type = cv2.NORM_L2
    else:  # BINARY
        norm_type = cv2.NORM_HAMMING

    # Define the 4 configurations explicitly
    scorer_configs = [
        # Config 1: Standard tight - good baseline
        {
            'ratio_threshold': 0.75,
            'ransac_thresh': 5.0,
            'max_reproj_error': 5.0,
            'use_reproj_error_penalty': True,
            'min_points': 20
        },
        # Config 2: Lenient - more matches, less strict
        {
            'ratio_threshold': 0.8,
            'ransac_thresh': 8.0,
            'max_reproj_error': 8.0,
            'use_reproj_error_penalty': True,
            'min_points': 15
        },
        # Config 3: Tight without penalty - pure inlier ratio
        {
            'ratio_threshold': 0.7,
            'ransac_thresh': 3.0,
            'max_reproj_error': 3.0,
            'use_reproj_error_penalty': False,
            'min_points': 20
        },
        # Config 4: Balanced without penalty - middle ground
        {
            'ratio_threshold': 0.75,
            'ransac_thresh': 5.0,
            'max_reproj_error': 5.0,
            'use_reproj_error_penalty': False,
            'min_points': 20
        }
    ]

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