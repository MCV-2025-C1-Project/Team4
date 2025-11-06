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
    DescriptorMatcher, KeypointAndDescriptorMaker, SURFDescriptor
)
from libs_week3.color_conversion import ColorConversion, ColorSpace
import libs_week3.preprocessing as preprocessing


def generate_orb_configs() -> Iterator[Dict[str, Any]]:
    # param_grid = {
    #     'n_features': [500, 1500, 3000],
    #     'scale_factor': [1.2, 1.5],
    #     'n_levels': [8, 12]
    # }
    param_grid = {
        'n_features': [500, 1000, 2000, 3000, 5000],
        'scale_factor': [1.1, 1.2, 1.5],
        'n_levels': [8, 10, 12],
        'wta_k': [2, 3, 4],  # WTA_K: 2=default (256 bit), 3=384 bit, 4=512 bit descriptors
        'score_type': [cv2.ORB_HARRIS_SCORE],  # Could also try cv2.ORB_FAST_SCORE
        'patch_size': [31]  # Default is 31, could try [15, 31, 45] for variation
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
    # param_grid = {
    #     'n_features': [0, 500, 1500],
    #     'n_octave_layers': [3, 4],
    #     'contrast_threshold': [0.04, 0.06],
    #     'edge_threshold': [10, 15]
    # }
    param_grid = {
        'n_features': [500, 1000, 2000, 0],
        'n_octave_layers': [ 4],
        'contrast_threshold': [0.03, 0.04, 0.06],
        'sigma': [1.6, 2.0],
        'edge_threshold': [15]
    }
    keys, values = zip(*param_grid.items())
    for v in itertools.product(*values):
        yield dict(zip(keys, v))

def generate_rootsift_configs() -> Iterator[Dict[str, Any]]:
    # RootSIFT uses the same parameters as SIFT
    param_grid = {
        'n_features': [500, 1000, 2000, 0],
        'n_octave_layers': [ 4],
        'contrast_threshold': [0.03, 0.04, 0.06],
        'sigma': [1.6, 2.0],
        'edge_threshold': [15]
    }
    keys, values = zip(*param_grid.items())
    for v in itertools.product(*values):
        yield dict(zip(keys, v))

def generate_brisk_configs() -> Iterator[Dict[str, Any]]:
    # param_grid = {
    #     'thresh': [30, 50, 70],
    #     'octaves': [3, 4],
    #     'pattern_scale': [1.0, 1.2]
    # }
    param_grid = {
        'thresh': [50, 70, 100],
        'octaves': [3, ],
        'pattern_scale': [1.0,]
    }
    keys, values = zip(*param_grid.items())
    for v in itertools.product(*values):
        yield dict(zip(keys, v))

def generate_kaze_configs() -> Iterator[Dict[str, Any]]:
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
    param_grid = {
        'threshold': [0.003, 0.005, 0.007],
        'n_octaves': [4, 5],
        'n_octave_layers': [4, 5]
    }
    keys, values = zip(*param_grid.items())
    for v in itertools.product(*values):
        yield dict(zip(keys, v))

def generate_daisy_configs() -> Iterator[Dict[str, Any]]:
    param_grid = {
        'step': [16, 32, 64],
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
            preprocessing.CropToMask(),
        ])
    ]

def generate_descriptor_matchers() -> Iterator[DescriptorMatcher]:
    # Vary ratio test threshold to test matching strictness
    for ratio_threshold in [0.7, 0.75, 0.8]:
        for cross_check in [False]:
            yield DescriptorMatcher(matcher_type='BF', norm_type=cv2.NORM_L2, ratio_test_threshold=ratio_threshold, cross_check=cross_check)
            # yield DescriptorMatcher(matcher_type='FLANN', norm_type=cv2.NORM_L2, ratio_test_threshold=ratio_threshold, cross_check=cross_check)

            yield DescriptorMatcher(matcher_type='BF', norm_type=cv2.NORM_HAMMING, ratio_test_threshold=ratio_threshold, cross_check=cross_check)
            # yield DescriptorMatcher(matcher_type='FLANN', norm_type=cv2.NORM_HAMMING, ratio_test_threshold=ratio_threshold, cross_check=cross_check)

def generate_homography_scorer_configs() -> Iterator[Dict[str, Any]]:
    """
    Generate configurations for HomographyScorer parameters.
    NOTE: These will be combined with matchers in the grid search.
    For now keeping a simple grid, but can be expanded later.
    """
    param_grid = {
        'ransac_thresh': [3.0, 5.0, 8.0],
        'max_reproj_error': [3.0, 5.0, 8.0],
        'use_reproj_error_penalty': [True],
        'min_points': [15, 20, 30]
    }
    keys, values = zip(*param_grid.items())
    for v in itertools.product(*values):
        yield dict(zip(keys, v))

def keypoint_hyperparameter_grid_search() -> Iterator[dict]:
    """
    Main generator for the grid search. It yields dictionaries of
    hyperparameter combinations for keypoint-based descriptors.
    """
    float_descriptors = ("SIFT", "RootSIFT", "KAZE", "DAISY", "HOG", "ArticleGLOH", "PCA-SIFT", "SURF")

    for color_spaces in generate_color_space_combinations():
        for preprocess in generate_preprocess_strategies():
            # The 'generate_keypoint_descriptors' function now does all the hard work
            for descriptor in generate_keypoint_descriptors():
                for matcher in generate_descriptor_matchers():
                    for scorer_config in generate_homography_scorer_configs():
                        descriptor_type_name = descriptor.to_dict()['type']
                        is_float_desc = any(s in descriptor_type_name for s in float_descriptors)

                        # --- Compatibility Check ---
                        if is_float_desc and matcher.norm_type == cv2.NORM_L2:
                            yield {
                                'color_conversion': ColorConversion(targets=color_spaces, normalize=True),
                                'preprocess': preprocess,
                                'keypoint_descriptor': descriptor,
                                'keypoint_and_descriptor_maker': KeypointAndDescriptorMaker(descriptor_computer=descriptor, color_conversion=ColorConversion(targets=color_spaces, normalize=True), preprocess=preprocess),
                                'matcher': matcher,
                                'scorer': HomographyScorer(matcher, **scorer_config)
                            }
                        elif not is_float_desc and matcher.norm_type == cv2.NORM_HAMMING:
                            yield {
                                'color_conversion': ColorConversion(targets=color_spaces, normalize=True),
                                'preprocess': preprocess,
                                'keypoint_descriptor': descriptor,
                                'keypoint_and_descriptor_maker': KeypointAndDescriptorMaker(descriptor_computer=descriptor, color_conversion=ColorConversion(targets=color_spaces, normalize=True), preprocess=preprocess),
                                'matcher': matcher,
                                'scorer': HomographyScorer(matcher, **scorer_config)
                            }

if __name__ == '__main__':
    print("="*50)
    print("ESTIMATING GRID SEARCH SIZE")
    print("="*50)
    
    
    total_combinations = len(list(keypoint_hyperparameter_grid_search()))
    
    print(f"\nTOTAL CONFIGURATIONS TO TEST: {total_combinations}\n")
    
    print("Breakdown per descriptor type:")
    desc_counts = {}
    for params in keypoint_hyperparameter_grid_search():
        desc_type = params['keypoint_descriptor'].to_dict()['type']
        desc_counts[desc_type] = desc_counts.get(desc_type, 0) + 1
        
    for desc_type, count in desc_counts.items():
        print(f"  - {desc_type:<25}: {count} combinations")