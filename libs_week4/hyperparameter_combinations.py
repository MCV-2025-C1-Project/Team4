# libs_week4/hyperparameter_combinations.py

import sys
from pathlib import Path
import itertools
from typing import Iterator, Dict, Any
import cv2

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from libs_week4.descriptor import (
    DescriptorComputer, HomographyScorer, ORBDescriptor, DaisyDescriptor, SIFTDescriptor, BRISKDescriptor,
    AKAZEDescriptor, PCASIFTDescriptor, HOGDescriptor, GLOHDescriptor,
    DescriptorMatcher, KeypointAndDescriptorMaker
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
        'n_features': [500,],
        'scale_factor': [1.2,],
        'n_levels': [8,]
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
        'n_features': [ 500, ],
        'n_octave_layers': [ 4],
        'contrast_threshold': [ 0.06],
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
        'thresh': [30, ],
        'octaves': [3, ],
        'pattern_scale': [1.0,]
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
        'step': [4, 8],
        'radius': [15, 25],
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
        'n_features': [500] # SIFT param
    }
    keys, values = zip(*param_grid.items())
    for v in itertools.product(*values):
        yield dict(zip(keys, v))

def generate_gloh_configs() -> Iterator[Dict[str, Any]]:
    param_grid = {
        'nbins': [36, 48],
        'n_features': [500] # SIFT param
    }
    keys, values = zip(*param_grid.items())
    for v in itertools.product(*values):
        yield dict(zip(keys, v))
        
def generate_pcasift_configs() -> Iterator[Dict[str, Any]]:
    param_grid = {
        'num_components': [24, 36, 48],
        'n_features': [500, 1500] # SIFT param
    }
    keys, values = zip(*param_grid.items())
    for v in itertools.product(*values):
        yield dict(zip(keys, v))


def generate_keypoint_descriptors() -> Iterator[DescriptorComputer]:
    """
    Generates instances of different keypoint descriptors by iterating
    through all their specified hyperparameter configurations.
    """
    for config in generate_sift_configs():
        yield SIFTDescriptor(**config)
        
    for config in generate_orb_configs():
        yield ORBDescriptor(**config)

    for config in generate_brisk_configs():
        yield BRISKDescriptor(**config)

    for config in generate_akaze_configs():
        yield AKAZEDescriptor(**config)
        
    for config in generate_daisy_configs():
        yield DaisyDescriptor(**config)
        
    for config in generate_hog_configs():
        yield HOGDescriptor(**config)

    for config in generate_gloh_configs():
        yield GLOHDescriptor(**config)
        
    for config in generate_pcasift_configs():
        yield PCASIFTDescriptor(**config)


def generate_color_space_combinations() -> list[list[ColorSpace]]:
    return [[ColorSpace.RGB]]

def generate_preprocess_strategies() -> list[preprocessing.ImagePreprocessStep | None]:
    return [
        preprocessing.Preprocess([
            preprocessing.CropToMask(),
        ])
    ]

def generate_descriptor_matchers() -> Iterator[DescriptorMatcher]:
    yield DescriptorMatcher(matcher_type='BF', norm_type=cv2.NORM_L2, ratio_test_threshold=0.75)
    yield DescriptorMatcher(matcher_type='FLANN', norm_type=cv2.NORM_L2, ratio_test_threshold=0.75)

    yield DescriptorMatcher(matcher_type='BF', norm_type=cv2.NORM_HAMMING, ratio_test_threshold=0.75)
    yield DescriptorMatcher(matcher_type='FLANN', norm_type=cv2.NORM_HAMMING, ratio_test_threshold=0.75)


def keypoint_hyperparameter_grid_search() -> Iterator[dict]:
    """
    Main generator for the grid search. It yields dictionaries of
    hyperparameter combinations for keypoint-based descriptors.
    """
    float_descriptors = ("SIFT", "DAISY", "HOG", "ArticleGLOH", "PCA-SIFT")

    for color_spaces in generate_color_space_combinations():
        for preprocess in generate_preprocess_strategies():
            # The 'generate_keypoint_descriptors' function now does all the hard work
            for descriptor in generate_keypoint_descriptors():
                for matcher in generate_descriptor_matchers():
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
                            'scorer': HomographyScorer(matcher)
                        }
                    elif not is_float_desc and matcher.norm_type == cv2.NORM_HAMMING:
                        yield {
                            'color_conversion': ColorConversion(targets=color_spaces, normalize=True),
                            'preprocess': preprocess,
                            'keypoint_descriptor': descriptor,
                            'keypoint_and_descriptor_maker': KeypointAndDescriptorMaker(descriptor_computer=descriptor, color_conversion=ColorConversion(targets=color_spaces, normalize=True), preprocess=preprocess),
                            'matcher': matcher,
                            'scorer': HomographyScorer(matcher)
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