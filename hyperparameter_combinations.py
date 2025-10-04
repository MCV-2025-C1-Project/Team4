import itertools
from typing import Iterator
import cv2

from descriptor import ColorSpace, WeightStrategy


def generate_gamma_corrections():
    return [0.8, 1.0, 1.5]
    return [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]


def generate_blur_functions():
    blur_options = [False]
    
    kernel_sizes = [3]
    
    sigma_values = [1]
    
    for ksize in kernel_sizes:
        for sigma in sigma_values:
            blur_func = lambda img, k=ksize, s=sigma: cv2.GaussianBlur(img, (k, k), s)
            blur_options.append(blur_func)
    
    return blur_options

def generate_color_space_combinations():
    all_spaces = list(ColorSpace)
    
    similar_groups = [
        {ColorSpace.RGB, ColorSpace.CMYK, ColorSpace.XYZ, ColorSpace.LAB, ColorSpace.LUV}, # All primary color based
        {ColorSpace.HSV, ColorSpace.HLS},  # Similar because use hue
        {ColorSpace.LAB, ColorSpace.LUV},  # Both perceptual
        {ColorSpace.YCRCB, ColorSpace.YUV},  # Both luma-chroma
    ]
    
    combinations = []
    
    for space in all_spaces:
        combinations.append([space])
    
    for space1, space2 in itertools.combinations(all_spaces, 2):
        is_similar = any(space1 in group and space2 in group 
                        for group in similar_groups)
        if not is_similar:
            combinations.append([space1, space2])
    
    """
    for space1, space2, space3 in itertools.combinations(all_spaces, 3):
        pairs = [(space1, space2), (space1, space3), (space2, space3)]
        is_similar = any(
            any(s1 in group and s2 in group for group in similar_groups)
            for s1, s2 in pairs
        )
        if not is_similar:
            combinations.append([space1, space2, space3])
    """
    
    return combinations

def generate_color_space_combinations():
    all_spaces = list(ColorSpace)
    combinations = []
    
    for space in all_spaces:
        combinations.append([space])
    
    rgb_pairs = [
        # ColorSpace.GRAY,    # Luminance info
        ColorSpace.HSV,     # Hue/saturation info
        ColorSpace.LAB,     # Perceptual color
        ColorSpace.YCRCB,   # Luma/chroma separation
        ColorSpace.HLS,     # Alternative hue representation
    ]
    
    for other_space in rgb_pairs:
        combinations.append([ColorSpace.RGB, other_space])
    
    return combinations



def generate_keep_discard_patterns(color_spaces: list[ColorSpace]):
    """
    Generate smart keep/discard patterns for given color spaces.
    
    Rules:
    - RGB, CMYK, XYZ, LAB, LUV: Always keep all channels (primary/perceptual)
    - GRAY: Always keep (only 1 channel)
    - HSV, HLS: Keep all if alone, otherwise keep only H or V/L when paired
    - YCrCb, YUV: Keep all if alone, otherwise keep Y or CrCb/UV when paired
    """
    is_alone = len(color_spaces) == 1
    
    # Define color space categories
    primary_spaces = {ColorSpace.RGB, ColorSpace.CMYK, ColorSpace.XYZ, 
                     ColorSpace.LAB, ColorSpace.LUV}
    hue_based = {ColorSpace.HSV, ColorSpace.HLS}
    luma_chroma = {ColorSpace.YCRCB, ColorSpace.YUV}
    
    pattern_parts = []
    
    for space in color_spaces:
        """
        if space == ColorSpace.GRAY:
            # Grayscale: always keep
            pattern_parts.append(['K'])
        """
        
        if space in primary_spaces:
            # Primary/perceptual color spaces: always keep all
            if space == ColorSpace.CMYK:
                pattern_parts.append(['KKKK'])
            else:
                pattern_parts.append(['KKK'])
        
        elif space in hue_based:
            if is_alone:
                # If alone, keep all channels
                pattern_parts.append(['KKK'])
            else:
                # If paired, use either H or V/L (last channel)
                if space == ColorSpace.HSV:
                    # HSV: H, S, V -> keep H or V
                    pattern_parts.append(['KDD', 'DDK'])  # H only or V only
                else:  # HLS
                    # HLS: H, L, S -> keep H or L
                    pattern_parts.append(['KDD', 'DKD'])  # H only or L only
        
        elif space in luma_chroma:
            if is_alone:
                # If alone, keep all channels
                pattern_parts.append(['KKK'])
            else:
                # If paired, use either Y or CrCb/UV
                # Y, Cr, Cb or Y, U, V -> keep Y or CrCb/UV
                pattern_parts.append(['KDD', 'DKK'])  # Y only or CrCb/UV only
        
        else:
            # Fallback: keep all channels (3 for any other space)
            pattern_parts.append(['KKK'])
    
    # Generate all combinations from the pattern parts
    patterns = []
    for combo in itertools.product(*pattern_parts):
        patterns.append(''.join(combo))
    
    return patterns


def generate_bins():
    return [8, 16, 32, 64]
    return [4, 8, 16, 32, 64, 128]


def generate_weights():
    return [None] + list(WeightStrategy)


def hyperparameter_grid_search() -> Iterator[dict]:
    gamma_values = generate_gamma_corrections()
    blur_functions = generate_blur_functions()
    color_space_combos = generate_color_space_combinations()
    bin_values = generate_bins()
    weight_options = generate_weights()
    
    total_combinations = 0
    
    for gamma in gamma_values:
        for color_spaces in color_space_combos:
            keep_discard_options = generate_keep_discard_patterns(color_spaces)
            
            for keep_discard in keep_discard_options:
                for bins in bin_values:
                    total_combinations += 1
                    for weight in weight_options:
                        yield {
                            'gamma_correction': gamma,
                            'color_spaces': color_spaces,
                            'bins': bins,
                            'keep_or_discard': keep_discard,
                            'weight': weight
                        }


def estimate_grid_size():
    gamma_count = len(generate_gamma_corrections())
    blur_count = len(generate_blur_functions())
    bin_count = len(generate_bins())
    weight_count = len(generate_weights())
    
    color_space_combos = generate_color_space_combinations()
    
    total_keep_discard = 0
    for color_spaces in color_space_combos:
        keep_discard_patterns = generate_keep_discard_patterns(color_spaces)
        total_keep_discard += len(keep_discard_patterns)
    
    avg_keep_discard = total_keep_discard / len(color_space_combos)
    
    print("gamma_count", gamma_count)
    print("len(color_space_combos)", len(color_space_combos))
    print("avg_keep_discard", avg_keep_discard)
    print("bin_count", bin_count)
    print("weight_count", weight_count)

    estimated_total = (gamma_count * len(color_space_combos) * 
                      avg_keep_discard * bin_count)
    
    return int(estimated_total)

if __name__ == '__main__':
    print(estimate_grid_size())
