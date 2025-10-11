import itertools
from typing import Iterator
import cv2

from libs_week2.descriptor import ColorSpace, WeightStrategy
import libs_week2.descriptor as descriptor


def generate_gamma_corrections():
    return [1.0] # third search (W2)
    return [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1] # second search
    return [0.8, 1.0, 1.5] # first search (1.5 performs badly)
    return [0.5, 0.8, 1.0, 1.2, 1.5, 2.0] # hmmmmm


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
    all_spaces = list(ColorSpace) # first search
    all_spaces = [ColorSpace.RGB, ColorSpace.LAB, ColorSpace.YCRCB] # second search
    combinations = []
    
    for space in all_spaces:
        combinations.append([space])
    
    rgb_pairs = [
        # ColorSpace.GRAY,    # Luminance info
        # ColorSpace.HSV,     # Hue/saturation info # first search
        ColorSpace.LAB,     # Perceptual color
        ColorSpace.YCRCB,   # Luma/chroma separation
        # ColorSpace.HLS,     # Alternative hue representation # first search
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
    return [4, 8, 16, 32, 64] # third search (W2)
    return [8, 12, 16, 24, 32, 48, 64] # second search
    return [4, 8, 16, 32, 64] # first search
    return [4, 8, 16, 32, 64, 128]


def generate_weights():
    return [None] # third search
    return [WeightStrategy.CENTER_CROP_05, WeightStrategy.CENTER_CROP_10, WeightStrategy.CENTER_CROP_15, WeightStrategy.PYRAMID]
    return [None] + list(WeightStrategy) # first search


def generate_block_splitting_strategies() -> list[descriptor.ImageBlockSplitter]:
    strategies = []
    strategies.append(descriptor.IdentityImageBlockSplitter()) # the base strategy
    
    strategies.append(descriptor.GridImageBlockSplitter((2, 2)))
    strategies.append(descriptor.GridImageBlockSplitter((3, 3)))
    strategies.append(descriptor.GridImageBlockSplitter((4, 4)))
    
    strategies.append(descriptor.PyramidImageBlockSplitter([(1, 1), (2, 2)]))
    strategies.append(descriptor.PyramidImageBlockSplitter([(1, 1), (2, 2), (3, 3)]))
    # strategies.append(descriptor.PyramidImageBlockSplitter([(1, 1), (2, 2), (3, 3), (4, 4)]))
    
    return strategies

def generate_channels(color_spaces: list[ColorSpace]) -> list[list[int]]:
    """
    Generate channel combinations for 1D histograms.
    Returns different configurations like all channels separately, or combined.
    """
    # Count total channels across all color spaces
    total_channels = sum(
        4 if space == ColorSpace.CMYK else 3
        for space in color_spaces
    )

    if total_channels == 0:
        return []

    # Option 1: Use all channels separately (one list with all channel indices)
    all_channels = list(range(total_channels))

    # Option 2: Use each channel individually
    individual_channels = [[i] for i in range(total_channels)]

    # Return different strategies
    return [
        all_channels,  # Use all channels together
        *individual_channels,  # Use each channel separately
    ]

def generate_channel_pairs(color_spaces: list[ColorSpace]) -> list[list[tuple[int, int]]]:
    """
    Generate channel pair combinations for 2D histograms.
    Returns different configurations of channel pairs.
    """
    total_channels = sum(
        4 if space == ColorSpace.CMYK else 3
        for space in color_spaces
    )

    if total_channels < 2:
        return []

    # Generate all possible pairs
    all_pairs = list(itertools.combinations(range(total_channels), 2))

    # Option 1: Use all pairs together
    # Option 2: Use pairs within each color space
    pairs_per_space = []
    offset = 0
    for space in color_spaces:
        n_channels = 4 if space == ColorSpace.CMYK else 3
        space_pairs = list(itertools.combinations(range(offset, offset + n_channels), 2))
        if space_pairs:
            pairs_per_space.append(space_pairs)
        offset += n_channels

    results = []

    # All pairs together
    if all_pairs:
        results.append(all_pairs)

    # Pairs within each color space
    for pairs in pairs_per_space:
        results.append(pairs)

    return results

def generate_channel_triplets(color_spaces: list[ColorSpace]) -> list[list[tuple[int, int, int]]]:
    """
    Generate channel triplet combinations for 3D histograms.
    Returns different configurations of channel triplets.
    """
    total_channels = sum(
        4 if space == ColorSpace.CMYK else 3
        for space in color_spaces
    )

    if total_channels < 3:
        return []

    # Generate all possible triplets
    all_triplets = list(itertools.combinations(range(total_channels), 3))

    # Option 1: Use all triplets together
    # Option 2: Use triplets within each color space
    triplets_per_space = []
    offset = 0
    for space in color_spaces:
        n_channels = 4 if space == ColorSpace.CMYK else 3
        space_triplets = list(itertools.combinations(range(offset, offset + n_channels), 3))
        if space_triplets:
            triplets_per_space.append(space_triplets)
        offset += n_channels

    results = []

    # All triplets together
    if all_triplets:
        results.append(all_triplets)

    # Triplets within each color space
    for triplets in triplets_per_space:
        results.append(triplets)

    return results

def generate_histogram_computers(bins: int, color_spaces: list[ColorSpace], block_splitter: descriptor.ImageBlockSplitter) -> list[descriptor.HistogramComputer]:
    computers = []
    channels = generate_channels(color_spaces)
    if channels:
        for channel in channels:
            computers.append(descriptor.Histogram1D(channel, bins, weight_strategy=None, block_splitter=block_splitter))
            
    if bins <= 32:
        channel_pairs = generate_channel_pairs(color_spaces)
        if channel_pairs:
            for pairs in channel_pairs:
                computers.append(descriptor.Histogram2D(pairs, bins, weight_strategy=None, block_splitter=block_splitter))
        
    if bins <= 16:
        channel_triplets = generate_channel_triplets(color_spaces)
        if channel_triplets:
            for triplets in channel_triplets:
                computers.append(descriptor.Histogram3D(triplets, bins, weight_strategy=None, block_splitter=block_splitter))

    return computers

def hyperparameter_grid_search() -> Iterator[dict]:
    gamma_values = generate_gamma_corrections()
    blur_functions = generate_blur_functions()
    color_space_combos = generate_color_space_combinations()
    bin_values = generate_bins()
    weight_options = generate_weights()
    
    total_combinations = 0
    
    for gamma in gamma_values:
        for color_spaces in color_space_combos:
            for bins in bin_values:
                for block_split_strategy in generate_block_splitting_strategies():
                    for histogram_computer in generate_histogram_computers(bins, color_spaces, block_split_strategy):
                        total_combinations += 1
                        yield {
                            'gamma_correction': gamma,
                            'color_spaces': color_spaces,
                            'histogram_computer': histogram_computer,
                            'block_split_strategy': block_split_strategy
                        }


def estimate_grid_size():
    gamma_count = len(generate_gamma_corrections())
    blur_count = len(generate_blur_functions())
    bin_count = len(generate_bins())
    weight_count = len(generate_weights())
    block_split_count = len(generate_block_splitting_strategies())
    
    color_space_combos = generate_color_space_combinations()
    
    print("gamma_count", gamma_count)
    print("len(color_space_combos)", len(color_space_combos))
    print("bin_count", bin_count)
    print("weight_count", weight_count)
    print("block_split_count", block_split_count)

    estimated_total = (gamma_count * len(color_space_combos) * bin_count * weight_count * block_split_count)
    
    return int(estimated_total)

def actual_grid_size():
    total = 0
    for _ in hyperparameter_grid_search():
        total += 1
    return total

if __name__ == '__main__':
    print(estimate_grid_size())
    print(actual_grid_size())

