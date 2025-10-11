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

    # Use all channels together (individual channels don't work well)
    all_channels = list(range(total_channels))

    # Return only the all-channels strategy
    return [
        all_channels,  # Use all channels together
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
            
    if False and bins <= 16:
        channel_pairs = generate_channel_pairs(color_spaces)
        if channel_pairs:
            for pairs in channel_pairs:
                computers.append(descriptor.Histogram2D(pairs, bins, weight_strategy=None, block_splitter=block_splitter))
        
    if False and bins <= 8:
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

    # Track distinct parameter values
    distinct_gamma = set()
    distinct_color_spaces = set()
    distinct_bins = set()
    distinct_block_splitters = set()
    distinct_histogram_types = set()
    distinct_histogram_channels = set()

    for config in hyperparameter_grid_search():
        total += 1

        # Extract parameters
        distinct_gamma.add(config['gamma_correction'])

        # Color spaces (convert list to tuple for hashing)
        color_space_tuple = tuple(sorted([cs.value for cs in config['color_spaces']]))
        distinct_color_spaces.add(color_space_tuple)

        # Histogram computer details
        histo_computer = config['histogram_computer']
        histogram_type = type(histo_computer).__name__
        distinct_histogram_types.add(histogram_type)

        # Bins (inside histogram computer)
        if hasattr(histo_computer, 'bins'):
            distinct_bins.add(histo_computer.bins)

        # Histogram channels configuration
        if hasattr(histo_computer, 'channels'):
            distinct_histogram_channels.add(tuple(histo_computer.channels))
        elif hasattr(histo_computer, 'channel_pairs'):
            distinct_histogram_channels.add(tuple(tuple(p) for p in histo_computer.channel_pairs))
        elif hasattr(histo_computer, 'channel_triplets'):
            distinct_histogram_channels.add(tuple(tuple(t) for t in histo_computer.channel_triplets))

        # Block splitter details
        block_splitter = config['block_split_strategy']
        splitter_type = type(block_splitter).__name__

        if hasattr(block_splitter, 'shape'):
            splitter_desc = f"{splitter_type}_{block_splitter.shape}"
        elif hasattr(block_splitter, 'shapes'):
            splitter_desc = f"{splitter_type}_{tuple(block_splitter.shapes)}"
        else:
            splitter_desc = splitter_type

        distinct_block_splitters.add(splitter_desc)

    # Print detailed analysis
    print("\n" + "="*80)
    print("PARAMETER DIVERSITY ANALYSIS")
    print("="*80)
    print(f"Total configurations: {total}")
    print("\nDistinct parameter values:")
    print(f"  Gamma corrections:       {len(distinct_gamma):4d} distinct values")
    print(f"    Values: {sorted(distinct_gamma)}")
    print(f"  Color space combos:      {len(distinct_color_spaces):4d} distinct combinations")
    print(f"  Bins:                    {len(distinct_bins):4d} distinct values")
    print(f"    Values: {sorted(distinct_bins)}")
    print(f"  Block splitters:         {len(distinct_block_splitters):4d} distinct strategies")
    for splitter in sorted(distinct_block_splitters):
        print(f"    - {splitter}")
    print(f"  Histogram types:         {len(distinct_histogram_types):4d} distinct types")
    for htype in sorted(distinct_histogram_types):
        print(f"    - {htype}")
    print(f"  Histogram channel configs: {len(distinct_histogram_channels):4d} distinct configurations")

    print("\nTop 10 most common color space combinations:")
    color_space_counts = {}
    for config in hyperparameter_grid_search():
        cs_tuple = tuple(sorted([cs.value for cs in config['color_spaces']]))
        color_space_counts[cs_tuple] = color_space_counts.get(cs_tuple, 0) + 1

    for idx, (cs, count) in enumerate(sorted(color_space_counts.items(), key=lambda x: x[1], reverse=True)[:10], 1):
        print(f"  {idx:2d}. {'+'.join(cs):30s} : {count:4d} configurations")

    print("="*80)

    return total

if __name__ == '__main__':
    print("Estimated grid size:", estimate_grid_size())
    print("\nActual grid size:", actual_grid_size())

