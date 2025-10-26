import itertools
from typing import Iterator
import cv2
import pywt

from libs_week3.color_conversion import ColorConversion
from libs_week3.descriptor import ColorSpace, WeightStrategy
import libs_week3.descriptor as descriptor
import libs_week3.preprocessing as preprocessing
import libs_week3.denoising as denoising
from libs_week3.descriptor import IdentityImageBlockSplitter


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
    all_spaces = [ColorSpace.RGB, ColorSpace.LAB, ColorSpace.YCRCB]  # second search
    combinations = []
    
    for space in all_spaces:
        combinations.append([space])
    
    rgb_pairs = [
        # ColorSpace.GRAY,    # Luminance info
        # ColorSpace.HSV,     # Hue/saturation info # first search
        # ColorSpace.LAB,     # Perceptual color
        # ColorSpace.YCRCB,   # Luma/chroma separation
        # ColorSpace.HLS,     # Alternative hue representation # first search
    ]
    
    for other_space in rgb_pairs:
        combinations.append([ColorSpace.RGB, other_space])
    
    return combinations


def generate_bins():
    return [4, 8, 16, 32] # third search (W2)
    return [8, 12, 16, 24, 32, 48, 64] # second search
    return [4, 8, 16, 32, 64] # first search
    return [4, 8, 16, 32, 64, 128]


def generate_weights():
    return [None] # third search
    return [WeightStrategy.CENTER_CROP_05, WeightStrategy.CENTER_CROP_10, WeightStrategy.CENTER_CROP_15, WeightStrategy.PYRAMID]
    return [None] + list(WeightStrategy) # first search


def generate_block_splitting_strategies(bins: int, color_spaces: list[ColorSpace]) -> list[descriptor.ImageBlockSplitter]:
    strategies = []
    strategies.append(descriptor.IdentityImageBlockSplitter()) # the base strategy
    
    strategies.append(descriptor.GridImageBlockSplitter((2, 2)))
    if bins <= 32 and len(color_spaces) == 1:
        strategies.append(descriptor.GridImageBlockSplitter((3, 3)))
        strategies.append(descriptor.GridImageBlockSplitter((4, 4)))

    if bins <= 32 and len(color_spaces) == 1:
        strategies.append(descriptor.PyramidImageBlockSplitter([(1, 1), (2, 2)]))
    if bins <= 16 and len(color_spaces) == 1:
        strategies.append(descriptor.PyramidImageBlockSplitter([(1, 1), (2, 2), (3, 3)]))
    # strategies.append(descriptor.PyramidImageBlockSplitter([(1, 1), (2, 2), (3, 3), (4, 4)]))
    
    return strategies


def generate_preprocess_strategies() -> list[descriptor.ImagePreprocessStep | None]:
    """Generate preprocessing pipeline options."""
    strategies = []

    # Option 1: No preprocessing
    # strategies.append(None)

    # Option 2: Open mask (erode from edges) + crop
    # for gamma in [0.8, 1.0]:
    #     strategies.append(preprocessing.Preprocess([
    #         preprocessing.CropToMask(),
    #         preprocessing.ApplyGamma(gamma)
    #     ]))

    # Option 3: Just crop without erosion
    # strategies.append(descriptor.Preprocess([
    #     descriptor.CropToMask()
    # ]))

    strategies.append(preprocessing.Preprocess([
        preprocessing.CropToMask(),
    ]))

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
    # channels = generate_channels(color_spaces)
    # if channels:
    #     for channel in channels:
    #         computers.append(descriptor.Histogram1D(channel, bins, weight_strategy=None, block_splitter=block_splitter))
    
    if bins <= 8 and block_splitter.num_blocks() <= 4:
        channel_pairs = generate_channel_pairs(color_spaces)
        if channel_pairs:
            for pairs in channel_pairs:
                computers.append(descriptor.Histogram2D(pairs, bins, weight_strategy=None, block_splitter=block_splitter))
    
    if bins <= 4 and block_splitter.num_blocks() <= 4:
        channel_triplets = generate_channel_triplets(color_spaces)
        if channel_triplets:
            for triplets in channel_triplets:
                computers.append(descriptor.Histogram3D(triplets, bins, weight_strategy=None, block_splitter=block_splitter))

    return computers

def generate_texture_descriptor_computers(color_spaces: list[ColorSpace]) -> list[descriptor.HistogramComputer]:
    computers = []
    channels = generate_channels(color_spaces)
    """
    for channel in channels:
        for method in ['default', 'ror', 'uniform', 'nri_uniform', 'var']:
            for radius in [1, 3, 5]:
                computer = descriptor.LBPHistogramDescriptor(channels=channel, bins=256, n_points=8, radius=radius, method=method, block_splitter=IdentityImageBlockSplitter())
                computers.append(computer)
    """
    def diag_to_coeffs(diag: int):
        return int(diag * (diag + 1) / 2)
    
    for channel in channels:
        for n_diags in range(2, 31):
            n_coeffs = diag_to_coeffs(n_diags)
            computer = descriptor.DCTDescriptor(channels=channel, n_coeffs=n_coeffs, block_splitter=IdentityImageBlockSplitter())
            computers.append(computer)
    
    """
    for channel in channels:
        # print(f"wavelet count = {len(pywt.wavelist())}")
        # for name in pywt.wavelist():
            # print(f"\t{name}")
        for wavelet in ['db2', 'db4', 'db8', 'sym3', 'sym6', 'sym8', 'coif1', 'coif3', 'bior2.2', 'bior3.3', 'bior4.4', 'rbio2.2', 'rbio3.3', 'haar']:
            for level in [1, 2, 3, 4]:
                computer = descriptor.WaveletDescriptor(channels=channel, block_splitter=IdentityImageBlockSplitter(), wavelet=wavelet, level=level)
                computers.append(computer)
    """

    return computers
    

def hyperparameter_grid_search() -> Iterator[dict]:
    color_space_combos = generate_color_space_combinations()
    # bin_values = generate_bins()
    preprocess_strategies = generate_preprocess_strategies()

    total_combinations = 0

    for color_spaces in color_space_combos:
        for histogram_computer in generate_texture_descriptor_computers(color_spaces):
            for preprocess in preprocess_strategies:
                    total_combinations += 1
                    yield {
                        'color_conversion': ColorConversion(targets=color_spaces, normalize=False),
                        'histogram_computer': histogram_computer,
                        'block_split_strategy': IdentityImageBlockSplitter(),
                        'preprocess': preprocess
                    }


def estimate_grid_size():
    bin_count = len(generate_bins())
    preprocess_count = len(generate_preprocess_strategies())
    color_space_combos = generate_color_space_combinations()

    # Average block splitters per bin value (depends on bins)
    avg_block_split = sum(len(generate_block_splitting_strategies(b)) for b in generate_bins()) / len(generate_bins())

    print("len(color_space_combos)", len(color_space_combos))
    print("bin_count", bin_count)
    print("avg_block_split_count", avg_block_split)
    print("preprocess_count", preprocess_count)

    # Note: histogram_computer count varies based on color_spaces and bins
    # This is a rough estimate
    estimated_total = (len(color_space_combos) * bin_count * avg_block_split * preprocess_count)

    return int(estimated_total)

def actual_grid_size():
    total = 0

    # Track distinct parameter values
    distinct_color_spaces = set()
    distinct_bins = set()
    distinct_block_splitters = set()
    distinct_histogram_types = set()
    distinct_histogram_channels = set()
    distinct_preprocess = set()
    early_total = 0
    for config in hyperparameter_grid_search():
        early_total += 1

    print(f"early total = {early_total}")

    for config in hyperparameter_grid_search():
        total += 1

        # Color spaces (convert list to tuple for hashing)
        color_space_tuple = tuple(sorted([cs.value for cs in config['color_conversion']]))
        distinct_color_spaces.add(color_space_tuple)

        # Preprocessing (convert to string representation)
        if config['preprocess'] is None:
            distinct_preprocess.add('None')
        else:
            preprocess_str = str(config['preprocess'].to_dict())
            distinct_preprocess.add(preprocess_str)

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
    print(f"  Preprocessing strategies: {len(distinct_preprocess):4d} distinct strategies")
    for i, prep in enumerate(sorted(distinct_preprocess), 1):
        # Truncate long preprocessing strings
        prep_display = prep if len(prep) < 100 else prep[:97] + "..."
        print(f"    {i}. {prep_display}")

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
    print("\nActual grid size:", actual_grid_size())

