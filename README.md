# TEAM 4

This project provides scripts for content-based image retrieval using color histograms and various distance metrics. Follow the instructions below to set up and execute the code.

## Installation

1. Clone the repository or download the files.
2. Install the required dependencies:

```bash
pip install -r requirements.txt
# or, using uv
uv pip install -r requirements.txt
```

## Usage

### Main Script: `query_by_sample.py`

This is the main script for querying images with advanced options.

```bash
python query_by_sample.py dataset_path queries_path [params]
```

#### Parameters

- `dataset_path`
  Path to the dataset directory. **(Required, positional argument)**

- `queries_path`
  Path to the queries directory. **(Required, positional argument)**

- `--gamma`
  Gamma correction factor. Default: `0.8`.

- `--color_spaces`
  Color space(s) to use. Options: `RGB`, `HSV`, `LAB`, `YCRCB`, `HLS`, `CMYK`, `LUV`, `XYZ`, `YUV`. Default: `LAB`.

- `--keep_or_discard`
  String of 'K' (keep) or 'D' (discard) for each channel. Default: all 'K's.

- `--weight_strategy`
  Weighting strategy. Options: `PYRAMID`, `CENTER_CROP_05`, `CENTER_CROP_10`, `CENTER_CROP_15`. Default: `CENTER_CROP_15`.

- `--bins`
  Number of bins in histogram. Default: `32`.

- `--distance`
  Distance function to use (e.g., `canberra_distance`, `l1_distance`, `euclidean_distance`). Default: `canberra_distance`.

- `--k`
  Number of top results to retrieve. Default: `10`.

- `--pkl_output_path`
  Path to save predictions as pickle file (optional).

#### Examples

```bash
# Basic usage
python query_by_sample.py ./data/BBDD ./data/qsd1_w1

# With multiple color spaces and custom parameters
python query_by_sample.py ./data/BBDD ./data/qsd1_w1 --color_spaces LAB HSV --weight_strategy CENTER_CROP_10 --bins 64 --k 5

# Save results to pickle file
python query_by_sample.py ./data/BBDD ./data/qsd1_w1 --pkl_output_path results.pkl --k 10
```

### Grid Search: `parameter_grid_search.py`

Run a comprehensive grid search over hyperparameters.

```bash
python parameter_grid_search.py database_path queries_path --results_folder PATH [params]
```

#### Parameters

- `database_path`
  Path to the database directory. **(Required, positional argument)**

- `queries_path`
  Path to the queries directory. **(Required, positional argument)**

- `--results_folder`
  Output folder to store JSON files with results. **(Required)**

- `--from_iter`
  Index of the first hyperparameter combination to process. Default: `0`.

- `--every`
  Process and save results every N iterations. Default: `1`.

#### Example

```bash
python parameter_grid_search.py ./data/BBDD ./data/qsd1_w1 --results_folder results/grid_search

# Resume from iteration 100, process every 5th configuration
python parameter_grid_search.py ./data/BBDD ./data/qsd1_w1 --results_folder results/grid_search --from_iter 100 --every 5
```

This will save JSON files (one per configuration) with MAP@1 and MAP@5 results for all distance metrics.

## Additional Details

### Core Library (`libs_week1/`)

- **`database.py`** - `ImageDatabase` class for loading and querying images
- **`descriptor.py`** - Image descriptor computation with multiple color spaces and weighting strategies
  - `ImageDescriptor` - Legacy descriptor class (1D histograms)
  - `ImageDescriptorMaker` - Advanced descriptor with gamma correction, multiple color spaces, weighting
- **`distances.py`** - Distance and similarity functions
  - Euclidean, L1, Canberra
  - Chi-squared (χ²), Histogram Intersection
  - Hellinger Similarity, KL Divergence, Jensen-Shannon Divergence
  - Earth Mover's Distance (EMD)
  - Quadratic Form Distance
- **`average_precision.py`** - MAP@k evaluation metrics
- **`hyperparameter_combinations.py`** - Grid search parameter generation

### Visualization Scripts (`plot_results/`)

- **`visualize_results.py`** - Generates comprehensive analysis plots from CSV results
- **`visualize_best_worst.py`** - Shows best and worst performing configurations
- **`visualize_cropping.py`** - Visualizes different weighting strategies
- **`compare_full_vs_center.py`** - Compares full image vs center-weighted descriptors

### Supported Color Spaces

RGB, HSV, LAB, YCrCb, HLS, CMYK, LUV, XYZ, YUV

### Weighting Strategies

- `PYRAMID` - Pyramid-shaped weights from center
- `CENTER_CROP_05/10/15` - Binary weights keeping center (5%, 10%, or 15% border discarded)
