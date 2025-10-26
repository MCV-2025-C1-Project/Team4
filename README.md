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

- `--color_spaces`
  Color space(s) to use. Options: `RGB`, `HSV`, `LAB`, `YCRCB`, `HLS`, `CMYK`, `LUV`, `XYZ`, `YUV`. Default: `LAB`.

- `--distance`
  Distance function to use (e.g., `canberra_distance`, `l1_distance`, `euclidean_distance`). Default: `canberra_distance`.

- `--generate_masks`
  If present, indicates that it has to generate masks per each query image instead of using the full image.

- `--multiple_paintings`
  If present, indicates that there can be more than one painting per image, and the program tries to split them.

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

### Mask generator: `wall_remover.py`

Generates the mask for a set of images in a folder.

#### Example

```bash
python wall_remover.py
```

In the file has to be specified the name of the folder where the images are.
The script generates a set of .png masks on the same folder.

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

### Core Library (`libs_week2/`)

- **`database.py`**

  - The `Image` class now includes a `mask` attribute to handle image masks.
  - The `ImageDatabase.load` method now loads an associated `.png` mask for each image if it exists. If no mask is found, a default mask of all white pixels is created.
  - `compute_descriptors` now passes both the image and its corresponding mask to the descriptor maker.

- **`descriptor.py`**
  - **Image Preprocessing**: `ImagePreprocessStep`:
    - `ApplyGamma`: Adjusts the gamma of the image.
    - `OpenMask`: Erodes the mask to remove noisy edges.
    - `CropToMask`: Crops the image to the bounding box of the mask content.
  - **Image Block Splitting**:
    - `ImageBlockSplitter`:
      - `IdentityImageBlockSplitter`: Treats the entire image as a single block (the default).
      - `GridImageBlockSplitter`: Divides the image into a grid of a specified shape (e.g., 2x2, 3x3).
      - `PyramidImageBlockSplitter`: Creates a spatial pyramid by combining grids of different resolutions.
  - **Multi-dimensional Histograms**:
    - `Histogram1D`, `Histogram2D`, `Histogram3D`: Classes for computing histograms of different dimensionalities.
  - **`ImageDescriptorMaker`**: Now it takes a `histogram_computer` and a `preprocess` pipeline as arguments. The `make_descriptor` method processes the image and mask through the preprocessing steps, generates the multi-channel color representation, and then uses the specified histogram computer to create the final descriptor.
