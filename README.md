# TEAM 4
## Final presentation
[Slides link](https://docs.google.com/presentation/d/1-dzYacbGVDKyQR7x3ZBHYJwrAK_Wy8d3cXOw-Ck6wKY/edit?usp=sharing)

This project provides scripts for content-based image retrieval using both global (histogram-based) and local (keypoint-based) descriptors.

## Week 4 - Keypoint-Based Image Retrieval

Week 4 introduces **keypoint-based descriptors** for image matching, transitioning from global histogram methods to local feature detection and matching:

- **Keypoint Descriptors**: RootSIFT, SIFT, ORB, BRISK, KAZE, AKAZE, SURF, and more
- **Feature Matching**: BruteForce and FLANN-based matchers with ratio test filtering
- **Geometric Verification**: Homography-based scoring using RANSAC
- **Multiple Painting Detection**: Automatic splitting of images containing multiple artworks
- **Mask Generation**: Variance-based background removal (HSV S+V channels)
- **Visualization**: Interactive and saved visualizations of query results

See [`query_by_sample.py`](#main-script-query_by_samplepy-week-4) for the main Week 4 script and [`libs_week4/`](#core-library-libs_week4) for implementation details.


## Installation

1. Clone the repository or download the files.
2. Install the required dependencies:

```bash
pip install -r requirements.txt
# or, using uv
uv pip install -r requirements.txt
```

## Usage

### Main Script: `query_by_sample.py` (Week 4)

This is the main script for querying images using keypoint-based descriptors (RootSIFT).

```bash
python query_by_sample.py dataset_path queries_path [params]
```

#### Parameters

- `dataset_path`
  Path to the dataset directory. **(Required, positional argument)**

- `queries_path`
  Path to the queries directory. **(Required, positional argument)**

- `--k`
  Number of top results to retrieve. Default: `10`.

- `--pkl_output_path`
  Path to save predictions as pickle file (optional).

- `--generate_masks`
  If present, generates masks using variance-based background removal (HSV S+V channels). Default: `False`.

- `--multiple_paintings`
  If present, detects and splits images containing multiple paintings. Default: `True`.

- `--visualize`
  If present, displays interactive visualizations of query results. Default: `False`.

- `--save_visualizations`
  Directory path to save visualization images (optional).

#### Examples

```bash
# Basic usage with RootSIFT descriptor
python query_by_sample.py ./data/BBDD ./data/qsd1_w4

# With mask generation and multiple painting detection
python query_by_sample.py ./data/BBDD ./data/qsd1_w4 --generate_masks --multiple_paintings

# Save results and visualizations
python query_by_sample.py ./data/BBDD ./data/qsd1_w4 --pkl_output_path results.pkl --save_visualizations ./visualizations

# Interactive visualization mode
python query_by_sample.py ./data/BBDD ./data/qsd1_w4 --visualize --k 10
```

### Grid Search: `parameter_grid_search_w4.py` (Week 4)

Run a comprehensive grid search over keypoint descriptor hyperparameters.

```bash
python parameter_grid_search_w4.py database_path queries_path --results_folder PATH [params]
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
python parameter_grid_search_w4.py ./data/BBDD ./data/qsd1_w4 --results_folder results/grid_search_w4

# Resume from iteration 100, process every 5th configuration
python parameter_grid_search_w4.py ./data/BBDD ./data/qsd1_w4 --results_folder results/grid_search_w4 --from_iter 100 --every 5
```

This will save JSON files (one per configuration) with MAP@1 and MAP@5 results for all keypoint descriptor and scorer combinations.


### Noise removal: `noise_removal.py`

This script is used to detect and remove image noise. The solution is general, but in our project all the images have impulse noise, removed
with different filters. It includes a bunch of batch testing and evaluation functions.

## Filters available
- **Median Filter** (default) - Best for salt & pepper noise, used with 3x3 kernel size
- **Adaptive Median** - Heavy impulse noise, works just in some images
- **Bilateral** - Edge-preserving smoothing
- **Non-Local Means** - High quality, slower
- **Morphological** - Binary noise patterns
- **Cascaded** - Multi-stage, applies 3 filters gradually

## Workflow

In the main method of the script you can select different functions, depending on the
task you want to do.

1. **Analyze**: Run noise detection to understand your dataset and see the metrics(kurtosis value, snr, impuls ratio)
2. **Evaluate**: Test denoising quality with ground truth
3. **Optimize**: Run grid search to find best parameters across all methods
4. **Deploy**: Process your final image set or just a specific image with a specified filter
   
## Output
- Denoised images in specified output folder
- JSON files with detailed metrics


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

### Core Library (`libs_week3/`)

- **`preprocessing.py`** - Enhanced image preprocessing pipeline
  - `Preprocess`: Chain multiple preprocessing steps together
  - `ApplyGamma`: Gamma correction for brightness adjustment
  - `OpenMask`: Erodes mask boundaries to remove noisy edges
  - `CropToMask`: Crops image to the bounding box of the mask
  - `Crop`: Crops image by a specified ratio from all sides

- **`denoised.py`** - Advanced noise detection and removal
  - `detect_noise`: Automatically detects noise type (Gaussian, salt-and-pepper) and noise level using statistical analysis (kurtosis, SNR, variance)
  - `DenoiseWithNonLocalMeans`: Applies non-local means denoising when salt-and-pepper noise is detected
  - `DenoiseWithMedianFilter`: Applies median filtering for noise reduction

- **`color_conversion.py`** - Color space conversion utilities
  - `ColorSpace`: Enum supporting RGB, HSV, LAB, YCrCb, HLS, CMYK, LUV, XYZ, YUV
  - `ColorConversion`: Preprocessing step for converting images to multiple color spaces
  - `bgr_to_cmyk`: Custom CMYK conversion function

- **`descriptor.py`** - Extended descriptor computation with texture features
  - All histogram computers from week 2 (Histogram1D, 2D, 3D)
  - **New texture descriptors**:
    - `LBPHistogramDescriptor`: Local Binary Pattern histogram for texture analysis
    - `DCTDescriptor`: Discrete Cosine Transform for texture/frequency features
    - `WaveletDescriptor`: Discrete Wavelet Transform for multi-resolution texture analysis
  - Weighting strategies: `PYRAMID`, `CONE`, `CENTER_CROP`
  - Image block splitters: `IdentityImageBlockSplitter`, `GridImageBlockSplitter`, `PyramidImageBlockSplitter`

- **`database.py`** - Image database management (similar to week 2 with mask support)

- **`distances.py`** - Distance metrics for comparing descriptors

- **`average_precision.py`** - MAP@k evaluation metrics

- **`hyperparameter_combinations.py`** - Grid search parameter generation

### Core Library (`libs_week4/`)

Week 4 introduces keypoint-based descriptors for image retrieval, moving from global histogram descriptors to local feature matching.

- **`descriptor.py`** - Keypoint detection and descriptor computation
  - **Keypoint Descriptors**:
    - `RootSIFTDescriptor`: L1-normalized and square-rooted SIFT (improved SIFT variant)
    - `SIFTDescriptor`: Scale-Invariant Feature Transform
    - `ORBDescriptor`: Oriented FAST and Rotated BRIEF (fast binary descriptor)
    - `PCASIFTDescriptor`: SIFT with PCA dimensionality reduction
  - **Descriptor Matcher**:
    - `DescriptorMatcher`: Matches keypoint descriptors between images
      - BruteForce (BF) matcher with various norms (L1, L2, Hamming)
      - FLANN-based matcher for fast approximate matching
      - Lowe's ratio test for filtering matches
      - Cross-check option for bidirectional matching
  - **Scoring Methods**:
    - `HomographyScorer`: Scores matches using homography estimation (RANSAC)
      - Reprojection error penalty
      - Configurable min_points threshold
    - `MatchRatioScorer`: Scores based on ratio of good matches to query keypoints
    - `SymmetricMatchRatioScorer`: Bidirectional match ratio scoring
    - `HomographyDistanceScorer`: Combines homography quality with geometric distance
  - `KeypointAndDescriptorMaker`: Integrates preprocessing, color conversion, and descriptor computation

- **`database.py`** - Image database management for keypoint-based retrieval
  - `Image` class with keypoints, descriptors, and score attributes
  - `ImageDatabase` class supporting keypoint descriptor workflow
    - `compute_keypoints_and_descriptors()`: Batch keypoint detection
    - `compute_keypoint_descriptor_statistics()`: Compute keypoint/descriptor statistics
    - `query()`: Find similar images using keypoint matching and scoring

- **`hyperparameter_combinations.py`** - Grid search generators for keypoint descriptors
  - `descriptor_maker_grid_search()`: Iterator over descriptor configurations
  - `scorer_grid_search()`: Iterator over scorer configurations
