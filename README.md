# TEAM 4
# Query by Sample Distance Comparison

This project provides a script, `query_by_sample_dist_comparison.py`, for comparing sample distances. Follow the instructions below to set up and execute the code.

## Installation

1. Clone the repository or download the files.
2. Install the required dependencies:
```bash
pip install -r requirements.txt
# or, using uv
uv pip install -r requirements.txt
```


## Usage

Run the script from the command line:

```bash
python query_by_sample_dist_comparison.py dataset_path queries_path [params]
```
### Parameters

- `dataset_path`  
    Path to the dataset file. **(Required, positional argument)**

- `queries_path`  
    Path to the queries file. **(Required, positional argument)**

- `-k`, `--top_k`  
    Number of top results to return. Default: `1`.

- `--bins_per_channel`  
    Number of bins per channel for histogram calculation. Default: `32`.

- `--color_space`  
    Color space to use (e.g., `HSV`, `RGB`). Default: `HSV`.

- `--test`  
    Run with a range of bins and color spaces.

- `--output_file`  
    Path to save the results CSV file. Default: `results1.csv`.

### Example

```bash
python query_by_sample_dist_comparison.py ./data/BBDD ./data/qsd1_w1 --top_k 5 --bins_per_channel 16 --color_space RGB --output_file results/test_output.csv --metric cosine
```

### Repo

# Script Explanations for Team4 Repository

## 1. `visualize_results.py`

**Purpose:**
- Loads a CSV file containing performance metrics, typically from image retrieval experiments.
- Generates visualizations (e.g., bar plots, box plots, heatmaps, line plots, and histograms) to analyze the performance under different configurations.
- Saves the generated figures into an output directory (`results_analysis`) and creates a text summary report.

**Key Components:**
- Data loading and preprocessing functions (e.g., `load_and_prepare_data`).
- Functions to print summary statistics and create various plots.
- Main function to tie the process together and provide command-line arguments for customization.

## 2. `query_by_sample_dist_comparison.py`

**Purpose:**
- Compares image samples by computing image descriptors and evaluating different distance metrics.
- Reads a dataset and query images, computes descriptors, and then assesses performance using a set of distance and similarity functions.
- Generates a CSV file with results and produces visual plots summarizing the performance comparisons.

**Key Components:**
- Functions to add descriptors to datasets (`add_descriptors_to_dataset`).
- Distance evaluation functions including `find_closest` and `calculate_average_precision`.
- Execution flow that supports both single setting and extensive testing across multiple configurations.

## 3. `distances.py`

**Purpose:**
- Contains a collection of functions to compute various distance or similarity measures between image histograms.

**Key Distance Functions:**
- **Euclidean Distance:** Standard L2 norm.
- **L1 Distance:** Manhattan distance.
- **Chi-Squared (`x2_distance`):** Statistical measure for histogram comparison.
- **Histogram Intersection:** Measures the overlap between histograms.
- **Hellinger Similarity:** Based on the Bhattacharyya coefficient.
- **KL Divergence and Jensen-Shannon Divergence:** Statistical divergences to measure differences.
- **Earth Moving Distance (END):** Implemented in several ways (custom implementation, using the OT library, and multichannel extension).

## 4. `descriptor.py`

**Purpose:**
- The `descriptor.py` script is responsible for computing image descriptors from input images.
- It leverages a specified color space (e.g., GRAY, RGB, LAB, YCbCr, etc.) and computes 1D histograms per channel to form a complete descriptor for each image.

## 5. `query_descriptor_grid_search.py`

**Purpose:**
This script automates a **hyperparameter grid search** over different image descriptor configurations and evaluates multiple distance metrics for image retrieval performance. For each combination of descriptor settings (e.g., color spaces, bins, weighting, gamma correction), it computes descriptors for the database, performs queries on a set of test images, calculates `MAP@1` and `MAP@5`, and saves the results as JSON files.

**Key Components:**

* **`ImageDatabase`** – Loads and stores dataset images, manages descriptor computation, and performs distance-based queries.
* **`ImageDescriptorMaker`** – Generates image descriptors based on configurable parameters (color spaces, histogram bins, weighting, etc.).
* **`hyperparameter_grid_search()`** – Yields combinations of descriptor hyperparameters to be evaluated.
* **`mapk()`** – Computes the mean average precision at `k` (for `k=1` and `k=5`).
* **Multiple distance metrics** – Each configuration is tested with several distance functions (L1, Chi-Squared, Intersection, Hellinger, etc.), plus special multichannel distances such as **Earth Mover’s Distance (EMD)** and **Multichannel Quadratic Form Distance**.
* **Result logging** – Each configuration’s results (descriptor settings, distance type, MAP@1, MAP@5) are stored in JSON files.

**Usage:**

```bash
python query_descriptor_grid_search.py database_path queries_path --results_folder PATH [--from_iter N] [--every M]
```

### Parameters

* `database_path`
  Path to the image database file (used with `ImageDatabase.load`). **(Required, positional argument)**
* `queries_path`
  Path to the query images directory containing `.jpg` files and a `gt_corresps.pkl` file. **(Required, positional argument)**

  * The directory must include:

    * `gt_corresps.pkl`: ground truth correspondences (used by `mapk()`).
    * Query images (`.jpg`): filenames are used to extract query IDs.
* `--results_folder`
  Output folder to store JSON files with results for each iteration. **(Required)**
* `--from_iter`
  Index of the first hyperparameter combination to process. Use this to resume from a given point. Default: `0`.
* `--every`
  Process and save results every `M` iterations (e.g., `--every 2` evaluates every second configuration). Default: `1`.


### Workflow Summary

1. **Load database** using `ImageDatabase.load(database_path)`.
2. **Load queries** and ground truth from the provided directory (`gt_corresps.pkl` + `.jpg` images).
3. **Iterate through hyperparameter combinations** from `hyperparameter_grid_search()`.
4. For each combination:

   * Initialize an `ImageDescriptorMaker` with the current parameters.
   * Compute descriptors for all database images.
   * For each distance metric:

     * Compute query descriptors.
     * Retrieve top-5 closest matches for each query.
     * Calculate `MAP@1` and `MAP@5`.
   * Append results to a structured list.
   * Save results as a JSON file in `--results_folder`.

### Example

```bash
python query_descriptor_grid_search.py ./data/BBDD ./data/qsd1_w1 --results_folder results/grid_search --from_iter 0 --every 1
```
