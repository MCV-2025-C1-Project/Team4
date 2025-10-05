# Import necessary libraries
import cv2
import numpy as np
import pickle
import os
import ot
import scipy


# Define distance and similarity functions

def euclidean_distance(h1: np.ndarray, h2: np.ndarray) -> float:
    """
    Compute the Euclidean (L2) distance between two histograms.

    Parameters:
        h1 (np.ndarray): First histogram.
        h2 (np.ndarray): Second histogram.

    Returns:
        float: The Euclidean distance.
    """
    return np.linalg.norm(h1 - h2, ord=2)


def l1_distance(h1: np.ndarray, h2: np.ndarray) -> float:
    """
    Compute the L1 distance between two histograms.

    Parameters:
        h1 (np.ndarray): First histogram.
        h2 (np.ndarray): Second histogram.

    Returns:
        float: The L1 distance.
    """
    return np.linalg.norm(h1 - h2, ord=1)


def x2_distance(h1: np.ndarray, h2: np.ndarray) -> float:
    """
    Compute the Chi-squared distance between two histograms.

    Parameters:
        h1 (np.ndarray): First histogram.
        h2 (np.ndarray): Second histogram.

    Returns:
        float: The Chi-squared distance.
    """
    return np.sum((h1 - h2) ** 2 / (h1 + h2 + 1e-10))


def canberra_distance(h1: np.ndarray, h2: np.ndarray) -> float:
    """
    Canberra Distance between two histograms.

    Parameters:
        h1 (np.ndarray): First histogram.
        h2 (np.ndarray): Second histogram.

    Returns:
        float: The Canberra distance.
    """
    epsilon = 1e-10
    numerator = np.abs(h1 - h2)
    denominator = np.abs(h1) + np.abs(h2) + epsilon
    return np.sum(numerator / denominator)


def spearman_correlation(h1: np.ndarray, h2: np.ndarray) -> float:
    """
    Compute the Spearman rank correlation coefficient between two histograms.
    
    Parameters:
        h1 (np.ndarray): First histogram.
        h2 (np.ndarray): Second histogram.

    Returns:
        float: The Spearman correlation coefficient  - higher value means more similar histograms.
    """
    # Ranks
    rank_h1 = scipy.stats.rankdata(h1)
    rank_h2 = scipy.stats.rankdata(h2)
    
    # Compute Pearson correlation of the ranks
    mean_rank_h1 = np.mean(rank_h1)
    mean_rank_h2 = np.mean(rank_h2)
    
    numerator = np.sum((rank_h1 - mean_rank_h1) * (rank_h2 - mean_rank_h2))
    denominator = np.sqrt(np.sum((rank_h1 - mean_rank_h1) ** 2) * np.sum((rank_h2 - mean_rank_h2) ** 2))
    
    if denominator == 0:
        return 0.0
    
    return numerator / denominator


def hist_intersection(h1: np.ndarray, h2: np.ndarray) -> float:
    """
    Compute the histogram intersection between two histograms.

    Parameters:
        h1 (np.ndarray): First histogram.
        h2 (np.ndarray): Second histogram.

    Returns:
        float: Sum of minimum values per bin.
    """
    return np.sum(np.minimum(h1, h2))


def bhattacharyya_similarity(h1: np.ndarray, h2: np.ndarray) -> float:
    """
    Compute the Bhattacharyya similarity between two histograms.

    Parameters:
        h1 (np.ndarray): First histogram.
        h2 (np.ndarray): Second histogram.

    Returns:
        float: Hellinger similarity score.
    """
    return np.sum(np.sqrt(h1 * h2))

def hellinger_similarity(h1: np.ndarray, h2: np.ndarray) -> float:
    """
    Compute the Hellinger similarity between two histograms.

    Parameters:
        h1 (np.ndarray): First histogram.
        h2 (np.ndarray): Second histogram.

    Returns:
        float: Hellinger similarity score.
    """
    return np.sqrt(np.sum((np.sqrt(h1) - np.sqrt(h2)) ** 2)) / np.sqrt(2)


def kl_divergence(h1: np.ndarray, h2: np.ndarray) -> float:
    """
    Compute the Kullback-Leibler divergence between two histograms.

    Parameters:
        h1 (np.ndarray): First histogram.
        h2 (np.ndarray): Second histogram.

    Returns:
        float: The KL divergence.
    """
    epsilon = 1e-10
    h1 = h1 + epsilon
    h2 = h2 + epsilon
    return np.sum(h1 * np.log(h1 / h2))


def jensen_shannon_divergence(h1: np.ndarray, h2: np.ndarray) -> float:
    """
    Compute the Jensen-Shannon divergence between two histograms.

    Parameters:
        h1 (np.ndarray): First histogram.
        h2 (np.ndarray): Second histogram.

    Returns:
        float: The Jensen-Shannon divergence.
    """
    m = 0.5 * (h1 + h2)
    return 0.5 * (kl_divergence(h1, m) + kl_divergence(h2, m))


def earth_movers_distance(h1: np.ndarray, h2: np.ndarray, bin_locations: np.ndarray) -> float:
    """
    Compute the Earth Movers Distance (EMD) between two histograms using Wasserstein distance.

    Parameters:
        h1 (np.ndarray): First histogram.
        h2 (np.ndarray): Second histogram.
        bin_locations (np.ndarray): Array of bin locations.

    Returns:
        float: The Earth Movers Distance (EMD).
    """
    return scipy.stats.wasserstein_distance(bin_locations, bin_locations, h1, h2)



def get_similarity_matrix(h1: np.ndarray, h2: np.ndarray) -> np.ndarray:
    """
    Compute a similarity matrix between two histograms by taking the minimum per bin pair.

    Parameters:
        h1 (np.ndarray): First histogram.
        h2 (np.ndarray): Second histogram.

    Returns:
        np.ndarray: A similarity matrix.
    """
    num_bins = len(h1)
    similarity_matrix = np.zeros((num_bins, num_bins))
    for i in range(num_bins):
        for j in range(num_bins):
            similarity_matrix[i, j] = min(h1[i], h2[j])
    return similarity_matrix


def quadratic_form_distance(h1: np.ndarray, h2: np.ndarray) -> float:
    """
    Compute the quadratic form distance between two histograms using a similarity matrix.

    Parameters:
        h1 (np.ndarray): First histogram.
        h2 (np.ndarray): Second histogram.

    Returns:
        float: The quadratic form distance.
    """
    A = get_similarity_matrix(h1, h2)
    diff = h1 - h2
    return np.sqrt(diff.T @ A @ diff)

def simple_quadratic_form_distance(h1: np.ndarray, h2: np.ndarray, sigma: float = 1.0):
    num_bins = len(h1)
    indices = np.arange(num_bins)
    
    bin_distances = np.abs(indices[:, np.newaxis] - indices[np.newaxis, :])
    A = np.exp(-(bin_distances ** 2) / (2 * sigma ** 2))
    
    diff = h1 - h2
    return np.sqrt(diff.T @ A @ diff)


def multichannel_quadratic_form_distance(
        h1: np.ndarray, h2: np.ndarray, 
        num_channels: int, 
        bins_per_channel: int,
        sigma: float = 1.0,
    ) -> float:

    indices = np.arange(bins_per_channel)
    bin_distances = np.abs(indices[:, np.newaxis] - indices[np.newaxis, :])
    A_channel = np.exp(-(bin_distances ** 2) / (2 * sigma ** 2))
    
    h1_channels = h1.reshape(num_channels, bins_per_channel)
    h2_channels = h2.reshape(num_channels, bins_per_channel)
    
    total_distance_squared = 0.0
    for i in range(num_channels):
        diff = h1_channels[i] - h2_channels[i]
        total_distance_squared += diff.T @ A_channel @ diff
    
    return np.sqrt(total_distance_squared)


def emd(a: list[float], b: list[float]) -> float:
    """
    Compute the Earth Movers Distance (EMD) between two 1D histograms using a custom implementation.

    Parameters:
        a (list[float]): First histogram as a list.
        b (list[float]): Second histogram as a list.

    Returns:
        float: The custom computed EMD.
    """
    earth = 0
    diff_array = []
    s = len(a)
    for i in range(s):
        diff_array.append(a[i] - b[i])
    su = []
    for j in range(s):
        earth += diff_array[j]
        su.append(abs(earth))
    return sum(su) / (s - 1) if s > 1 else 0.0


def emd_ot(h1: np.ndarray, h2: np.ndarray, bin_locations: np.ndarray) -> float:
    """
    Compute the Earth Movers Distance (EMD) using the Optimal Transport (OT) library.

    Parameters:
        h1 (np.ndarray): First histogram.
        h2 (np.ndarray): Second histogram.
        bin_locations (np.ndarray): Array of bin locations.

    Returns:
        float: The computed EMD using OT.
    """
    n_bins = len(h1)
    cost_matrix = ot.dist(bin_locations.reshape((n_bins, 1)), bin_locations.reshape((n_bins, 1)))
    ot_emd = ot.emd2(h1, h2, cost_matrix)
    return ot_emd


def emd_multichannel(h1: np.ndarray, h2: np.ndarray, num_channels: int, bins_per_channel: int) -> float:
    """
    Compute multichannel Earth Movers Distance (EMD) by splitting histograms into channels and summing the EMD for each channel.

    Parameters:
        h1 (np.ndarray): Concatenated histogram for the first image.
        h2 (np.ndarray): Concatenated histogram for the second image.
        num_channels (int): Number of color channels.
        bins_per_channel (int): Number of bins per channel.

    Returns:
        float: Total Earth Movers Distance across all channels.
    """
    bin_locations = np.arange(bins_per_channel, dtype=np.float64)
    h1_channels = np.split(h1, num_channels)
    h2_channels = np.split(h2, num_channels)
    total_emd = 0.0
    for i in range(num_channels):
        total_emd += emd_ot(h1_channels[i], h2_channels[i], bin_locations)
    return total_emd


def iter_simple_distances():
    return [
        ("euclidean_distance", euclidean_distance),
        ("l1_distance", l1_distance),
        ("x2_distance", x2_distance),
        ("canberra_distance", canberra_distance),
        ("hist_intersection", hist_intersection),
        ("bhattacharyya_similarity", bhattacharyya_similarity),
        ("hellinger_similarity", hellinger_similarity),
        ("spearman_correlation", spearman_correlation),
        ("kl_divergence", kl_divergence),
        ("jensen_shannon_divergence", jensen_shannon_divergence),
        ("simple_quadratic_form_distance", simple_quadratic_form_distance),
    ]



if __name__ == "__main__":
    pass