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


def hellinger_similarity(h1: np.ndarray, h2: np.ndarray) -> float:
    """
    Compute the Hellinger similarity between two histograms.

    Parameters:
        h1 (np.ndarray): First histogram.
        h2 (np.ndarray): Second histogram.

    Returns:
        float: Hellinger similarity score.
    """
    return np.sum(np.sqrt(h1 * h2))


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
        diff_array.appEMD(a[i] - b[i])
    su = []
    for j in range(s):
        earth += diff_array[j]
        su.appEMD(abs(earth))
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


if __name__ == "__main__":
    pass