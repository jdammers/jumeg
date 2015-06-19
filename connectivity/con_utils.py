#!/usr/bin/env python

""" Utilities used for connectivity analysis. """

import numpy as np
import mne


def find_distances_matrix(con, epochs, picks_epochs):
    """ Function to calculate distances between sensors ( distance in mm mostly )

    Parameters
    ----------
    con : ndarray
        Connectivity matrix.
    epochs : Epochs object.
        Instance of mne.Epochs
    picks_epochs : list
        Picks of epochs to be considered for analysis.

    Returns
    -------
    con_dist : ndarray
         Connectivity distance matrix. Matrix of distances between various sensors.
    """

    ch_names = epochs.ch_names
    idx = [ch_names.index(name) for name in ch_names]
    sens_loc = [epochs.info['chs'][picks_epochs[i]]['loc'][:3] for i in idx]
    sens_loc = np.array(sens_loc)
    con_dist = np.zeros(con.shape)
    from scipy import linalg
    for i in range(0, 31):
        for j in range(0, 31):
            con_dist[i][j] = linalg.norm(sens_loc[i] - sens_loc[j])
    return con_dist


def weighted_con_matrix(con, epochs, picks_epochs, sigma=20):
    """ Function to compute the weighted connectivity matrix.
        A normalized gaussian weighted matrix is computed and
        added to the true connectivity matrix.

    Parameters
    ----------
    con : ndarray (n_channels x n_channels)
        Connectivity matrix.
    epochs : Epochs object.
        Instance of mne.Epochs
    picks_epochs : list
        Picks of epochs to be considered for analysis.
    sigma : int
        Standard deviation of the gaussian function used for weighting.

    Returns
    -------
    weighted_con_matrix : ndarray (n_channels x n_channels)
        Gaussian distance weighted connectivity matrix.
    """
    con_dist = find_distances_matrix(con, epochs, picks_epochs)

    con_dist_range = np.unique(con_dist.ravel())
    # gaussian function for weighting, sigma - standard deviation
    from scipy.signal import gaussian
    gaussian_function = gaussian(con_dist_range.size * 2, sigma)[:con_dist_range.size]
    # Calculate the weights
    normalized_weights = (con_dist_range * gaussian_function) / np.sum(con_dist_range * gaussian_function)
    # make a dictionary with distance and respective weights values
    # d{sensor_distances:gaussian_normalized_weights}
    d = {}
    for i in range(0, con_dist_range.size):
        d[con_dist_range[i]] = normalized_weights[i]
    # compute the distance weights matrix
    dist_weights_matrix = np.zeros(con_dist.shape)
    for j in range(0, con_dist.shape[0]):
        for k in range(0, con_dist.shape[0]):
            dist_weights_matrix[j][k] = d[con_dist[j][k]]
    # add the weights matrix to connectivity matrix to get the weighted connectivity matrix
    weighted_con_matrix = con + dist_weights_matrix
    return weighted_con_matrix
