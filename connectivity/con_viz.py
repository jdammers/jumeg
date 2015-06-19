#!/usr/bin/env python

""" Visualization functions for connectivity analysis. """

import numpy as np
import scipy as sci
import matplotlib.pyplot as pl
import mne


def sensor_connectivity_3d(raw, picks, con, idx, n_con=20, min_dist=0.05, scale_factor=0.005, tube_radius=0.001):
    """ Function to plot sensor connectivity showing strongest connections(n_con)
        excluding sensors that are less than min_dist apart.
        https://github.com/mne-tools/mne-python/blob/master/examples/connectivity/plot_sensor_connectivity.py

    Parameters
    ----------
    raw : Raw object
        Instance of mne.io.Raw
    picks : list
        Picks to be included.
    con : ndarray (n_channels, n_channels)
        Connectivity matrix.
    idx : list
        List of indices of sensors of interest.
    n_con : int
        Number of connections of interest.
    min_dist : float
        Minimum distance between sensors allowed.

    Note: Please modify scale factor and tube radius to appropriate sizes if the plot looks scrambled.
    """

    # Now, visualize the connectivity in 3D
    try:
        from enthought.mayavi import mlab
    except:
        from mayavi import mlab

    mlab.figure(size=(600, 600), bgcolor=(0.5, 0.5, 0.5))

    # Plot the sensor location
    sens_loc = [raw.info['chs'][picks[i]]['loc'][:3] for i in idx]
    sens_loc = np.array(sens_loc)

    pts = mlab.points3d(sens_loc[:, 0], sens_loc[:, 1], sens_loc[:, 2],
                        color=(1, 1, 1), opacity=1, scale_factor=scale_factor)

    # Get the strongest connections
    threshold = np.sort(con, axis=None)[-n_con]
    ii, jj = np.where(con >= threshold)

    # Remove close connections
    con_nodes = list()
    con_val = list()
    for i, j in zip(ii, jj):
        if sci.linalg.norm(sens_loc[i] - sens_loc[j]) > min_dist:
            con_nodes.append((i, j))
            con_val.append(con[i, j])

    con_val = np.array(con_val)

    # Show the connections as tubes between sensors
    vmax = np.max(con_val)
    vmin = np.min(con_val)
    for val, nodes in zip(con_val, con_nodes):
        x1, y1, z1 = sens_loc[nodes[0]]
        x2, y2, z2 = sens_loc[nodes[1]]
        points = mlab.plot3d([x1, x2], [y1, y2], [z1, z2], [val, val],
                             vmin=vmin, vmax=vmax, tube_radius=tube_radius,
                             colormap='RdBu')
        points.module_manager.scalar_lut_manager.reverse_lut = True

    mlab.scalarbar(title='Phase Lag Index (PLI)', nb_labels=4)

    # Add the sensor names for the connections shown
    nodes_shown = list(set([n[0] for n in con_nodes] +
                           [n[1] for n in con_nodes]))

    for node in nodes_shown:
        x, y, z = sens_loc[node]
        mlab.text3d(x, y, z, raw.ch_names[picks[node]], scale=0.005,
                    color=(0, 0, 0))

    view = (-88.7, 40.8, 0.76, np.array([-3.9e-4, -8.5e-3, -1e-2]))
    mlab.view(*view)
