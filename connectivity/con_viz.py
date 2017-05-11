#!/usr/bin/env python

""" Visualization functions for connectivity analysis. """

import sys
import os.path as op
import numpy as np
import scipy as sci
import matplotlib.pyplot as pl
import mne
import yaml
import pickle


def sensor_connectivity_3d(raw, picks, con, idx, n_con=20, min_dist=0.05,
                           scale_factor=0.005, tube_radius=0.001):
    """ Function to plot sensor connectivity showing strongest
        connections(n_con) excluding sensors that are less than min_dist apart.
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

    Note: Please modify scale factor and tube radius to appropriate sizes
          if the plot looks scrambled.
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


def plot_grouped_connectivity_circle(yaml_fname, con, orig_labels,
                                     node_order_size=68, indices=None,
                                     out_fname='circle.png', title=None,
                                     subplot=111, include_legend=False,
                                     n_lines=None, fig=None, show=True,
                                     vmin=None, vmax=None, colormap='hot',
                                     colorbar=False):
    '''
    Plot the connectivity circle grouped and ordered according to
    groups in the yaml input file provided.
    orig_labels : list of str
        Label names in the order as appears in con.
    '''
    # read the yaml file with grouping
    if op.isfile(yaml_fname):
        with open(yaml_fname, 'r') as f:
            labels = yaml.load(f)
    else:
        print '%s - File not found.' % yaml_fname
        sys.exit()

    cortex_colors = ['m', 'b', 'y', 'c', 'r', 'g',
                     'g', 'r', 'c', 'y', 'b', 'm']

    # make list of label_names (without individual cortex locations)
    label_names = list()
    for lab in labels:
        label_names.extend(labels[lab])

    lh_labels = [name + '-lh' for name in label_names]
    rh_labels = [name + '-rh' for name in label_names]

    # Save the plot order and create a circular layout
    node_order = list()
    node_order.extend(lh_labels[::-1])  # reverse the order
    node_order.extend(rh_labels)

    assert len(node_order) == node_order_size, 'Node order length is correct.'

    # the respective no. of regions in each cortex
    group_bound = [len(labels[key]) for key in labels.keys()]
    group_bound = [0] + group_bound[::-1] + group_bound
    group_boundaries = [sum(group_bound[:i+1]) for i in range(len(group_bound))]

    # remove the first element of group_bound
    # make label colours such that each cortex is of one colour
    group_bound.pop(0)
    label_colors = []
    for ind, rep in enumerate(group_bound):
        label_colors += [cortex_colors[ind]] * rep
    assert len(label_colors) == len(node_order), 'Number of colours do not match'

    # remove the last total sum of the list
    group_boundaries.pop()

    from mne.viz.circle import circular_layout
    node_angles = circular_layout(orig_labels, node_order, start_pos=90,
                                  group_boundaries=group_boundaries)

    # the order of the node_colors must match that of orig_labels
    # therefore below reordering is necessary
    reordered_colors = [label_colors[node_order.index(orig)]
                        for orig in orig_labels]

    # Plot the graph using node_order and colours
    # orig_labels is the order of nodes in the con matrix (important)
    from mne.viz import plot_connectivity_circle
    plot_connectivity_circle(con, orig_labels, n_lines=n_lines,
                             facecolor='white', textcolor='black',
                             node_angles=node_angles, colormap=colormap,
                             node_colors=reordered_colors,
                             node_edgecolor='white', fig=fig,
                             fontsize_names=6, vmax=vmax, vmin=vmin,
                             colorbar_size=0.2, colorbar_pos=(-0.3, 0.1),
                             colorbar=colorbar, show=show, subplot=subplot,
                             indices=indices, title=title)

    if include_legend:
        import matplotlib.patches as mpatches
        legend_patches = [mpatches.Patch(color=col, label=key)
                          for col, key in zip(['g', 'r', 'c', 'y', 'b', 'm'],
                                              labels.keys())]
        pl.legend(handles=legend_patches, loc=(0.02, 0.02), ncol=1,
                  mode=None, fontsize='small')
    if out_fname:
        pl.savefig(out_fname, facecolor='white', dpi=300)


def plot_generic_grouped_circle(yaml_fname, con, orig_labels,
                                node_order_size,
                                out_fname='circle.png', title=None,
                                subplot=111, include_legend=False,
                                n_lines=None, fig=None, show=True,
                                vmin=None, vmax=None,
                                colorbar=False):
    '''
    Plot the connectivity circle grouped and ordered according to
    groups in the yaml input file provided. This is not specific to aparc and
    does not automatically split the labels into left and right hemispheres.

    orig_labels : list of str
        Label names in the order as appears in con.
    '''
    # read the yaml file with grouping
    if op.isfile(yaml_fname):
        with open(yaml_fname, 'r') as f:
            labels = yaml.load(f)
    else:
        print '%s - File not found.' % yaml_fname
        sys.exit()

    cortex_colors = ['m', 'b', 'y', 'c', 'r', 'g']

    # make list of label_names (without individual cortex locations)
    label_names = list()
    for lab in labels:
        label_names.extend(labels[lab])

    # here label_names are the node_order
    node_order = label_names
    assert len(node_order) == node_order_size, 'Node order length is correct.'

    # the respective no. of regions in each cortex
    group_bound = [len(labels[key]) for key in labels.keys()]
    group_bound = [0] + group_bound
    group_boundaries = [sum(group_bound[:i+1]) for i in range(len(group_bound))]

    # remove the first element of group_bound
    # make label colours such that each cortex is of one colour
    group_bound.pop(0)
    label_colors = []
    for ind, rep in enumerate(group_bound):
        label_colors += [cortex_colors[ind]] * rep
    assert len(label_colors) == len(node_order), 'Number of colours do not match'

    # remove the last total sum of the list
    group_boundaries.pop()

    from mne.viz.circle import circular_layout
    node_angles = circular_layout(orig_labels, label_names, start_pos=90,
                                  group_boundaries=group_boundaries)

    # the order of the node_colors must match that of orig_labels
    # therefore below reordering is necessary
    reordered_colors = [label_colors[node_order.index(orig)]
                        for orig in orig_labels]

    # Plot the graph using node_order and colours
    # orig_labels is the order on nodes in the con matrix (important)
    from mne.viz import plot_connectivity_circle
    plot_connectivity_circle(con, orig_labels, n_lines=n_lines,
                             facecolor='white', textcolor='black',
                             node_angles=node_angles,
                             node_colors=reordered_colors,
                             node_edgecolor='white', fig=fig,
                             fontsize_names=8, vmax=vmax, vmin=vmin,
                             colorbar_size=0.2, colorbar_pos=(-0.3, 0.1),
                             colorbar=colorbar, show=show, subplot=subplot,
                             title=title)

    if include_legend:
        import matplotlib.patches as mpatches
        legend_patches = [mpatches.Patch(color=col, label=key)
                          for col, key in zip(['g', 'r', 'c', 'y', 'b', 'm'],
                                              labels.keys())]
        pl.legend(handles=legend_patches, loc=(0.02, 0.02), ncol=1,
                  mode=None, fontsize='small')
    if out_fname:
        pl.savefig(out_fname, facecolor='white', dpi=300)
