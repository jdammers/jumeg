#!/usr/bin/env python

""" Visualization functions for connectivity analysis. """

import sys
import os.path as op

from itertools import cycle
from functools import partial

import numpy as np
import scipy as sci

from mne.viz.utils import plt_show
from mne.viz.circle import (circular_layout, _plot_connectivity_circle_onpick)

import yaml
import time


def sensor_connectivity_3d(raw, picks, con, idx, n_con=20, min_dist=0.05,
                           scale_factor=0.005, tube_radius=0.001,
                           title='Sensor connectivity', vmin=None, vmax=None,
                           out_fname='sensor_connectivity.png'):
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

    # mlab.figure(size=(600, 600), bgcolor=(0.5, 0.5, 0.5))
    mlab.figure(size=(600, 600), bgcolor=(1, 1, 1), fgcolor=(0.5, 0.5, 0.5))

    # Plot the sensor location
    sens_loc = [raw.info['chs'][picks[i]]['loc'][:3] for i in idx]
    sens_loc = np.array(sens_loc)

    pts = mlab.points3d(sens_loc[:, 0], sens_loc[:, 1], sens_loc[:, 2],
                        color=(0.5, 0.5, 0.5), opacity=1, scale_factor=scale_factor)

    # do the distance based thresholding first
    import itertools
    for (i, j) in itertools.combinations(list(range(247)), 2):
        # print sci.linalg.norm(sens_loc[i] - sens_loc[j])
        if sci.linalg.norm(sens_loc[i] - sens_loc[j]) < min_dist:
            con[i, j] = con[j, i ] = 0.

    # Get the strongest connections
    threshold = np.sort(con, axis=None)[-n_con]
    assert threshold > 0.0, 'No surviving connections.'
    ii, jj = np.where(con >= threshold)

    # Remove close connections
    con_nodes = list()
    con_val = list()
    for i, j in zip(ii, jj):
        if sci.linalg.norm(sens_loc[i] - sens_loc[j]) > min_dist:
            con_nodes.append((i, j))
            con_val.append(con[i, j])

    con_val = np.array(con_val)
    print(con_val.shape)

    # Show the connections as tubes between sensors
    if not vmax:
        vmax = np.max(con_val)
    if not vmin:
        vmin = np.min(con_val)
    print(vmin, vmax)

    for val, nodes in zip(con_val, con_nodes):
        x1, y1, z1 = sens_loc[nodes[0]]
        x2, y2, z2 = sens_loc[nodes[1]]
        points = mlab.plot3d([x1, x2], [y1, y2], [z1, z2], [val, val],
                             vmin=vmin, vmax=vmax, tube_radius=tube_radius,
                             colormap='Blues')
        points.module_manager.scalar_lut_manager.reverse_lut = False

    mlab.scalarbar(title=title, nb_labels=2)

    # Add the sensor names for the connections shown
    nodes_shown = list(set([n[0] for n in con_nodes] +
                           [n[1] for n in con_nodes]))

    for node in nodes_shown:
        x, y, z = sens_loc[node]
        mlab.text3d(x, y, z, raw.ch_names[picks[node]], scale=scale_factor,
                    color=(0, 0, 0))

    view = (-88.7, 40.8, 0.76, np.array([-3.9e-4, -8.5e-3, -1e-2]))
    mlab.view(*view)
    time.sleep(1)
    mlab.savefig(out_fname)
    time.sleep(1)
    mlab.close()


def plot_connectivity_circle(con, node_names, indices=None, n_lines=None,
                             node_angles=None, node_width=None,
                             node_colors=None, facecolor='black',
                             textcolor='white', node_edgecolor='black',
                             linewidth=1.5, colormap='hot', vmin=None,
                             vmax=None, colorbar=True, title=None, title_pad=15,
                             colorbar_size=0.2, colorbar_pos=(-0.25, 0.05),
                             symmetric_cbar=False, fontsize_title=12,
                             fontsize_names=8, fontsize_colorbar=8, padding=6.,
                             fig=None, subplot=111, interactive=True,
                             node_linewidth=2., show=True, arrow=False,
                             arrowstyle='->,head_length=0.7,head_width=0.4', **kwargs):
    """Visualize connectivity as a circular graph.

    Note: This code is based on the circle graph example by Nicolas P. Rougier
    http://www.labri.fr/perso/nrougier/coding/.

    Parameters
    ----------
    con : array
        Connectivity scores. Can be a square matrix, or a 1D array. If a 1D
        array is provided, "indices" has to be used to define the connection
        indices.
    node_names : list of str
        Node names. The order corresponds to the order in con.
    indices : tuple of arrays | None
        Two arrays with indices of connections for which the connections
        strenghts are defined in con. Only needed if con is a 1D array.
    n_lines : int | None
        If not None, only the n_lines strongest connections (strength=abs(con))
        are drawn.
    node_angles : array, shape=(len(node_names,)) | None
        Array with node positions in degrees. If None, the nodes are equally
        spaced on the circle. See mne.viz.circular_layout.
    node_width : float | None
        Width of each node in degrees. If None, the minimum angle between any
        two nodes is used as the width.
    node_colors : list of tuples | list of str
        List with the color to use for each node. If fewer colors than nodes
        are provided, the colors will be repeated. Any color supported by
        matplotlib can be used, e.g., RGBA tuples, named colors.
    facecolor : str
        Color to use for background. See matplotlib.colors.
    textcolor : str
        Color to use for text. See matplotlib.colors.
    node_edgecolor : str
        Color to use for lines around nodes. See matplotlib.colors.
    linewidth : float
        Line width to use for connections.
    colormap : str
        Colormap to use for coloring the connections.
    vmin : float | None
        Minimum value for colormap. If None, it is determined automatically.
    vmax : float | None
        Maximum value for colormap. If None, it is determined automatically.
    colorbar : bool
        Display a colorbar or not.
    title : str
        The figure title.
    title_pad : float
        The offset of the title from the top of the axes, in points.
        matplotlib default is None to use rcParams['axes.titlepad'].
    colorbar_size : float
        Size of the colorbar.
    colorbar_pos : 2-tuple
        Position of the colorbar.
    symmetric_cbar : bool
        Symmetric colorbar around 0.
    fontsize_title : int
        Font size to use for title.
    fontsize_names : int
        Font size to use for node names.
    fontsize_colorbar : int
        Font size to use for colorbar.
    padding : float
        Space to add around figure to accommodate long labels.
    fig : None | instance of matplotlib.pyplot.Figure
        The figure to use. If None, a new figure with the specified background
        color will be created.
    subplot : int | 3-tuple
        Location of the subplot when creating figures with multiple plots. E.g.
        121 or (1, 2, 1) for 1 row, 2 columns, plot 1. See
        matplotlib.pyplot.subplot.
    interactive : bool
        When enabled, left-click on a node to show only connections to that
        node. Right-click shows all connections.
    node_linewidth : float
        Line with for nodes.
    show : bool
        Show figure if True.
    arrow: bool
        Include arrows at end of connection.
    arrowstyle: str
        The style params of the arrow head to be drawn.

    Returns
    -------
    fig : instance of matplotlib.pyplot.Figure
        The figure handle.
    axes : instance of matplotlib.axes.PolarAxesSubplot
        The subplot handle.

    Code taken from mne-python v0.14.
    URL: https://github.com/mne-tools/mne-python
    """
    import matplotlib.pyplot as plt
    import matplotlib.path as m_path
    import matplotlib.patches as m_patches

    n_nodes = len(node_names)

    if node_angles is not None:
        if len(node_angles) != n_nodes:
            raise ValueError('node_angles has to be the same length '
                             'as node_names')
        # convert it to radians
        node_angles = node_angles * np.pi / 180
    else:
        # uniform layout on unit circle
        node_angles = np.linspace(0, 2 * np.pi, n_nodes, endpoint=False)

    if node_width is None:
        # widths correspond to the minimum angle between two nodes
        dist_mat = node_angles[None, :] - node_angles[:, None]
        dist_mat[np.diag_indices(n_nodes)] = 1e9
        node_width = np.min(np.abs(dist_mat))
    else:
        node_width = node_width * np.pi / 180

    if node_colors is not None:
        if len(node_colors) < n_nodes:
            node_colors = cycle(node_colors)
    else:
        # assign colors using colormap
        node_colors = [plt.cm.spectral(i / float(n_nodes))
                       for i in range(n_nodes)]

    # handle 1D and 2D connectivity information
    if con.ndim == 1:
        if indices is None:
            raise ValueError('indices has to be provided if con.ndim == 1')
    elif con.ndim == 2:
        if con.shape[0] != n_nodes or con.shape[1] != n_nodes:
            raise ValueError('con has to be 1D or a square matrix')
        # we use the lower-triangular part

        is_symmetric = np.all(np.abs(con - con.T) < 1e-8)
        if is_symmetric:
            indices = np.tril_indices(n_nodes, -1)
        else:
            if not arrow:
                import warnings
                warnings.warn('Since the con matrix is asymmetric it will be '
                              'treated as a causality matrix and arrow will '
                              'be set to True.', Warning)
                arrow = True
            # get off-diagonal indices
            indices = np.where(~np.eye(con.shape[0], dtype=bool))

        con = con[indices]
    else:
        raise ValueError('con has to be 1D or a square matrix')

    # get the colormap
    if isinstance(colormap, str):
        colormap = plt.get_cmap(colormap)

    # Make figure background the same colors as axes
    if fig is None:
        fig = plt.figure(figsize=(8, 8), facecolor=facecolor)

    # Use a polar axes
    if not isinstance(subplot, tuple):
        subplot = (subplot,)
    axes = plt.subplot(*subplot, polar=True, facecolor=facecolor)

    # No ticks, we'll put our own
    plt.xticks([])
    plt.yticks([])

    # Set y axes limit, add additional space if requested
    plt.ylim(0, 10 + padding)

    # Remove the black axes border which may obscure the labels
    axes.spines['polar'].set_visible(False)

    con_abs = np.abs(con)
    n_nonzero_cons = len(np.where(con_abs)[0])
    # Draw lines between connected nodes, only draw the strongest connections
    if n_lines is not None and len(con) > n_lines:

        if n_nonzero_cons > n_lines:
            con_thresh = np.sort(np.abs(con).ravel())[-n_lines]
        elif n_nonzero_cons > 0:
            con_thresh = np.sort(np.abs(con).ravel())[-n_nonzero_cons]
        else:
            # there are no significant connections, set minimum threshold to
            # avoid plotting everything
            con_thresh = 0.001

    else:
        if n_nonzero_cons > 0:
            con_thresh = con_abs[np.where(con_abs)].min()
        else:
            # there are no significant connections, set minimum threshold to
            # avoid plotting everything
            con_thresh = 0.001

    # get the connections which we are drawing and sort by connection strength
    # this will allow us to draw the strongest connections first

    con_draw_idx = np.where(con_abs >= con_thresh)[0]

    con = con[con_draw_idx]
    con_abs = con_abs[con_draw_idx]
    indices = [ind[con_draw_idx] for ind in indices]

    # now sort them
    sort_idx = np.argsort(con_abs)
    del con_abs
    con = con[sort_idx]
    indices = [ind[sort_idx] for ind in indices]

    # Get vmin vmax for color scaling
    if vmin is None:
        if n_nonzero_cons > 0:
            vmin = np.min(con)
        else:
            vmin = 0.
    if vmax is None:
        if n_nonzero_cons > 0:
            vmax = np.max(con)
        else:
            vmax = 0.2

    if symmetric_cbar:
        if np.fabs(vmin) > np.fabs(vmax):
            vmin = -np.fabs(vmin)
            vmax = np.fabs(vmin)
        else:
            vmin = -np.fabs(vmax)
            vmax = np.fabs(vmax)
    vrange = vmax - vmin

    # We want to add some "noise" to the start and end position of the
    # edges: We modulate the noise with the number of connections of the
    # node and the connection strength, such that the strongest connections
    # are closer to the node center
    nodes_n_con = np.zeros((n_nodes), dtype=np.int)
    for i, j in zip(indices[0], indices[1]):
        nodes_n_con[i] += 1
        nodes_n_con[j] += 1

    # initialize random number generator so plot is reproducible
    rng = np.random.mtrand.RandomState(seed=0)

    n_con = len(indices[0])
    noise_max = 0.25 * node_width
    start_noise = rng.uniform(-noise_max, noise_max, n_con)
    end_noise = rng.uniform(-noise_max, noise_max, n_con)

    nodes_n_con_seen = np.zeros_like(nodes_n_con)
    for i, (start, end) in enumerate(zip(indices[0], indices[1])):
        nodes_n_con_seen[start] += 1
        nodes_n_con_seen[end] += 1

        start_noise[i] *= ((nodes_n_con[start] - nodes_n_con_seen[start]) /
                           float(nodes_n_con[start]))
        end_noise[i] *= ((nodes_n_con[end] - nodes_n_con_seen[end]) /
                         float(nodes_n_con[end]))

    # scale connectivity for colormap (vmin<=>0, vmax<=>1)
    con_val_scaled = (con - vmin) / vrange

    # Finally, we draw the connections
    for pos, (i, j) in enumerate(zip(indices[0], indices[1])):
        # Start point
        t0, r0 = node_angles[i], 10

        # End point
        if arrow:
            # make shorter to accomodate arrowhead
            t1, r1 = node_angles[j], 9
        else:
            t1, r1 = node_angles[j], 10

        # Some noise in start and end point
        t0 += start_noise[pos]
        t1 += end_noise[pos]

        verts = [(t0, r0), (t0, 5), (t1, 5), (t1, r1)]
        codes = [m_path.Path.MOVETO, m_path.Path.CURVE4, m_path.Path.CURVE4,
                 m_path.Path.LINETO]
        path = m_path.Path(verts, codes)

        color = colormap(con_val_scaled[pos])

        if arrow:
            # add an arrow to the patch
            patch = m_patches.FancyArrowPatch(path=path,
                                              arrowstyle=arrowstyle,
                                              fill=False, edgecolor=color,
                                              mutation_scale=10,
                                              linewidth=linewidth, alpha=1.)
        else:
            patch = m_patches.PathPatch(path, fill=False, edgecolor=color,
                                        linewidth=linewidth, alpha=1.)

        axes.add_patch(patch)

    # Draw ring with colored nodes
    height = np.ones(n_nodes) * 1.0
    bars = axes.bar(node_angles, height, width=node_width, bottom=9,
                    edgecolor=node_edgecolor, lw=node_linewidth,
                    facecolor='.9', align='center')

    for bar, color in zip(bars, node_colors):
        bar.set_facecolor(color)

    # Draw node labels
    angles_deg = 180 * node_angles / np.pi
    for name, angle_rad, angle_deg in zip(node_names, node_angles, angles_deg):
        if angle_deg >= 270:
            ha = 'left'
        else:
            # Flip the label, so text is always upright
            angle_deg += 180
            ha = 'right'

        axes.text(angle_rad, 10.4, name, size=fontsize_names,
                  rotation=angle_deg, rotation_mode='anchor',
                  horizontalalignment=ha, verticalalignment='center',
                  color=textcolor)

    if title is not None:
        plt.title(title, color=textcolor, fontsize=fontsize_title,
                  axes=axes, pad=title_pad)

    if colorbar:
        sm = plt.cm.ScalarMappable(cmap=colormap,
                                   norm=plt.Normalize(vmin, vmax))
        sm.set_array(np.linspace(vmin, vmax))
        cb = plt.colorbar(sm, ax=axes, use_gridspec=False,
                          shrink=colorbar_size,
                          anchor=colorbar_pos)
        cb_yticks = plt.getp(cb.ax.axes, 'yticklabels')
        cb.ax.tick_params(labelsize=fontsize_colorbar)
        plt.setp(cb_yticks, color=textcolor)

    # Add callback for interaction
    if interactive:
        callback = partial(_plot_connectivity_circle_onpick, fig=fig,
                           axes=axes, indices=indices, n_nodes=n_nodes,
                           node_angles=node_angles)

        fig.canvas.mpl_connect('button_press_event', callback)

    plt_show(show)
    return fig, axes


def plot_grouped_connectivity_circle(yaml_fname, con, orig_labels,
                                     replacer_dict, labels_mode=None,
                                     node_order_size=68, indices=None,
                                     out_fname='circle.png', title=None,
                                     subplot=111, include_legend=False,
                                     n_lines=None, fig=None, show=True,
                                     vmin=None, vmax=None, colormap='hot',
                                     colorbar=False, colorbar_pos=(-0.25, 0.05),
                                     symmetric_cbar=False, bbox_inches=None,
                                     tight_layout=None, **kwargs):
    """
    Plot the connectivity circle grouped and ordered according to
    groups in the yaml input file provided.

    orig_labels : list of str
        Label names in the order as appears in con.

    labels_mode : str | None
        'blank' mode plots no labels on the circle plot,
        'cortex_only' plots only the name of the cortex on one representative
        node and None plots all of the orig_label names provided.

    bbox_inches : None | 'tight'

    tight_layout : bool

    NOTE: yaml order fix helps preserves the order of entries in the yaml file.
    """
    import matplotlib.pyplot as plt
    # read the yaml file with grouping
    if op.isfile(yaml_fname):
        with open(yaml_fname, 'r') as f:
            labels = yaml.safe_load(f)
    else:
        print('%s - File not found.' % yaml_fname)
        sys.exit()

    node_angles, node_colors = _get_group_node_angles_and_colors(labels, orig_labels, node_order_size)

    my_labels = _get_circular_plot_labels(labels_mode, orig_labels, replacer_dict)

    # Plot the graph using node_order and colours
    # orig_labels is the order of nodes in the con matrix (important)
    fig, axes = plot_connectivity_circle(con, my_labels, n_lines=n_lines,
                                         facecolor='white', textcolor='black',
                                         node_angles=node_angles, colormap=colormap,
                                         node_colors=node_colors,
                                         node_edgecolor='white', fig=fig,
                                         vmax=vmax, vmin=vmin, colorbar_size=0.2,
                                         colorbar_pos=colorbar_pos,
                                         colorbar=colorbar, symmetric_cbar=symmetric_cbar,
                                         show=show, subplot=subplot, indices=indices,
                                         title=title, **kwargs)

    if include_legend:
        import matplotlib.patches as mpatches
        # yaml order fix
        legend_patches = [mpatches.Patch(color=col, label=list(llab.keys())[0])
                          for col, llab in zip(['g', 'r', 'c', 'y', 'b', 'm'], labels)]
        # legend_patches = [mpatches.Patch(color=col, label=key)
        #                   for col, key in zip(['g', 'r', 'c', 'y', 'b', 'm'],
        #                                       labels.keys())]
        plt.legend(handles=legend_patches, loc=3, ncol=1,
                   mode=None, fontsize='medium')

    if tight_layout:
        fig.tight_layout()

    if out_fname:
        fig.savefig(out_fname, facecolor='white',
                    dpi=600, bbox_inches=bbox_inches)

    return fig


def _get_circular_plot_labels(labels_mode, orig_labels, replacer_dict):
    """
    Parameters:
    -----------
    labels_mode : str | None
        'blank' mode plots no labels on the circle plot,
        'cortex_only' plots only the name of the cortex on one representative
        node and None plots all of the orig_label names provided.
    orig_labels : list of str
        Label names in the order as appears in con.
    replacer_dict :
        Dictionary to replace the individual label names with cortex
        names.

    Returns:
    --------
    my_labels : list of str
        The label names used in the circular plot.
    """

    # labels mode decides the labels printed for each of the nodes
    if labels_mode == 'blank':
        # show nothing, only the empty circle plot
        my_labels = ['' for _ in orig_labels]

    elif labels_mode == 'cortex_only':
        if isinstance(replacer_dict, dict):
            # show only the names of cortex areas on one representative node
            replacer = replacer_dict
        else:
            raise RuntimeError('Replacer dict with cortex names not set, \
                                cannot choose cortex_only labels_mode.')

        replaced_labels = []
        for myl in orig_labels:
            if myl.split('-lh')[0] in list(replacer.keys()):
                replaced_labels.append(replacer[myl.split('-lh')[0]] + '-lh')
            elif myl.split('-rh')[0] in list(replacer.keys()):
                replaced_labels.append(replacer[myl.split('-rh')[0]] + '-rh')
            else:
                replaced_labels.append('')
        my_labels = replaced_labels
    else:
        # show all the node labels as originally given
        my_labels = orig_labels

    return my_labels


def _get_group_node_angles_and_colors(labels, orig_labels, node_order_size, cortex_colors=None):

    if cortex_colors is None:
        cortex_colors = ['m', 'b', 'y', 'c', 'r', 'g',
                         'g', 'r', 'c', 'y', 'b', 'm']

    ######################################################################
    # Get labels in left and right hemisphere
    ######################################################################

    label_names = list()
    for lab in labels:
        # label_names.extend(labels[lab])
        label_names += list(lab.values())[0]  # yaml order fix

    lh_labels = [name + '-lh' for name in label_names if name + '-lh' in orig_labels]
    rh_labels = [name + '-rh' for name in label_names if name + '-rh' in orig_labels]

    ######################################################################
    # Get number of labels per group in a list
    ######################################################################

    group_numbers = []
    # left first in reverse order, then right hemi labels
    for i in reversed(range(len(labels))):
        cortical_region = list(labels[i].keys())[0]
        actual_num_lh = len([rlab for rlab in labels[i][cortical_region] if rlab + '-lh' in lh_labels])
        # print(cortical_region, actual_num_lh)
        group_numbers.append(actual_num_lh)

    for i in range(len(labels)):
        cortical_region = list(labels[i].keys())[0]
        actual_num_rh = len([rlab for rlab in labels[i][cortical_region] if rlab + '-rh' in rh_labels])
        # print(cortical_region, actual_num_rh)
        group_numbers.append(actual_num_rh)

    assert np.sum(group_numbers) == len(orig_labels), 'Mismatch in number of labels when computing group boundaries.'

    ######################################################################
    # assign a color and angle to each label based on the group
    ######################################################################

    node_order = list()
    node_order.extend(reversed(lh_labels))  # reverse the order
    node_order.extend(rh_labels)

    assert len(node_order) == node_order_size, 'Node order length is correct.'

    node_angles, node_colors = _get_node_angles_and_colors(group_numbers, cortex_colors,
                                                           node_order, orig_labels)

    return node_angles, node_colors


def _get_node_angles_and_colors(group_numbers, cortex_colors, node_order, orig_labels):

    # the respective no. of regions in each cortex
    group_boundaries = np.cumsum([0] + group_numbers)[:-1]

    label_colors = []
    for ind, rep in enumerate(group_numbers):
        label_colors += [cortex_colors[ind]] * rep

    assert len(label_colors) == len(node_order), 'Number of colours do not match'

    # the order of the node_colors must match that of orig_labels
    # therefore below reordering is necessary

    node_colors = [label_colors[node_order.index(orig)] for orig in orig_labels]

    node_angles = circular_layout(orig_labels, node_order, start_pos=90,
                                  group_boundaries=group_boundaries)

    return node_angles, node_colors


def plot_generic_grouped_circle(yaml_fname, con, orig_labels,
                                node_order_size,
                                out_fname='circle.png', title=None,
                                subplot=111, include_legend=False,
                                n_lines=None, fig=None, show=True,
                                vmin=None, vmax=None,
                                colorbar=False, **kwargs):
    """
    Plot the connectivity circle grouped and ordered according to
    groups in the yaml input file provided. This is not specific to aparc and
    does not automatically split the labels into left and right hemispheres.

    orig_labels : list of str
        Label names in the order as appears in con.

    NOTE: The order of entries in the yaml file is not preserved.
    """
    import matplotlib.pyplot as pl
    # read the yaml file with grouping
    if op.isfile(yaml_fname):
        with open(yaml_fname, 'r') as f:
            labels = yaml.safe_load(f)
    else:
        print('%s - File not found.' % yaml_fname)
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
    group_numbers = [len(labels[key]) for key in list(labels.keys())]

    node_angles, node_colors = _get_node_angles_and_colors(group_numbers, cortex_colors,
                                                           node_order, orig_labels)

    # Plot the graph using node_order and colours
    # orig_labels is the order on nodes in the con matrix (important)
    plot_connectivity_circle(con, orig_labels, n_lines=n_lines,
                             facecolor='white', textcolor='black',
                             node_angles=node_angles,
                             node_colors=node_colors,
                             node_edgecolor='white', fig=fig,
                             fontsize_names=8, vmax=vmax, vmin=vmin,
                             colorbar_size=0.2, colorbar_pos=(-0.3, 0.1),
                             colorbar=colorbar, show=show, subplot=subplot,
                             title=title, **kwargs)

    if include_legend:
        import matplotlib.patches as mpatches
        legend_patches = [mpatches.Patch(color=col, label=key)
                          for col, key in zip(['g', 'r', 'c', 'y', 'b', 'm'],
                                              list(labels.keys()))]
        pl.legend(handles=legend_patches, loc=(0.02, 0.02), ncol=1,
                  mode=None, fontsize='small')
    if out_fname:
        pl.savefig(out_fname, facecolor='white', dpi=600)


def get_vmin_vmax_causality(vmin, vmax, cau):
    """
    Get the minimum and maximum off-diagonal values that
    are different from 0.

    Parameters:
    -----------
    vmin : None | float
        If vmin is None, the minimum value is taken
        from the data.
    vmax : None | float
        If vmax is None, the maximum value is taken
        from the data.
    cau : np.array of shape (n_rois, n_rois)
        The causality data.

    Returns:
    --------
    vmin : float
        The minimum value.
    vmax : float
        The maximum value.
    """

    # off diagonal elements
    cau_od = cau[np.where(~np.eye(cau.shape[0], dtype=bool))]
    if vmax is None:
        vmax = cau_od.max()

    if vmin is None:
        if cau_od.max() == 0:
            # no significant connections found
            vmin = 0
            vmax = 0.2
        else:
            # get minimum value that is different from 0
            vmin = cau_od[np.where(cau_od)].min()

    return vmin, vmax


def plot_grouped_causality_circle(caus, yaml_fname, label_names, n_lines=None,
                                  labels_mode='cortex_only', title='Causal Metric',
                                  out_fname='causality_circle.png', colormap='Blues',
                                  figsize=(10, 6), show=False, colorbar=False, fig=None,
                                  vmin=None, vmax=None, tight_layout=False, **kwargs):

    vmin, vmax = get_vmin_vmax_causality(vmin, vmax, caus)

    if not fig:
        import matplotlib.pyplot as plt
        fig = plt.figure(num=None, figsize=figsize)

    fig = plot_grouped_connectivity_circle(yaml_fname, caus, label_names,
                                           out_fname=out_fname, labels_mode=labels_mode,
                                           node_order_size=len(label_names), show=show,
                                           title=title, fig=fig, subplot=(1, 1, 1),
                                           vmin=vmin, vmax=vmax, n_lines=n_lines,
                                           colormap=colormap, colorbar=colorbar,
                                           arrow=True, tight_layout=tight_layout, **kwargs)

    return fig


def plot_degree_circle(degrees, yaml_fname, orig_labels_fname,
                       node_order_size=68, fig=None, subplot=111,
                       color='b', cmap='Blues', tight_layout=False,
                       alpha=0.5, fontsize_groups=6, textcolor_groups='k',
                       radsize=1., degsize=1, show_group_labels=True,
                       out_fname='degree.png', show=True):
    """
    Given degree values of various nodes of a network, plot a grouped circle
    plot a scatter plot around a circle.
    """

    cortex_colors = ['m', 'b', 'y', 'c', 'r', 'g',
                     'g', 'r', 'c', 'y', 'b', 'm']

    n_nodes = len(degrees)

    with open(orig_labels_fname, 'r') as f:
        orig_labels = yaml.safe_load(f)['label_names']

    assert n_nodes == len(orig_labels), 'Mismatch in node names and number.'

    # read the yaml file with grouping of the various nodes
    if op.isfile(yaml_fname):
        with open(yaml_fname, 'r') as f:
            labels = yaml.safe_load(f)
    else:
        print('%s - File not found.' % yaml_fname)
        sys.exit()

    # make list of label_names (without individual cortex locations)
    label_names = [list(lab.values())[0] for lab in labels]
    label_names = [la for l in label_names for la in l]

    lh_labels = [name + '-lh' for name in label_names]
    rh_labels = [name + '-rh' for name in label_names]

    # save the plot order
    node_order = list()
    node_order.extend(lh_labels[::-1])  # reverse the order
    node_order.extend(rh_labels)
    assert len(node_order) == node_order_size, 'Node order length is correct.'

    # the respective no. of regions in each cortex
    # yaml fix order change
    group_numbers = [len(list(key.values())[0]) for key in labels]
    group_numbers = group_numbers[::-1] + group_numbers

    node_angles, node_colors = _get_node_angles_and_colors(group_numbers, cortex_colors,
                                                           node_order, orig_labels)

    # prepare group label positions
    group_labels = [list(lab.keys())[0] for lab in labels]
    grp_lh_labels = [name + '-lh' for name in group_labels]
    grp_rh_labels = [name + '-rh' for name in group_labels]
    all_group_labels = grp_lh_labels + grp_rh_labels

    # save the group order
    group_node_order = list()
    group_node_order.extend(grp_lh_labels[::-1])
    group_node_order.extend(grp_rh_labels)
    n_groups = len(group_node_order)

    group_node_angles = circular_layout(group_node_order, group_node_order,
                                        start_pos=90.)

    import matplotlib.pyplot as plt
    if fig is None:
        fig = plt.figure()

    # use a polar axes
    if not isinstance(subplot, tuple):
        subplot = (subplot,)

    ax = plt.subplot(*subplot, polar=True, facecolor='white')

    # first plot the circle showing the degree
    theta = np.deg2rad(node_angles)
    radii = np.ones(len(node_angles)) * radsize
    c = ax.scatter(theta, radii, c=node_colors, s=degrees * degsize,
                   cmap=None, alpha=alpha)

    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['polar'].set_visible(False)

    # add group labels
    if show_group_labels:
        for i in range(group_node_angles.size):
            # to modify the position of the labels
            theta = group_node_angles[i] + np.pi/n_groups
            ax.text(np.deg2rad(theta), radsize + radsize/5., group_node_order[i],
                    rotation=theta-90.,
                    size=fontsize_groups, horizontalalignment='center',
                    verticalalignment='center', color=textcolor_groups)
            # to draw lines
            # ax.bar(np.deg2rad(theta), 1, bottom=0., width=(np.pi/180), color='r')
            # ax.text(np.deg2rad(theta), 1.2, all_group_labels[i])

    if tight_layout:
        fig.tight_layout()

    if out_fname:
        fig.savefig(out_fname, dpi=300.)

    if show:
        plt.show()

    return fig, ax


def plot_lines_and_blobs(con, degrees, yaml_fname, orig_labels_fname,
                         replacer_dict,
                         node_order_size=68, fig=None, subplot=111,
                         color='b', cmap='Blues', tight_layout=False,
                         alpha=0.5, fontsize_groups=6, textcolor_groups='k',
                         radsize=1., degsize=1, labels_mode=None,
                         linewidth=1.5, n_lines=50, node_width=None,
                         arrow=False, out_fname='lines_and_blobs.png',
                         vmin=None, vmax=None, figsize=None,
                         fontsize_colorbar=8, textcolor='black',
                         fontsize_title=12, title=None, fontsize_names=6,
                         show_node_labels=False, colorbar=True,
                         colorbar_size=0.2, colorbar_pos=(-0.3, 0.1),
                         show=True, **kwargs):
    '''
    Plot connectivity circle plot with a centrality index per node shown as
    blobs along the circulference of the circle, hence the lines and the blobs.
    '''

    import yaml

    if isinstance(orig_labels_fname, str):
        with open(orig_labels_fname, 'r') as f:
            orig_labels = yaml.safe_load(f)['label_names']
    else:
        orig_labels = orig_labels_fname

    n_nodes = len(degrees)

    assert n_nodes == len(orig_labels), 'Mismatch in node names and number.'
    assert n_nodes == len(con), 'Mismatch in n_nodes for con and degrees.'

    # read the yaml file with grouping of the various nodes
    if op.isfile(yaml_fname):
        with open(yaml_fname, 'r') as f:
            labels = yaml.safe_load(f)
    else:
        print('%s - File not found.' % yaml_fname)
        sys.exit()

    # make list of label_names (without individual cortex locations)
    label_names = [list(lab.values())[0] for lab in labels]
    label_names = [la for l in label_names for la in l]

    lh_labels = [name + '-lh' for name in label_names if name + '-lh' in orig_labels]
    rh_labels = [name + '-rh' for name in label_names if name + '-rh' in orig_labels]

    # Save the plot order and create a circular layout
    node_order = list()
    node_order.extend(lh_labels[::-1])  # reverse the order
    node_order.extend(rh_labels)

    assert len(node_order) == node_order_size, 'Node order length is correct.'

    group_bound = [0]
    # left first in reverse order, then right hemi labels
    for i in range(len(labels))[::-1]:
        cortical_region = list(labels[i].keys())[0]
        actual_num_lh = [rlab for rlab in labels[i][cortical_region] if rlab + '-lh' in lh_labels]
        group_bound.append(len(actual_num_lh))

    for i in range(len(labels)):
        cortical_region = list(labels[i].keys())[0]
        actual_num_rh = [rlab for rlab in labels[i][cortical_region] if rlab + '-rh' in rh_labels]
        group_bound.append(len(actual_num_rh))

    assert np.sum(group_bound) == len(orig_labels), 'Mismatch in number of labels when computing group boundaries.'

    # the respective no. of regions in each cortex
    # group_bound = [len(list(key.values())[0]) for key in labels]  # yaml order fix
    # group_bound = [0] + group_bound[::-1] + group_bound
    group_boundaries = [sum(group_bound[:i+1]) for i in range(len(group_bound))]

    # remove the first element of group_bound
    # make label colours such that each cortex is of one colour
    group_bound.pop(0)

    # remove the last total sum of the list
    group_boundaries.pop()

    from mne.viz.circle import circular_layout
    node_angles = circular_layout(orig_labels, node_order, start_pos=90,
                                  group_boundaries=group_boundaries)

    if node_width is None:
        # widths correspond to the minimum angle between two nodes
        dist_mat = node_angles[None, :] - node_angles[:, None]
        dist_mat[np.diag_indices(n_nodes)] = 1e9
        node_width = np.min(np.abs(dist_mat))
    else:
        node_width = node_width * np.pi / 180

    # prepare group label positions
    group_labels = [list(lab.keys())[0] for lab in labels]
    grp_lh_labels = [name + '-lh' for name in group_labels]
    grp_rh_labels = [name + '-rh' for name in group_labels]
    all_group_labels = grp_lh_labels + grp_rh_labels

    # save the group order
    group_node_order = list()
    group_node_order.extend(grp_lh_labels[::-1])
    group_node_order.extend(grp_rh_labels)
    n_groups = len(group_node_order)

    group_node_angles = circular_layout(group_node_order, group_node_order,
                                        start_pos=90.)

    import matplotlib.pyplot as plt
    import matplotlib.path as m_path
    import matplotlib.patches as m_patches

    if fig is None:
        fig = plt.figure(figsize=figsize)

    # use a polar axes
    if not isinstance(subplot, tuple):
        subplot = (subplot,)

    ax = plt.subplot(*subplot, polar=True, facecolor='white')

    cortex_colors = ['m', 'b', 'y', 'c', 'r', 'g',
                     'g', 'r', 'c', 'y', 'b', 'm']

    label_colors = []
    for ind, rep in enumerate(group_bound):
        label_colors += [cortex_colors[ind]] * rep
    assert len(label_colors) == len(node_order), 'Number of colours do not match'

    # the order of the node_colors must match that of orig_labels
    # therefore below reordering is necessary
    reordered_colors = [label_colors[node_order.index(orig)]
                        for orig in orig_labels]

    # first plot the circle showing the degree
    theta = np.deg2rad(node_angles)
    radii = np.ones(len(node_angles)) * radsize
    c = ax.scatter(theta, radii, c=reordered_colors, s=degrees * degsize,
                   cmap=cmap, alpha=alpha)

    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['polar'].set_visible(False)

    # handle 1D and 2D connectivity information
    if con.ndim == 1:
        if indices is None:
            raise ValueError('indices has to be provided if con.ndim == 1')
    elif con.ndim == 2:
        if con.shape[0] != n_nodes or con.shape[1] != n_nodes:
            raise ValueError('con has to be 1D or a square matrix')
        # we use the lower-triangular part
        indices = np.tril_indices(n_nodes, -1)
        con = con[indices]
    else:
        raise ValueError('con has to be 1D or a square matrix')

    # Draw lines between connected nodes, only draw the strongest connections
    if n_lines is not None and len(con) > n_lines:
        con_thresh = np.sort(np.abs(con).ravel())[-n_lines]
    else:
        con_thresh = 0.

    # get the connections which we are drawing and sort by connection strength
    # this will allow us to draw the strongest connections first
    con_abs = np.abs(con)
    con_draw_idx = np.where(con_abs >= con_thresh)[0]

    con = con[con_draw_idx]
    con_abs = con_abs[con_draw_idx]
    indices = [ind[con_draw_idx] for ind in indices]

    # now sort them
    sort_idx = np.argsort(con_abs)
    con_abs = con_abs[sort_idx]
    con = con[sort_idx]
    indices = [ind[sort_idx] for ind in indices]

    # get the colormap
    if isinstance(cmap, str):
        colormap = plt.get_cmap(cmap)

    # Get vmin vmax for color scaling
    if vmin is None:
        vmin = np.min(con[np.abs(con) >= con_thresh])
    if vmax is None:
        vmax = np.max(con)
    vrange = vmax - vmin

    # We want to add some "noise" to the start and end position of the
    # edges: We modulate the noise with the number of connections of the
    # node and the connection strength, such that the strongest connections
    # are closer to the node center
    nodes_n_con = np.zeros((n_nodes), dtype=np.int)
    for i, j in zip(indices[0], indices[1]):
        nodes_n_con[i] += 1
        nodes_n_con[j] += 1

    # initialize random number generator so plot is reproducible
    rng = np.random.mtrand.RandomState(seed=0)

    n_con = len(indices[0])
    noise_max = 0.25 * node_width
    start_noise = rng.uniform(-noise_max, noise_max, n_con)
    end_noise = rng.uniform(-noise_max, noise_max, n_con)

    nodes_n_con_seen = np.zeros_like(nodes_n_con)
    for i, (start, end) in enumerate(zip(indices[0], indices[1])):
        nodes_n_con_seen[start] += 1
        nodes_n_con_seen[end] += 1

        start_noise[i] *= ((nodes_n_con[start] - nodes_n_con_seen[start]) /
                           float(nodes_n_con[start]))
        end_noise[i] *= ((nodes_n_con[end] - nodes_n_con_seen[end]) /
                         float(nodes_n_con[end]))

    # scale connectivity for colormap (vmin<=>0, vmax<=>1)
    con_val_scaled = (con - vmin) / vrange

    # convert to radians for below code
    node_angles = node_angles * np.pi / 180

    # Finally, we draw the connections
    for pos, (i, j) in enumerate(zip(indices[0], indices[1])):
        # Start point
        t0, r0 = node_angles[i], radsize - 0.05

        # End point
        if arrow:
            # make shorter to accomodate arrowhead
            t1, r1 = node_angles[j], radsize - 1.
        else:
            t1, r1 = node_angles[j], radsize - 0.05

        # Some noise in start and end point
        t0 += start_noise[pos]
        t1 += end_noise[pos]

        verts = [(t0, r0), (t0, radsize/2.), (t1, radsize/2.), (t1, r1)]
        codes = [m_path.Path.MOVETO, m_path.Path.CURVE4, m_path.Path.CURVE4,
                 m_path.Path.LINETO]
        path = m_path.Path(verts, codes)

        color = colormap(con_val_scaled[pos])

        if arrow:
            # add an arrow to the patch
            patch = m_patches.FancyArrowPatch(path=path,
                                              arrowstyle=arrowstyle,
                                              fill=False, edgecolor=color,
                                              mutation_scale=10,
                                              linewidth=linewidth, alpha=1.)
        else:
            patch = m_patches.PathPatch(path, fill=False, edgecolor=color,
                                        linewidth=linewidth, alpha=1.)

        ax.add_patch(patch)

    # labels mode decides the labels printed for each of the nodes
    if labels_mode == 'blank':
        # show nothing, only the empty circle plot
        my_labels = ['' for orig in orig_labels]
    elif labels_mode == 'cortex_only':
        if isinstance(replacer_dict, dict):
            # show only the names of cortex areas on one representative node
            replacer = replacer_dict
        else:
            raise RuntimeError('Replacer dict with cortex names not set, \
                                cannot choose cortex_only labels_mode.')

        replaced_labels = []
        for myl in orig_labels:
            if myl.split('-lh')[0] in list(replacer.keys()):
                replaced_labels.append(replacer[myl.split('-lh')[0]] + '-lh')
            elif myl.split('-rh')[0] in list(replacer.keys()):
                replaced_labels.append(replacer[myl.split('-rh')[0]] + '-rh')
            else:
                replaced_labels.append('')
        my_labels = replaced_labels
    else:
        # show all the node labels as originally given
        my_labels = orig_labels

    # draw node labels
    if show_node_labels:
        angles_deg = 180 * node_angles / np.pi
        for name, angle_rad, angle_deg in zip(my_labels, node_angles,
                                              angles_deg):
            if angle_deg >= 270:
                ha = 'left'
            else:
                # Flip the label, so text is always upright
                angle_deg += 180
                ha = 'right'

            ax.text(angle_rad, radsize + 0.2, name, size=fontsize_names,
                    rotation=angle_deg, rotation_mode='anchor',
                    horizontalalignment=ha, verticalalignment='center',
                    color='k')

    # # add group labels
    # if show_group_labels:
    #     for i in range(group_node_angles.size):
    #         # to modify the position of the labels
    #         theta = group_node_angles[i] + np.pi/n_groups
    #         ax.text(np.deg2rad(theta), radsize + radsize/5., group_node_order[i],
    #                 rotation=theta-90.,
    #                 size=fontsize_groups, horizontalalignment='center',
    #                 verticalalignment='center', color=textcolor_groups)

    if colorbar:
        sm = plt.cm.ScalarMappable(cmap=colormap,
                                   norm=plt.Normalize(vmin, vmax))
        sm.set_array(np.linspace(vmin, vmax))
        cb = plt.colorbar(sm, ax=ax, use_gridspec=False,
                          shrink=colorbar_size,
                          anchor=colorbar_pos)
        cb_yticks = plt.getp(cb.ax.axes, 'yticklabels')
        cb.ax.tick_params(labelsize=fontsize_colorbar)
        plt.setp(cb_yticks, color=textcolor)

    if title is not None:
        plt.title(title, color=textcolor, fontsize=fontsize_title,
                  axes=ax)

    if show:
        plt.show()

    # if tight_layout:
    #     fig.tight_layout()

    # if out_fname:
    #     fig.savefig(out_fname, dpi=300.)

    return fig, ax


def _plot_connectivity_circle_group_bars(con, node_names,
                                         indices=None, n_lines=None,
                                         node_angles=None, node_width=None,
                                         node_colors=None, facecolor='black',
                                         textcolor='white', node_edgecolor='black',
                                         linewidth=1.5, colormap='hot', vmin=None,
                                         vmax=None, colorbar=True, title=None,
                                         group_node_order=None, group_node_angles=None,
                                         group_node_width=None, group_colors=None,
                                         fontsize_groups=8,
                                         colorbar_size=0.2, colorbar_pos=(-0.3, 0.1),
                                         fontsize_title=12, fontsize_names=8,
                                         fontsize_colorbar=8, padding=6.,
                                         fig=None, subplot=111,
                                         node_linewidth=2., show=True):
    """Visualize connectivity as a circular graph.

    Circle connectivity plot with external labels ring with group names.

    Note: This code is based on the circle graph example by Nicolas P. Rougier
    http://www.labri.fr/perso/nrougier/coding/.

    Parameters
    ----------
    con : array
        Connectivity scores. Can be a square matrix, or a 1D array. If a 1D
        array is provided, "indices" has to be used to define the connection
        indices.
    node_names : list of str
        Node names. The order corresponds to the order in con.
    indices : tuple of arrays | None
        Two arrays with indices of connections for which the connections
        strenghts are defined in con. Only needed if con is a 1D array.
    n_lines : int | None
        If not None, only the n_lines strongest connections (strength=abs(con))
        are drawn.
    node_angles : array, shape=(len(node_names,)) | None
        Array with node positions in degrees. If None, the nodes are equally
        spaced on the circle. See mne.viz.circular_layout.
    node_width : float | None
        Width of each node in degrees. If None, the minimum angle between any
        two nodes is used as the width.
    node_colors : list of tuples | list of str
        List with the color to use for each node. If fewer colors than nodes
        are provided, the colors will be repeated. Any color supported by
        matplotlib can be used, e.g., RGBA tuples, named colors.
    group_node_order : list of str
        Group node names in correct order.
    group_node_angles : array, shape=(len(group_node_order,)) | None
        Array with node positions in degrees. If None, the nodes are equally
        spaced on the circle. See mne.viz.circular_layout.
    group_node_width : float | None
        Width of each group node in degrees. If None, the minimum angle between
        any two nodes is used as the width.
    group_colors : None
        List with colours to use for each group node.
    fontsize_groups : int
        The font size of the text used for group node labels.
    facecolor : str
        Color to use for background. See matplotlib.colors.
    textcolor : str
        Color to use for text. See matplotlib.colors.
    node_edgecolor : str
        Color to use for lines around nodes. See matplotlib.colors.
    linewidth : float
        Line width to use for connections.
    colormap : str
        Colormap to use for coloring the connections.
    vmin : float | None
        Minimum value for colormap. If None, it is determined automatically.
    vmax : float | None
        Maximum value for colormap. If None, it is determined automatically.
    colorbar : bool
        Display a colorbar or not.
    title : str
        The figure title.
    colorbar_size : float
        Size of the colorbar.
    colorbar_pos : 2-tuple
        Position of the colorbar.
    fontsize_title : int
        Font size to use for title.
    fontsize_names : int
        Font size to use for node names.
    fontsize_colorbar : int
        Font size to use for colorbar.
    padding : float
        Space to add around figure to accommodate long labels.
    fig : None | instance of matplotlib.pyplot.Figure
        The figure to use. If None, a new figure with the specified background
        color will be created.
    subplot : int | 3-tuple
        Location of the subplot when creating figures with multiple plots. E.g.
        121 or (1, 2, 1) for 1 row, 2 columns, plot 1. See
        matplotlib.pyplot.subplot.
    node_linewidth : float
        Line with for nodes.
    show : bool
        Show figure if True.

    Returns
    -------
    fig : instance of matplotlib.pyplot.Figure
        The figure handle.
    axes : instance of matplotlib.axes.PolarAxesSubplot
        The subplot handle.

    Modified from mne-python v0.14.
    """
    import matplotlib.pyplot as plt
    import matplotlib.path as m_path
    import matplotlib.patches as m_patches

    n_nodes = len(node_names)
    n_groups = len(group_node_order)

    if group_node_angles is not None:
        if len(group_node_angles) != n_groups:
            raise ValueError('group_node_angles has to be the same length '
                             'as group_node_order')
        # convert it to radians
        group_node_angles = group_node_angles * np.pi / 180

    if node_angles is not None:
        if len(node_angles) != n_nodes:
            raise ValueError('node_angles has to be the same length '
                             'as node_names')
        # convert it to radians
        node_angles = node_angles * np.pi / 180
    else:
        # uniform layout on unit circle
        node_angles = np.linspace(0, 2 * np.pi, n_nodes, endpoint=False)

    if node_width is None:
        # widths correspond to the minimum angle between two nodes
        dist_mat = node_angles[None, :] - node_angles[:, None]
        dist_mat[np.diag_indices(n_nodes)] = 1e9
        node_width = np.min(np.abs(dist_mat))
    else:
        node_width = node_width * np.pi / 180

    if node_colors is not None:
        if len(node_colors) < n_nodes:
            node_colors = cycle(node_colors)
    else:
        # assign colors using colormap
        node_colors = [plt.cm.spectral(i / float(n_nodes))
                       for i in range(n_nodes)]

    # handle 1D and 2D connectivity information
    if con.ndim == 1:
        if indices is None:
            raise ValueError('indices has to be provided if con.ndim == 1')
    elif con.ndim == 2:
        if con.shape[0] != n_nodes or con.shape[1] != n_nodes:
            raise ValueError('con has to be 1D or a square matrix')
        # we use the lower-triangular part
        indices = np.tril_indices(n_nodes, -1)
        con = con[indices]
    else:
        raise ValueError('con has to be 1D or a square matrix')

    # get the colormap
    if isinstance(colormap, str):
        colormap = plt.get_cmap(colormap)

    # Make figure background the same colors as axes
    if fig is None:
        fig = plt.figure(figsize=(8, 8), facecolor=facecolor)

    # Use a polar axes
    if not isinstance(subplot, tuple):
        subplot = (subplot,)
    axes = plt.subplot(*subplot, polar=True, facecolor=facecolor)

    # No ticks, we'll put our own
    plt.xticks([])
    plt.yticks([])

    # Set y axes limit, add additonal space if requested
    # plt.ylim(0, 10 + padding)

    # increase space to allow for external group names
    plt.ylim(0, 18 + padding)

    # Remove the black axes border which may obscure the labels
    axes.spines['polar'].set_visible(False)

    # Draw lines between connected nodes, only draw the strongest connections
    if n_lines is not None and len(con) > n_lines:
        con_thresh = np.sort(np.abs(con).ravel())[-n_lines]
    else:
        con_thresh = 0.

    # get the connections which we are drawing and sort by connection strength
    # this will allow us to draw the strongest connections first
    con_abs = np.abs(con)
    con_draw_idx = np.where(con_abs >= con_thresh)[0]

    con = con[con_draw_idx]
    con_abs = con_abs[con_draw_idx]
    indices = [ind[con_draw_idx] for ind in indices]

    # now sort them
    sort_idx = np.argsort(con_abs)
    con_abs = con_abs[sort_idx]
    con = con[sort_idx]
    indices = [ind[sort_idx] for ind in indices]

    # Get vmin vmax for color scaling
    if vmin is None:
        vmin = np.min(con[np.abs(con) >= con_thresh])
    if vmax is None:
        vmax = np.max(con)
    vrange = vmax - vmin

    # We want to add some "noise" to the start and end position of the
    # edges: We modulate the noise with the number of connections of the
    # node and the connection strength, such that the strongest connections
    # are closer to the node center
    nodes_n_con = np.zeros((n_nodes), dtype=np.int)
    for i, j in zip(indices[0], indices[1]):
        nodes_n_con[i] += 1
        nodes_n_con[j] += 1

    # initalize random number generator so plot is reproducible
    rng = np.random.mtrand.RandomState(seed=0)

    n_con = len(indices[0])
    noise_max = 0.25 * node_width
    start_noise = rng.uniform(-noise_max, noise_max, n_con)
    end_noise = rng.uniform(-noise_max, noise_max, n_con)

    nodes_n_con_seen = np.zeros_like(nodes_n_con)
    for i, (start, end) in enumerate(zip(indices[0], indices[1])):
        nodes_n_con_seen[start] += 1
        nodes_n_con_seen[end] += 1

        start_noise[i] *= ((nodes_n_con[start] - nodes_n_con_seen[start]) /
                           float(nodes_n_con[start]))
        end_noise[i] *= ((nodes_n_con[end] - nodes_n_con_seen[end]) /
                         float(nodes_n_con[end]))

    # scale connectivity for colormap (vmin<=>0, vmax<=>1)
    con_val_scaled = (con - vmin) / vrange

    # Finally, we draw the connections
    for pos, (i, j) in enumerate(zip(indices[0], indices[1])):
        # Start point
        t0, r0 = node_angles[i], 10

        # End point
        t1, r1 = node_angles[j], 10

        # Some noise in start and end point
        t0 += start_noise[pos]
        t1 += end_noise[pos]

        verts = [(t0, r0), (t0, 5), (t1, 5), (t1, r1)]
        codes = [m_path.Path.MOVETO, m_path.Path.CURVE4, m_path.Path.CURVE4,
                 m_path.Path.LINETO]
        path = m_path.Path(verts, codes)

        color = colormap(con_val_scaled[pos])

        # Actual line
        patch = m_patches.PathPatch(path, fill=False, edgecolor=color,
                                    linewidth=linewidth, alpha=1.)
        axes.add_patch(patch)

    # Draw ring with colored nodes
    height = np.ones(n_nodes) * 1.0
    bars = axes.bar(node_angles, height, width=node_width, bottom=9,
                    edgecolor=node_edgecolor, lw=node_linewidth,
                    facecolor='.9', align='center')

    for bar, color in zip(bars, node_colors):
        bar.set_facecolor(color)

    # Draw node labels
    angles_deg = 180 * node_angles / np.pi
    for name, angle_rad, angle_deg in zip(node_names, node_angles, angles_deg):
        if angle_deg >= 270:
            ha = 'left'
        else:
            # Flip the label, so text is always upright
            angle_deg += 180
            ha = 'right'

        axes.text(angle_rad, 10.4, name, size=fontsize_names,
                  rotation=angle_deg, rotation_mode='anchor',
                  horizontalalignment=ha, verticalalignment='center',
                  color=textcolor)

    # draw outer ring with group names
    group_heights = np.ones(n_groups) * 1.5
    group_width = 2 * np.pi/n_groups

    # draw ring with group colours
    group_bars = axes.bar(group_node_angles, group_heights,
                          width=group_width, bottom=22,
                          linewidth=node_linewidth, facecolor='.9',
                          edgecolor=node_edgecolor)

    for gbar, color in zip(group_bars, group_colors):
        gbar.set_facecolor(color)

    # add group labels
    for i in range(group_node_angles.size):
        # to modify the position of the labels
        theta = group_node_angles[i] + np.pi/n_groups
        # theta = group_node_angles[n_groups-1-i] + np.pi/n_groups
        plt.text(theta, 22.5, group_node_order[i], rotation=180*theta/np.pi-90,
                 size=fontsize_groups, horizontalalignment='center',
                 verticalalignment='center', color=textcolor)

    if title is not None:
        plt.title(title, color=textcolor, fontsize=fontsize_title,
                  axes=axes)

    if colorbar:
        norm = plt.normalize_colors(vmin=vmin, vmax=vmax)
        sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
        sm.set_array(np.linspace(vmin, vmax))
        cb = plt.colorbar(sm, ax=axes, use_gridspec=False,
                          shrink=colorbar_size, orientation='horizontal',
                          anchor=colorbar_pos)
        cb_yticks = plt.getp(cb.ax.axes, 'yticklabels')
        cb.ax.tick_params(labelsize=fontsize_colorbar)
        plt.setp(cb_yticks, color=textcolor)

    plt_show(show)
    return fig, axes


def plot_labelled_group_connectivity_circle(yaml_fname, con, orig_labels,
                                            node_order_size=68,
                                            out_fname='circle.png', title=None,
                                            facecolor='white', fontsize_names=6,
                                            subplot=111, include_legend=False,
                                            n_lines=None, fig=None, show=True):
    '''
    Plot the connectivity circle grouped and ordered according to
    groups in the yaml input file provided.
    '''

    import matplotlib.pyplot as plt

    # read the yaml file with grouping
    if op.isfile(yaml_fname):
        with open(yaml_fname, 'r') as f:
            labels = yaml.safe_load(f)
    else:
        print('%s - File not found.' % yaml_fname)
        sys.exit()

    cortex_colors = ['m', 'b', 'y', 'c', 'r', 'g',
                     'g', 'r', 'c', 'y', 'b', 'm']

    # make list of label_names (without individual cortex locations)
    label_names = list()
    for lab in labels:
        label_names.extend(labels[lab])

    lh_labels = [name + '-lh' for name in label_names]
    rh_labels = [name + '-rh' for name in label_names]

    group_labels = list(labels.keys())
    grp_lh_labels = [name + '-lh' for name in group_labels]
    grp_rh_labels = [name + '-rh' for name in group_labels]

    # Save the plot order and create a circular layout
    node_order = list()
    node_order.extend(lh_labels[::-1])  # reverse the order
    node_order.extend(rh_labels)

    assert len(node_order) == node_order_size, 'Node order length is correct.'

    # save the group order
    group_node_order = list()
    group_node_order.extend(grp_lh_labels[::-1])
    group_node_order.extend(grp_rh_labels)

    from mne.viz.circle import circular_layout
    group_node_angles = circular_layout(group_node_order, group_node_order,
                                        start_pos=90.)

    # the respective no. of regions in each cortex
    group_bound = [len(labels[key]) for key in list(labels.keys())]
    group_bound = [0] + group_bound[::-1] + group_bound

    group_boundaries = [sum(group_bound[:i+1])
                        for i in range(len(group_bound))]

    # remove the first element of group_bound
    # make label colours such that each cortex is of one colour
    group_bound.pop(0)
    label_colors = []
    for ind, rep in enumerate(group_bound):
        label_colors += [cortex_colors[ind]] * rep
    assert len(label_colors) == len(node_order), 'Num. of colours do not match'

    # remove the last total sum of the list
    group_boundaries.pop()

    # obtain the node angles
    node_angles = circular_layout(orig_labels, node_order, start_pos=90,
                                  group_boundaries=group_boundaries)
    # the order of the node_colors must match that of orig_labels
    # therefore below reordering is necessary
    reordered_colors = [label_colors[node_order.index(orig)]
                        for orig in orig_labels]

    # Plot the graph using node_order and colours
    _plot_connectivity_circle_group_bars(con, orig_labels, n_lines=n_lines,
                              facecolor=facecolor, textcolor='black',
                              group_node_order=group_node_order,
                              group_node_angles=group_node_angles,
                              group_colors=cortex_colors,
                              fontsize_groups=6, node_angles=node_angles,
                              node_colors=reordered_colors, fontsize_names=8,
                              node_edgecolor='white', fig=fig,
                              colorbar=False, show=show, subplot=subplot,
                              title=title)

    if include_legend:
        import matplotlib.patches as mpatches
        legend_patches = [mpatches.Patch(color=col, label=key)
                          for col, key in zip(['g', 'r', 'c', 'y', 'b', 'm'],
                                              list(labels.keys()))]
        plt.legend(handles=legend_patches, loc=(0.02, 0.02), ncol=1,
                   mode=None, fontsize='small')
    if out_fname:
        plt.savefig(out_fname, facecolor='white', dpi=300)


def plot_fica_grouped_circle(yaml_fname, con, orig_labels, node_order_size,
                             out_fname='grouped_circle.png', title=None,
                             facecolor='white', fontsize_names=6,
                             subplot=111, include_legend=False,
                             n_lines=None, fig=None, show=True):
    '''
    Plot the connectivity circle grouped and ordered according to
    groups in the yaml input file provided.

    This is not specific to 'aparc' parcellation and does not split the labels
    into left and right hemispheres.

    Note: Currently requires fica_names.txt in jumeg/examples. This needs to
          be removed.
    '''

    import matplotlib.pyplot as plt

    # read the yaml file with grouping
    if op.isfile(yaml_fname):
        with open(yaml_fname, 'r') as f:
            labels = yaml.safe_load(f)
    else:
        print('%s - File not found.' % yaml_fname)
        sys.exit()

    cortex_colors = ['m', 'b', 'y', 'c', 'r', 'g']

    # make list of label_names (without individual cortex locations)
    label_names = list()
    for lab in labels:
        label_names.extend(labels[lab])

    group_labels = list(labels.keys())

    # Save the plot order and create a circular layout
    node_order = label_names

    assert len(node_order) == node_order_size, 'Node order length is correct.'

    # save the group order
    group_node_order = group_labels

    from mne.viz.circle import circular_layout
    group_node_angles = circular_layout(group_node_order, group_node_order,
                                        start_pos=75.)

    # the respective no. of regions in each cortex
    group_bound = [len(labels[key]) for key in list(labels.keys())]
    group_bound = [0] + group_bound
    # group_bound = [0] + group_bound[::-1] + group_bound

    group_boundaries = [sum(group_bound[:i+1])
                        for i in range(len(group_bound))]

    # remove the first element of group_bound
    # make label colours such that each cortex is of one colour
    group_bound.pop(0)
    label_colors = []
    for ind, rep in enumerate(group_bound):
        label_colors += [cortex_colors[ind]] * rep
    assert len(label_colors) == len(node_order), 'Num. of colours do not match'

    # remove the last total sum of the list
    group_boundaries.pop()

    # obtain the node angles
    node_angles = circular_layout(orig_labels, node_order, start_pos=90,
                                  group_boundaries=group_boundaries)
    # the order of the node_colors must match that of orig_labels
    # therefore below reordering is necessary
    reordered_colors = [label_colors[node_order.index(orig)]
                        for orig in orig_labels]

    # Plot the graph using node_order and colours
    _plot_connectivity_circle_group_bars(con, orig_labels, n_lines=n_lines,
                              facecolor=facecolor, textcolor='black',
                              group_node_order=group_node_order,
                              group_node_angles=group_node_angles,
                              group_colors=cortex_colors,
                              fontsize_groups=6, node_angles=node_angles,
                              node_colors=reordered_colors, fontsize_names=8,
                              node_edgecolor='white', fig=fig,
                              colorbar=False, show=show, subplot=subplot,
                              title=title)

    if include_legend:
        import matplotlib.patches as mpatches
        legend_patches = [mpatches.Patch(color=col, label=key)
                          for col, key in zip(['g', 'r', 'c', 'y', 'b', 'm'],
                                              list(labels.keys()))]
        plt.legend(handles=legend_patches, loc=(0.02, 0.02), ncol=1,
                   mode=None, fontsize='small')
    if out_fname:
        plt.savefig(out_fname, facecolor='white', dpi=300)
