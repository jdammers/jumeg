#!/usr/bin/env python

""" Visualization functions for connectivity analysis. """

import sys
import os.path as op

from itertools import cycle
from functools import partial

import numpy as np
import scipy as sci

import mne
from mne.viz.utils import plt_show
from mne.externals.six import string_types
from mne.viz.circle import (circular_layout, _plot_connectivity_circle_onpick)

import yaml


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


def plot_connectivity_circle(con, node_names, indices=None, n_lines=None,
                             node_angles=None, node_width=None,
                             node_colors=None, facecolor='black',
                             textcolor='white', node_edgecolor='black',
                             linewidth=1.5, colormap='hot', vmin=None,
                             vmax=None, colorbar=True, title=None,
                             colorbar_size=0.2, colorbar_pos=(-0.3, 0.1),
                             fontsize_title=12, fontsize_names=8,
                             fontsize_colorbar=8, padding=6.,
                             fig=None, subplot=111, interactive=True,
                             node_linewidth=2., show=True, arrow=False,
                             arrowstyle='->,head_length=3,head_width=3'):
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
        indices = np.tril_indices(n_nodes, -1)
        con = con[indices]
    else:
        raise ValueError('con has to be 1D or a square matrix')

    # get the colormap
    if isinstance(colormap, string_types):
        colormap = plt.get_cmap(colormap)

    # Make figure background the same colors as axes
    if fig is None:
        fig = plt.figure(figsize=(8, 8), facecolor=facecolor)

    # Use a polar axes
    if not isinstance(subplot, tuple):
        subplot = (subplot,)
    axes = plt.subplot(*subplot, polar=True, axisbg=facecolor)

    # No ticks, we'll put our own
    plt.xticks([])
    plt.yticks([])

    # Set y axes limit, add additional space if requested
    plt.ylim(0, 10 + padding)

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
                  axes=axes)

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
                                     labels_mode=None,
                                     node_order_size=68, indices=None,
                                     out_fname='circle.png', title=None,
                                     subplot=111, include_legend=False,
                                     n_lines=None, fig=None, show=True,
                                     vmin=None, vmax=None, colormap='hot',
                                     colorbar=False, colorbar_pos=(-0.3, 0.1),
                                     bbox_inches=None, tight_layout=None, **kwargs):
    '''
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
    '''
    import matplotlib.pyplot as plt
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
        # label_names.extend(labels[lab])
        label_names += lab.values()[0]  # yaml order fix

    lh_labels = [name + '-lh' for name in label_names]
    rh_labels = [name + '-rh' for name in label_names]

    # Save the plot order and create a circular layout
    node_order = list()
    node_order.extend(lh_labels[::-1])  # reverse the order
    node_order.extend(rh_labels)

    assert len(node_order) == node_order_size, 'Node order length is correct.'

    # the respective no. of regions in each cortex
    # group_bound = [len(labels[key]) for key in labels.keys()]
    group_bound = [len(key.values()[0]) for key in labels]  # yaml order fix
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

    node_angles = circular_layout(orig_labels, node_order, start_pos=90,
                                  group_boundaries=group_boundaries)

    # the order of the node_colors must match that of orig_labels
    # therefore below reordering is necessary
    reordered_colors = [label_colors[node_order.index(orig)]
                        for orig in orig_labels]

    # labels mode decides the labels printed for each of the nodes
    if labels_mode is 'blank':
        # show nothing, only the empty circle plot
        my_labels = ['' for orig in orig_labels]
    elif labels_mode is 'cortex_only':
        # show only the names of cortex areas on one representative node
        replacer = dict({'caudalanteriorcingulate': 'cingulate',
                         'insula': 'insula', 'parstriangularis': 'frontal',
                         'precuneus': 'parietal', 'lingual': 'occipital',
                         'transversetemporal': 'temporal'})

        replaced_labels = []
        for myl in orig_labels:
            if myl.split('-lh')[0] in replacer.keys():
                replaced_labels.append(replacer[myl.split('-lh')[0]] + '-lh')
            elif myl.split('-rh')[0] in replacer.keys():
                replaced_labels.append(replacer[myl.split('-rh')[0]] + '-rh')
            else:
                replaced_labels.append('')
        my_labels = replaced_labels
    else:
        # show all the node labels as originally given
        my_labels = orig_labels

    # Plot the graph using node_order and colours
    # orig_labels is the order of nodes in the con matrix (important)
    fig, axes = plot_connectivity_circle(con, my_labels, n_lines=n_lines,
                                         facecolor='white', textcolor='black',
                                         node_angles=node_angles, colormap=colormap,
                                         node_colors=reordered_colors,
                                         node_edgecolor='white', fig=fig,
                                         fontsize_title=12,
                                         fontsize_names=10, padding=6.,
                                         vmax=vmax, vmin=vmin, colorbar_size=0.2,
                                         colorbar_pos=colorbar_pos,
                                         colorbar=colorbar, show=show,
                                         subplot=subplot, indices=indices,
                                         title=title, **kwargs)

    if include_legend:
        import matplotlib.patches as mpatches
        # yaml order fix
        legend_patches = [mpatches.Patch(color=col, label=llab.keys()[0])
                          for col, llab in zip(['g', 'r', 'c', 'y', 'b', 'm'],
                                               labels)]
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


def plot_generic_grouped_circle(yaml_fname, con, orig_labels,
                                node_order_size,
                                out_fname='circle.png', title=None,
                                subplot=111, include_legend=False,
                                n_lines=None, fig=None, show=True,
                                vmin=None, vmax=None,
                                colorbar=False, **kwargs):
    '''
    Plot the connectivity circle grouped and ordered according to
    groups in the yaml input file provided. This is not specific to aparc and
    does not automatically split the labels into left and right hemispheres.

    orig_labels : list of str
        Label names in the order as appears in con.

    NOTE: The order of entries in the yaml file is not preserved.
    '''
    import matplotlib.pyplot as pl
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
    plot_connectivity_circle(con, orig_labels, n_lines=n_lines,
                             facecolor='white', textcolor='black',
                             node_angles=node_angles,
                             node_colors=reordered_colors,
                             node_edgecolor='white', fig=fig,
                             fontsize_names=8, vmax=vmax, vmin=vmin,
                             colorbar_size=0.2, colorbar_pos=(-0.3, 0.1),
                             colorbar=colorbar, show=show, subplot=subplot,
                             title=title, **kwargs)

    if include_legend:
        import matplotlib.patches as mpatches
        legend_patches = [mpatches.Patch(color=col, label=key)
                          for col, key in zip(['g', 'r', 'c', 'y', 'b', 'm'],
                                              labels.keys())]
        pl.legend(handles=legend_patches, loc=(0.02, 0.02), ncol=1,
                  mode=None, fontsize='small')
    if out_fname:
        pl.savefig(out_fname, facecolor='white', dpi=600)


def plot_grouped_causality_circle(caus, yaml_fname, label_names, n_lines=None,
                                  labels_mode='cortex_only', title='Causal Metric',
                                  out_fname='causality_circle.png', colormap='Blues',
                                  figsize=(10, 6), show=False, colorbar=False):

    con_l = np.tril(caus, k=-1)
    con_u = np.triu(caus, k=1).T

    vmin = np.min([np.min(con_l[con_l != 0]), np.min(con_u[con_u != 0])])
    vmax = np.max([np.max(con_l[con_l != 0]), np.max(con_u[con_u != 0])])

    import matplotlib.pyplot as plt
    fig = plt.figure(num=None, figsize=figsize)
    conds = [con_l, con_u]

    if colorbar:
        colorbar = [False, True]

    for ii, cond in enumerate(conds):
        plot_grouped_connectivity_circle(yaml_fname, conds[ii],
                                         label_names,
                                         out_fname=out_fname,
                                         labels_mode=labels_mode,
                                         show=show, title=title,
                                         fig=fig, subplot=(1, 1, 1),
                                         vmin=vmin, vmax=vmax,
                                         n_lines=n_lines, colormap=colormap,
                                         colorbar=colorbar[ii], arrow=True)
