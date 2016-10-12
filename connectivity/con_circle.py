'''
Circle connectivity plot with external labels ring with group names.

Note: This code is based on the circle graph example by Nicolas P. Rougier
http://www.labri.fr/perso/nrougier/coding/
and is a slightly modified version of the mne-python function
http://martinos.org/mne/stable/generated/mne.viz.plot_connectivity_circle.html.

'''

import numpy as np
import os.path as op
import yaml
import pickle

import matplotlib.pyplot as plt

from mne.viz import circular_layout
from mne.viz.utils import plt_show
from mne.viz.circle import _plot_connectivity_circle_onpick
from mne.externals.six import string_types

from ..jumeg_utils import get_jumeg_path


def _plot_connectivity_circle(con, node_names, indices=None, n_lines=None,
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
                              fig=None, subplot=111, interactive=False,
                              node_linewidth=2., show=True):
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
    interactive : bool
        When enabled, left-click on a node to show only connections to that
        node. Right-click shows all connections.
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
    """
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

    # Add callback for interaction
    if interactive:
        callback = partial(_plot_connectivity_circle_onpick, fig=fig,
                           axes=axes, indices=indices, n_nodes=n_nodes,
                           node_angles=node_angles)

        fig.canvas.mpl_connect('button_press_event', callback)

    plt_show(show)
    return fig, axes


def plot_labelled_group_connectivity_circle(yaml_fname, con, node_order_size=68,
                                            out_fname='circle.png', title=None,
                                            facecolor='white', fontsize_names=6,
                                            subplot=111, include_legend=False,
                                            n_lines=None, fig=None, show=True):
    '''
    Plot the connectivity circle grouped and ordered according to
    groups in the yaml input file provided.
    '''

    # load the label names in the original order (aparc)
    labels_fname = get_jumeg_path() + '/examples/label_names.list'
    with open(labels_fname, 'r') as f:
        orig_labels = pickle.load(f)

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

    group_labels = labels.keys()
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
                                        start_pos=75.)

    # the respective no. of regions in each cortex
    group_bound = [len(labels[key]) for key in labels.keys()]
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
    _plot_connectivity_circle(con, orig_labels, n_lines=n_lines,
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
                                              labels.keys())]
        plt.legend(handles=legend_patches, loc=(0.02, 0.02), ncol=1,
                   mode=None, fontsize='small')
    if out_fname:
        plt.savefig(out_fname, facecolor='white', dpi=300)


def plot_fica_grouped_circle(yaml_fname, con, node_order_size,
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

    # load the label names in the original order
    # TODO remove empty lines / strings if any
    # TODO remove below code from here
    labels_fname = get_jumeg_path() + '/examples/fica_names.txt'
    with open(labels_fname, 'r') as f:
        orig_labels = [line.rstrip('\n') for line in f]

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

    group_labels = labels.keys()

    # Save the plot order and create a circular layout
    node_order = label_names

    assert len(node_order) == node_order_size, 'Node order length is correct.'

    # save the group order
    group_node_order = group_labels

    from mne.viz.circle import circular_layout
    group_node_angles = circular_layout(group_node_order, group_node_order,
                                        start_pos=75.)

    # the respective no. of regions in each cortex
    group_bound = [len(labels[key]) for key in labels.keys()]
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
    _plot_connectivity_circle(con, orig_labels, n_lines=n_lines,
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
                                              labels.keys())]
        plt.legend(handles=legend_patches, loc=(0.02, 0.02), ncol=1,
                   mode=None, fontsize='small')
    if out_fname:
        plt.savefig(out_fname, facecolor='white', dpi=300)
