#!/usr/bin/env python

'''
Test script for trying plotting routines involving degree + Connectivity
circle plots.
'''

import numpy as np
import os.path as op
import mne

from jumeg import get_jumeg_path
from jumeg.connectivity import plot_degree_circle, plot_lines_and_blobs

import matplotlib.pyplot as plt

orig_labels_fname = get_jumeg_path() + '/data/desikan_label_names.yaml'
yaml_fname = get_jumeg_path() + '/data/desikan_aparc_cortex_based_grouping.yaml'
con_fname = get_jumeg_path() + '/data/sample,aparc-con.npy'

# real connectivity
con = np.load(con_fname)
con = con[0, :, :, 2] + con[0, :, :, 2].T
degrees = mne.connectivity.degree(con, threshold=0.2)

# test known connections
# con = np.zeros((68, 68))
# con[55, 47] = 0.9  # rostralmiddlefrontal-rh - posteriorcingulate-rh
# con[46, 22] = 0.6  # lateraloccipital-lh - posteriorcingulate-lh
# con = con + con.T
# degrees = mne.connectivity.degree(con, threshold=0.2)

fig, ax = plot_lines_and_blobs(con, degrees, yaml_fname, orig_labels_fname,
                               figsize=(8, 8), node_labels=True)

###############################################################################

# n_nodes = len(degrees)
# node_order_size = 68
# fig = None
# subplot = 111
# radsize = 5.
# degsize = 10.
# alpha = 0.5
# n_lines = 50
# vmin, vmax = None, None
# node_width = None
# arrow = False
# colormap = 'Blues'
# linewidth = 1.5


# def plot_lines_and_blobs(con, degrees, yaml_fname, orig_labels_fname,
#                          node_order_size=68, fig=None, subplot=111,
#                          color='b', cmap='Blues', tight_layout=False,
#                          alpha=0.5, fontsize_groups=6, textcolor_groups='k',
#                          radsize=1., degsize=1, show_group_labels=True,
#                          linewidth=1.5, n_lines=50, node_width=None,
#                          arrow=False, out_fname='lines_and_blobs.png',
#                          vmin=None, vmax=None, figsize=None, node_labels=False,
#                          show=True):
#     '''
#     Plot connectivity circle plot with a centrality index per node shown as
#     blobs along the circulference of the circle, hence the lines and the blobs.
#     '''
#
#     import yaml
#     with open(orig_labels_fname, 'r') as f:
#         orig_labels = yaml.safe_load(f)['label_names']
#
#     n_nodes = len(degrees)
#
#     assert n_nodes == len(orig_labels), 'Mismatch in node names and number.'
#     assert n_nodes == len(con), 'Mismatch in n_nodes for con and degrees.'
#
#     # read the yaml file with grouping of the various nodes
#     if op.isfile(yaml_fname):
#         with open(yaml_fname, 'r') as f:
#             labels = yaml.safe_load(f)
#     else:
#         print('%s - File not found.' % yaml_fname)
#         sys.exit()
#
#     # make list of label_names (without individual cortex locations)
#     label_names = [list(lab.values())[0] for lab in labels]
#     label_names = [la for l in label_names for la in l]
#
#     lh_labels = [name + '-lh' for name in label_names]
#     rh_labels = [name + '-rh' for name in label_names]
#
#     # save the plot order
#     node_order = list()
#     node_order.extend(lh_labels[::-1])  # reverse the order
#     node_order.extend(rh_labels)
#     assert len(node_order) == node_order_size, 'Node order length is correct.'
#
#     # the respective no. of regions in each cortex
#     # yaml fix order change
#     group_bound = [len(list(key.values())[0]) for key in labels]
#     group_bound = [0] + group_bound[::-1] + group_bound
#     group_boundaries = [sum(group_bound[:i+1]) for i in range(len(group_bound))]
#
#     # remove the first element of group_bound
#     # make label colours such that each cortex is of one colour
#     group_bound.pop(0)
#
#     # remove the last total sum of the list
#     group_boundaries.pop()
#
#     from mne.viz.circle import circular_layout
#     node_angles = circular_layout(orig_labels, node_order, start_pos=90,
#                                   group_boundaries=group_boundaries)
#
#     if node_width is None:
#         # widths correspond to the minimum angle between two nodes
#         dist_mat = node_angles[None, :] - node_angles[:, None]
#         dist_mat[np.diag_indices(n_nodes)] = 1e9
#         node_width = np.min(np.abs(dist_mat))
#     else:
#         node_width = node_width * np.pi / 180
#
#     # prepare group label positions
#     group_labels = [list(lab.keys())[0] for lab in labels]
#     grp_lh_labels = [name + '-lh' for name in group_labels]
#     grp_rh_labels = [name + '-rh' for name in group_labels]
#     all_group_labels = grp_lh_labels + grp_rh_labels
#
#     # save the group order
#     group_node_order = list()
#     group_node_order.extend(grp_lh_labels[::-1])
#     group_node_order.extend(grp_rh_labels)
#     n_groups = len(group_node_order)
#
#     group_node_angles = circular_layout(group_node_order, group_node_order,
#                                         start_pos=90.)
#
#     import matplotlib.pyplot as plt
#     import matplotlib.path as m_path
#     import matplotlib.patches as m_patches

#     if fig is None:
#         fig = plt.figure(figsize=figsize)
#
#     # use a polar axes
#     if not isinstance(subplot, tuple):
#         subplot = (subplot,)
#
#     ax = plt.subplot(*subplot, polar=True, facecolor='white')
#
#     cortex_colors = ['m', 'b', 'y', 'c', 'r', 'g',
#                      'g', 'r', 'c', 'y', 'b', 'm']
#
#     label_colors = []
#     for ind, rep in enumerate(group_bound):
#         label_colors += [cortex_colors[ind]] * rep
#     assert len(label_colors) == len(node_order), 'Number of colours do not match'
#
#     # the order of the node_colors must match that of orig_labels
#     # therefore below reordering is necessary
#     reordered_colors = [label_colors[node_order.index(orig)]
#                         for orig in orig_labels]
#
#     # first plot the circle showing the degree
#     theta = np.deg2rad(node_angles)
#     radii = np.ones(len(node_angles)) * radsize
#     c = ax.scatter(theta, radii, c=reordered_colors, s=degrees * degsize,
#                    cmap=cmap, alpha=alpha)
#
#     ax.grid(False)
#     ax.set_xticks([])
#     ax.set_yticks([])
#     ax.spines['polar'].set_visible(False)
#
#     # handle 1D and 2D connectivity information
#     if con.ndim == 1:
#         if indices is None:
#             raise ValueError('indices has to be provided if con.ndim == 1')
#     elif con.ndim == 2:
#         if con.shape[0] != n_nodes or con.shape[1] != n_nodes:
#             raise ValueError('con has to be 1D or a square matrix')
#         # we use the lower-triangular part
#         indices = np.tril_indices(n_nodes, -1)
#         con = con[indices]
#     else:
#         raise ValueError('con has to be 1D or a square matrix')
#
#     # Draw lines between connected nodes, only draw the strongest connections
#     if n_lines is not None and len(con) > n_lines:
#         con_thresh = np.sort(np.abs(con).ravel())[-n_lines]
#     else:
#         con_thresh = 0.
#
#     # get the connections which we are drawing and sort by connection strength
#     # this will allow us to draw the strongest connections first
#     con_abs = np.abs(con)
#     con_draw_idx = np.where(con_abs >= con_thresh)[0]
#
#     con = con[con_draw_idx]
#     con_abs = con_abs[con_draw_idx]
#     indices = [ind[con_draw_idx] for ind in indices]
#
#     # now sort them
#     sort_idx = np.argsort(con_abs)
#     con_abs = con_abs[sort_idx]
#     con = con[sort_idx]
#     indices = [ind[sort_idx] for ind in indices]
#
#     # get the colormap
#     if isinstance(cmap, str):
#         colormap = plt.get_cmap(cmap)
#
#     # Get vmin vmax for color scaling
#     if vmin is None:
#         vmin = np.min(con[np.abs(con) >= con_thresh])
#     if vmax is None:
#         vmax = np.max(con)
#     vrange = vmax - vmin
#
#     # We want to add some "noise" to the start and end position of the
#     # edges: We modulate the noise with the number of connections of the
#     # node and the connection strength, such that the strongest connections
#     # are closer to the node center
#     nodes_n_con = np.zeros((n_nodes), dtype=np.int)
#     for i, j in zip(indices[0], indices[1]):
#         nodes_n_con[i] += 1
#         nodes_n_con[j] += 1
#
#     # initialize random number generator so plot is reproducible
#     rng = np.random.mtrand.RandomState(seed=0)
#
#     n_con = len(indices[0])
#     noise_max = 0.25 * node_width
#     start_noise = rng.uniform(-noise_max, noise_max, n_con)
#     end_noise = rng.uniform(-noise_max, noise_max, n_con)
#
#     nodes_n_con_seen = np.zeros_like(nodes_n_con)
#     for i, (start, end) in enumerate(zip(indices[0], indices[1])):
#         nodes_n_con_seen[start] += 1
#         nodes_n_con_seen[end] += 1
#
#         start_noise[i] *= ((nodes_n_con[start] - nodes_n_con_seen[start]) /
#                            float(nodes_n_con[start]))
#         end_noise[i] *= ((nodes_n_con[end] - nodes_n_con_seen[end]) /
#                          float(nodes_n_con[end]))
#
#     # scale connectivity for colormap (vmin<=>0, vmax<=>1)
#     con_val_scaled = (con - vmin) / vrange
#
#     # convert to radians for below code
#     node_angles = node_angles * np.pi / 180
#     # Finally, we draw the connections
#     for pos, (i, j) in enumerate(zip(indices[0], indices[1])):
#         # Start point
#         t0, r0 = node_angles[i], radsize
#
#         # End point
#         if arrow:
#             # make shorter to accomodate arrowhead
#             t1, r1 = node_angles[j], radsize - 1.
#         else:
#             t1, r1 = node_angles[j], radsize
#
#         # Some noise in start and end point
#         t0 += start_noise[pos]
#         t1 += end_noise[pos]
#
#         verts = [(t0, r0), (t0, radsize/2.), (t1, radsize/2.), (t1, r1)]
#         codes = [m_path.Path.MOVETO, m_path.Path.CURVE4, m_path.Path.CURVE4,
#                  m_path.Path.LINETO]
#         path = m_path.Path(verts, codes)
#
#         color = colormap(con_val_scaled[pos])
#
#         if arrow:
#             # add an arrow to the patch
#             patch = m_patches.FancyArrowPatch(path=path,
#                                               arrowstyle=arrowstyle,
#                                               fill=False, edgecolor=color,
#                                               mutation_scale=10,
#                                               linewidth=linewidth, alpha=1.)
#         else:
#             patch = m_patches.PathPatch(path, fill=False, edgecolor=color,
#                                         linewidth=linewidth, alpha=1.)
#
#         ax.add_patch(patch)
#
#     # draw node labels
#     if node_labels:
#         angles_deg = 180 * node_angles / np.pi
#         for name, angle_rad, angle_deg in zip(orig_labels, node_angles,
#                                               angles_deg):
#             if angle_deg >= 270:
#                 ha = 'left'
#             else:
#                 # Flip the label, so text is always upright
#                 angle_deg += 180
#                 ha = 'right'
#
#             ax.text(angle_rad, radsize + 0.2, name, size=8,
#                     rotation=angle_deg, rotation_mode='anchor',
#                     horizontalalignment=ha, verticalalignment='center',
#                     color='k')
#
#     if show:
#         plt.show()
#
#     if tight_layout:
#         fig.tight_layout()
#
#     if out_fname:
#         fig.savefig(out_fname, dpi=300.)
#
#     return fig, ax
#
