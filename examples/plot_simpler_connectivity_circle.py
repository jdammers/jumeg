#!/usr/bin/env python

'''
Read grouped aparc labels from yaml file.
Plot grouped connectivity circle with these grouped labels.

TESTING script for the main function in con_viz.py.
'''

import os.path as op
import numpy as np
from jumeg import get_jumeg_path
from jumeg.connectivity import plot_grouped_connectivity_circle
import yaml


def test_plot_grouped_connectivity_circle(yaml_fname, con, con2, orig_labels,
                                     labels_mode=None,
                                     node_order_size=68, indices=None,
                                     out_fname='circle.png', title=None,
                                     subplot=111, include_legend=False,
                                     n_lines=None, fig=None, show=True,
                                     vmin=None, vmax=None, colormap='hot',
                                     colorbar=False, colorbar_pos=(-0.3, 0.1)):
    '''
    Plot the connectivity circle grouped and ordered according to
    groups in the yaml input file provided.
    orig_labels : list of str
        Label names in the order as appears in con.
    labels_mode : str | None
        'blank' mode plots no labels on the circle plot,
        'cortex_only' plots only the name of the cortex on one representative
        node and None plots all of the orig_label names provided.
    '''
    import matplotlib.pyplot as plt
    # read the yaml file with grouping
    if op.isfile(yaml_fname):
        with open(yaml_fname, 'r') as f:
            labels = yaml.load(f)
    else:
        print('%s - File not found.' % yaml_fname)
        sys.exit()

    cortex_colors = ['m', 'b', 'y', 'c', 'r', 'g',
                     'g', 'r', 'c', 'y', 'b', 'm']

    # make list of label_names (without individual cortex locations)
    label_names = list()
    for lab in labels:
        # label_names.extend(labels[lab])
        #NOTE modified yaml file to return list of dicts which preserves order
        label_names += list(lab.values())[0]  # lab is a dict now

    lh_labels = [name + '-lh' for name in label_names]
    rh_labels = [name + '-rh' for name in label_names]

    # Save the plot order and create a circular layout
    node_order = list()
    node_order.extend(lh_labels[::-1])  # reverse the order
    node_order.extend(rh_labels)

    assert len(node_order) == node_order_size, 'Node order length is correct.'

    # the respective no. of regions in each cortex
    # group_bound = [len(labels[key]) for key in labels.keys()]
    group_bound = [len(list(key.values())[0]) for key in labels]  # yaml fix order change
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

    # Plot the graph using node_order and colours
    # orig_labels is the order of nodes in the con matrix (important)
    from mne.viz import plot_connectivity_circle
    fig, axes = plot_connectivity_circle(con, my_labels, n_lines=n_lines,
                                         facecolor='white', textcolor='black',
                                         node_angles=node_angles, colormap=colormap,
                                         node_colors=reordered_colors,
                                         node_edgecolor='white', fig=fig,
                                         fontsize_names=10, padding=10.,
                                         vmax=vmax, vmin=vmin,
                                         colorbar_size=0.2, colorbar_pos=colorbar_pos,
                                         colorbar='Blues', show=show, subplot=subplot,
                                         indices=indices, title=title)

    fig, axes = plot_connectivity_circle(con2, my_labels, n_lines=n_lines,
                                         facecolor='white', textcolor='black',
                                         node_angles=node_angles, colormap=colormap,
                                         node_colors=reordered_colors,
                                         node_edgecolor='white', fig=fig,
                                         fontsize_names=10, padding=10.,
                                         vmax=vmax, vmin=vmin,
                                         colorbar_size=0.2, colorbar_pos=colorbar_pos,
                                         colorbar='Red', show=show, subplot=subplot,
                                         indices=indices, title=title)


    if include_legend:
        import matplotlib.patches as mpatches
        # legend_patches = [mpatches.Patch(color=col, label=key)
        #                   for col, key in zip(['g', 'r', 'c', 'y', 'b', 'm'],
        #                                       labels.keys())]
        #NOTE yaml change to preserve order
        legend_patches = [mpatches.Patch(color=col, label=list(llab.keys())[0])
                          for col, llab in zip(['g', 'r', 'c', 'y', 'b', 'm'],
                                               labels)]
        # plt.legend(handles=legend_patches, loc=(0.02, 0.02), ncol=1,
        #            mode=None, fontsize='small')
        plt.legend(handles=legend_patches, loc=4, ncol=1,
                   mode=None, fontsize='medium')

    # fig.tight_layout()
    if out_fname:
        fig.savefig(out_fname, facecolor='white', dpi=300)


labels_fname = get_jumeg_path() + '/data/desikan_label_names.yaml'
yaml_fname = get_jumeg_path() + '/data/desikan_aparc_cortex_based_grouping.yaml'

with open(labels_fname, 'r') as f:
    label_names = yaml.load(f)['label_names']

# make a random matrix with 68 nodes
# use simple seed for reproducibility
np.random.seed(42)
con = np.random.random((68, 68))
con2 = np.random.random((68, 68))
# con = np.zeros((68, 68))  # to check what happens when zeros con are provided
con[con < 0.6] = 0.
con2[con2 < 0.6] = 0.
indices = (np.array((1, 2, 3)), np.array((5, 6, 7)))

test_plot_grouped_connectivity_circle(yaml_fname, con, con2, label_names,
                                 n_lines=100, colorbar=True,
                                 labels_mode='cortex_only',
                                 colorbar_pos=(0.1, 0.1),
                                 include_legend=True, show=False)

# plot_grouped_connectivity_circle(yaml_fname, con, con2, label_names,
#                                  n_lines=10, colorbar=True,
#                                  labels_mode='cortex_only',
#                                  colorbar_pos=(0.1, 0.1),
#                                  include_legend=True, show=False,
#                                  bbox_inches='tight', tight_layout=True)
