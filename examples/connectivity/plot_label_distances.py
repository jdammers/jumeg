#!/usr/bin/env python

'''
Script to plot label distances on circle and connectome plots.
'''

import os.path as op
import numpy as np

import mne
from mne.datasets import sample

from jumeg import get_jumeg_path
from jumeg.connectivity import (get_label_distances,
                                plot_grouped_connectivity_circle)
import yaml

from nilearn import plotting

data_path = sample.data_path()
subjects_dir = data_path + '/subjects'
subject = 'sample'

parc = 'aparc_sub'

# labels_fname = get_jumeg_path() + '/data/desikan_label_names.yaml'
yaml_fname = get_jumeg_path() + '/data/desikan_%s_cortex_based_grouping.yaml' % parc

config_fname = op.join('/Users/psripad/megscripts/thesis_scripts/con_matrix_config.yaml')
with open(config_fname, 'r') as cf:
    config = yaml.safe_load(cf)

label_names = config['%s_label_names' % parc]

label_distances_fname = get_jumeg_path() + '/data/%s_label_com_distances.npy' % parc

replacer_dict = config['replacer_dict_%s' % parc]

# with open(labels_fname, 'r') as f:
#     label_names = yaml.safe_load(f)['label_names']

# load the distances matrix
con = np.load(label_distances_fname)
node_order_size = con.shape[0]

# forget long range connections, plot short neighbouring connections
neighbor_range = 10.  # cms
con[con > neighbor_range] = 0.

plot_grouped_connectivity_circle(yaml_fname, con, label_names,
                                 labels_mode='cortex_only',
                                 node_order_size=node_order_size,
                                 replacer_dict=replacer_dict,
                                 out_fname='label_com_distances_circle_%0.1f_%s.png' % (neighbor_range, parc),
                                 colorbar_pos=(0.1, 0.1),
                                 n_lines=None, colorbar=True,
                                 colormap='Reds')

# compute the distances between COM's of the labels
_, coords, _, _ = get_label_distances(subject, subjects_dir, parc=parc)

# compute the degree
degs = mne.connectivity.degree(con, threshold_prop=1)

# show the label ROIs using Nilearn plotting
fig = plotting.plot_connectome(np.zeros((node_order_size, node_order_size)),
                               coords, node_size=20, edge_threshold='99%',
                               node_color='cornflowerblue',
                               display_mode='ortho',
                               title='%s' % parc)


# fig.savefig('%s_label_distances_based_degrees.png' % parc)
fig.savefig('%s_labels_degrees.png' % parc)
