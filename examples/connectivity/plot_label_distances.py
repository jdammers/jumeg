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

parc = 'aparc'

yaml_fname = get_jumeg_path() + '/data/desikan_%s_cortex_based_grouping.yaml' % parc
label_distances_fname = get_jumeg_path() + '/data/desikan_%s_label_com_distances.npy' % parc

labels_fname = get_jumeg_path() + '/data/desikan_label_names.yaml'
with open(labels_fname, 'r') as f:
    label_names = yaml.safe_load(f)['label_names']

replacer_dict_fname = get_jumeg_path() + '/data/replacer_dictionaries.yaml'
with open(replacer_dict_fname, 'r') as f:
    replacer_dict = yaml.safe_load(f)['replacer_dict_aparc']

# load the distances matrix
con = np.load(label_distances_fname)

# forget long range connections, plot short neighbouring connections
neighbor_range = 30.  # millimetres 
con[con > neighbor_range] = 0.

plot_grouped_connectivity_circle(yaml_fname, con, label_names,
                                 labels_mode='cortex_only',
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
