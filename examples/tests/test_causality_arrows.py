#!/usr/bin/env python3

"""
=============
Test Causality Arrows Plotting
=============

Example showing how to plot a causality matrix on a circle plot.
"""

import numpy as np

from jumeg import get_jumeg_path
from jumeg.connectivity import plot_grouped_causality_circle
import yaml

# load the yaml grouping of Freesurfer labels
yaml_fname = get_jumeg_path() + '/data/desikan_aparc_cortex_based_grouping.yaml'
labels_fname = get_jumeg_path() + '/data/desikan_label_names.yaml'
replacer_dict_fname = get_jumeg_path() + '/data/replacer_dictionaries.yaml'

with open(labels_fname, 'r') as f:
    label_names = yaml.safe_load(f)['label_names']

with open(replacer_dict_fname, 'r') as f:
    replacer_dict = yaml.safe_load(f)['replacer_dict_aparc']

# make a random causality matrix
n_nodes = 68  # currently needs to be always this number
caus = np.random.random((n_nodes, n_nodes))
caus[np.diag_indices_from(caus)] = 0.
caus[caus < 0.7] = 0.

plot_grouped_causality_circle(caus, yaml_fname, label_names, n_lines=10,
                              labels_mode='replace', replacer_dict=replacer_dict,
                              out_fname='causality_circle.png',
                              colormap='Blues', colorbar=True,
                              figsize=(6, 6), show=False,
                              arrowstyle='->,head_length=1,head_width=1')
