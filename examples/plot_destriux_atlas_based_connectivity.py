#!/usr/bin/env python

'''
Read grouped aparc labels from yaml file.
Plot grouped connectivity circle with these grouped labels
for the Destriux atlas.
'''

import numpy as np
import mne
from jumeg import get_jumeg_path
from jumeg.connectivity import plot_grouped_connectivity_circle
from mne.viz import circular_layout, plot_connectivity_circle
import yaml

grouping_yaml_fname = get_jumeg_path() + '/data/destriux_aparc_cortex_based_grouping.yaml'
label_names_yaml_fname = get_jumeg_path() + '/data/destriux_label_names.yaml'

with open(label_names_yaml_fname, 'r') as f:
    label_names = yaml.safe_load(f)['label_names']

# make a random matrix with 68 nodes
# use simple seed for reproducibility
np.random.seed(42)
con = np.random.random((148, 148))
con[con < 0.5] = 0.

# plot grouped connnectivity
plot_grouped_connectivity_circle(grouping_yaml_fname, con, label_names,
                                 labels_mode=None, node_order_size=148,
                                 colorbar_pos=(0.1, 0.1), out_fname='destriux_circle.png',
                                 n_lines=50, colorbar=True)
