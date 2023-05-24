#!/usr/bin/env python

"""
=============
Plot Destriux Atlas
=============

Read grouped aparc labels from yaml file.

Plot grouped connectivity circle with these grouped labels
for the Destriux atlas.
"""

import os.path as op
from jumeg import get_jumeg_path
from jumeg.connectivity import (plot_grouped_connectivity_circle,
                                generate_random_connectivity_matrix)
import yaml

grouping_yaml_fname = op.join(get_jumeg_path(), 'data/destriux_aparc_cortex_based_grouping.yaml')
label_names_yaml_fname = op.join(get_jumeg_path(), 'data/destriux_label_names.yaml')

with open(label_names_yaml_fname, 'r') as f:
    label_names = yaml.safe_load(f)['label_names']

# make a random matrix with 148 nodes
con = generate_random_connectivity_matrix(size=(148, 148), symmetric=True)
con[con < 0.5] = 0.

# plot grouped connnectivity
plot_grouped_connectivity_circle(grouping_yaml_fname, con, label_names,
                                 labels_mode=None, colorbar_pos=(0.1, 0.1),
                                 replacer_dict=None,
                                 out_fname='fig_destriux_circle.png',
                                 n_lines=50, colorbar=True)
