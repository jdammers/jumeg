#!/usr/bin/env python

"""Plot grouped connectivity circle.

Example showing how to read grouped aparc labels from yaml file and plot
grouped connectivity circle with these labels.

Author: Praveen Sripad <pravsripad@gmail.com>
        Christian Kiefer <ch.kiefer@fz-juelich.de>

"""
import os.path as op
from jumeg import get_jumeg_path
from jumeg.connectivity import (plot_grouped_connectivity_circle,
                                generate_random_connectivity_matrix)
import yaml

labels_fname = get_jumeg_path() + '/data/desikan_label_names.yaml'
yaml_cortex_fname = get_jumeg_path() + '/data/desikan_aparc_cortex_based_grouping.yaml'
yaml_cluster_fname = get_jumeg_path() + '/data/desikan_aparc_cluster_based_grouping_example.yaml'
replacer_dict_fname = get_jumeg_path() + '/data/replacer_dictionaries.yaml'

with open(labels_fname, 'r') as f:
    label_names = yaml.safe_load(f)['label_names']

with open(replacer_dict_fname, 'r') as f:
    replacer_dict = yaml.safe_load(f)['replacer_dict_aparc']

# make a random matrix with 68 nodes
con = generate_random_connectivity_matrix(size=(68, 68), symmetric=True)

# plot simple connectivity circle with cortex based grouping and colors
plot_grouped_connectivity_circle(yaml_cortex_fname, con, label_names,
                                 labels_mode='replace', replacer_dict=replacer_dict,
                                 out_fname='fig_grouped_con_circle_cortex.png',
                                 colorbar_pos=(0.1, 0.1), n_lines=10, colorbar=True,
                                 colormap='viridis')

# plot connectivity circle with cluster-based grouping but same node colors as above
plot_grouped_connectivity_circle(yaml_cluster_fname, con, label_names,
                                 labels_mode=None, replacer_dict=None,
                                 yaml_color_fname=yaml_cortex_fname,
                                 out_fname='fig_grouped_con_circle_cluster.png',
                                 colorbar_pos=(0.1, 0.1), n_lines=10, colorbar=True,
                                 colormap='viridis')
