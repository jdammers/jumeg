#!/usr/bin/env python3

"""
==========
Plot labelled connectivity circle
==========

Example exposing the plot_labelled_group_connectivity_circle function.

Author: Praveen Sripad <pravsripad@gmail.com>

"""
import os.path as op
from jumeg.connectivity import (plot_labelled_group_connectivity_circle,
                                generate_random_connectivity_matrix)
from jumeg import get_jumeg_path
import yaml

# load the yaml grouping of Freesurfer labels
yaml_fname = op.join(get_jumeg_path(), 'data/rsn_desikan_aparc_cortex_grouping.yaml')
label_names_yaml_fname = op.join(get_jumeg_path(), 'data/desikan_label_names.yaml')

with open(label_names_yaml_fname, 'r') as f:
    label_names = yaml.safe_load(f)['label_names']

# make a random matrix with 68 nodes
con = generate_random_connectivity_matrix(size=(68, 68), symmetric=True)

# plotting within a subplot
plot_labelled_group_connectivity_circle(yaml_fname, con, label_names,
                                        out_fname='fig_rsn_circle.png',
                                        show=False, n_lines=20,
                                        fontsize_names=6,
                                        title='test RSN circ labels')
