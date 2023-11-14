#!/usr/bin/env python

'''
Example showing how to read grouped aparc labels from yaml file and plot
grouped connectivity circle with these labels.

Author: Praveen Sripad <pravsripad@gmail.com>
'''

import numpy as np
from jumeg import get_jumeg_path
from jumeg.connectivity.con_viz import plot_connectivity_circle
from jumeg.connectivity.con_utils import group_con_matrix_by_lobe
from jumeg.connectivity import plot_grouped_connectivity_circle

import yaml

labels_fname = get_jumeg_path() + '/data/desikan_label_names.yaml'
yaml_fname = get_jumeg_path() + '/data/desikan_aparc_cortex_based_grouping.yaml'
replacer_dict_fname = get_jumeg_path() + '/data/replacer_dictionaries.yaml'
lobes_fname = get_jumeg_path() + '/data/lobes_grouping.yaml'

with open(labels_fname, 'r') as f:
    label_names = yaml.safe_load(f)['label_names']

with open(replacer_dict_fname, 'r') as f:
    replacer_dict = yaml.safe_load(f)['replacer_dict_aparc']

# make a random matrix with 68 nodes
# use simple seed for reproducibility
np.random.seed(42)
con = np.random.random((68, 68))
con[np.diag_indices(68)] = 0.
con[con < 0.5] = 0.

# indices = (np.array((1, 2, 3)), np.array((5, 6, 7)))
# plot_grouped_connectivity_circle(yaml_fname, lcon, label_names,
#                                  labels_mode='cortex_only',
#                                  replacer_dict=replacer_dict,
#                                  out_fname='example_grouped_con_circle.png',
#                                  colorbar_pos=(0.1, 0.1),
#                                  n_lines=10, colorbar=True,
#                                  colormap='viridis')

# lcon = np.random.random((12, 12))
# lcon = np.round(lcon, 2)
# lcon[np.tril_indices(12, k=-1)] = 0
# lcon = lcon + lcon.T


lcon, full_grouping_labels = group_con_matrix_by_lobe(con, label_names, yaml_fname)

# lna = ['occipital', 'parietal', 'temporal', 'cingulate', 'insula', 'frontal']
# label_names = [lab + '-lh' for lab in lna]
# label_names.extend([lab + '-rh' for lab in lna])

# plot_connectivity_circle(lcon, label_names, n_lines=20,
#                          arrow=False, ignore_diagonal=False)

# plot_connectivity_circle(cau, label_names, vmin=0, vmax=1, colorbar=True,
#                          title='test', fig=None, subplot=111, interactive=False,
#                          show=True, arrow=True, arrowstyle='->,head_length=0.7,head_width=0.4',
#                          ignore_diagonal=False)
#
plot_grouped_connectivity_circle(lobes_fname, lcon, full_grouping_labels, n_lines=25,
                              title='test', colormap='magma_r', colorbar=True,
                              colorbar_pos=(-0.25, 0.05), replacer_dict=None,
                              arrowstyle='->,head_length=0.7,head_width=0.4',
                              figsize=(9.1, 6), vmin=0.75, ignore_diagonal=False,
                              node_order_size=12,
                              show=True)
