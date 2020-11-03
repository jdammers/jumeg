'''
Script to show functionality to plot centrality indices along with
connectivity circle plot.
'''

import numpy as np
import os.path as op
import mne

from jumeg import get_jumeg_path
from jumeg.connectivity import plot_degree_circle, plot_lines_and_blobs

import bct

import matplotlib.pyplot as plt

import yaml

orig_labels_fname = get_jumeg_path() + '/data/desikan_label_names.yaml'
yaml_fname = get_jumeg_path() + '/data/desikan_aparc_cortex_based_grouping.yaml'
con_fname = get_jumeg_path() + '/data/sample,aparc-con.npy'

replacer_dict_fname = get_jumeg_path() + '/data/replacer_dictionaries.yaml'

with open(replacer_dict_fname, 'r') as f:
    replacer_dict = yaml.safe_load(f)['replacer_dict_aparc']

# real connectivity
con = np.load(con_fname)
con = con[0, :, :, 2] + con[0, :, :, 2].T
degrees = mne.connectivity.degree(con, threshold_prop=0.2)

eigenvec_centrality = bct.eigenvector_centrality_und(con)

fig, ax = plot_lines_and_blobs(con, degrees, yaml_fname,
                               orig_labels_fname, replacer_dict=replacer_dict,
                               figsize=(8, 8), show_node_labels=False,
                               show_group_labels=True, n_lines=100,
                               out_fname=None, degsize=10)
ax.set_title('Eigen vector centrality: Coh,alpha')
fig.tight_layout()

