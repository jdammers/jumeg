'''
Script to show functionality to plot centrality indices along with
connectivity circle plot.
'''

import numpy as np
import os.path as op
import mne

from jumeg import get_jumeg_path
from jumeg.connectivity import plot_degree_circle, plot_lines_and_blobs

import matplotlib.pyplot as plt

orig_labels_fname = get_jumeg_path() + '/data/desikan_label_names.yaml'
yaml_fname = get_jumeg_path() + '/data/desikan_aparc_cortex_based_grouping.yaml'
con_fname = get_jumeg_path() + '/data/sample,aparc-con.npy'

# real connectivity
con = np.load(con_fname)
con = con[0, :, :, 2] + con[0, :, :, 2].T
degrees = mne.connectivity.degree(con, threshold=0.2)

import bct
eigenvec_centrality = bct.eigenvector_centrality_und(con)

fig, ax = plot_lines_and_blobs(con, degrees, yaml_fname,
                               orig_labels_fname,
                               figsize=(8, 8), show_node_labels=False,
                               show_group_labels=True, n_lines=100,
                               out_fname=None, degsize=10)
ax.set_title('Eigen vector centrality: Coh,alpha')
fig.tight_layout()

# test connections
# con = np.zeros((68, 68))
# con[55, 47] = 0.9  # rostralmiddlefrontal-rh - posteriorcingulate-rh
# con[46, 22] = 0.6  # lateraloccipital-lh - posteriorcingulate-lh
# con = con + con.T
# degrees = mne.connectivity.degree(con, threshold=0.2)
# fig, ax = plot_lines_and_blobs(con, degrees, yaml_fname, orig_labels_fname,
#                                figsize=(8, 8), node_labels=True,
#                                out_fname=None)
