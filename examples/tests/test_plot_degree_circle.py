#!/usr/bin/env python

'''
Plot degree values for a given set of nodes in a simple circle plot.
'''

import numpy as np
import matplotlib.pyplot as plt

import mne
from jumeg import get_jumeg_path
from jumeg.connectivity import plot_degree_circle

import bct

orig_labels_fname = get_jumeg_path() + '/data/desikan_label_names.yaml'
yaml_fname = get_jumeg_path() + '/data/desikan_aparc_cortex_based_grouping.yaml'
con_fname = get_jumeg_path() + '/data/sample,aparc-con.npy'

# con_fname = '/Users/psripad/Downloads/test_env_correlations/101716_MEG94T_121219_1310_1_c,rfDC_EO_bcc,nr,ar,1,MNE,snr1,1s,epo,aparc-con.npy'

con = np.load(con_fname)
con_ = con[0, :, :, 2] + con[0, :, :, 2].T

# test connections
# con = np.zeros((68, 68))
# con[55, 47] = 0.9  # rostralmiddlefrontal-rh - posteriorcingulate-rh
# con[46, 22] = 0.6  # lateraloccipital-lh - posteriorcingulate-lh
# con_ = con + con.T
# degrees = mne.connectivity.degree(con_, threshold=0.2)

# make a random matrix with 68 nodes
# use simple seed for reproducibility
# np.random.seed(42)
# con = np.random.random((68, 68))
# con[con < 0.6] = 0.
degrees = mne.connectivity.degree(con_, threshold=0.2)

n_per = 0.2
n_nodes = 68
max_nodes = n_nodes * (n_nodes - 1)
n_top = int(max_nodes - max_nodes * n_per)
n_thresh = np.sort(np.abs(con_).ravel())[-n_top]
x_ = con_.copy()
x_[x_ < n_thresh] = 0.
print(x_.nonzero()[0].shape)


eig_vector_centrality = bct.eigenvector_centrality_und(con_)

# for i in range(5):
#     fig = plt.figure(i, figsize=(18, 4))
#     for j in range(7):
#         con_ = con[i, :, :, j]
#         degrees = mne.connectivity.degree(con_ + con_.T, threshold=0.2)
#         print(degrees.std())
#         fig, ax = plot_degree_circle(degrees, yaml_fname, orig_labels_fname,
#                                      radsize=4, degsize=3, tight_layout=True,
#                                      fig=fig, subplot=171+j, out_fname=None,
#                                      show=False, show_group_labels=False)
#         ax.set_title('Method: %d / Band: %d' % (i, j))
#     fig.savefig('degree_circle_%d.png' % i)
#     plt.close(fig)

fig, ax = plot_degree_circle(degrees, yaml_fname, orig_labels_fname)

# fig = plt.figure()
# ax = plt.subplot(111, projection='polar')
# for i in range(5):
#      print(theta[i], orig_labels[i], reordered_colors[i])
#      ax.scatter(theta[i], radii[i], c=reordered_colors[i], s=degrees[i] * 4)
