#!/usr/bin/env python

'''
Plot connectivity on a glass brain using 'plot_connectome' function from
Nilearn (https://nilearn.github.io/).

Author: Praveen Sripad <pravsripad@gmail.com>
'''

import numpy as np
import mne
from mne.datasets import sample

from nilearn import plotting
import nibabel as nib

import matplotlib.pyplot as plt

data_path = sample.data_path()
subjects_dir = data_path + '/subjects'
subject = 'fsaverage'

aparc = mne.read_labels_from_annot(subject, subjects_dir=subjects_dir, parc='aparc')

# nodes in one hemisphere can be plotted as well
aparc_lh = [lab for lab in aparc if lab.hemi == 'lh']

coords = []

# plot 10 nodes from left hemisphere only for better viz
for lab in aparc_lh[:10]:
    if lab.name is 'unknown-lh':
        continue
    # get the center of mass
    com = lab.center_of_mass('fsaverage')
    # obtain mni coordinated to the vertex from left hemi
    coords_ = mne.vertex_to_mni(com, hemis=0, subject=subject, subjects_dir=subjects_dir)[0]
    coords.append(coords_)

n_nodes = np.array(coords).shape[0]

# make a random connectivity matrix
con = np.random.random((n_nodes, n_nodes))
con[np.diag_indices(5)] = 0.
con[np.triu_indices(5, k=1)] = 0.
con += con.T
con[con < 0.6] = 0.

# plot the connectome on a glass brain background
plotting.plot_connectome(con, coords)
plt.show()
