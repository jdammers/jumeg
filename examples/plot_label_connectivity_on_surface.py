#!/usr/bin/env python2

'''
Plot label based connectivity on the brain surface by choosing a seed label
from an all-to-all labels connectivity matrix.

1. Prepare a connectivity matrix.
2. Choose label of interest to serve as seed.
3. Plot the connectivity between the seed label and all other labels on the
   brain surface with the connectivity values colour coded.

'''

import numpy as np

import mne
from mne.datasets import sample

from surfer import Brain
from jumeg.jumeg_utils import get_jumeg_path, get_cmap
import time

data_path = sample.data_path()
subject = 'sample'
subjects_dir = data_path + '/subjects'

# load labels and choose seed label
labels = mne.read_labels_from_annot('sample', parc='aparc',
                                    subjects_dir=subjects_dir)
label_names = [lab.name for lab in labels]
seed_label = 'posteriorcingulate-lh'
# seed_label = 'postcentral-lh'
seed_idx = label_names.index(seed_label)

# load the connectivity matrix
con_fname = 'combined_aparc,5meth,EO,1s,orig.npy'
con_ = np.load(con_fname)

con = con_[:, 0, :, :, 2].mean(axis=0)  # mean alpha coherence
seed_con = con[seed_idx]

# set the colors
# get colours for connectivity values in bins of 0.05
N = 20
con_cmap = get_cmap(N)
bins = np.linspace(0, 1, N)
seed_cols = np.digitize(seed_con, bins=bins)  # assign con values to bins
# for i, s in enumerate(seed_con):
#     plt.plot(i, s, 'o', color=con_cmap(seed_cols[i]))
for i, lab in enumerate(labels):
    # set the color based on the con value
    lab.color = con_cmap(seed_cols[i])

# plot the brain surface
brain = Brain(subject, hemi='both', surf='inflated',
              subjects_dir=subjects_dir)

for xlab in labels:
    brain.add_label(xlab)

time.sleep(2)

brain.save_montage(con_fname.split('.npy')[0] + ',mean,coh,alpha.png',
                   order=['lat', 'dor', 'med'])

time.sleep(2)
brain.close()
