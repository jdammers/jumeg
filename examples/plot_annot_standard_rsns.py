#!/usr/bin/env python2

'''
Grow and visualize standard resting state ROIs from literature.

1. Read ROIs of standard regions involved in resting state networks from literature.
   (the data is provided as a csv file with list of regions with seed MNI coordinates)
2. Grow labels of 1cm radius (approx) in the surface source space.
3. Make annotation and visualize the labels.

Uses RSNs provided by [1]
[1] P. Garcés, M. C. Martín-Buro, and F. Maestú,
“Quantifying the Test-Retest Reliability of Magnetoencephalography
Resting-State Functional Connectivity,” Brain Connect., vol. 6, no. 6, pp. 448–460, 2016.
'''

import os.path as op
import numpy as np
import mne

from mne.datasets import sample
from jumeg.jumeg_utils import get_jumeg_path
from jumeg.connectivity import make_annot_from_csv

from nilearn import plotting
from surfer import Brain

data_path = sample.data_path()
subject = 'sample'
subjects_dir = data_path + '/subjects'
parc_fname = 'standard_garces_2016'
csv_fname = op.join(get_jumeg_path(), 'data', 'standard_rsns.csv')

# set make_annot to True to save the annotation to disk
labels, coords = make_annot_from_csv(subject, subjects_dir, csv_fname,
                                     parc_fname=parc_fname, make_annot=False,
                                     return_label_coords=True)

# to plot mni coords on glass brain
n_nodes = np.array(coords).shape[0]
# make a random zero valued connectivity matrix
con = np.zeros((n_nodes, n_nodes))
# plot the connectome on a glass brain background
plotting.plot_connectome(con, coords)
plotting.show()

# plot the brain surface, foci and labels
brain = Brain(subject, hemi='both', surf='white', subjects_dir=subjects_dir)
for mni_coord, mylabel in zip(coords, labels):
    brain.add_foci(mni_coord, coords_as_verts=False, hemi=mylabel.hemi,
                   color='red', map_surface='white', scale_factor=0.6)
    brain.add_label(mylabel, hemi=mylabel.hemi)
