#!/usr/bin/env python2

import numpy as np
from scipy import stats
import mne
from jumeg.connectivity import get_label_distances

from mne.datasets import sample
data_path = sample.data_path()
subjects_dir = data_path + '/subjects'
subject = 'sample'

# compute the distances between COM's of the labels
rounded_coms, coords, coms_lh, coms_rh = get_label_distances(subject, subjects_dir)
# np.save('%s_distances.npy' % subject, rounded_com)

# get maximum distance between ROIs
print 'Max distance between ROIs', rounded_coms.ravel().max()

# do plotting using PySurfer
from surfer import Brain
brain = Brain(subject, hemi='both', surf='inflated', subjects_dir=subjects_dir)
brain.add_foci(coms_lh, coords_as_verts=True, hemi='lh')
brain.add_foci(coms_rh, coords_as_verts=True, hemi='rh')
brain.save_montage('%s_coms.png' % subject, order=['lat', 'ven', 'med'],
                   orientation='h', border_size=15, colorbar='auto',
                   row=-1, col=-1)
brain.close()

# show the label ROIs using Nilearn plotting
from nilearn import plotting
fig = plotting.plot_connectome(rounded_coms, coords,
                               edge_threshold='99%', node_color='cornflowerblue',
                               title='aparc - label distances')
fig.savefig('aparc_label_distances.png')
