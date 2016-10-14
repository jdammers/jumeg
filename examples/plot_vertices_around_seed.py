#!/usr/bin/env python

'''
Find distances between vertices and plot vertices in a small region.

mainly using functions from within mne.label.grow_labels
'''

import mne
from mne.datasets import sample

data_path = sample.data_path()
subjects_dir = data_path + '/subjects'

tris, vert, dist = {}, {}, {}
hemi = 0  # lh

# read the surface
vert[hemi], tris[hemi] = mne.read_surface(subjects_dir + '/fsaverage/surf/lh.inflated')

# obtain distance matrix
dist[hemi] = mne.label.mesh_dist(tris[hemi], vert[hemi])

# choose seed vertex as 20 and plot vertices within 5mm radius around it
# obtain neighbouring vertices within 5mm distance
my_verts, my_dist = mne.label._verts_within_dist(dist[hemi], [20], 5)

# number of vertices in a given radius
print len(my_verts)

from surfer import Brain
brain = Brain('fsaverage', hemi='lh', surf='inflated',
              subjects_dir='/Users/psripad/sciebo/resting_state_analysis/')

for myv in my_verts:
    brain.add_foci(myv, coords_as_verts=True, color='b', scale_factor=0.1)
