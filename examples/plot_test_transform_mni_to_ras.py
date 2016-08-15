'''
To plot a vertex point, convert it to MNI coordinates and then reconvert it back to RAS to obtain the vertex number.

It works when 'fsaverage' subject is used, but does not when any other subjects are used.
'''

import os
import mne
import matplotlib.pyplot as plt
from mne.datasets import sample

print(__doc__)

data_path = sample.data_path()
fname_inv = data_path + '/MEG/sample/sample_audvis-meg-oct-6-meg-inv.fif'
fname_evoked = data_path + '/MEG/sample/sample_audvis-ave.fif'
subjects_dir = data_path + '/subjects'
os.environ['SUBJECTS_DIR'] = subjects_dir

stc_fname = data_path + '/MEG/sample/sample_audvis-meg'
stc = mne.read_source_estimate(stc_fname)

new_stc = mne.morph_data('sample', 'fsaverage', stc, grade=5, subjects_dir=subjects_dir, n_jobs=2)

subject = 'fsaverage'

# Plot brain in 3D with PySurfer if available
brain = new_stc.plot(subject, hemi='lh', subjects_dir=subjects_dir)
brain.show_view('lateral')

# use peak getter to move vizualization to the time point of the peak
vertno_max, time_idx = new_stc.get_peak(hemi='lh', time_as_index=True)

brain.set_data_time_index(time_idx)

# draw marker at maximum peaking vertex
brain.add_foci(vertno_max, coords_as_verts=True, hemi='lh', color='blue',
               scale_factor=0.6, map_surface='white')

mni_coords = mne.vertex_to_mni(vertno_max, hemis=0, subject=subject,
                               subjects_dir=subjects_dir)
print 'The MNI coords are ', mni_coords

#my_trans = mne.read_trans(?)
#src_pts = apply_trans(trans, some_tgt_pts)

from surfer import utils
utils.coord_to_label(subject, mni_coords[0], label='mycoord',
                     hemi='lh', n_steps=25, map_surface="white")
brain.add_label('mycoord-lh.label', color="darkseagreen", alpha=.8)

# if the new mni_coords are computed
brain.add_foci(mni_coords[0], coords_as_verts=False, hemi='lh', color='red',
               map_surface='white', scale_factor=0.6)
