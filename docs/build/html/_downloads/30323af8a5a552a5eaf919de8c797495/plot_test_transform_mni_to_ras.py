"""
To plot a vertex point, convert it to MNI coordinates and then reconvert it back to RAS to obtain the vertex number.

It works when 'fsaverage' subject is used, but does not when any other subjects are used.
"""

import os
import os.path as op
import mne
from mne.datasets import sample
from surfer import utils

print(__doc__)

data_path = sample.data_path()
fname_inv = op.join(data_path, 'MEG/sample/sample_audvis-meg-oct-6-meg-inv.fif')
fname_evoked = op.join(data_path, 'MEG/sample/sample_audvis-ave.fif')
subjects_dir = op.join(data_path, 'subjects')
os.environ['SUBJECTS_DIR'] = subjects_dir

stc_fname = op.join(data_path, 'MEG/sample/sample_audvis-meg')
stc = mne.read_source_estimate(stc_fname)

morph = mne.compute_source_morph(stc, subject_from='sample',
                                 subject_to='fsaverage', spacing=5, subjects_dir=subjects_dir)
new_stc = morph.apply(stc)

subject = 'fsaverage'

# Plot brain in 3D with PySurfer if available
brain = new_stc.plot(subject, hemi='lh',
                     subjects_dir=subjects_dir, backend='pyvistaqt')
brain.show_view('lateral')

# use peak getter to move vizualization to the time point of the peak
vertno_max, time_idx = new_stc.get_peak(hemi='lh', time_as_index=True)

brain.set_time_point(time_idx)

# draw marker at maximum peaking vertex
brain.add_foci(vertno_max, coords_as_verts=True, hemi='lh', color='blue',
               scale_factor=0.6, map_surface='white')

mni_coords = mne.vertex_to_mni(vertno_max, hemis=0, subject=subject,
                               subjects_dir=subjects_dir)
print('The MNI coords are ', mni_coords)

# my_trans = mne.read_trans(?)
# src_pts = apply_trans(trans, some_tgt_pts)

utils.coord_to_label(subject, mni_coords, label='mycoord',
                     hemi='lh', n_steps=25, map_surface="white")
brain.add_label('mycoord-lh.label', color="darkseagreen", alpha=.8)

# if the new mni_coords are computed
brain.add_foci(mni_coords, coords_as_verts=False, hemi='lh',
               color='red', scale_factor=0.6)
