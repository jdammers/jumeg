import mne, os, random
import numpy as np
import glob
from surfer import Brain
subjects_dir = os.environ['SUBJECTS_DIR']
fn_avg = subjects_dir+'/fsaverage/dSPM_ROIs/common-lh.stc'
stc_avg = mne.read_source_estimate(fn_avg)
color = ['#990033', '#9900CC', '#FF6600', '#FF3333', '#00CC33']
hemi = "split"
subject_id = 'fsaverage'
surf = 'inflated'
brain = Brain(subject_id, hemi, surf)
# Get common labels
list_file = subjects_dir+'/fsaverage/dSPM_ROIs/anno_ROIs/func_label_list.txt'
with open(list_file, 'r') as fl:
            file_list = [line.rstrip('\n') for line in fl]
fl.close()

for f in file_list:
    label = mne.read_label(f)
    stc_label = stc_avg.in_label(label)
    stc_label.data[stc_label.data < 0] = 0
    brain.add_label(label, color='red', alpha=0.5)
    if label.hemi == 'lh':
        vtx, _, _ = stc_label.center_of_mass(subject_id, hemi=0)
        mni_lh = mne.vertex_to_mni(vtx, 0, subject_id)[0]
        brain.add_foci(vtx, coords_as_verts=True, hemi='lh', color=random.choice(color),
                        scale_factor=0.8)
        print 'label:%s coordinate: ' %label.name + str(mni_lh)
        
    if label.hemi == 'rh':
        vtx, _, _ = stc_label.center_of_mass(subject_id, hemi=1)
        mni_rh = mne.vertex_to_mni(vtx, 1, subject_id)[0]
        brain.add_foci(vtx, coords_as_verts=True, hemi='rh', color=random.choice(color),
                        scale_factor=0.8)
        print 'label:%s coordinate: '%label.name + str(mni_rh)                
