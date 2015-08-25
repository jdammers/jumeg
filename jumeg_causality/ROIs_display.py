import mne, os
from surfer import Brain
subjects_dir = os.environ['SUBJECTS_DIR']
subject_path = subjects_dir + 'fsaverage'
labels_dir = subject_path + '/Group_ROIs/common/'
subject_id = 'fsaverage'
hemi = "split"
#surf = "smoothwm"
surf = 'inflated'

brain = Brain(subject_id, hemi, surf)
list_dirs = os.walk(labels_dir) 
for root, dirs, files in list_dirs: 
    for f in files:
        label_fname = os.path.join(root, f) 
        label = mne.read_label(label_fname)
        brain.add_label(label, color='red', subdir=root)
       