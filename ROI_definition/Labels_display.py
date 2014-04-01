print __doc__

import mne, os, sys
from surfer import Brain
import random
try:
    subject = sys.argv[1]#Get the subject
    trigger = sys.argv[2]#Get the trigger is stim or resp
except:
    print "Please run with input file provided. Exiting"
    sys.exit()

res_ch_name = 'STI 013'
sti_ch_name = 'STI 014'

subjects_dir = '/home/qdong/freesurfer/subjects/' 
subject_path = subjects_dir + subject#Set the data path of the subject

subject_id = "fsaverage"
hemi = "both"
#surf = "smoothwm"
surf = 'inflated'
brain = Brain(subject_id, hemi, surf)
list_dirs = os.walk(subject_path + '/func_labels/')#Get the folder of defined ROIs
color = ['#990033', '#9900CC', '#FF6600', '#FF3333', '#00CC33']
#Load Functional ROIs and display them
for root, dirs, files in list_dirs: 
    for f in files: 
        label_fname = os.path.join(root, f) 
        label = mne.read_label(label_fname)
        if label.hemi == 'lh':
           brain.add_label(label, color=random.choice(color))
        elif label.hemi == 'rh':
           brain.add_label(label, color=random.choice(color))

brain.show_view("lateral")
brain.save_image('/home/qdong/freesurfer/subjects/101611/lh_ROI2.tiff')
