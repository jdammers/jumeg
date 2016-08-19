'''
ROIs definition using STCs from spatial-temporal clustering.
'''

import os
import numpy as np
import mne
import glob
from apply_merge import apply_merge
from jumeg.jumeg_utils import reset_directory

subjects_dir = os.environ['SUBJECTS_DIR']
stcs_path = subjects_dir + '/fsaverage/conf_stc/' # the path of STC
labels_path = stcs_path + 'STC_ROI/' # path where ROIs are saved

conditions = [('sti', 'LLst'), ('sti', 'RRst'), ('sti', 'RLst'), ('sti', 'LRst'),
              ('res', 'LLrt'), ('res', 'RRrt'), ('res', 'RLrt'), ('res', 'LRrt')]
fn_stcdata = stcs_path + 'stcsdata.npy'
thr = 0 # Threshold of the time span of STC
vert_num = 0 # minimum vertices amount for ROIs

do_ROIs = True
if do_ROIs:
    reset_directory(labels_path+'ini/')
    fn_src = subjects_dir + '/fsaverage/bem/fsaverage-ico-5-src.fif'
    src = mne.read_source_spaces(fn_src)
    for cond in conditions:
        side = cond[0]
        conf_type = cond[1]
        fn_stc = stcs_path + 'Ttestpermu16384_pthr0.0001_%s_%s,temp_0.001,tmin_0.000-rh.stc' %(side, conf_type)
        stc = mne.read_source_estimate(fn_stc)
        data = stc.data
        print '%s_%s condition, the time span is: min %.4f, max %.4f, median %.4f' \
               %(side, conf_type, data[np.nonzero(data)].min(), data[np.nonzero(data)].max(),np.median(data[np.nonzero(data)]))
        data[data < thr] = 0
        stc.data.setfield(data, np.float32)
        lh_labels, rh_labels = mne.stc_to_label(stc, src=src, smooth=True,
                                        subjects_dir=subjects_dir, connected=True)
        i = 0
        while i < len(lh_labels):
            lh_label = lh_labels[i]
            print 'left hemisphere ROI_%d has %d vertices' %(i, lh_label.vertices.shape[0])
            lh_label.save(labels_path + 'ini/%s,%s_%s' % (side, conf_type, str(i)))
            i = i + 1

        j = 0
        while j < len(rh_labels):
            rh_label = rh_labels[j]
            print 'right hemisphere ROI_%d has %d vertices' % (j, rh_label.vertices.shape[0])
            rh_label.save(labels_path + 'ini/%s,%s_%s' % (side, conf_type, str(j)))
            j = j + 1

# Merge ROIs across conditions
do_merge = True
if do_merge:
    apply_merge(labels_path, vert_num, red_little=False)

# Split large cluster
do_split = True
if do_split:
    ''' Before this conduction, we need to check the large ROI which are
        necessary for splitting. Collect the anatomical labels involve in
        the large ROIs. And split them one by one using the following scripts.
    '''
    # The large ROI
    fn_par = labels_path + '/merge/RLst_3-rh.label'
    # The corresponding anatomical labels
    fnana_list = glob.glob(labels_path + '/func_ana/rh/*')
    par_label = mne.read_label(fn_par)
    # The path to save splited ROIs
    chis_path = labels_path + '/func_ana/%s/' % par_label.name
    reset_directory(chis_path)
    for fnana in fnana_list:
        ana_label = mne.read_label(fnana)
        overlapped = len(np.intersect1d(ana_label.vertices,
                                        par_label.vertices))
        if overlapped > 0:
            chi_label = ana_label - (ana_label - par_label)
            chi_label.save(chis_path+ana_label.name)

# Merge small pieces ROIs again.
