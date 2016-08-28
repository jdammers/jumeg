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
#fn_stcdata = stcs_path + 'stcsdata.npy'
thr = 0 # Threshold of the time span of STC
vert_num = 0 # minimum vertices amount for ROIs

do_ROIs = False
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
do_merge = False
if do_merge:
    apply_merge(labels_path, vert_num, red_little=False)

# Split large cluster
do_split = False
if do_split:
    ''' Before this conduction, we need to check the large ROI which are
        necessary for splitting. Collect the anatomical labels involve in
        the large ROIs. And split them one by one using the following scripts.
    '''
    # The large ROI
    fn_par_list = glob.glob(labels_path + '/merge/*')
    # The corresponding anatomical labels
    #fnana_list = glob.glob(labels_path + '/func_ana/rh/*')
    analabel_list = mne.read_labels_from_annot('fsaverage')

    for fn_par in fn_par_list:
        par_label = mne.read_label(fn_par, subject='fsaverage')
        # The path to save splited ROIs
        chis_path = labels_path + '/func_ana/%s/' % par_label.name
        reset_directory(chis_path)
        for ana_label in analabel_list[:-1]:
            #ana_label = mne.read_label(fnana)
            overlapped = len(np.intersect1d(ana_label.vertices,
                                            par_label.vertices))
            if overlapped > 0 and ana_label.hemi == par_label.hemi:  
                chi_label = ana_label - (ana_label - par_label)
                chi_label.save(chis_path+ana_label.name)

# Merge ROIs with the same anatomical labels.   
do_merge_ana = False
if do_merge_ana:
    import shutil
    ana_path = glob.glob(labels_path + 'func_ana/*')
    tar_path = labels_path + 'func/'
    reset_directory(tar_path)
    path_list = ana_path[0]
    ana_list = glob.glob(path_list + '/*')
    for filename in ana_list:
        shutil.copy(filename, tar_path)
    for path_list in ana_path[1:]:
        ana_list = glob.glob(path_list + '/*')
        for fn_ana in ana_list:
            ana_label = mne.read_label(fn_ana)
            label_name = ana_label.name
            fn_tar = tar_path + '%s.label' %label_name
            if os.path.isfile(fn_tar):
                tar_label = mne.read_label(fn_tar)
                mer_label = ana_label + tar_label
                os.remove(fn_tar)
                mer_label.save(tar_path + label_name)
            else:
                shutil.copy(fn_ana, tar_path)
tar_path = labels_path + 'func/'
func_list1 = glob.glob(tar_path + '*-lh.label')
func_list2 = glob.glob(tar_path + '*-rh.label')
func_list = func_list1 + func_list2
fn_func = labels_path + 'func_list.txt'
text_file = open(fn_func, "w")
for func in func_list:
    text_file.write("%s\n" %func)
text_file.close()
