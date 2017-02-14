# -*- coding: utf-8 -*-
"""
===========================================================
Compute ROIs using the common STCs averaged across subjects
===========================================================
Epochs from 4 experimental conditions are firstly morphed into
the common brain space, and then averaged to form the common STCs.
STCs larger than 95% strength are used for ROIs definition.
Then we use Euclidean_norms and least distance for ROIs selection[1].

[1] D. W. G. J. Et.al., “Lexical mediation of phonotactic frequency effects
on spoken word recognition: A Granger causality analysis of MRI-constrained 
MEG/EEG data.” pp. 41–45, 2015.  
"""
from jumeg.jumeg_preprocessing import get_files_from_list
import mne
import os
import numpy as np
from dirs_manage import reset_directory, set_directory
subjects_dir = os.environ['SUBJECTS_DIR']
def apply_inverse(fnepo, method='dSPM', event='LLst', min_subject='fsaverage', snr=5.0):
    ''' 
       Compute the averaged individual STCs and then morph them into the
       common brain sapce.
          
        Parameter
        ---------
        fnepo: string or list
            The epochs file with ECG, EOG and environmental noise free.
            Confirm 'empty' and 'trans' files are in the same path as 'fname'.
        method: inverse method, 'MNE' or 'dSPM'
        event: string
            The event name related with epochs.
        min_subject: string
            The subject name as the common brain.
        STC_US: string
            The using of the inversion for further analysis.
            'ROI' stands for ROIs definition, 'CAU' stands for causality analysis.
    '''
    #Get the default subjects_dir
    from mne.minimum_norm import (apply_inverse, apply_inverse_epochs)
    fnlist = get_files_from_list(fnepo)
    # loop across all filenames
    for fname in fnlist:
        fn_path = os.path.split(fname)[0]
        name = os.path.basename(fname)
        stc_name = name[:name.rfind('-epo.fif')] 
        subject = name.split('_')[0]
        subject_path = subjects_dir + '/%s' %subject
        min_dir = subjects_dir + '/%s' %min_subject
        fn_trans = fn_path + '/%s-trans.fif' % subject
        fn_cov = fn_path + '/%s_empty,nr-cov.fif' % subject
        fn_src = subject_path + '/bem/%s-ico-4-src.fif' % subject
        fn_bem = subject_path + '/bem/%s-5120-5120-5120-bem-sol.fif' % subject
        snr = snr
        lambda2 = 1.0 / snr ** 2 
        #noise_cov = mne.read_cov(fn_cov)
        epochs = mne.read_epochs(fname)
        noise_cov = mne.read_cov(fn_cov)

        # this path used for ROI definition
        stc_path = min_dir + '/%s_ROIs/%s' %(method,subject)
        #fn_cov = meg_path + '/%s_empty,fibp1-45,nr-cov.fif' % subject
        evoked = epochs.average()
        set_directory(stc_path)
        noise_cov = mne.cov.regularize(noise_cov, evoked.info,
                                        mag=0.05, grad=0.05, proj=True)
        fwd_ev = mne.make_forward_solution(evoked.info, trans=fn_trans,
                                            src=fn_src, bem=fn_bem,
                                            fname=None, meg=True, eeg=False,
                                            mindist=5.0, n_jobs=2,
                                            overwrite=True)
        fwd_ev = mne.convert_forward_solution(fwd_ev, surf_ori=True)
        forward_meg_ev = mne.pick_types_forward(fwd_ev, meg=True, eeg=False)
        inverse_operator_ev = mne.minimum_norm.make_inverse_operator(
            evoked.info, forward_meg_ev, noise_cov,
            loose=0.2, depth=0.8)
        # Compute inverse solution
        stc = apply_inverse(evoked, inverse_operator_ev, lambda2, method,
                            pick_ori=None)
        # Morph STC
        subject_id = min_subject
        stc_morph = mne.morph_data(subject, subject_id, stc, grade=5, smooth=5)
        stc_morph.save(stc_path + '/%s' % (stc_name), ftype='stc')
    

def apply_stcs(method='dSPM', event='LLst'):
    
    '''
       Normalize the individual STCs and average them across subjects.
        
       Parameters
       ----------
       method: string
          'dSPM' or 'MNE'.
       event: string
          the event name in the experimental conditions.
    '''
    import glob
    from scipy.signal import detrend
    from scipy.stats.mstats import zscore
    fn_list = glob.glob(subjects_dir+'/fsaverage/%s_ROIs/*/*,evtW_%s_bc-lh.stc' % (method, event))
    stcs = []
    for fname in fn_list:
        stc = mne.read_source_estimate(fname)
        #stc = stc.crop(tmin, tmax)
        cal_data = stc.data
        dt_data = detrend(cal_data, axis=-1)
        zc_data = zscore(dt_data, axis=-1)
        stc.data.setfield(zc_data, np.float32)
        stcs.append(stc)
    stcs = np.array(stcs)
    stc_avg = np.sum(stcs, axis=0)/stcs.shape[0]
    fn_avg = subjects_dir+'/fsaverage/%s_ROIs/%s' %(method,event)
    stc_avg.save(fn_avg, ftype='stc')
    
def apply_rois(fn_stc, tmin, tmax, thr, min_subject='fsaverage'):

    '''
       Make ROIs using the common STCs.
        
       Parameters
       ----------
       fn_stc: string.
            The path of the common STCs
       tmin, tmax: float (s).
            The interest time range.
       thr: float or int
            The percentile of STCs strength.
       min_subject: string.
            The common subject.
       
    '''
    stc_avg = mne.read_source_estimate(fn_stc)
    stc_avg = stc_avg.crop(tmin, tmax)
    src_pow = np.sum(stc_avg.data ** 2, axis=1)
    stc_avg.data[src_pow < np.percentile(src_pow, thr)] = 0.
    fn_src = subjects_dir+'/%s/bem/fsaverage-ico-5-src.fif' %min_subject
    src_inv = mne.read_source_spaces(fn_src)
    func_labels_lh, func_labels_rh = mne.stc_to_label(
                    stc_avg, src=src_inv, smooth=True,
                    subjects_dir=subjects_dir,
                    connected=True)
    # Left hemisphere definition
    i = 0
    labels_path = fn_stc[:fn_stc.rfind('-')] + '/ini'
    reset_directory(labels_path)
    while i < len(func_labels_lh):
        func_label = func_labels_lh[i]
        func_label.save(labels_path + '/ROI_%d' %(i))
        i = i + 1
    # right hemisphere definition
    j = 0
    while j < len(func_labels_rh):
        func_label = func_labels_rh[j]
        func_label.save(labels_path + '/ROI_%d' %(j))
        j = j + 1

def _cluster_sel(sel_path, label_list, stc, src, min_dist, weight, mni_subject='fsaverage'):
    """
    subfunctions of sel_ROIs
    ----------
    sel_path: string or list
        The directory for storing selected ROIs.
    label_list: list
        Labels to be selected.
    stc: the object of source estimates.
    src: the object of the common source space
    min_dist: int (mm)
        Least distance between ROIs candidates.
    weight: float
        Euclidean_norms weight related to the larger candidate's standard deviation.
    """
    class_list = []
    class_list.append(label_list[0])
    for test_fn in label_list[1:]:
        test_label = mne.read_label(test_fn)
        i = 0
        belong = False
        while (i < len(class_list)) and (belong is False):
            class_label = mne.read_label(class_list[i])
            if test_label.hemi != class_label.hemi:
                i = i + 1
                continue
            else:
                class_stc = stc.in_label(class_label)
                test_stc = stc.in_label(test_label)
                class_pow = np.sum(class_stc.data ** 2, axis=1)
                test_pow = np.sum(test_stc.data ** 2, axis=1)
                if class_label.hemi == 'lh':
                    h = 0
                elif class_label.hemi == 'rh':
                    h = 1
                class_seed = class_stc.vertices[h][np.argmax(class_pow)]
                test_seed = test_stc.vertices[h][np.argmax(test_pow)]
                class_vtx = np.searchsorted(stc.vertices[h], class_seed)
                test_vtx = np.searchsorted(stc.vertices[h], test_seed) 
                class_mni = mne.vertex_to_mni(class_vtx, h, mni_subject)[0]
                test_mni = mne.vertex_to_mni(test_vtx, h, mni_subject)[0]
                class_ts = stc.data[class_vtx, :]
                test_ts = stc.data[test_vtx, :]
                max_ts = class_ts
                exch = False
                if np.max(class_pow) < np.max(test_pow):
                    max_ts = test_ts
                    exch = True
                nearby = False
                if np.linalg.norm(class_mni - test_mni) < min_dist:
                    if exch == True:
                        os.remove(class_list[i])
                        class_list[i] = test_fn
                    elif exch == False:
                        os.remove(test_fn)
                    nearby = True
                    belong = True
                    
                if nearby == False:
                    thre = max_ts.std() * weight
                    diff =  np.abs(np.linalg.norm(class_ts) - np.linalg.norm(test_ts))
                    if diff < thre:
                        if exch == True:
                            os.remove(class_list[i])
                            class_list[i] = test_fn
                        elif exch == False:
                            os.remove(test_fn)
                        belong = True
                i = i + 1
        if belong is False:
            class_list.append(test_fn)
                
    return len(class_list)



def sele_rois(fn_stc_list, fn_src, min_dist, weight, tmin=0.1, tmax=0.5):
    """
    Select ROIs based on the least distance and the difference of Euclidean_norms
    between ROIs candidates.
    
    Parameters
    ----------
    fn_stc_list: string or list 
        The path of the common STCs.
    fn_src: string
       The path of the common source space, such as: '*/fsaverage/bem/*-src.fif'
    min_dist: int (mm)
        Least distance between ROIs candidates.
    weight: float
        Euclidean_norms weight related to the larger candidate's standard deviation.
    tmin, tmax: float (s).
        The interest time range.
    """
    fn_stc_list = get_files_from_list(fn_stc_list)
    # loop across all filenames
    for fn_stc in fn_stc_list:
        import glob, shutil
        labels_path = fn_stc[:fn_stc.rfind('-')] 
        source_path = labels_path + '/ini/' 
        sel_path = labels_path + '/ROIs/'        
        reset_directory(sel_path)
        for filename in glob.glob(os.path.join(source_path, '*.*')):
            shutil.copy(filename, sel_path)
        reducer = True
        stc = mne.read_source_estimate(fn_stc)
        stc = stc.crop(tmin, tmax)
        src = mne.read_source_spaces(fn_src)
        while reducer:
            list_dirs = os.walk(sel_path)
            label_list = ['']
            for root, dirs, files in list_dirs:
                for f in files:
                    label_fname = os.path.join(root, f)
                    label_list.append(label_fname)
            label_list = label_list[1:]
            len_class = _cluster_sel(sel_path, label_list, stc, src, min_dist, weight)
            if len_class == len(label_list):
                reducer = False  

def apply_stand(fn_stc, radius=5.0, min_subject='fsaverage', tmin=0.1, tmax=0.5):

    """
    Standardize the size of the selected ROIs.
    Parameters
    ----------
    fn_stc: string or list
        The path of the common STCs.
    radius: the radius of every ROI. 
    tmin, tmax: float (s).
        The interest time range.
    """
    fnlist = get_files_from_list(fn_stc)
    # loop across all filenames
    for fn_stc in fnlist:
        stc_path = fn_stc[:fn_stc.rfind('-')]
        stc = mne.read_source_estimate(fn_stc, subject=min_subject)
        stc = stc.crop(tmin, tmax)
        #min_path = subjects_dir + '/%s' %min_subject
        # extract the subject infromation from the file name
        source_path = stc_path + '/ROIs/'
        stan_path = stc_path + '/standard/'
        reset_directory(stan_path)
        list_dirs = os.walk(source_path)
        for root, dirs, files in list_dirs:
            for f in files:
                label_fname = os.path.join(root, f)
                label = mne.read_label(label_fname)
                stc_label = stc.in_label(label)
                src_pow = np.sum(stc_label.data ** 2, axis=1)
                if label.hemi == 'lh':
                    # Get the max MNE value within each ROI
                    seed_vertno = stc_label.vertices[0][np.argmax(src_pow)]
                    func_label = mne.grow_labels(min_subject, seed_vertno,
                                                 extents=radius, hemis=0,
                                                 subjects_dir=subjects_dir,
                                                 n_jobs=1)
                    func_label = func_label[0]
                    func_label.save(stan_path + '%s' %f)
                elif label.hemi == 'rh':
                    seed_vertno = stc_label.vertices[1][np.argmax(src_pow)]
                    func_label = mne.grow_labels(min_subject, seed_vertno,
                                                 extents=radius, hemis=1,
                                                 subjects_dir=subjects_dir,
                                                 n_jobs=1)
                    func_label = func_label[0]
                    func_label.save(stan_path + '%s' %f)

def _cluster_merge(mer_path, label_list):
    """
    subfunctions of apply_merge
    ----------
    mer_path: str
        The directory for storing merged ROIs.
    label_list: list
        Labels to be merged
    """
    class_list = []
    class_list.append(label_list[0])
    for test_fn in label_list[1:]:
        test_label = mne.read_label(test_fn)
        i = 0
        belong = False
        while (i < len(class_list)) and (belong is False):
            class_label = mne.read_label(class_list[i])
            label_name = class_label.name
            if test_label.hemi != class_label.hemi:
                i = i + 1
                continue
            overlapped = len(np.intersect1d(test_label.vertices,
                                            class_label.vertices))
            if overlapped > 0:
                com_label = test_label + class_label
                pre_test = test_label.name.split('_')[0]
                pre_class = class_label.name.split('_')[0]
                #label_name = pre_class + '_%s-%s' %(pre_test,class_label.name.split('-')[-1])
                if pre_test != pre_class:
                    pre_class += ',%s' % pre_test
                    pre_class = list(set(pre_class.split(',')))
                    new_pre = ''
                    for pre in pre_class[:-1]:
                        new_pre += '%s,' % pre
                    new_pre += pre_class[-1]
                    label_name = '%s' % (new_pre) + \
                        class_label.name.split('_')[-1]
                os.remove(class_list[i])
                os.remove(test_fn)
                fn_newlabel = mer_path + '%s.label' %label_name
                if os.path.isfile(fn_newlabel):
                    fn_newlabel = fn_newlabel[:fn_newlabel.rfind('_')] + '_new,%s' %fn_newlabel.split('_')[-1]
                mne.write_label(fn_newlabel, com_label)
                class_list[i] = fn_newlabel
                belong = True
            i = i + 1
        if belong is False:
            class_list.append(test_fn)
    return len(class_list)

def apply_merge(labels_path, evt_list):
    '''Every condition have two kinds of events, merge the two kinds of standard 
       ROIs together to form the terminal ROIs for causality analyze.
       
       Parameters
       ----------
       labels_path: string.
            The name of the total labels path.
       evt_list: list.
            The name of the subpath under 'labels_path', such as: '['LLst', 'LLrt']'
    '''
    import glob, shutil
    for evt in evt_list:
        mer_path = labels_path + '%s/merged/' %evt[0]
        reset_directory(mer_path)
        source0_path = labels_path + '%s/standard/' %evt[0]
        source1_path = labels_path + '%s/standard/' %evt[1]
        source = glob.glob(os.path.join(source0_path, '*.*'))
        source = source + glob.glob(os.path.join(source1_path, '*.*'))
        for filename in source:
            shutil.copy(filename, mer_path)
             
        reducer = True
        while reducer:
            list_dirs = os.walk(mer_path)
            label_list = ['']
            for root, dirs, files in list_dirs:
                for f in files:
                    label_fname = os.path.join(root, f)
                    label_list.append(label_fname)
            label_list = label_list[1:]
            len_class = _cluster_merge(mer_path, label_list)
            if len_class == len(label_list):
                reducer = False  
