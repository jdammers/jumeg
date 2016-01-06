'''ROIs definition using MNE or dSPM
'''
from jumeg.jumeg_preprocessing import get_files_from_list
import mne
import os
import numpy as np
from dirs_manage import reset_directory, set_directory
''' Before running these functions, please make sure essential files in the
    correct pathes as 'bem-sol' and 'src' files: subject_path/bem/. 'trans'
    file and 'noise-cov'are in the same directory as fname_raw.The way of 
    applying these functions please refer 'test.py'.
'''
def apply_create_noise_covariance(fname_empty_room, verbose=None):
    
    '''
    Creates the noise covariance matrix from an empty room file.

    Parameters
    ----------
    fname_empty_room : String containing the filename
        of the de-noise, empty room file (must be a fif-file)
    require_filter: bool
        If true, the empy room file is filtered before calculating
        the covariance matrix. (Beware, filter settings are fixed.)
    verbose : bool, str, int, or None
        If not None, override default verbose level
        (see mne.verbose).
        default: verbose=None
    '''

    # -------------------------------------------
    # import necessary modules
    # -------------------------------------------
    from mne import compute_raw_data_covariance as cp_covariance
    from mne import write_cov, pick_types
    from mne.io import Raw
    from jumeg.jumeg_noise_reducer import noise_reducer
    fner = get_files_from_list(fname_empty_room)
    nfiles = len(fner)
    ext_empty_raw = '-raw.fif'
    ext_empty_cov = '-cov.fif'
    # loop across all filenames
    for ifile in range(nfiles):
        fn_in = fner[ifile]
        print ">>> create noise covariance using file: "
        path_in, name = os.path.split(fn_in)
        print name   
        fn_empty_nr = fn_in[:fn_in.rfind('-raw.fif')] + ',nr-raw.fif'
        noise_reducer(fn_in, refnotch=50, detrending=False, fnout=fn_empty_nr)
        noise_reducer(fn_empty_nr, refnotch=60, detrending=False, fnout=fn_empty_nr) 
        noise_reducer(fn_empty_nr, reflp=5, fnout=fn_empty_nr)
        # file name for saving noise_cov
        fn_out = fn_empty_nr[:fn_empty_nr.rfind(ext_empty_raw)] + ext_empty_cov
        # read in data
        raw_empty = Raw(fn_empty_nr, preload=True, verbose=verbose)
        raw_empty.interpolate_bads()
        # pick MEG channels only
        picks = pick_types(raw_empty.info, meg=True, ref_meg=False, eeg=False,
                           stim=False, eog=False, exclude='bads')

        # calculate noise-covariance matrix
        noise_cov_mat = cp_covariance(raw_empty, picks=picks, verbose=verbose)

        # write noise-covariance matrix to disk
        write_cov(fn_out, noise_cov_mat)
        
def apply_inverse(fnepo, method='dSPM', event='LLst', min_subject='fsaverage', STC_US='ROI', 
                  snr=5.0):
    '''  
        Parameter
        ---------
        fnepo: string or list
            The epochs file with ECG, EOG and environmental noise free.
        method: inverse method, 'MNE' or 'dSPM'
        event: string
            The event name related with epochs.
        min_subject: string
            The subject name as the common brain.
        STC_US: string
            The using of the inversion for further analysis.
            'ROI' stands for ROIs definition, 'CAU' stands for causality analysis.
        snr: signal to noise ratio for inverse solution. 
    '''
    #Get the default subjects_dir
    from mne.minimum_norm import (apply_inverse, apply_inverse_epochs)
    subjects_dir = os.environ['SUBJECTS_DIR']
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
        if STC_US == 'ROI':
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
            stc_morph = mne.morph_data(subject, subject_id, stc, grade=4, smooth=4)
            stc_morph.save(stc_path + '/%s' % (stc_name), ftype='stc')
    
        elif STC_US == 'CAU':
            stcs_path = min_dir + '/stcs/%s/%s/' % (subject,event)
            reset_directory(stcs_path)
            noise_cov = mne.cov.regularize(noise_cov, epochs.info,
                                            mag=0.05, grad=0.05, proj=True)
            fwd = mne.make_forward_solution(epochs.info, trans=fn_trans,
                                            src=fn_src, bem=fn_bem,
                                            meg=True, eeg=False, mindist=5.0,
                                            n_jobs=2, overwrite=True)
            fwd = mne.convert_forward_solution(fwd, surf_ori=True)
            forward_meg = mne.pick_types_forward(fwd, meg=True, eeg=False)
            inverse_operator = mne.minimum_norm.make_inverse_operator(
                epochs.info, forward_meg, noise_cov, loose=0.2,
                depth=0.8)
            # Compute inverse solution
            stcs = apply_inverse_epochs(epochs, inverse_operator, lambda2,
                                        method=method, pick_ori='normal')
            s = 0
            while s < len(stcs):
                stc_morph = mne.morph_data(
                    subject, min_subject, stcs[s], grade=4, smooth=4)
                stc_morph.save(stcs_path + '/trial%s_fsaverage'
                                % (subject, str(s)), ftype='stc')
                s = s + 1
            
def apply_rois(fn_stc, event, tmin=0.0, tmax=0.3, tstep=0.05, window=0.2, 
                min_subject='fsaverage', thr=99):
    """
    Compute regions of interest (ROI) based on events
    ----------
    fn_stc : string
        evoked and morphed STC.
    event: string
        event of the related STC.
    tmin, tmax: float
        segment for ROIs definition.
    min_subject: string
        the subject as the common brain space.
    thr: float or int
        threshold of STC used for ROI identification.
    """
    fnlist = get_files_from_list(fn_stc)
    # loop across all filenames
    for ifn_stc in fnlist:
        subjects_dir = os.environ['SUBJECTS_DIR']
        # extract the subject infromation from the file name
        stc_path = os.path.split(ifn_stc)[0]
        #name = os.path.basename(fn_stc)
        #tri = name.split('_')[1].split('-')[0]
        min_path = subjects_dir + '/%s' % min_subject
        fn_src = min_path + '/bem/fsaverage-ico-4-src.fif'
        # Make sure the target path is exist
        labels_path = stc_path + '/%s/' %event
        reset_directory(labels_path)
        # Read the MNI source space
        src_inv = mne.read_source_spaces(fn_src)
        stc = mne.read_source_estimate(ifn_stc, subject=min_subject)
        stc = stc.crop(tmin, tmax)
        src_pow = np.sum(stc.data ** 2, axis=1)
        stc.data[src_pow < np.percentile(src_pow, thr)] = 0.
        tbeg = tmin
        count = 1
        while tbeg < tmax:
            tend = tbeg + window
            if tend > tmax:
                break
            win_stc = stc.copy().crop(tbeg, tend)
            stc_data = win_stc.data 
            src_pow = np.sum(stc_data ** 2, axis=1)
            win_stc.data[src_pow < np.percentile(src_pow, thr)] = 0.
            func_labels_lh, func_labels_rh = mne.stc_to_label(
                win_stc, src=src_inv, smooth=True,
                subjects_dir=subjects_dir,
                connected=True)
            # Left hemisphere definition
            i = 0
            while i < len(func_labels_lh):
                func_label = func_labels_lh[i]
                func_label.save(labels_path + '%s_%s_win%d' % (event, str(i), count))
                i = i + 1
            # right hemisphere definition
            j = 0
            while j < len(func_labels_rh):
                func_label = func_labels_rh[j]
                func_label.save(labels_path +  '%s_%s_win%d' % (event, str(j), count))
                j = j + 1
            tbeg = tbeg + tstep
            count = count + 1
            
def _cluster_rois(mer_path, label_list, count):
    """
    subfunctions of merge_ROIs
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
                    label_name = '%s_%d_' % (new_pre, count) + \
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


def merge_rois(labels_path_list, group=False, evelist=['LLst','LLrt']):
    """
    merge ROIs, so that the overlapped lables merged into one. 
    If 'group' is False, ROIs from all the events are merged and 
    saved in the folder 'ROIs' under the 'labels_path'.
    If 'group' is True, ROIs from all the subjects are merged and
    saved in the folder 'merged' under the 'labels_path'.
    ----------
    labels_path: the total path of all the ROIs' folders.
    group: if 'group' is False, merge ROIs from different events within one
           subject, if 'group' is True, merge ROIs across subjects.
    evelist: events name of all subfolders
    """
    path_list = get_files_from_list(labels_path_list)
    # loop across all filenames
    for labels_path in path_list:
        import glob, shutil
        if group is False:
            mer_path = labels_path + '/ROIs/'
            reset_directory(mer_path)
            for eve in evelist:
                source_path = labels_path + '/%s' %eve
                for filename in glob.glob(os.path.join(source_path, '*.*')):
                    shutil.copy(filename, mer_path)
        elif group is True:
            mer_path = labels_path + 'merged/'
            reset_directory(mer_path)
            source_path = labels_path + 'standard/' 
            for filename in glob.glob(os.path.join(source_path, '*.*')):
                shutil.copy(filename, mer_path)
        # Merge the individual subject's ROIs
        reducer = True
        count = 1
        while reducer:
            list_dirs = os.walk(mer_path)
            label_list = ['']
            for root, dirs, files in list_dirs:
                for f in files:
                    label_fname = os.path.join(root, f)
                    label_list.append(label_fname)
            label_list = label_list[1:]
            len_class = _cluster_rois(mer_path, label_list, count)
            if len_class == len(label_list):
                reducer = False  
            count = count + 1
def stan_rois(fname=None, stan_path=None, size=8.0, min_subject='fsaverage'):
    """
    Before merging all ROIs together, the size of ROIs will be standardized.
    Keep every ROIs in a same size
    ----------
    fname: averaged STC of the trials.
    stan_path: path to store all subjects standarlized labels
    size: the radius of every ROI.
    min_subject: the subject for the common brain space. 
    """
    fnlist = get_files_from_list(fname)
    subjects_dir = os.environ['SUBJECTS_DIR']
    # loop across all filenames
    for fn_stc in fnlist:
        stc_path = os.path.split(fn_stc)[0]
        stc_morph = mne.read_source_estimate(fn_stc, subject=min_subject)
        #min_path = subjects_dir + '/%s' %min_subject
        # extract the subject infromation from the file name
        name = os.path.basename(fn_stc)
        subject = name.split('_')[0]
        mer_path = stc_path + '/ROIs/'
        #stan_path = min_path + '/Group_ROIs/standard/'
        #set_directory(stan_path)
        list_dirs = os.walk(mer_path)
        for root, dirs, files in list_dirs:
            for f in files:
                label_fname = os.path.join(root, f)
                label = mne.read_label(label_fname)
                stc_label = stc_morph.in_label(label)
                src_pow = np.sum(stc_label.data ** 2, axis=1)
                if label.hemi == 'lh':
                    # Get the max MNE value within each ROI
                    seed_vertno = stc_label.vertices[0][np.argmax(src_pow)]
                    func_label = mne.grow_labels(min_subject, seed_vertno,
                                                 extents=size, hemis=0,
                                                 subjects_dir=subjects_dir,
                                                 n_jobs=1)
                    func_label = func_label[0]
                    func_label.save(stan_path + '%s_%s' % (subject, f))
                elif label.hemi == 'rh':
                    seed_vertno = stc_label.vertices[1][np.argmax(src_pow)]
                    func_label = mne.grow_labels(min_subject, seed_vertno,
                                                 extents=size, hemis=1,
                                                 subjects_dir=subjects_dir,
                                                 n_jobs=1)
                    func_label = func_label[0]
                    func_label.save(stan_path + '%s_%s' % (subject, f))
                    
def group_rois(am_sub=0, com_path=None, mer_path=None):
    """
    choose commont ROIs come out in at least 'sum_sub' subjects
    ----------
    am_sub: the least amount of subjects have the common ROIs.
    com_path: the directory of the common labels.
    mer_path: the directory of the merged rois.
    """
    import shutil
    reset_directory(com_path)
    list_dirs = os.walk(mer_path)
    label_list = ['']
    for root, dirs, files in list_dirs:
        for f in files:
            label_fname = os.path.join(root, f)
            label_list.append(label_fname)
    label_list = label_list[1:]
    for fn_label in label_list:
        fn_name = os.path.basename(fn_label)
        subjects = (fn_name.split('_')[0]).split(',')
        if len(subjects) >= am_sub:
            shutil.copy(fn_label, com_path)