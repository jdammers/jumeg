from jumeg.jumeg_preprocessing import get_files_from_list
import mne
import os
import numpy as np
''' Before running these functions, please make sure essential files in the
    correct pathes as 'bem-sol' and 'src' files: subject_path/bem/. 'trans'
    file is in the same directory as fname_raw.The way of applying these
    functions please refer 'test.py'.
'''


def reset_directory(path=None):
    """
    check whether the directory exits, if yes, recreat the directory
    ----------
    path : the target directory.
    """
    import shutil
    isexists = os.path.exists(path)
    if isexists:
        shutil.rmtree(path)
    os.makedirs(path)


def set_directory(path=None):
    """
    check whether the directory exits, if no, creat the directory
    ----------
    path : the target directory.

    """
    isexists = os.path.exists(path)
    if not isexists:
        os.makedirs(path)


def apply_inverse(fname_raw, min_subject='fsaverage',
                  subjects_dir=None, unfiltered=False):
    """
    Compute source time courses in MIN source space on conditions of
    unfiltered.
    ----------
    fn_raw : raw data with interest components.
    min_subject: the subject for the common brain space.
    subjects_dir: the directory of subjects
    unfiltered:'True' for unfiltered data, get STCs for causality analysis and
               define the exact active positions.
               'False' for filtered data, get STCs for ROIs identification.

    Returns
    -------
    if unfiltered is 'True', it will return the amount of morphed trials
    """
    from mne.minimum_norm import (apply_inverse, apply_inverse_epochs)
    from mne.fiff import Raw
    fnlist = get_files_from_list(fname_raw)
    # loop across all filenames
    for fn_raw in fnlist:
        # extract the subject infromation from the file name
        # the path including MEG information
        meg_path = os.path.split(fn_raw)[0]
        name = os.path.basename(fn_raw)
        subject = name.split('_')[0]
        fn_inv = fn_raw.split('.fif')[0] + '-inv.fif'
        subject_path = subjects_dir + subject
        min_dir = subjects_dir + min_subject
        fn_trans = meg_path + '/%s-trans.fif' % subject
        fn_src = subject_path + '/bem/%s-ico-4-src.fif' % subject
        fn_bem = subject_path + '/bem/%s-5120-5120-5120-bem-sol.fif' % subject
        snr = 3.0
        lambda2 = 1.0 / snr ** 2

        if unfiltered is False:
            method = 'dSPM'
            # this path used for ROI definition
            stc_path = min_dir + '/STC_ROI/%s' % subject
            fn_cov = meg_path + '/%s,fibp1-45,empty-cov.fif' % subject
            evoked = mne.read_evokeds(fn_raw, condition=0, baseline=(None, 0))
            sti_name = fn_raw.split('ar,')[1].split(',ctpsbr')[0]

        elif unfiltered is True:
            method = 'MNE'
            # this path used for focus plotting
            stc_path = min_dir + '/STC_FOC/%s' % subject
            fn_cov = meg_path + '/%s,empty-cov.fif' % subject
            # Load data
            tmin, tmax = 0., 0.6
            raw = Raw(fn_raw, preload=True)
            events = mne.find_events(raw, stim_channel='STI 014')
            sti_name = 'trigger'
            picks = mne.fiff.pick_types(raw.info, meg=True, exclude='bads')
            epochs = mne.Epochs(raw, events, 1, tmin, tmax, picks=picks,
                                reject=dict(mag=4e-12))
            fwd = mne.make_forward_solution(epochs.info, mri=fn_trans,
                                            src=fn_src, bem=fn_bem, fname=None,
                                            meg=True, eeg=False, mindist=5.0,
                                            n_jobs=2, overwrite=True)

            fwd = mne.convert_forward_solution(fwd, surf_ori=True)

            forward_meg = mne.pick_types_forward(fwd, meg=True, eeg=False)
            noise_cov = mne.read_cov(fn_cov)
            noise_cov = mne.cov.regularize(noise_cov, epochs.info,
                                           mag=0.05, grad=0.05, proj=True)
            inverse_operator = mne.minimum_norm.make_inverse_operator(
                epochs.info, forward_meg, noise_cov, loose=0.2,
                depth=0.8)
            # Compute inverse solution
            stcs = apply_inverse_epochs(epochs, inverse_operator, lambda2,
                                        method=method, pick_ori='normal')
            # Make new folder for storing morphed Epochs
            stcs_path = min_dir + '/stcs/%s' % subject
            reset_directory(stcs_path)
            s = 0
            while s < len(stcs):
                stc_morph = mne.morph_data(
                    subject, min_subject, stcs[s], grade=4, smooth=4)
                stc_morph.save(stcs_path + '/%s_trial%s_fsaverage'
                               % (subject, str(s)), ftype='stc')
                s = s + 1
            evoked = epochs.average()
        set_directory(stc_path)
        noise_cov = mne.read_cov(fn_cov)
        noise_cov = mne.cov.regularize(noise_cov, evoked.info,
                                       mag=0.05, grad=0.05, proj=True)
        fwd_ev = mne.make_forward_solution(evoked.info, mri=fn_trans,
                                           src=fn_src, bem=fn_bem,
                                           fname=None, meg=True, eeg=False,
                                           mindist=5.0, n_jobs=2,
                                           overwrite=True)
        fwd_ev = mne.convert_forward_solution(fwd_ev, surf_ori=True)
        forward_meg_ev = mne.pick_types_forward(fwd_ev, meg=True, eeg=False)
        inverse_operator_ev = mne.minimum_norm.make_inverse_operator(
            evoked.info, forward_meg_ev, noise_cov,
            loose=0.2, depth=0.8)
        mne.minimum_norm.write_inverse_operator(fn_inv, inverse_operator_ev)
        # Compute inverse solution
        stc = apply_inverse(evoked, inverse_operator_ev, lambda2, method,
                            pick_ori=None)
        # Morph STC
        subject_id = 'fsaverage'
        stc_morph = mne.morph_data(subject, subject_id, stc, 4, smooth=4)

        stc_morph.save(stc_path + '/%s_%s' % (subject, sti_name), ftype='stc')


def apply_rois(fn_stc, subjects_dir=None, min_subject='fsaverage', thr=99):
    """
    Compute regions of interest (ROI) based on events
    ----------
    fn_stc : evoked and morphed STC.
    subjects_dir: the directory of the subjects.
    min_subject: the subject for the common brain space.
    thr: threshold of STC used for ROI identification.
    """
    fnlist = get_files_from_list(fn_stc)
    # loop across all filenames
    for fn_stc in fnlist:
        # extract the subject infromation from the file name
        stc_path = os.path.split(fn_stc)[0]
        name = os.path.basename(fn_stc)
        tri = name.split('_')[1].split('-')[0]
        min_path = subjects_dir + '/%s' % min_subject
        fn_src = min_path + '/bem/fsaverage-ico-4-src.fif'
        # Make sure the target path is exist
        labels_path = stc_path + '/evROIs/'
        set_directory(labels_path)
        # Read the MNI source space
        src_inv = mne.read_source_spaces(fn_src, add_geom=True)
        stc_morph = mne.read_source_estimate(fn_stc, subject=min_subject)
        src_pow = np.sum(stc_morph.data ** 2, axis=1)
        stc_morph.data[src_pow < np.percentile(src_pow, thr)] = 0.
        func_labels_lh, func_labels_rh = mne.stc_to_label(
            stc_morph, src=src_inv, smooth=5,
            subjects_dir=subjects_dir,
            connected=True)
        # Left hemisphere definition
        i = 0
        while i < len(func_labels_lh):
            func_label = func_labels_lh[i]
            func_label.save(labels_path + '%s_%s' % (tri, str(i)))
            i = i + 1
        # right hemisphere definition
        j = 0
        while j < len(func_labels_rh):
            func_label = func_labels_rh[j]
            func_label.save(labels_path + '%s_%s' % (tri, str(j)))
            j = j + 1


def _cluster_rois(mer_path, label_list):
    """
    subfunctions of merge_ROIs
    ----------
    mer_path: the directory for storing merged ROIs.
    label_list: labels to be merged
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
                if pre_test != pre_class:
                    pre_class += ',%s' % pre_test
                    pre_class = list(set(pre_class.split(',')))
                    new_pre = ''
                    for pre in pre_class[:-1]:
                        new_pre += '%s,' % pre
                    new_pre += pre_class[-1]
                    label_name = '%s_' % new_pre + \
                        class_label.name.split('_')[-1]
                os.remove(class_list[i])
                os.remove(test_fn)
                mne.write_label(mer_path + '/%s.label' % label_name, com_label)
                print label_name
                class_list[i] = mer_path + '/%s.label' % label_name
                belong = True
            i = i + 1
        if belong is False:
            class_list.append(test_fn)
    return len(class_list)


def merge_rois(subjects_dir=None, min_subject='fsaverage', group=False,
               subject=None):
    """
    merge ROIs, so that the overlapped lables merged into one.
    ----------
    subject: target subject for ROIs merging, if 'group' is False, is parameter
             is unused.
    subjects_dir: the directory of the subjects.
    min_subject: the subject for the common brain space.
    group: if 'group' is False, merge ROIs from different events within one
           subject, if 'group' is True, merge ROIs among subjects.
    """
    min_path = subjects_dir + '%s' % min_subject
    if group is False:
        labels_path = min_path + '/STC_ROI/%s' % subject
        mer_path = labels_path + '/evROIs'
    elif group is True:
        labels_path = min_path + '/Group_ROIs'
        mer_path = min_path + '/Group_ROIs/standard'
    # Merge the individual subject's ROIs
    reducer = True
    while reducer:
        list_dirs = os.walk(mer_path)
        label_list = ['']
        for root, dirs, files in list_dirs:
            for f in files:
                label_fname = os.path.join(root, f)
                label_list.append(label_fname)
        label_list = label_list[1:]
        len_class = _cluster_rois(mer_path, label_list)
        if len_class == len(label_list):
            reducer = False


def stan_rois(fname_stc, size=8.0, subjects_dir=None, min_subject='fsaverage'):
    """
    standardize ROIs, keep every ROIs in a same size
    ----------
    fname_stc: averaged STC includes all the events.
    size: the radius of every ROI.
    subjects_dir: the directory of the subjects.
    min_subject: the subject for the common brain space.
    """
    fnlist = get_files_from_list(fname_stc)
    # loop across all filenames
    for fn_stc in fnlist:
        stc_path = os.path.split(fn_stc)[0]
        stc_morph = mne.read_source_estimate(fn_stc, subject=min_subject)
        min_path = subjects_dir + min_subject
        # extract the subject infromation from the file name
        name = os.path.basename(fn_stc)
        subject = name.split('_')[0]
        mer_path = stc_path + '/evROIs/'
        stan_path = min_path + '/Group_ROIs/standard/'
        set_directory(stan_path)
        list_dirs = os.walk(mer_path)
        for root, dirs, files in list_dirs:
            for f in files:
                label_fname = os.path.join(root, f)
                label = mne.read_label(label_fname)
                stc_label = stc_morph.in_label(label)
                src_pow = np.sum(stc_label.data ** 2, axis=1)
                if label.hemi == 'lh':
                    # Get the max MNE value within each ROI
                    seed_vertno = stc_label.vertno[0][np.argmax(src_pow)]
                    func_label = mne.grow_labels(min_subject, seed_vertno,
                                                 extents=size, hemis=0,
                                                 subjects_dir=subjects_dir,
                                                 n_jobs=1)
                    func_label = func_label[0]
                    func_label.save(stan_path + '%s_%s' % (subject, f))
                elif label.hemi == 'rh':
                    seed_vertno = stc_label.vertno[1][np.argmax(src_pow)]
                    func_label = mne.grow_labels(min_subject, seed_vertno,
                                                 extents=size, hemis=1,
                                                 subjects_dir=subjects_dir,
                                                 n_jobs=1)
                    func_label = func_label[0]
                    func_label.save(stan_path + '%s_%s' % (subject, f))


def group_rois(am_sub=0, subjects_dir=None, min_subject='fsaverage'):
    """
    choose commont ROIs come out in at least 'sum_sub' subjects
    ----------
    am_sub: amount of subjects.
    subjects_dir: the directory of the subjects.
    min_subject: the subject for the common brain space.
    """
    import shutil
    min_path = subjects_dir + 'fsaverage'
    com_path = min_path + '/Group_ROIs/common/'
    mer_path = min_path + '/Group_ROIs/standard/'
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


def causal_analysis(subject, subjects_dir, min_subject='fsaverage',
                    method='PDC'):
    """
    causality computing for every subject.
    ----------
    subject: the subject for causality analysis.
    subjects_dir: the directory of the subjects.
    min_subject: the subject for the common brain space.
    method: the method for causality analysis, such as 'PDC', 'DTF'.
    """
    min_path = subjects_dir + min_subject
    stcs_path = min_path + '/stcs/%s/' % subject
    cau_path = min_path + '/causality/'
    set_directory(cau_path)
    trials = len(sum([i[2] for i in os.walk(stcs_path)], [])) / 2
    src_path = min_path + '/bem/fsaverage-ico-4-src.fif'
    src_inv = mne.read_source_spaces(src_path, add_geom=True)
    fs = 1017.25
    nfft = fs / 2
    # Get unfiltered and morphed stcs
    stcs = []
    i = 0
    while i < trials:
        fn_stc = stcs_path + '%s_trial%s_fsaverage' % (subject, str(i))
        stc = mne.read_source_estimate(fn_stc + '-lh.stc', subject='fsaverage')
        stcs.append(stc)
        i = i + 1
    # Get common labels
    list_dirs = os.walk(min_path + '/Group_ROIs/common/')
    labels = []
    rois = []
    for root, dirs, files in list_dirs:
        for f in files:
            label_fname = os.path.join(root, f)
            label = mne.read_label(label_fname)
            labels.append(label)
            rois.append(f)
    # Extract stcs in common labels
    label_ts = mne.extract_label_time_course(stcs, labels, src_inv,
                                             mode='pca_flip',
                                             return_generator=False)
    # Causal analysis
    from scot.connectivity import connectivity
    import scot.connectivity_statistics as scs
    label_ts = np.asarray(label_ts).transpose(2, 1, 0)

    # remove mean over epochs (evoked response) to improve stationarity
    label_ts -= label_ts.mean(axis=2, keepdims=True)
    import scot
    mvar = scot.var.VAR(1)
    mvar.optimize_order(label_ts)
    mvar.optimize_delta_bisection(label_ts)
    # generate connectivity surrogates under the null-hypothesis of no
    # connectivity
    c_surrogates = scs.surrogate_connectivity(
        method, label_ts, mvar, nfft=nfft, repeats=1000)
    mvar.fit(label_ts)
    con = connectivity(method, mvar.coef, mvar.rescov, nfft=nfft)
    np.save(cau_path + '%s_%s-cau_con.npy' % (subject, method), con)
    np.save(cau_path + '%s_%s-surrogates_con.npy' %
            (subject, method), c_surrogates)


def sig_causality(fn_cau, thresh=99):
    """
    significant matrices are made based on the surrogate threshold across
    frequency band (1,4), (4, 8), (8, 12), (12, 18), (18, 30), (30, 40).
    ----------
    fn_cau: the path for individual causality matrices.
    threshold: the threshold of surrogates matrices.
    """
    fnlist = get_files_from_list(fn_cau)
    # loop across all filenames
    for fn_stc in fnlist:
        cau_path = os.path.split(fn_cau)[0]
        sig_path = cau_path + '/sig_con/'
        set_directory(sig_path)
        pre_name = os.path.basename(fn_cau).split('-')[0]
        fn_surr = fn_cau.split('-')[0] + '-surrogates_con.npy'

        cau_subject = np.load(fn_cau)
        surr_subject = np.load(fn_surr)
        c0 = np.percentile(surr_subject, thresh, axis=0)
        con_dif = cau_subject - c0
        freqs = [(1, 4), (4, 8), (8, 12), (12, 18), (18, 30), (30, 40)]
        con_freqs = []
        nfreq = len(freqs)
        for ifreq in range(nfreq):
            fmin, fmax = freqs[ifreq][0], freqs[ifreq][1]
            con_band = np.mean(con_dif[:, :, fmin:fmax], axis=-1)
            np.fill_diagonal(con_band, 0)  # ignore the dignoal values
            # ignore the value less than significance
            con_band[con_band < 0] = 0
            con_band[con_band > 0] = 1  # set the significant element as 1
            con_freqs.append(con_band)
        con_freqs = np.array(con_freqs)
        np.save(sig_path + '%s_sig_con_band.npy' % pre_name, con_freqs)


def group_causality(subjects_dir, min_subject='fsaverage'):
    """
    make group causality analysis, by evaluating significant matrices across
    subjects.
    ----------
    subjects_dir: the directory of the subjects.
    min_subject: the subject for the common brain space.
    """
    import matplotlib.pylab as plt
    min_path = subjects_dir + min_subject
    cau_path = min_path + '/causality'
    sig_path = cau_path + '/sig_con'
    list_dirs = os.walk(sig_path)
    sig_caus = []
    for root, dirs, files in list_dirs:
        for f in files:
            fn_sig = os.path.join(root, f)
            sig_cau = np.load(fn_sig)
            sig_caus.append(sig_cau)
    sig_caus = np.array(sig_caus)
    sig_group = sig_caus.sum(axis=0)
    freqs = [(1, 4), (4, 8), (8, 12), (12, 18), (18, 30), (30, 40)]
    for i in xrange(len(sig_group)):
        fmin, fmax = freqs[i][0], freqs[i][1]
        cau_band = sig_group[i]
        cau_band[cau_band < 4] = 0  # ignore the value less than significance
        plt.imshow(cau_band, interpolation='nearest')
        v = np.arange(cau_band.max())
        plt.colorbar(ticks=v)
        # plt.colorbar()
        plt.show()
        plt.savefig(cau_path + '/band_%s_%s.png' %
                    (str(fmin), str(fmax)), dpi=100)
        plt.close()
    reset_directory(sig_path)
