from jumeg.jumeg_preprocessing import get_files_from_list
import mne, math
import os
import numpy as np
from dirs_manage import reset_directory, set_directory

########################################
#  ROIs definition
#######################################
''' We normalized the STCs, to improve th SNR, then average them into a common 
    STCs. Anatomical labels from Freesurfer provide the general interest regions.
    Normalized STCs identifiy the focal regions.
'''
subjects_dir = os.environ['SUBJECTS_DIR']
def apply_inverse_ave(fnevo, method='dSPM', snr=3.0, event='LLst', 
                      baseline=True, btmin=-0.3, btmax=-0.1, 
                      min_subject='fsaverage'):
    ''' Inverse evoked data into the source space. 
        Parameter
        ---------
        fnevo: string or list
            The evoked file with ECG, EOG and environmental noise free.
        method:string
            Inverse method, 'MNE' or 'dSPM'
        snr: float
            Signal to noise ratio for inverse solution.
        event: string
            The event name related with evoked data.
        baseline: bool
            If true, prestimulus segment from 'btmin' to 'btmax' will be saved 
            for future baseline correction. If false, no baseline segment is saved.
        btmin: float
            The start time point (second) of baseline.
        btmax: float
            The end time point(second) of baseline.
        min_subject: string
            The subject name as the common brain.
    '''
    #Get the default subjects_dir
    from mne.minimum_norm import apply_inverse
    fnlist = get_files_from_list(fnevo)
    # loop across all filenames
    for fname in fnlist:
        fn_path = os.path.split(fname)[0]
        name = os.path.basename(fname)
        stc_name = name[:name.rfind('-ave.fif')] 
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
        [evoked] = mne.read_evokeds(fname)
        noise_cov = mne.read_cov(fn_cov)
        # this path used for ROI definition
        stc_path = min_dir + '/%s_ROIs/%s' %(method,subject)
        #fn_cov = meg_path + '/%s_empty,fibp1-45,nr-cov.fif' % subject
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
        stc_morph = mne.morph_data(subject, min_subject, stc, grade=4, smooth=5)
        stc_morph.save(stc_path + '/%s' % (stc_name), ftype='stc')
        if baseline == True:
            stc_base = stc_morph.crop(btmin, btmax)
            stc_base.save(stc_path + '/%s_%s_baseline' % (subject, event), ftype='stc')

def apply_norm(fn_stc, event, ref_event, thr=95):   
    ''' 
        Normalize the inversed STC 
        Parameter
        ---------
        fnstc: string or list
            The file path of STC
        event: string
            The event name related with STC.
        ref_event: string
            The reference event name of baseline.
        thr: int 
            Percentile of baseline STC for nomalization.
    '''     
    fn_list = get_files_from_list(fn_stc)
    stcs = []
    for fname in fn_list:
        fn_path = os.path.split(fname)[0]
        stc = mne.read_source_estimate(fname)       
        name = os.path.basename(fname)
        subject = name.split('_')[0]
        fn_base = fn_path + '/%s_%s_baseline-lh.stc' %(subject,ref_event)
        stc = mne.read_source_estimate(fname)
        stc_base = mne.read_source_estimate(fn_base)
        thre = np.percentile(stc_base.data, thr, axis=-1)
        data = stc.data
        cal_mean = data.mean(axis=-1)
        norm_data = (data.T / thre) - 1
        norm_data[norm_data < 0] = 0
        norm_data = norm_data.T
        norm_data[cal_mean == 0, :] = 0
        norm_mean = norm_data.mean(axis=-1)
        zc_data = norm_data.T/norm_data.max(axis=-1)
        zc_data = zc_data.T
        zc_data[norm_mean == 0, :] = 0
        #import pdb
        #pdb.set_trace()
        print zc_data.min()
        stc.data.setfield(zc_data, np.float32)
        stcs.append(stc)
        fn_nr = fname[:fname.rfind('-lh')] + '_norm_1'
        stc.save(fn_nr, ftype='stc')
    stcs_path = subjects_dir+'/fsaverage/dSPM_ROIs/' 
    fn_avg = stcs_path + '%s' %(event)
    stcs = np.array(stcs)
    stc_avg = np.sum(stcs, axis=0)/stcs.shape[0]
    stc_avg.save(fn_avg, ftype='stc')
    
def apply_comSTC(stcs_path, evt_list):
    ''' 
        Average the normalized STCs across conditions 
        Parameter
        ---------
        stcs_path: list
            Pathes of all conditions' STCs.
        evt_list: list
            List of all the conditions.
    '''     
    stcs = []
    for evt in evt_list:
        fname = stcs_path + '%s-lh.stc' %evt
        stc = mne.read_source_estimate(fname)
        stcs.append(stc) 
    stcs = np.array(stcs)
    stc_avg = np.sum(stcs, axis=0)/stcs.shape[0]
    fn_avg = subjects_dir+'/fsaverage/dSPM_ROIs/common' 
    stc_avg.save(fn_avg, ftype='stc')


def focal_ROIs(fn_stc, radius, tmin, tmax, fn_ana_list, min_subject='fsaverage'):
    ''' 
        Concentrate the anatomical labels. 
        Parameter
        ---------
        fn_stc: string
            Path of common STC.
        radius: float 
            The radius size of each functional labels.
        tmin: float
            The start timepoint(second) of common STC.
        tmax: float
            The end timepoint(second) of common STC.
        min_subject: string
            The common subject.
        fn_ana_list: string
            The path of the file including pathes of anatomical labels.
    '''     
    min_path = subjects_dir+'/fsaverage'
    # Make sure the target path is exist
    funcs_path = min_path + '/dSPM_ROIs/anno_ROIs/funcs/'
    reset_directory(funcs_path)
    # Read the stc for concentrating the anatomical labels
    stc = mne.read_source_estimate(fn_stc)
    with open(fn_ana_list, 'r') as fl:
                file_list = [line.rstrip('\n') for line in fl]
    fl.close()
    for f in file_list:
        label = mne.read_label(f)
        #stc_mean_label = stc_mean.in_label(label)
        stc_label = stc.in_label(label)
        src_pow = np.sum(stc_label.data ** 2, axis=1)
        if label.hemi == 'lh':
            h = 0
        elif label.hemi == 'rh':
            h = 1
        # Get the max MNE value within each ROI
        seed_vertno = stc_label.vertices[h][np.argmax(src_pow)]
        func_label = mne.grow_labels(min_subject, seed_vertno,
                                    extents=radius, hemis=h,
                                    subjects_dir=subjects_dir,
                                    n_jobs=1)
        func_label = func_label[0]
        func_label.save(funcs_path + '%s' %label.name)

def _merge_rois(mer_path, label_list):
    """
    subfunctions of apply_merge
    Parameter
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
                    label_name = '%s_' % (new_pre) + \
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

def apply_merge(labels_path):
    ''' 
        Merge the concentrated ROIs. 
        Parameter
        ---------
        labels_path: string.
            The path of concentrated labels.
    ''' 
    import glob, shutil
    mer_path = labels_path + '/merge/'
    reset_directory(mer_path)
    source = []
    source_path = labels_path + '/funcs/'
    source = glob.glob(os.path.join(source_path, '*.*'))
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
        len_class = _merge_rois(mer_path, label_list)
        if len_class == len(label_list):
            reducer = False  
           
########################################
#  Causality analysis
#######################################
''' 
   Epochs of each condition are inversed into STCs.
'''
def apply_inverse_epo(fnepo, method='MNE', event='LLst', min_subject='fsaverage', 
                  snr=3.0, tmax=0.7):
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
        snr: float
            Signal to noise ratio for inverse solution. 
        tmax:float
            The end timepoint(second) of each epoch.
    '''
    #Get the default subjects_dir
    from mne.minimum_norm import apply_inverse_epochs
    fnlist = get_files_from_list(fnepo)
    # loop across all filenames
    for fname in fnlist:
        fn_path = os.path.split(fname)[0]
        name = os.path.basename(fname)
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
        epochs.crop(0, tmax)
        noise_cov = mne.read_cov(fn_cov)
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
                            % (str(s)), ftype='stc')
            s = s + 1
            
def cal_labelts(stcs_path, condition='LLst', min_subject='fsaverage'):
    '''Extract stcs from special ROIs, and store them for funther causality
       analysis.
       Parameter
       ---------
       stcs_path: string
            The path of stc's epochs.
       condition: string
            The condition for experiments.
       min_subject: the subject for common brain
    '''
    path_list = get_files_from_list(stcs_path)
    # loop across all filenames
    for stcs_path in path_list:
        subjects_dir = os.environ['SUBJECTS_DIR']
        minpath = subjects_dir + '/%s' % (min_subject)
        caupath = stcs_path[:stcs_path.rfind('/%s' %condition)] 
        fn_stcs_labels = caupath + '/%s_labels_ts.npy' % (condition)
        _, _, files = os.walk(stcs_path).next()
        trials = len(files) / 2
        srcpath = minpath + '/bem/fsaverage-ico-4-src.fif'
        src_inv = mne.read_source_spaces(srcpath)
        # Get unfiltered and morphed stcs
        stcs = []
        i = 0
        while i < trials:
            fn_stc = stcs_path + 'trial%s_fsaverage' % (str(i))
            stc = mne.read_source_estimate(fn_stc + '-lh.stc',
                                           subject=min_subject)
            stcs.append(stc)
            i = i + 1
        # Get common labels
        list_file = minpath + '/dSPM_ROIs/anno_ROIs/func_label_list.txt'
        with open(list_file, 'r') as fl:
                  file_list = [line.rstrip('\n') for line in fl]
        fl.close()
        rois = []
        labels = []
        for f in file_list:
            label = mne.read_label(f)
            labels.append(label)
            rois.append(label.name)
        # Extract stcs in common labels
        label_ts = mne.extract_label_time_course(stcs, labels, src_inv,
                                                 mode='pca_flip',
                                                 return_generator=False)
        #make label_ts's shape as (sources, samples, trials)
        label_ts = np.asarray(label_ts).transpose(1, 2, 0)
        np.save(fn_stcs_labels, label_ts)

def _mvdetrend(data):
    '''Multivariate polynomial detrend of time series data.
       refer:http://users.sussex.ac.uk/~lionelb/MVGC/html/mvdetrend.html.
       Parameter
       ---------
       data: narray
            multi-trial time series data, shape as (variables, obsevations,
             trials).
       Return
       ------
       Y: detrended time series.
    '''
    n, m, N = data.shape
    pdeg = 1
    pdeg = pdeg * np.ones(n)
    x = np.arange(m) + 1
    mu = np.mean(x)
    sig = np.std(x, ddof=1)
    x = (x - mu) / sig
    P = np.zeros((n, m))
    p = []
    for i in xrange(n):
        d = pdeg[i]
        d1 = int(d + 1)
        V = np.zeros((d1, m))
        V[d1 - 1, :] = np.ones(m)
        j = int(d)
        while j > 0:
            V[j - 1, :] = x * V[j, :]
            j = j - 1
        temp = np.mean(data[i, :, :], axis=-1)
        p.append(np.linalg.lstsq(V.T, temp)[0].T)
        P[i, :] = p[i][0] * np.ones((1, m))
        for j in range(1, d1):
            P[i, :] = x * P[i, :] + p[i][j]
    Y = np.zeros(data.shape)
    for r in xrange(N):
        Y[:, :, r] = data[:, :, r] - P
    return Y

def normalize_data(fn_ts, factor=1):
    '''
       Before causal model construction, labelts need to be normalized further:
        1) Downsampling for reducing the time consuming.
        2) Apply Z-scoring to each STC.
       Parameter
       ---------
       fnts: string
           The file name of representative STCs for each ROI.
       factor: int
          The factor for downsampling.
    '''
    from scipy import signal
    path_list = get_files_from_list(fn_ts)
    # loop across all filenames
    for fnts in path_list:
        fnnorm = fnts[:fnts.rfind('.npy')] + ',%d-norm.npy' % factor
        label_ts = np.load(fnts)
        assert label_ts.shape[1] / factor > label_ts.shape[-1], \
            ('Trial length can not be smaller than the amount of trials.')
        if factor > 1:
            #downsampled
            dw_data = signal.decimate(label_ts, q=factor, axis=1)
        else:
            dw_data = label_ts
        #zscore
        d_mu = dw_data.mean(axis=1, keepdims=True)
        d_std = dw_data.std(axis=1, ddof=1, keepdims=True)
        z_data = (dw_data - d_mu) / d_std
        np.save(fnnorm, z_data)

def _plot_morder(bic, morder, figmorder):
    '''
       Parameter
       ---------
       bic: array
           BIC values for each model order lower than 'p_max'.
       morder: int
          The optimized model order.
       figmorder: string
          The path for storing the plot.
    '''
    import matplotlib.pyplot as plt
    plt.figure()
    h0, = plt.plot(np.arange(len(bic)) + 1, bic, 'r', linewidth=3)
    plt.legend([h0], ['BIC: %d' %morder])
    plt.xlabel('order')
    plt.ylabel('BIC')
    plt.title('Model Order')
    plt.show()
    plt.savefig(figmorder, dpi=100)
    plt.close()

def _model_order(fnnorm, p_max=100):

    """ Calculate the optimized model order for VAR 
        models from time series data.

        Parameters
        ----------
        fnnorm: string
            The file name of model order estimation.
        p_max: int 
            The upper limit for model order estimation.
        Returns
        ----------
        morder: int
            The optimized BIC model order.

    """
    figmorder = fnnorm[:fnnorm.rfind('.npy')] + '.png'
    X = np.load(fnnorm)
    n, m, N = X.shape
    if p_max == 0:
        p_max = m - 1
    q = p_max
    q1 = q + 1
    XX = np.zeros((n, q1, m + q, N))
    for k in xrange(q1):
        XX[:, k, k:k + m, :] = X
    q1n = q1 * n
    bic = np.empty((q, 1))
    bic.fill(np.nan)
    I = np.identity(n)
    # initialise recursion
    AF = np.zeros((n, q1n))#forward AR coefficients
    AB = np.zeros((n, q1n))#backward AR coefficients
    k = 1
    kn = k * n
    M = N * (m - k)
    kf = range(0, kn)
    kb = range(q1n - kn, q1n)
    XF = np.reshape(XX[:, 0:k, k:m, :], (kn, M), order='F')
    XB = np.reshape(XX[:, 0:k, k - 1:m - 1, :], (kn, M), order='F')
    #import pdb
    #pdb.set_trace()
    CXF = np.linalg.cholesky(XF.dot(XF.T)).T
    CXB = np.linalg.cholesky(XB.dot(XB.T)).T
    AF[:, kf] = np.linalg.solve(CXF.T, I)
    AB[:, kb] = np.linalg.solve(CXB.T, I)
    while k <= q - 1:
        #print('model order = %d' % k)
        #import pdb
        #pdb.set_trace()
        tempF = np.reshape(XX[:, 0:k, k:m, :], (kn, M), order='F')
        af = AF[:, kf]
        EF = af.dot(tempF)
        tempB = np.reshape(XX[:, 0:k, k - 1:m - 1, :], (kn, M), order='F')
        ab = AB[:, kb]
        EB = ab.dot(tempB)
        CEF = np.linalg.cholesky(EF.dot(EF.T)).T
        CEB = np.linalg.cholesky(EB.dot(EB.T)).T
        R = np.dot(np.linalg.solve(CEF.T, EF.dot(EB.T)), np.linalg.inv(CEB))
        CRF = np.linalg.cholesky(I - R.dot(R.T)).T
        CRB = np.linalg.cholesky(I - (R.T).dot(R)).T
        k = k + 1
        kn = k * n
        M = N * (m - k)
        kf = np.arange(kn)
        kb = range(q1n - kn, q1n)
        AFPREV = AF[:, kf]
        ABPREV = AB[:, kb]
        AF[:, kf] = np.linalg.solve(CRF.T, AFPREV - R.dot(ABPREV))
        AB[:, kb] = np.linalg.solve(CRB.T, ABPREV - R.T.dot(AFPREV))
        E = np.linalg.solve(AF[:, :n], AF[:, kf]).dot(np.reshape(XX[:, :k, k:m,
                                                      :], (kn, M), order='F'))
        DSIG = np.linalg.det((E.dot(E.T)) / (M - 1))
        i = k - 1
        K = i * n * n
        L = -(M / 2) * math.log(DSIG)
        bic[i - 1] = -2 * L + K * math.log(M)
    #morder = np.nanmin(bic), np.nanargmin(bic) + 1
    morder = np.nanargmin(bic) + 1
    _plot_morder(bic, morder, p_max, figmorder)
    return morder

def _tsdata_to_var(X,p):
    """ Calculate coefficients and recovariance and noise covariance of 
        the optimized model order.
        ref: http://users.sussex.ac.uk/~lionelb/MVGC/html/tsdata_to_var.html
        Parameters
        ----------
        X: narray, shape (n_sources, n_times, n_epochs)
              The data to estimate the model order for.
        p: int, the optimized model order.
        Returns
        ----------
        A: array, coefficients of the specified model
        SIG:array, recovariance of this model
        E:  array, noise covariance of this model
    """
    n, m, N = X.shape
    p1 = p + 1
    A = np.nan
    SIG = np.nan
    E = np.nan
    q1n = p1 * n
    I = np.eye(n)
    XX = np.zeros((n, p1, m + p, N))
    for k in xrange(p1):
        XX[:, k, k:k + m, :] = X
    AF = np.zeros((n, q1n))
    AB = np.zeros((n, q1n))
    k = 1
    kn = k * n
    M = N * (m - k)
    kf = range(0, kn)
    kb = range(q1n - kn, q1n)
    XF = np.reshape(XX[:, 0:k, k:m, :], (kn, M), order='F')
    XB = np.reshape(XX[:, 0:k, k - 1:m - 1, :], (kn, M), order='F')
    CXF = np.linalg.cholesky(XF.dot(XF.T)).T
    CXB = np.linalg.cholesky(XB.dot(XB.T)).T
    AF[:, kf] = np.linalg.solve(CXF.T, I)
    AB[:, kb] = np.linalg.solve(CXB.T, I)
    while k <= p:
        tempF = np.reshape(XX[:, 0:k, k:m, :], (kn, M), order='F')
        af = AF[:, kf]
        EF = af.dot(tempF)
        tempB = np.reshape(XX[:, 0:k, k - 1:m - 1, :], (kn, M), order='F')
        ab = AB[:, kb]
        EB = ab.dot(tempB)
        CEF = np.linalg.cholesky(EF.dot(EF.T)).T
        CEB = np.linalg.cholesky(EB.dot(EB.T)).T
        R = np.dot(np.linalg.solve(CEF.T, EF.dot(EB.T)), np.linalg.inv(CEB))
        RF = np.linalg.cholesky(I - R.dot(R.T)).T
        RB = np.linalg.cholesky(I - (R.T).dot(R)).T
        k = k + 1
        kn = k * n
        M = N * (m - k)
        kf = np.arange(kn)
        kb = range(q1n - kn, q1n)
        AFPREV = AF[:, kf]
        ABPREV = AB[:, kb]
        AF[:, kf] = np.linalg.solve(RF.T, AFPREV - R.dot(ABPREV))
        AB[:, kb] = np.linalg.solve(RB.T, ABPREV - R.T.dot(AFPREV))
    E = np.linalg.solve(AFPREV[:, :n], EF)
    SIG = (E.dot(E.T)) / (M - 1)
    E = np.reshape(E, (n, m - p, N), order='F')
    temp = np.linalg.solve(-AF[:, :n], AF[:, n:])
    A = np.reshape(temp, (n, n, p), order='F')
    return A, SIG, E

# Whiteness test                      
def _erfcc(x):
    """Complementary error function."""
    z = abs(x)
    t = 1. / (1. + 0.5 * z)
    r = t * math.exp(-z * z - 1.26551223 + t * (1.00002368 + t * (.37409196 +
                     t * (.09678418 + t * (-.18628806 + t * (.27886807 +
                    t * (-1.13520398 + t * (1.48851587 + t * (-.82215223 +
                                                      t * .17087277)))))))))
    if (x >= 0.):
    	return r
    else:
    	return 2. - r
    	
def _normcdf(x, mu, sigma):
    t = x - mu
    y = 0.5 * _erfcc(-t / (sigma * math.sqrt(2.0)))
    if y > 1.0:
        y = 1.0
    return y

def _durbinwatson(X, E):
    n, m = X.shape
    dw = np.sum(np.diff(E, axis=0) ** 2) / np.sum(E ** 2, axis=0)
    A = np.dot(X, X.T)
    from scipy.signal import lfilter
    B = lfilter(np.array([-1, 2, -1]), 1, X.T, axis=0)
    temp = X[:, [0, m - 1]] - X[:, [1, m - 2]]
    B[[0, m - 1], :] = temp.T
    D = np.dot(B, np.linalg.pinv(A))
    C = X.dot(D)
    nu1 = 2 * (m - 1) - np.trace(C)
    nu2 = 2 * (3 * m - 4) - 2 * np.trace(B.T.dot(D)) + np.trace(C.dot(C))
    mu = nu1 / (m - n)
    sigma = math.sqrt(2./((m - n) * (m - n + 2)) * (nu2 - nu1 * mu))
    pval = _normcdf(dw, mu, sigma)
    pval = 2 * min(pval, 1-pval)
    return dw, pval

def _whiteness(X,E):

    """Durbin-Watson test for whiteness (no serial correlation) of
       VAR residuals.
       Prarameters
       -----------
       X: array
          Multi-trial time series data.
       E: array
          Residuals time series.

       Returns
       -------
       dw: array
           Vector of Durbin-Watson statistics.
       pval: array
             Vector of p-values.
    """
    n, m, N = X.shape
    dw = np.zeros(n)
    pval = np.zeros(n)
    for i in xrange(n):
        Ei = np.squeeze(E[i, :, :])
        e_a, e_b = Ei.shape
        tempX = np.reshape(X, (n, m * N), order='F')
        tempE = np.reshape(Ei, (e_a * e_b), order='F')
        dw[i], pval[i] = _durbinwatson(tempX, tempE)
    return dw, pval

# Consistency test                    
def _consistency(X, E):
    '''
       Prarameters
       -----------
       X: array
          Multi-trial time series data.
       E: array
          Residuals time series.
       Returns
       -------
       cons: float
        consistency test measurement.

    '''
    n, m, N = X.shape
    p = m - E.shape[1]
    X = X[:, p:m, :]
    n1, m1, N1 = X.shape
    X = np.reshape(X, (n1, m1 * N1), order='F')
    E = np.reshape(E, (n1, m1 * N1), order='F')
    s = N * (m - p)
    Y = X - E
    Rr = X.dot(X.T) / (s - 1)
    Rs = Y.dot(Y.T) / (s - 1)
    cons = 1 - np.linalg.norm(Rs - Rr, 2) / np.linalg.norm(Rr, 2)
    return cons

def model_estimation(fn_norm, thr_cons=0.8, whit_min=1., whit_max=3., pmax=100):
    '''
       Check the statistical evalutions of the MVAR model corresponding the
       optimized morder.
       Reference
       ---------
       Granger Causal Connectivity Analysis: A MATLAB Toolbox, Anil K. Seth
       (2009)
       Parameters
       ----------
        fn_norm: string
            The file name of model order estimation.
        thr_cons:float
            The threshold of consistency evaluation.
        whit_min:float
            The lower limit for whiteness evaluation.
        whit_max:float
            The upper limit for whiteness evaluation.
        pmax: int
            The maximum value for model order estimation.
    '''
    import scot
    path_list = get_files_from_list(fn_norm)
    # loop across all filenames
    for fnnorm in path_list:
        fneval = fnnorm[:fnnorm.rfind('.npy')] + '_evaluation.txt'
        morder = _model_order(fnnorm, pmax)
        X = np.load(fnnorm)
        A, SIG, E = _tsdata_to_var(X, morder)
        whi = False
        dw, pval = _whiteness(X, E)
        if np.all(dw < whit_max) and np.all(dw > whit_min):
            whi = True
        cons = _consistency(X, E)
        X = X.transpose(1, 0, 2)
        mvar = scot.var.VAR(morder)
        mvar.fit(X)
        is_st = mvar.is_stable()
        if cons < thr_cons or is_st == False or whi == False:
            print fnnorm
        #assert cons > thr_cons and is_st and whi, ('Consistency, whiteness, stability:\
        #                                    %f, %s, %s' %(cons, str(whi), str(is_st)))
        with open(fneval, "a") as f:
            f.write('model_order, whiteness, consistency, stability: %d, %s, %f, %s\n' 
                    %(morder, str(whi), cons, str(is_st)))

def causal_analysis(fn_norm, pmax=100, method='PDC'):
    '''
        Calculate causality matrices of real data and surrogates.
        Parameters
        ----------
        fnnorm: string
            The file name of model order estimation.
        morder: int
            The optimized model order.
        method: string
            causality measures.
    '''
    import scot.connectivity_statistics as scs
    from scot.connectivity import connectivity
    import scot
    path_list = get_files_from_list(fn_norm)
    # loop across all filenames
    for fnnorm in path_list:
        fncau = fnnorm[:fnnorm.rfind('.npy')] + ',cau.npy'
        fnsurr = fnnorm[:fnnorm.rfind('.npy')] + ',surrcau.npy'
        X = np.load(fnnorm)
        X = X.transpose(1, 0, 2)
        morder = _model_order(fnnorm, pmax)
        mvar = scot.var.VAR(morder)
        c_surrogates = scs.surrogate_connectivity(method, X, mvar,
                                                  repeats=1000)
        mvar.fit(X)
        con = connectivity(method, mvar.coef, mvar.rescov)
        np.save(fncau, con)
        np.save(fnsurr, c_surrogates)

def _plot_hist(con_b, surr_b, fig_out):
    '''
     Plot the distribution of real and surrogates' causality results.
     Parameter
     ---------
     con_b: array
            Causality matrix.
     surr_b: array
            Surrogates causality matix.
     fig_out: string
            The path to store this distribution.
    '''
    import matplotlib.pyplot as pl
    fig = pl.figure('Histogram - surrogate vs real')
    c = con_b  # take a representative freq point
    fig.add_subplot(211, title='Histogram - real connectivity')
    pl.hist(c, bins=100)  # plot histogram with 100 bins (representative)
    s = surr_b
    fig.add_subplot(212, title='Histogram - surrogate connectivity')
    pl.hist(s, bins=100)  # plot histogram
    pl.show()
    pl.savefig(fig_out)
    pl.close()

def _plot_thr(con_b, fdr_thr, max_thr, fig_out):
    '''plot the significant threshold of causality analysis.
       Parameter
       ---------
       con_b: array
            Causality matrix.
       fdr_thr: float
           Threshold combining with FDR.
       max_thr: float
           Threshold from the maximum causality value of surrogates.
       fig_out: string
           The path to store the threshold plots.
    '''
    import matplotlib.pyplot as pl
    pl.close('all')
    c = np.unique(con_b)
    pl.plot(c, 'k', label='real con')
    xmin, xmax = pl.xlim()
    pl.hlines(fdr_thr, xmin, xmax, linestyle='--', colors='k',
              label='p=0.05(FDR):%.2f' % fdr_thr, linewidth=2)
    pl.hlines(max_thr, xmin, xmax, linestyle='--', colors='g',
              label='Max surr', linewidth=2)
    pl.legend()
    pl.xlabel('points')
    pl.ylabel('causality values')
    pl.show()
    pl.savefig(fig_out)
    pl.close()

def sig_thresh(cau_list, condition=None, freqs = [(8, 12), (30, 40)],
               sfreq=678, alpha=0.05, factor=1, min_subject='fsaverage'):
    '''
       Evaluate the significance for each pair's causal interactions.
       Parameter
       ---------
       fn_cau: string
            The file path of causality matrices.
       freqs: list
            The list of interest frequency band.
       sfreq: float
            The sampling rate.
       alpha: significant factor
       factor: int
            The downsampled factor, it is used to compute the frequency
            resolution.
    '''
    from mne.stats import fdr_correction
    from scipy import stats
    path_list = get_files_from_list(cau_list)
    # loop across all filenames
    for fncau in path_list:
        fnsurr = fncau[:fncau.rfind(',cau.npy')] + ',surrcau.npy'
        cau_path = os.path.split(fncau)[0]
        assert condition != None, ('Please assign one condition of the\
                                    experiments.')
        sig_path = cau_path + '/sig_con/'
        set_directory(sig_path)
        con = np.load(fncau)
        surr_subject = np.load(fnsurr)
        nfft = con.shape[-1]
        delta_F = sfreq / float(2 * nfft * factor)
        #freqs = [(4, 8), (8, 12), (12, 18), (18, 30), (30, 70), (60, 90)]
        sig_freqs = []
        nfreq = len(freqs)
        for ifreq in range(nfreq):
            fmin, fmax = int(freqs[ifreq][0] / delta_F), int(freqs[ifreq][1] /
                                                             delta_F)
            con_band = np.mean(con[:, :, fmin:fmax + 1], axis=-1)
            np.fill_diagonal(con_band, 0)
            surr_band = np.mean(surr_subject[:, :, :, fmin:fmax + 1], axis=-1)
            r, s, _ = surr_band.shape
            for i in xrange(r):
                ts = surr_band[i]
                np.fill_diagonal(ts, 0)
            con_b = con_band.flatten()
            con_b = con_b[con_b > 0]
            surr_b = surr_band.reshape(r, s * s)
            surr_b = surr_b[surr_b > 0]
            zscore = (con_b - np.mean(surr_b, axis=0)) / np.std(surr_b, axis=0)
            p_values = stats.norm.pdf(zscore)
            accept, _ = fdr_correction(p_values, alpha)
            assert accept.any() == True, ('Normalized amplitude values are not\
            statistically significant. Please try with lower alpha value.')
            z_thre = np.abs(con_b[accept]).min()
            histout = sig_path + '%s,%d-%d,distribution.png'\
                                % (condition, freqs[ifreq][0], freqs[ifreq][1])
            throut = sig_path + '%s,%d-%d,threshold.png'\
                        % (condition, freqs[ifreq][0], freqs[ifreq][1])
            _plot_hist(con_b, surr_b, histout)
            _plot_thr(con_b, z_thre, surr_band.max(), throut)
            con_band[con_band < z_thre] = 0
            con_band[con_band >= z_thre] = 1
            sig_freqs.append(con_band)
        sig_freqs = np.array(sig_freqs)
        np.save(sig_path + '%s_sig_con_band.npy' %condition, sig_freqs)
        #return sig_freqs

def group_causality(sig_list, condition=None, freqs = [(8, 12), (30, 40)], 
                    min_subject='fsaverage', submount=10):

    """
        make group causality analysis, by evaluating significant matrices across
        subjects.
        ----------
        sig_list: list
            The path list of individual significant causal matrix.
        condition: string
            One condition of the experiments.
        freqs: list
            The list of interest frequency band.
        min_subject: string
            The subject for the common brain space.
        submount: int
            Significant interactions come out at least in 'submount' subjects.
    """
    import matplotlib.pylab as plt
    cau_path = subjects_dir + '/%s/causality' %min_subject
    set_directory(cau_path)
    assert condition != None, ('Please assign one condition of the\
                               experiments.')
    sig_caus = []
    for f in sig_list:
        sig_cau = np.load(f)
        sig_caus.append(sig_cau)
    sig_caus = np.array(sig_caus)
    sig_group = sig_caus.sum(axis=0)
    #freqs = [(4, 8), (8, 12), (12, 18), (18, 30), (30, 40)]
    plt.close()
    for i in xrange(len(sig_group)):
        fmin, fmax = freqs[i][0], freqs[i][1]
        cau_band = sig_group[i]
        cau_band[cau_band < submount] = 0
        plt.imshow(cau_band, interpolation='nearest')
        np.save(cau_path + '/%s_%s_%sHz.npy' %
                    (condition, str(fmin), str(fmax)), cau_band)
        v = np.arange(cau_band.max())
        plt.colorbar(ticks=v)
        # plt.colorbar()
        plt.show()
        plt.savefig(cau_path + '/%s_%s_%sHz.png' %
                    (condition, str(fmin), str(fmax)), dpi=100)
        plt.close()

def diff_mat(incon_event, fmin, fmax, min_subject='fsaverage', con_event=['LLst', 'RRst']):
    """
        make comparisons between two conditions' group causal matrices
        ----------
        con_event: list
            The list of congruent conditions.
        incon_event: string
            The name of incongruent condition.  
        fmin, fmax:int
            The interest bandwidth.    
        min_subject: string
            The subject for the common brain space.
        
    """
    mat_dir = subjects_dir + '/fsaverage/causality'
    fn_con1 = mat_dir + '/%s_%d_%dHz.npy' %(con_event[0], fmin, fmax)
    fn_con2 = mat_dir + '/%s_%d_%dHz.npy' %(con_event[1], fmin, fmax)
    fn_incon = mat_dir + '/%s_%d_%dHz.npy' %(incon_event, fmin, fmax)
    con_cau1 = np.load(fn_con1)
    con_cau2 = np.load(fn_con2)
    con_cau = con_cau1 + con_cau2
    incon_cau = np.load(fn_incon)
    con_cau[con_cau > 0] = 1
    incon_cau[incon_cau > 0] = 1
    dif_cau = incon_cau - con_cau
    dif_cau[dif_cau < 0] = 0
    fn_dif = mat_dir + '/%s_%s_%d.npy' %(incon_event, con_event, fmin)
    np.save(fn_dif, dif_cau)
    print np.argwhere(dif_cau)