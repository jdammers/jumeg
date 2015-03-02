# -*- coding: utf-8 -*-

subjects_dir = '/home/qdong/freesurfer/subjects/'
MNI_dir = subjects_dir + 'fsaverage/'
fn_inv = MNI_dir + 'bem/fsaverage-ico-4-src.fif' 
subject_id = 'fsaverage'
'''1)'make_inverse_epochs':morph unfiltered individual 
      epochs(including 'trigger' and 'response' events)
   2) 'causal_analysis':make causality analysis and statistical evaluation
   
'''
#######################################################
#                                                     #
# small utility function to handle file lists         #
#                                                     #
#######################################################
def get_files_from_list(fin):
    ''' Return files as iterables lists '''
    if isinstance(fin, list):
        fout = fin
    else:
        if isinstance(fin, str):
            fout = list([fin])
        else:
            fout = list(fin)
    return fout
    
####################################################################
# inverse the epochs of individual raw data and morph into the
# common brain space
####################################################################        
def make_inverse_epochs(fname_raw):
    from mne.minimum_norm import (apply_inverse_epochs)
    import mne, os
    from mne.fiff import Raw
    fnlist = get_files_from_list(fname_raw)
    # loop across all filenames
    for fn_raw in fnlist:
        #extract the subject infromation from the file name
        name = os.path.basename(fn_raw)
        subject = name.split('_')[0]
        
        fn_inv = fn_raw.split('.fif')[0] + '-inv.fif'
        subject_path = subjects_dir + subject
        fn_cov = subject_path + '/MEG/%s,empty-cov.fif' %subject
        fn_trans = subject_path + '/MEG/%s-trans.fif' %subject
        fn_src = subject_path + '/bem/%s-ico-4-src.fif' %subject
        fn_bem = subject_path + '/bem/%s-5120-5120-5120-bem-sol.fif' %subject
        snr = 3.0
        lambda2 = 1.0 / snr ** 2
        # Load data
        raw = Raw(fn_raw, preload=True)
        tmin, tmax = -0.2, 0.5
        events = mne.find_events(raw, stim_channel='STI 014')
        picks = mne.fiff.pick_types(raw.info, meg=True, exclude='bads')
        epochs = mne.Epochs(raw, events, 1, tmin, tmax, proj=False, picks=picks, preload=True, reject=None)
        fwd = mne.make_forward_solution(epochs.info, mri=fn_trans, src=fn_src, 
                                    bem=fn_bem,fname=None, meg=True, eeg=False, 
                                    mindist=5.0,n_jobs=2, overwrite=True)
                                    
        fwd = mne.convert_forward_solution(fwd, surf_ori=True)
        noise_cov = mne.read_cov(fn_cov)
        noise_cov = mne.cov.regularize(noise_cov, epochs.info,
                                    mag=0.05, grad=0.05, proj=True)
        forward_meg = mne.pick_types_forward(fwd, meg=True, eeg=False)
        inverse_operator = mne.minimum_norm.make_inverse_operator(epochs.info, 
                                    forward_meg, noise_cov, loose=0.2, depth=0.8)
        mne.minimum_norm.write_inverse_operator(fn_inv, inverse_operator)
        # Compute inverse solution
        stcs = apply_inverse_epochs(epochs, inverse_operator, lambda2, "dSPM",
                                pick_ori=None)
        stcs_path = MNI_dir+'/stcs/%s' %subject
        isExists=os.path.exists(stcs_path)
        if not isExists:
            os.makedirs(stcs_path) 
        s = 0
        while s < len(stcs):
            stc_morph = mne.morph_data(subject, 'fsaverage', stcs[s], 4, smooth=4)
            stc_morph.save(stcs_path+'/%s_trial%s_fsaverage' %(subject, str(s)), ftype='stc')
            s = s +1
##################################################################
# Causal analysis
# 1) read the morphed individual epochs
# 2) compute the model order
# 3) make the significant threshold
# 4）get the plot of effective connectivities
##################################################################
def causal_analysis(subject,top_c=8):
    import numpy as np
    import mne, os
    subject_path = subjects_dir + 'fsaverage'
    stcs_path = subject_path + '/stcs/%s/' %subject
    inv_path = subject_path + '/bem/fsaverage-ico-4-src.fif'
    src_inv = mne.read_source_spaces(inv_path, add_geom=True)
    
    #Get unfiltered and morphed stcs
    stcs = []
    i = 0
    while i < 120:
        fn_stc = stcs_path + '%s_trial%s_fsaverage' %(subject, str(i))
        stc = mne.read_source_estimate(fn_stc+'-lh.stc', subject='fsaverage')
        stcs.append(stc)
        i = i + 1
    #Get common labels
    list_dirs = os.walk(subject_path + '/func_labels/common/') 
    labels = []
    rois = []
    for root, dirs, files in list_dirs: 
        for f in files: 
            label_fname = os.path.join(root, f) 
            label = mne.read_label(label_fname)
            labels.append(label)
            rois.append(f)
    #Extract stcs in common labels
    label_ts = mne.extract_label_time_course(stcs, labels, src_inv, mode='mean_flip',
                                         return_generator=False)
    
    # Causal analysis
    from scot.connectivity import connectivity
    import scot.connectivity_statistics as scs
    from scot.backend_sklearn import VAR
    import scot.plotting as splt
    from scipy import linalg
    import heapq
    import matplotlib.pylab as plt
    import make_model_order
    # rearrange data to fit scot's format
    label_ts = np.asarray(label_ts).transpose(2, 1, 0)
    # Model order estimation
    label_or = np.mean(label_ts, -1)
    label_or = label_or.T
    mu = np.mean(label_or, axis=1)
    label_or = label_or - mu[:, None]
    p, bic = make_model_order.compute_order(label_or, p_max=20)
    mvar = VAR(p)
    
    # generate connectivity surrogates under the null-hypothesis of no connectivity
    c_surrogates = scs.surrogate_connectivity('dDTF', label_ts, mvar, repeats=1000)
    c0 = np.percentile(c_surrogates, 99, axis=0)
    #sfreq = 1017.25 
    freqs=[(4, 7), (6, 9), (8, 12), (11, 15), (14, 20),(19, 30)]
    nfreq = len(freqs)
    mvar.fit(label_ts)
    con = connectivity('dDTF', mvar.coef, mvar.rescov)
    con_dif = con - c0

    for ifreq in range(nfreq):             
        fmin,fmax = freqs[ifreq][0],freqs[ifreq][1]
        #fig = splt.plot_connectivity_spectrum([c0], fs=sfreq, freq_range=[fmin, fmax], diagonal=-1)
        #splt.plot_connectivity_spectrum(con, fs=sfreq, freq_range=[fmin, fmax], diagonal=-1, fig=fig)
        #splt.show_plots()
        fig = plt.figure()
        con_band = np.mean(con_dif[:, :, fmin:fmax], axis=-1)
        np.fill_diagonal(con_band, 0)#ignore the dignoal values
        con_band[con_band<0] = 0#ignore the value less than significance
        sig_thr = heapq.nlargest(top_c,con_band.flatten())[-1]#get the top top_c largest significance CA
        con_band[con_band > sig_thr] = con_band.max()
        plt.imshow(con_band, interpolation='nearest', cmap=plt.cm.gray)
        v = np.linspace(0.0, con_band.max(), 10, endpoint=True)
        plt.colorbar(ticks=v)
        #plt.colorbar()
        plt.show()
        plt.savefig(stcs_path+'dDTF_%s_%s.png' %(str(fmin),str(fmax)), dpi=100)
        plt.close()