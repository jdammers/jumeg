import jumeg_causality
import jumeg
subjects_dir = '/home/qdong/freesurfer/subjects/'
MIN_path = subjects_dir + 'fsaverage'
path = MIN_path + '/Group_ROIs/standard/'
jumeg_causality.reset_directory(path)
for subject in ['101611', '110061', '109925', '201394', '202825', '108815']:
    subject_path = subjects_dir + subject
    raw_fname = subject_path + '/MEG/%s_audi_cued-raw.fif' % subject
    basename = raw_fname.split('-')[0]
    res_name, tri_name = 'STI 013', 'STI 014'
    #########################
    # ECG and EOG rejection #
    #########################
    fn_filt = basename + ',fibp1-45-raw.fif'
    fn_ica = basename + ',fibp1-45-ica.fif'
    jumeg.jumeg_preprocessing.apply_filter(raw_fname)
    jumeg.jumeg_preprocessing.apply_ica(fn_filt, n_components=0.99, decim=None)
    # perform cleaning on filtered data
    jumeg.jumeg_preprocessing.apply_ica_cleaning(fn_ica, n_pca_components=0.99,
                                                 unfiltered=False)
    # perform cleaning on unfiltered data
    jumeg.jumeg_preprocessing.apply_ica_cleaning(fn_ica, n_pca_components=0.99,
                                                 unfiltered=True)

    ###########################################################################
    # Extract interest components from filtered raw data without ECG and EOG#
    ###########################################################################
    fn_clean = basename + ',fibp1-45,ar-raw.fif'
    fn_ica2 = basename + ',fibp1-45,ar-ica.fif'
    fn_ctps_tri = basename + ',fibp1-45,ar,ctpsbr-trigger.npy'
    fn_ctps_res = basename + ',fibp1-45,ar,ctpsbr-response.npy'
    fn_ics_tri = basename + ',fibp1-45,ar,ctpsbr-trigger-ic_selection.txt'
    fn_ics_res = basename + ',fibp1-45,ar,ctpsbr-response-ic_selection.txt'
    
    # second ICA decomposition
    jumeg.jumeg_preprocessing.apply_ica(
        fn_clean, n_components=0.95, decim=None)
    # stimulus components extraction
    tmin, tmax = 0, 0.3
    conditions = ['trigger']
    jumeg.jumeg_preprocessing.apply_ctps(fn_ica2, tmin=tmin, tmax=tmax,
                                         name_stim=tri_name)
    jumeg.jumeg_preprocessing.apply_ctps_select_ic(fname_ctps=fn_ctps_tri,
                                                   threshold=0.1)
    jumeg.jumeg_preprocessing.apply_ica_select_brain_response(
        fn_clean, conditions=conditions, n_pca_components=0.95)

    # response components extraction
    tmin, tmax = -0.15, 0.15
    conditions = ['response']
    jumeg.jumeg_preprocessing.apply_ctps(fn_ica2, tmin=tmin, tmax=tmax,
                                         name_stim=res_name)
    jumeg.jumeg_preprocessing.apply_ctps_select_ic(fname_ctps=fn_ctps_res,
                                                   threshold=0.1)
    jumeg.jumeg_preprocessing.apply_ica_select_brain_response(
        fn_clean, conditions=conditions, n_pca_components=0.95)
    # interest components extraction
    conditions = ['trigger', 'response']
    jumeg.jumeg_preprocessing.apply_ica_select_brain_response(
        fn_clean, conditions=conditions, n_pca_components=0.95)
    #############################################################################
    # Extract interest components from unfiltered raw data without ECG and EOG  #
    ############################################################################
    #Decompose the raw data
    fn_clean_unfilt = basename + ',ar-raw.fif' 
    fn_ica2_unfilt = basename + ',ar-ica.fif' 
    fn_ctps_tri_unfilt = basename + ',ar,ctpsbr-trigger.npy'
    fn_ctps_res_unfilt = basename + ',ar,ctpsbr-response.npy'
    fn_ics_tri_unfilt = basename + ',ar,ctpsbr-trigger-ic_selection.txt' 
    jumeg.jumeg_preprocessing.apply_ica(fn_clean_unfilt,n_components=0.95, 
                                        decim=None)
    #Extract interest ICs related with Auditory events
    tmin, tmax = 0, 0.3
    jumeg.jumeg_preprocessing.apply_ctps(fn_ica2_unfilt, tmin=tmin, tmax=tmax, 
                                                    name_stim=tri_name)
    jumeg.jumeg_preprocessing.apply_ctps_select_ic(fname_ctps=fn_ctps_tri_unfilt)  
                                                
    #Extract interest ICs related with Motor events                                                 
    tmin, tmax = -0.15, 0.15 
    jumeg.jumeg_preprocessing.apply_ctps(fn_ica2_unfilt, tmin=tmin, tmax=tmax, 
                                                    name_stim=res_name)
    jumeg.jumeg_preprocessing.apply_ctps_select_ic(fname_ctps=fn_ctps_res_unfilt) 
    
    #recompose interest ICs 
    conditions=['trigger', 'response']                
    jumeg.jumeg_preprocessing.apply_ica_select_brain_response(fn_clean_unfilt, 
                                    conditions=conditions, n_pca_components=0.95) 
    ###########################
    # Noise covariance making #
    ###########################
    fn_empty_room = subject_path + '/MEG/%s-empty.fif' % subject
    jumeg.jumeg_preprocessing.apply_create_noise_covariance(
        fn_empty_room, require_filter=True)
    jumeg.jumeg_preprocessing.apply_create_noise_covariance(
        fn_empty_room, require_filter=False)
    ############################
    # inverse operator makeing #
    ############################
    # make average evoked data
    fn_tri = basename + ',fibp1-45,ar,trigger,ctpsbr-raw.fif'
    fn_res = basename + ',fibp1-45,ar,response,ctpsbr-raw.fif'
    fn_both = basename + ',fibp1-45,ar,trigger,response,ctpsbr-raw.fif'
    jumeg.jumeg_preprocessing.apply_average(fn_tri, name_stim=tri_name,
                                            tmin=0., tmax=0.3)
    jumeg.jumeg_preprocessing.apply_average(fn_res, name_stim=res_name,
                                            tmin=-0.15, tmax=0.15)
    jumeg.jumeg_preprocessing.apply_average(fn_both)

    # prepare STC for ROI definition
    fn_tri_evoked = basename + ',fibp1-45,ar,trigger,ctpsbr,trigger-ave.fif'
    jumeg_causality.apply_inverse(fn_tri_evoked, MIN_subject='fsaverage',
                                  subjects_dir=subjects_dir, unfiltered=False)
    fn_res_evoked = basename + ',fibp1-45,ar,response,ctpsbr,response-ave.fif'
    jumeg_causality.apply_inverse(fn_res_evoked, MIN_subject='fsaverage',
                                  subjects_dir=subjects_dir, unfiltered=False)
    fn_both_evoked = basename + ',fibp1-45,ar,trigger,response,ctpsbr,trigger-ave.fif'
    jumeg_causality.apply_inverse(fn_both_evoked, MIN_subject='fsaverage',
                                  subjects_dir=subjects_dir, unfiltered=False)
    # prepare STCs for causality analysis
    fn_unfilt = basename + ',ar,trigger,response,ctpsbr-raw.fif'
    jumeg_causality.apply_inverse(fn_unfilt,subjects_dir=subjects_dir,
                                unfiltered=True)
    # ROI definition
    path = MIN_path + '/STC_ROI/%s/evROIs/' % subject
    jumeg_causality.reset_directory(path)
    fn_stc_res = MIN_path + \
        '/STC_ROI/%s/%s_response-lh.stc' % (subject, subject)
    fn_stc_tri = MIN_path + \
        '/STC_ROI/%s/%s_trigger-lh.stc' % (subject, subject)
    jumeg_causality.apply_rois(fn_stc_res, subjects_dir=subjects_dir, thr=99)
    jumeg_causality.apply_rois(fn_stc_tri, subjects_dir=subjects_dir, thr=99)
    jumeg_causality.merge_rois(subjects_dir=subjects_dir, subject=subject)
    fn_stc_both = MIN_path + \
        '/STC_ROI/%s/%s_trigger,response-lh.stc' % (subject, subject)
    jumeg_causality.stan_rois(fn_stc_both, subjects_dir=subjects_dir)

jumeg_causality.merge_rois(subjects_dir=subjects_dir, group=True)
jumeg_causality.group_rois(am_sub=4, subjects_dir=subjects_dir)



