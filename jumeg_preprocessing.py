#################################################################
#
# apply filter on (raw) data
#
#################################################################
def apply_filter(fname_raw, flow=1, fhigh=45, order=4, njobs=4):
    
    ''' Applies the MNE butterworth filter to a list of raw files. '''

    import mne

    filter_type = 'butter'
    filt_method = 'fft'

    if isinstance(fname_raw, list):
        fnraw = fname_raw
    else:
        if isinstance(fname_raw, str):
            fnraw = list([fname_raw]) 
        else:
            fnraw = list(fname_raw)

    # loop across all filenames
    for fname in fnraw:        
        print ">>> filter raw data: %0.1f - %0.1fHz..." % (flow, fhigh)
        # load raw data
        raw = mne.io.Raw(fname,preload=True)
        # filter raw data
        raw.filter(flow, fhigh,n_jobs=njobs,method=filt_method)
        # raw.filter(l_freq=flow_raw, h_freq=fhigh_raw, n_jobs=njobs, method='iir',
        #     iir_params={'ftype': filter_type, 'order': order})
        print ">>>> writing filtered data to disk..."
        #name_raw = fname[0:len(fname)-4]
        name_raw = fname.split('-')[0]
        fnfilt = name_raw+',bp' + "%d-%dHz" % (flow, fhigh)
        fnfilt = fnfilt + '-raw.fif'
        print 'saving: '+ fnfilt
        raw.save(fnfilt, overwrite=True)


#######################################################
# 
#  apply average on list of files
# 
#######################################################
def apply_average(filenames, name_stim='STI 014', event_id =None, postfix=None,
    tmin=-0.2, tmax=0.4, baseline = (None,0), save_plot=True, show_plot=False):

    ''' Performs averaging to a list of raw files. '''

    import mne, os
    import numpy as np
    import matplotlib.pylab as pl


    # Trigger or Response ?
    if name_stim == 'STI 014':      # trigger
        trig_name = 'trigger'    
    else:
        if name_stim == 'STI 013':   # response
            trig_name = 'response'
        else:
            trig_name = 'trigger'

    # check list of filenames        
    if isinstance(filenames, list):
        fnlist = filenames
    else:
        if isinstance(filenames, str):
            fnlist = list([filenames]) 
        else:
            fnlist = list(filenames)

    # loop across raw files
    fnavg = []    # collect output filenames
    for fname in fnlist:        
        name = os.path.split(fname)[1]
        print '>>> average raw data'
        print name
        # load raw data
        raw = mne.io.Raw(fname, preload=True)
        picks = mne.pick_types(raw.info, meg=True, exclude='bads')

        # stim events
        stim_events = mne.find_events(raw, stim_channel=name_stim) 
        nevents = len(stim_events)
        
        if nevents > 0:
            # for a specific event ID
            if event_id:
                ix = np.where(stim_events[:,2] == event_id)[0]
                stim_events = stim_events[ix,:]
            else:
                event_id = stim_events[0,2]  
                
            epochs = mne.Epochs(raw, events=stim_events,
                        event_id=event_id, tmin=tmin, tmax=tmax,
                        picks=picks, preload=True, baseline=baseline)
            avg = epochs.average()

            # save averaged data
            if (postfix):
                fnout = fname[0:len(fname)-4] + postfix+'.fif'
            else:    
                fnout = fname[0:len(fname)-4] + ',avg,'+trig_name+'.fif'

            avg.save(fnout)
            print 'saved:'+fnout
            fnavg.append(fnout)

            if (save_plot):
                plot_average(fnavg, show_plot=show_plot)

        else: 
            event_id = None
            print '>>> Warning: Event not found in file: '+fname   


#######################################################
# 
#  plot average from a list of files
# 
#######################################################
def plot_average(filenames, save_plot=True, show_plot=False):

    ''' Plot Signal average from a list of averaged files. '''

    import mne, os
    import matplotlib.pylab as pl


    # check list of filenames        
    if isinstance(filenames, list):
        fname = filenames
    else:
        if isinstance(filenames, str):
            fname = list([filenames]) 
        else:
            fname = list(filenames)

    # plot averages
    pl.ioff()  # switch off (interactive) plot visualisation
    factor = 1e15
    for fnavg in fname:
        name = fnavg[0:len(fnavg)-4] 
        basename = os.path.splitext(os.path.basename(name))[0]
        print fnavg
        # mne.read_evokeds provides a list or a single evoked based on the condition.
        # here we assume only one evoked is returned (requires further handling)
        avg = mne.read_evokeds(fnavg)[0]
        ymin, ymax = avg.data.min(), avg.data.max()
        ymin  *= factor*1.1
        ymax  *= factor*1.1
        fig = pl.figure(basename,figsize=(10,8), dpi=100) 
        pl.clf()
        pl.ylim([ymin, ymax])
        pl.xlim([avg.times.min(), avg.times.max()])
        pl.plot(avg.times, avg.data.T*factor, color='black')
        pl.title(basename)

        # save figure
        fnfig = os.path.splitext(fnavg)[0]+'.png'
        pl.savefig(fnfig,dpi=100)

    pl.ion()  # switch on (interactive) plot visualisation


#######################################################
# 
#  apply ICA for artifact rejection
# 
#######################################################
def apply_ica(fname_filtered, n_components=0.99, decim=None):
    ''' Applies ICA to a list of (filtered) raw files. '''

    import mne
    from mne.preprocessing import ICA
    import os


    if isinstance(fname_filtered, list):
        fnfilt = fname_filtered
    else:
        if isinstance(fname_filtered, str):
            fnfilt = list([fname_filtered]) 
        else:
            fnfilt = list(fname_filtered)

    # loop across all filenames
    for fname in fnfilt:                    
        name  = os.path.split(fname)[1]
        print ">>>> perform ICA signal decomposition on :  "+name
        # load filtered data
        raw = mne.io.Raw(fname,preload=True)
        picks = mne.pick_types(raw.info, meg=True, exclude='bads')
        # ICA decomposition
        ica = ICA(n_components=n_components, max_pca_components=None)
        ica.fit(raw, picks=picks, decim=decim, reject={'mag': 5e-12})
        # save ICA object 
        fnica_out = fname.strip('-raw.fif') + '-ica.fif'
      # fnica_out = fname[0:len(fname)-4]+'-ica.fif'
        ica.save(fnica_out)


#######################################################
# 
#  apply ICA-cleaning for artifact rejection
# 
#######################################################
def apply_ica_cleaning(fname_ica, n_pca_components=None, 
    flow_ecg=10, fhigh_ecg=20, flow_eog=1, fhigh_eog=10, threshold=0.3):

    ''' Performs artifact rejection based on ICA to a list of (ICA) files. '''


    import mne, os

    if isinstance(fname_ica, list):
        fnlist = fname_ica
    else:
        if isinstance(fname_ica, str):
            fnlist = list([fname_ica]) 
        else:
            fnlist = list(fname_ica)

    # loop across all filenames
    for fnica in fnlist:
        name  = os.path.split(fnica)[1]
        #basename = fnica[0:len(fnica)-4]
        basename = fnica.strip('-ica.fif')
        fnfilt = basename+'-raw.fif'
        #fnfilt = basename + '.fif'
        fnclean = basename+',ar-raw.fif'
        fnica_ar = basename+',ica-performance'
        print ">>>> perform artifact rejection on :"
        print '   '+name

        # load filtered data
        meg_raw = mne.io.Raw(fnfilt,preload=True)
        picks = mne.pick_types(meg_raw.info, meg=True, exclude='bads')
        # ICA decomposition
        ica = mne.preprocessing.read_ica(fnica)
        
        # get ECG and EOG related components
        ic_ecg = get_ics_cardiac(meg_raw, ica,
                                flow=flow_ecg, fhigh=fhigh_ecg, thresh=threshold)
        ic_eog = get_ics_ocular(meg_raw, ica,
                                flow=flow_eog, fhigh=fhigh_eog, thresh=threshold)
        ica.exclude += list(ic_ecg) + list(ic_eog)
        # ica.plot_topomap(ic_artifacts)
        ica.save(fnica)  # save again to store excluded

        # clean and save MEG data 
        if n_pca_components:
            npca = n_pca_components
        else:
            npca = picks.size
        print npca
        meg_clean = ica.apply(meg_raw, exclude=ica.exclude,
                              n_pca_components=npca, copy=True)
        meg_clean.save(fnclean, overwrite=True)

        # plot ECG, EOG averages before and after ICA 
        print ">>>> create performance image..."
        plot_performance_artifact_rejection(meg_raw, ica, fnica_ar,
                                show=False, verbose=False)


#######################################################
# 
#  determine occular related ICs
# 
#######################################################
def get_ics_ocular(meg_raw, ica, flow=1, fhigh=10,
    name_eog_hor = 'EOG 001', name_eog_ver = 'EOG 002',
    score_func = 'pearsonr', thresh=0.3):

    import mne
    import numpy as np

    # -----------------------------------
    # ICs related to ocular artifacts
    # -----------------------------------


    # Note: when using the following:
    #   - the filter settings are different
    #   - here we cannot define the filter range

    # vertical EOG
    # idx_eog_ver = [meg_raw.ch_names.index(name_eog_ver)]
    # eog_scores = ica.score_sources(meg_raw, meg_raw[idx_eog_ver][0])
    # eogv_idx = np.where(np.abs(eog_scores) > thresh)[0]
    # ica.exclude += list(eogv_idx)
    # ica.plot_topomap(eog_idx)
    
    # horizontal EOG
    # idx_eog_hor = [meg_raw.ch_names.index(name_eog_hor)]
    # eog_scores = ica.score_sources(meg_raw, meg_raw[idx_eog_hor][0])
    # eogh_idx = np.where(np.abs(eog_scores) > thresh)[0]
    # ica.exclude += list(eogh_idx)
    # ica.plot_topomap(eog_idx)
    # print [eogv_idx, eogh_idx]


    # vertical EOG
    idx_eog_ver = [meg_raw.ch_names.index(name_eog_ver)]
    eog_ver_filtered = mne.filter.band_pass_filter(meg_raw[idx_eog_ver, :][0], \
                            meg_raw.info['sfreq'], Fp1=flow, Fp2=fhigh)
    eog_ver_scores = ica.score_sources(meg_raw, \
                        target=eog_ver_filtered, score_func=score_func)
    ic_eog_ver = np.where(np.abs(eog_ver_scores) >= thresh)[0] +1  # plus 1 for any()
    if not ic_eog_ver.any(): 
        ic_eog_ver = np.array([0])

    # horizontal EOG
    idx_eog_hor = [meg_raw.ch_names.index(name_eog_hor)]
    eog_hor_filtered = mne.filter.band_pass_filter(meg_raw[idx_eog_hor, :][0], \
                            meg_raw.info['sfreq'], Fp1=flow, Fp2=fhigh)
    eog_hor_scores = ica.score_sources(meg_raw, \
                        target=eog_hor_filtered, score_func=score_func)
    ic_eog_hor = np.where(np.abs(eog_hor_scores) >= thresh)[0] +1 # plus 1 for any()
    if not ic_eog_hor.any(): 
        ic_eog_hor = np.array([0])
    
    # combine both  
    idx_eog = []
    for i in range(ic_eog_ver.size):
        ix = ic_eog_ver[i] -1
        if (ix >= 0):
            idx_eog.append(ix)
    for i in range(ic_eog_hor.size):
        ix = ic_eog_hor[i] -1
        if (ix >= 0):
            idx_eog.append(ix)

    return idx_eog


#######################################################
# 
#  determine cardiac related ICs
# 
#######################################################
def get_ics_cardiac(meg_raw, ica, flow=10, fhigh=20, tmin=-0.3, tmax=0.3, \
                    name_ecg = 'ECG 001', use_CTPS=True, \
                    score_func = 'pearsonr', thresh=0.3):
    '''
    Identify components with cardiac artefacts
    '''
    import mne, ctps
    import numpy as np

    event_id_ecg = 999

    # get and filter ICA signals
    ica_raw = ica.get_sources(meg_raw)
    ica_raw.filter(l_freq=flow, h_freq=fhigh, n_jobs=2, method='fft')
    # get R-peak indices in ECG signal
    idx_R_peak, _, _ = mne.preprocessing.find_ecg_events(meg_raw, \
                        ch_name=name_ecg, event_id=event_id_ecg, \
                        l_freq=flow, h_freq=fhigh,verbose=False)


    # -----------------------------------
    # default method:  CTPS
    #           else:  correlation
    # -----------------------------------
    if use_CTPS:
        # create epochs
        picks = np.arange(ica.n_components_)
        ica_epochs = mne.Epochs(ica_raw, events=idx_R_peak, \
                                event_id=event_id_ecg, tmin=tmin, \
                                tmax=tmax, baseline=None, \
                                proj=False, picks=picks, verbose=False)
        # compute CTPS
        _, pk, _ = ctps.compute_ctps(ica_epochs.get_data())

        pk_max = np.max(pk, axis=1)
        idx_ecg = np.where(pk_max >= thresh)[0]
    else:
        # use correlation
        idx_ecg = [meg_raw.ch_names.index(name_ecg)]
        ecg_filtered = mne.filter.band_pass_filter(meg_raw[idx_ecg, :][0], \
                                meg_raw.info['sfreq'], Fp1=flow, Fp2=fhigh)
        ecg_scores = ica.score_sources(meg_raw, \
                            target=ecg_filtered, score_func=score_func)
        idx_ecg = np.where(np.abs(ecg_scores) >= thresh)[0]


    return idx_ecg


#######################################################
# 
#  calculate the performance of artifact rejection
# 
#######################################################
def calc_performance(evoked_raw, evoked_clean):
    ''' Gives a measure of the performance of the artifact reduction. 
        Percentage value returned as output. 
    '''
    from jumeg import jumeg_math as jmath

    diff = evoked_raw.data - evoked_clean.data
    rms_diff = jmath.calc_rms(diff, average=1)
    rms_meg = jmath.calc_rms(evoked_raw.data, average=1)
    arp = (rms_diff / rms_meg) * 100.0
    return arp


#######################################################
# 
#  make/save plots to show the performance 
#            of the ICA artifact rejection
# 
#######################################################
def plot_performance_artifact_rejection(meg_raw, ica, fnout_fig, \
                                        show=False, verbose=False):

    ''' Creates a performance image of the data before 
    and after the cleaning process.
    '''

    import mne
    from jumeg import jumeg_math as jmath
    import matplotlib.pylab as pl
    import numpy as np

    name_ecg = 'ECG 001'
    name_eog_hor = 'EOG 001'
    name_eog_ver = 'EOG 002'
    event_id_ecg = 999
    event_id_eog = 998
    tmin_ecg = -0.4
    tmax_ecg =  0.4
    tmin_eog = -0.4
    tmax_eog =  0.4

    picks = mne.pick_types(meg_raw.info, meg=True, exclude='bads')
    # Why is the parameter below n_components_ instead of n_pca_components?
    meg_clean = ica.apply(meg_raw, exclude=ica.exclude, n_pca_components=ica.n_components_, copy=True)

    # plotting parameter
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    xFigSize = 12
    nrange = 2


    # ToDo:  How can we avoid popping up the window if show=False ?
    pl.ioff()
    pl.figure('performance image', figsize=(xFigSize, 12))
    pl.clf()


    # ECG, EOG:  loop over all artifact events
    for i in range(nrange):
        # get event indices
        if i == 0:
            baseline = (None, None)
            event_id = event_id_ecg
            idx_event, _, _ = mne.preprocessing.find_ecg_events(meg_raw,
                                event_id, ch_name=name_ecg, verbose=verbose)
            idx_ref_chan = meg_raw.ch_names.index(name_ecg)
            tmin = tmin_ecg
            tmax = tmax_ecg
            pl1 = nrange * 100 + 21
            pl2 = nrange * 100 + 22
            text1 = "CA: original data"
            text2 = "CA: cleaned data"
        elif i == 1:
            baseline = (None, None)
            event_id = event_id_eog
            idx_event = mne.preprocessing.find_eog_events(meg_raw,
                                event_id, ch_name=name_eog_ver, verbose=verbose)
            idx_ref_chan = meg_raw.ch_names.index(name_eog_ver)
            tmin = tmin_eog
            tmax = tmax_eog
            pl1 = nrange * 100 + 23
            pl2 = nrange * 100 + 24
            text1 = "OA: original data"
            text2 = "OA: cleaned data"

        # average the signals
        raw_epochs = mne.Epochs(meg_raw, idx_event, event_id, tmin, tmax,
                            picks=picks, baseline=baseline, verbose=verbose)
        cleaned_epochs = mne.Epochs(meg_clean, idx_event, event_id, tmin, tmax,
                            picks=picks, baseline=baseline, verbose=verbose)
        ref_epochs = mne.Epochs(meg_raw, idx_event, event_id, tmin, tmax,
                            picks=[idx_ref_chan], baseline=baseline, verbose=verbose)

        raw_epochs_avg = raw_epochs.average()
        cleaned_epochs_avg = cleaned_epochs.average()
        ref_epochs_avg = np.average(ref_epochs.get_data(), axis=0).flatten() * -1.0
        times = raw_epochs_avg.times*1e3
        if np.max(raw_epochs_avg.data) < 1:
            factor = 1e15
        else:
            factor = 1
        ymin = np.min(raw_epochs_avg.data) * factor
        ymax = np.max(raw_epochs_avg.data) * factor

        # plotting data before cleaning
        pl.subplot(pl1)
        pl.plot(times, raw_epochs_avg.data.T * factor, 'k')
        pl.title(text1)
        # plotting reference signal
        pl.plot(times, jmath.rescale(ref_epochs_avg, ymin, ymax), 'r')
        pl.xlim(times[0], times[len(times)-1])
        pl.ylim(1.1*ymin, 1.1*ymax)
        # print some info
        textstr1 = 'num_events=%d\nEpochs: tmin, tmax = %0.1f, %0.1f' \
                   %(len(idx_event), tmin, tmax)
        pl.text(times[10], 1.09*ymax, textstr1, fontsize=10, verticalalignment='top', bbox=props)


        # plotting data after cleaning
        pl.subplot(pl2)
        pl.plot(times, cleaned_epochs_avg.data.T * factor, 'k')
        pl.title(text2)
        # plotting reference signal again
        pl.plot(times, jmath.rescale(ref_epochs_avg, ymin, ymax), 'r')
        pl.xlim(times[0], times[len(times)-1])
        pl.ylim(1.1*ymin, 1.1*ymax)
        # print some info
        #ToDo: would be nice to add info about ica.excluded
        textstr1 = 'Performance: %f\nNum of components used: %d\nn_pca_components: %f' \
                   %(calc_performance(raw_epochs_avg, cleaned_epochs_avg), \
                   ica.n_components_, ica.n_pca_components)
        pl.text(times[10], 1.09*ymax, textstr1, fontsize=10, verticalalignment='top', bbox=props)

    if show:
        pl.show()

    # save image
    pl.savefig(fnout_fig + '.tif', format='tif')
    pl.close('performance image')
    pl.ion()


#######################################################
# 
#  apply CTPS (for brain responses)
# 
#######################################################
def apply_ctps(fname_ica, freqs=[(1, 4), (4, 8), (8, 12), (12, 16), (16, 20)],
    tmin=-0.2, tmax=0.4, name_stim='STI 014', event_id =None, baseline=(None,0)):

    ''' Applies CTPS to a list of ICA files. '''

    import mne, ctps, os
    import numpy as np

    from jumeg.filter import jumeg_filter


    fiws = jumeg_filter.jumeg_filter(filter_method="ws")
    fiws.filter_type        = 'bp'   # bp, lp, hp
    fiws.dcoffset           = True
    fiws.filter_attenuation_factor = 1

    nfreq = len(freqs)

    print freqs
    print nfreq


    # Trigger or Response ?
    if name_stim == 'STI 014':      # trigger
        trig_name = 'trigger'
    else:
        if name_stim == 'STI 013':   # response
            trig_name = 'response'
        else:
            trig_name = 'auxillary'

    # check list of filenames
    if isinstance(fname_ica, list):
        fnlist = fname_ica
    else:
        if isinstance(fname_ica, str):
            fnlist = list([fname_ica]) 
        else:
            fnlist = list(fname_ica)

    # loop across all filenames
    for fnica in fnlist:
        name  = os.path.split(fnica)[1]
        #fname = fnica[0:len(fnica)-4]
        basename = fnica.strip('-ica.fif')
        fnraw = basename+'-raw.fif'
        #basename = os.path.splitext(os.path.basename(fnica))[0]
        # load cleaned data
        raw = mne.io.Raw(fnraw,preload=True)
        picks = mne.pick_types(raw.info, meg=True, exclude='bads')

        # read (second) ICA
        print ">>>> working on: "+basename
        ica = mne.preprocessing.read_ica(fnica)
        ica_picks = np.arange(ica.n_components_)
        ncomp = len(ica_picks)

        # stim events
        stim_events = mne.find_events(raw, stim_channel=name_stim)
        nevents = len(stim_events)

        if (nevents > 0):
            # for a specific event ID
            if event_id:
                ix = np.where(stim_events[:,2] == event_id)[0]
                stim_events = stim_events[ix,:]
            else:
                event_id = stim_events[0,2]  
            # create ctps dictionary
            dctps = {'fnica': fnica,
                'basename': basename,
                'stim_channel': name_stim,
                'trig_name': trig_name,
                'ncomp': ncomp,
                'nevent': nevents,
                'event_id': event_id,
                'nfreq': nfreq,
                'freqs': freqs,
                }
            # loop across all filenames
            pkarr = []
            ptarr = []
            pkmax_arr = []
            for ifreq in range(nfreq):            
                ica_raw = ica.get_sources(raw)
                flow,fhigh = freqs[ifreq][0],freqs[ifreq][1]
                bp = str(flow)+'_'+str(fhigh)
                # filter ICA data and create epochs
                #tw=0.1
                # ica_raw.filter(l_freq=flow, h_freq=fhigh, picks=ica_picks,
                #     method='fft',l_trans_bandwidth=tw, h_trans_bandwidth=tw)
                # ica_raw.filter(l_freq=flow, h_freq=fhigh, picks=ica_picks,
                #                                                 method='fft')

                
                # filter ws settings
                # later we will make this as a one line call
                data_length = raw._data[0,:].size
                fiws.sampling_frequency = raw.info['sfreq']
                fiws.fcut1              = flow
                fiws.fcut2              = fhigh
                #fiws.init_filter_kernel(data_length)
                #fiws.init_filter(data_length)
                for ichan in ica_picks:
                    fiws.apply_filter(ica_raw._data[ichan,:])


                ica_epochs = mne.Epochs(ica_raw, events=stim_events,
                        event_id=event_id, tmin=tmin, tmax=tmax, verbose=False,
                        picks=ica_picks, baseline=baseline)
                # compute CTPS
                _, pk, pt = ctps.compute_ctps(ica_epochs.get_data())
                pkmax = pk.max(1)
                times = ica_epochs.times * 1e3
                pkarr.append(pk)
                ptarr.append(pt)
                pkmax_arr.append(pkmax)
            pkarr = np.array(pkarr)    
            ptarr = np.array(ptarr)
            pkmax_arr = np.array(pkmax_arr)            
            dctps['pk'] = pkarr
            dctps['pt'] = ptarr
            dctps['pkmax'] = pkmax_arr
            dctps['nsamp'] = len(times)
            dctps['times'] = times
            dctps['tmin'] = ica_epochs.tmin
            dctps['tmax'] = ica_epochs.tmax
            fnctps = basename + ',ctps-'+trig_name
            np.save(fnctps, dctps)
            # Note; loading example: dctps = np.load(fnctps).items()
        else:
            event_id=None


#######################################################
# 
#  Select ICs from CTPS anaysis (for brain responses)
# 
#######################################################
def apply_ctps_select_ic(fname_ctps, threshold=0.1):

    ''' Select ICs based on CTPS analysis. '''

    import mne, os, ctps, string
    import numpy as np
    import matplotlib.pylab as pl

    # check list of filenames        
    if isinstance(fname_ctps, list):
        fnlist = fname_ctps
    else:
        if isinstance(fname_ctps, str):
            fnlist = list([fname_ctps]) 
        else:
            fnlist = list(fname_ctps)

    # loop across all filenames
    pl.ioff()  # switch off (interactive) plot visualisation
    ifile = 0
    for fnctps in fnlist:        
        name  = os.path.splitext(fnctps)[0]
        basename = os.path.splitext(os.path.basename(fnctps))[0]
        print '>>> working on: '+basename
        # load CTPS data
        dctps = np.load(fnctps).item()
        freqs = dctps['freqs']
        nfreq = len(freqs)
        ncomp = dctps['ncomp']
        trig_name = dctps['trig_name']
        times = dctps['times']
        ic_sel = []
        # loop acros all freq. bands
        fig=pl.figure(ifile+1,figsize=(16,9), dpi=100) 
        pl.clf()
        fig.subplots_adjust(left=0.08, right=0.95, bottom=0.05,
                            top=0.93, wspace=0.2, hspace=0.2)            
        fig.suptitle(basename, fontweight='bold')
        nrow = np.ceil(float(nfreq)/2)
        for ifreq in range(nfreq):
            pk = dctps['pk'][ifreq]
            pt = dctps['pt'][ifreq]
            pkmax = pk.max(1)
            ixmax = np.where(pkmax == pkmax.max())[0]
            ix = (np.where(pkmax >= threshold))[0]
            if np.any(ix):
                if (ifreq > 0):
                    ic_sel = np.append(ic_sel,ix+1)
                else:
                    ic_sel = ix+1

            # do construct names for title, fnout_fig, fnout_ctps
            frange = ' @'+str(freqs[ifreq][0])+'-'+str(freqs[ifreq][1])+'Hz'
            x = np.arange(ncomp)+1
            # do make bar plots for ctps thresh level plots
            ax = fig.add_subplot(nrow,2,ifreq+1)
            pl.bar(x,pkmax,color='steelblue')
            pl.bar(x[ix],pkmax[ix],color='red')
            pl.title(trig_name+frange, fontsize='small')
            pl.xlim([1,ncomp])
            pl.ylim([0,0.5])
            pl.text(2,0.45, 'ICs: '+str(ix+1))
        ic_sel = np.unique(ic_sel)
        nic = np.size(ic_sel)
        info = 'ICs (all): '+str(ic_sel).strip('[]')
        fig.text(0.02,0.01, info,transform=ax.transAxes)

        # save CTPS components found
        fntxt = name+'-ic_selection.txt'
        ic_sel = np.reshape(ic_sel,[1,nic])
        np.savetxt(fntxt,ic_sel,fmt='%i',delimiter=', ')
        ifile += 1

        # save figure
        fnfig = name+'-ic_selection.png'
        pl.savefig(fnfig,dpi=100)
    pl.ion()  # switch on (interactive) plot visualisation


#######################################################
#                                                     #
# interface for creating the noise-covariance matrix  #
#                                                     #
#######################################################
def apply_create_noise_covariance(fname_empty_room, fname_out, verbose=None):

    '''
    Creates the noise covariance matrix from an
    empty room file.

        Parameters
        ----------
        fname_empty_room : String containing the filename
            of the empty room file (must be a fif-file)
        fname_out : String containing the output filename
            of the noise-covariance estimation (must also
            be a fif-file)
        verbose : bool, str, int, or None
            If not None, override default verbose level
            (see mne.verbose).
            default: verbose=None
    '''

    # -------------------------------------------
    # import necessary modules
    # -------------------------------------------
    from mne import compute_raw_data_covariance as cp_covariance
    from mne import write_cov
    from mne.io import Raw
    from mne import pick_types
    import os


    if isinstance(fname_empty_room, list):
        fner = fname_empty_room
    else:
        if isinstance(fname_empty_room, str):
            fner = list([fname_empty_room]) 
        else:
            fner = list(fname_empty_room)
    nfiles = len(fner)

    if isinstance(fname_out, list):
        fnout = fname_out
    else:
        if isinstance(fname_out, str):
            fnout = list([fname_out]) 
        else:
            fnout = list(fname_out)
    if (len(fnout) != nfiles):
        print ">>> Error (create_noise_covariance_matrix):"
        print "    Number of in/out files do not match"
        print
        exit()
    

    # loop across all filenames
    for ifile in range(nfiles):
        fn_in = fner[ifile]
        fn_out = fnout[ifile]
        print ">>> create noise covariance using file: " 
        path_in , name  = os.path.split(fn_in)
        print name

        # read in data
        raw_empty = Raw(fn_in, verbose=verbose)

        # pick MEG channels only
        picks = pick_types(raw_empty.info, meg=True, eeg=False, stim=False,
                           eog=False, exclude='bads')

        # calculate noise-covariance matrix
        noise_cov_mat = cp_covariance(raw_empty, picks=picks, verbose=verbose)

        # write noise-covariance matrix to disk
        write_cov(fn_out, noise_cov_mat)
