import mne
from mne.filter import band_pass_filter
from mne.preprocessing import find_ecg_events
import ctps
import numpy as np
from jumeg import math as jmath
import matplotlib.pylab as pl


#######################################################
# 
#  determine occular related ICs
# 
#######################################################
def get_ics_ocular(meg_raw, ica, flow=1, fhigh=10,
    name_eog_hor = 'EOG 001', name_eog_ver = 'EOG 002',
    score_func = 'pearsonr', thresh=0.3):


    # -----------------------------------
    # ICs related to ocular artifacts
    # -----------------------------------


    # Note: when using the following:
    #   - the filter settings are different
    #   - here we cannot define the filter range

    # vertical EOG
    # idx_eog_ver = [meg_raw.ch_names.index(name_eog_ver)]
    # eog_scores = ica.find_sources_raw(meg_raw, meg_raw[idx_eog_ver][0])
    # eogv_idx = np.where(np.abs(eog_scores) > thresh)[0]
    # ica.exclude += list(eogv_idx)
    # ica.plot_topomap(eog_idx)
    
    # horizontal EOG
    # idx_eog_hor = [meg_raw.ch_names.index(name_eog_hor)]
    # eog_scores = ica.find_sources_raw(meg_raw, meg_raw[idx_eog_hor][0])
    # eogh_idx = np.where(np.abs(eog_scores) > thresh)[0]
    # ica.exclude += list(eogh_idx)
    # ica.plot_topomap(eog_idx)
    # print [eogv_idx, eogh_idx]


    # vertical EOG
    idx_eog_ver = [meg_raw.ch_names.index(name_eog_ver)]
    eog_ver_filtered = band_pass_filter(meg_raw[idx_eog_ver, :][0], \
                            meg_raw.info['sfreq'], Fp1=flow, Fp2=fhigh)
    eog_ver_scores = ica.find_sources_raw(meg_raw, \
                        target=eog_ver_filtered, score_func=score_func)
    idx_eog_ver = np.where(np.abs(eog_ver_scores) >= thresh)[0]
    if not idx_eog_ver.any(): 
        idx_eog_ver = np.array([-1])

    # horizontal EOG
    idx_eog_hor = [meg_raw.ch_names.index(name_eog_hor)]
    eog_hor_filtered = band_pass_filter(meg_raw[idx_eog_hor, :][0], \
                            meg_raw.info['sfreq'], Fp1=flow, Fp2=fhigh)
    eog_hor_scores = ica.find_sources_raw(meg_raw, \
                        target=eog_hor_filtered, score_func=score_func)
    idx_eog_hor = np.where(np.abs(eog_hor_scores) >= thresh)[0]
    if not idx_eog_hor.any(): 
        idx_eog_hor = np.array([-1])
    
    # combine both  
    idx_eog = []
    for i in range(idx_eog_ver.size):
        ix = idx_eog_ver[i]
        if (ix >= 0):
            idx_eog.append(ix)
    for i in range(idx_eog_hor.size):
        ix = idx_eog_hor[i]
        if (ix >= 0):
            idx_eog.append(ix)

    return idx_eog






#######################################################
# 
#  determine cardiac related ICs
# 
#######################################################
def get_ics_cardiac(meg_raw, ica, flow=10, fhigh=20, tmin=-0.3, tmax=0.3,
    name_ecg = 'ECG 001', use_CTPS=True, score_func = 'pearsonr', thresh=0.3):

    event_id_ecg = 999

    # get and filter ICA signals
    ica_raw = ica.sources_as_raw(meg_raw)
    ica_raw.filter(l_freq=flow, h_freq=fhigh, n_jobs=2, method='fft')
    # get R-peak indices in ECG signal
    idx_R_peak, _, _ = find_ecg_events(meg_raw,
                        ch_name=name_ecg, event_id=event_id_ecg,
                        l_freq=flow, h_freq=fhigh,verbose=False)


    # -----------------------------------
    # default method:  CTPS
    #           else:  correlation
    # -----------------------------------
    if (use_CTPS):
        # create epochs
        picks = np.arange(ica.n_components_)
        ica_epochs = mne.Epochs(ica_raw, events=idx_R_peak, event_id=event_id_ecg,
                                tmin=tmin, tmax=tmax, baseline=None, 
                                proj=False, picks=picks, verbose=False)
        # compute CTPS
        _,pk,_ = ctps.compute_ctps(ica_epochs.get_data())

        pk_max = np.max(pk, axis=1)
        idx_ecg = np.where(pk_max >= thresh)[0]
    else:
        # use correlation
        idx_ecg = [meg_raw.ch_names.index(name_ecg)]
        ecg_filtered = band_pass_filter(meg_raw[idx_ecg, :][0], \
                                meg_raw.info['sfreq'], Fp1=flow, Fp2=fhigh)
        ecg_scores = ica.find_sources_raw(meg_raw, \
                            target=ecg_filtered, score_func=score_func)
        idx_ecg = np.where(np.abs(ecg_scores) >= thresh)[0]


    return idx_ecg







#######################################################
# 
#  calculate the performance of artifact rejection
# 
#######################################################
def calc_performance(evoked_raw, evoked_clean):
    """ Gives a measure of the performance of the artifact reduction. Percentage value returned as output. """
    diff = evoked_raw.data - evoked_clean.data
    rms_diff = jmath.calc_rms(diff, average=1)
    rms_meg  = jmath.calc_rms(evoked_raw.data, average=1)
    arp = (rms_diff / rms_meg) * 100.0
    return arp





#######################################################
# 
#  make/save plots to show the performance 
#            of the ICA artifact rejection
# 
#######################################################
def plot_performance_artifact_rejection(meg_raw, ica, fnout_fig,
                                        show=False, verbose=False):

    """ Creates a performance image of the data before and after the cleaning process. """
    name_ecg = 'ECG 001'
    name_eog_hor = 'EOG 001'
    name_eog_ver = 'EOG 002'
    event_id_ecg = 999
    event_id_eog = 998
    tmin_ecg = -0.4
    tmax_ecg =  0.4
    tmin_eog = -0.4
    tmax_eog =  0.4

    picks = mne.fiff.pick_types(meg_raw.info, meg=True, exclude='bads')
    meg_clean = ica.pick_sources_raw(meg_raw,n_pca_components=ica.n_components_)

    # plotting parameter
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    xFigSize = 12
    nrange = 2


    # ToDo:  How can we avoid popping up the window if show=False ?
    pl.figure('performance image', figsize=(xFigSize, 12))
    

    # ECG, EOG:  loop over all artifact events
    for i in range(nrange):
        # get event indices
        if i == 0:
            baseline = (None, None)
            event_id = event_id_ecg
            idx_event, _, _ = mne.preprocessing.find_ecg_events(meg_raw,
                                event_id, ch_name=name_ecg,  verbose=verbose)
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