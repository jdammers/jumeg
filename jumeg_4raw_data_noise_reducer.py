'''
----------------------------------------------------------------------
--- jumeg.jumeg_4raw_data_noise_reducer
--- modified by FB
----> interface for use of only one raw obj or one fif file
----> use most jumeg_base cls functions  
----------------------------------------------------------------------

--- !!! featuring the one and olny magic EE <jumeg.jumeg_noise_reducer>
----------------------------------------------------------------------
 author     : Eberhard Eich
 email      : e.eich@fz-juelich.de
 last update: 17.07.2015
 version    : 1.3

----------------------------------------------------------------------
 Based on following publications:
----------------------------------------------------------------------

Robinson, Stephen E., 'Environmental Noise Cancellation for
Biomagnetic Measurements', Advances in Biomagnetism,
Plenum Press, New York, 1989

----------------------------------------------------------------------

  s'_i(t) = s_i(t) - sum(w_ij*r_j(t), j=1,nref)
 where
  s_i  are the   signal  traces, i=1,nsig
  r_j  are the reference traces, j=1,nref after DC removal
  w_ij are weights determined by minimizing
       <(s'_i(t)-<s'_i>)^2> with <x> temporal mean
 Typically s_i are magnetic signal channels and
 r_j (selected) magnetic reference channels, but
 other refs are possible.

 ----------------------------------------------------------------------
 How to use the jumeg_noise_reducer?
----------------------------------------------------------------------

from jumeg import jumeg_noise_reducer

jumeg_noise_reducer.noise_reducer(fname_raw)


--> for further comments we refer directly to the functions
----------------------------------------------------------------------
'''
# Author: EE
#   150203/EE/
#   150619/EE/ fix for tmin/tmax-arg
#
# License: BSD (3-clause)

import numpy as np
import time
import copy
import warnings

import os
from math import floor, ceil
import mne
from mne.utils import logger
from mne.epochs import _is_good
from mne.io.pick import channel_indices_by_type
from jumeg.jumeg_utils import get_files_from_list
from jumeg.jumeg_base  import jumeg_base

TINY = 1.e-38
SVD_RELCUTOFF = 1.e-08


##################################################
#
# generate plot of power spectrum before and
# after noise reduction
#
##################################################
def plot_denoising_4raw_data(fname_raw, fmin=0, fmax=300, tmin=0.0, tmax=60.0,
                   proj=False, n_fft=4096, color='blue',
                   stim_name=None, event_id=1,
                   tmin_stim=-0.2, tmax_stim=0.5,
                   area_mode='range', area_alpha=0.33, n_jobs=1,
                   title1='before denoising', title2='after denoising',
                   info=None, show=True, fnout=None):
    """Plot the power spectral density across channels to show denoising.

    Parameters
    ----------
    fname_raw : list or str
        List of raw files, without denoising and with for comparison.
    tmin : float
        Start time for calculations.
    tmax : float
        End time for calculations.
    fmin : float
        Start frequency to consider.
    fmax : float
        End frequency to consider.
    proj : bool
        Apply projection.
    n_fft : int
        Number of points to use in Welch FFT calculations.
    color : str | tuple
        A matplotlib-compatible color to use.
    area_mode : str | None
        Mode for plotting area. If 'std', the mean +/- 1 STD (across channels)
        will be plotted. If 'range', the min and max (across channels) will be
        plotted. Bad channels will be excluded from these calculations.
        If None, no area will be plotted.
    area_alpha : float
        Alpha for the area.
    info : bool
        Display information in the figure.
    show : bool
        Show figure.
    fnout : str
        Name of the saved output figure. If none, no figure will be saved.
    title1, title2 : str
        Title for two psd plots.
    n_jobs : int
        Number of jobs to use for parallel computation.
    stim_name : str
        Name of the stim channel. If stim_name is set, the plot of epochs average
        is also shown alongside the PSD plots.
    event_id : int
        ID of the stim event. (only when stim_name is set)

    Example Usage
    -------------
    plot_denoising(['orig-raw.fif', 'orig,nr-raw.fif', fnout='example')
    """

    from matplotlib import gridspec as grd
    import matplotlib.pyplot as plt
    from mne.time_frequency import compute_raw_psd

    fnraw = get_files_from_list(fname_raw)

    # ---------------------------------
    # estimate power spectrum
    # ---------------------------------
    psds_all = []
    freqs_all = []

    # loop across all filenames
    for fname in fnraw:

        # read in data
        raw = mne.io.Raw(fname, preload=True)
        picks = mne.pick_types(raw.info, meg='mag', eeg=False,
                               stim=False, eog=False, exclude='bads')

        if area_mode not in [None, 'std', 'range']:
            raise ValueError('"area_mode" must be "std", "range", or None')

        psds, freqs = compute_raw_psd(raw, picks=picks, fmin=fmin, fmax=fmax,
                                      tmin=tmin, tmax=tmax, n_fft=n_fft,
                                      n_jobs=n_jobs, proj=proj)
        psds_all.append(psds)
        freqs_all.append(freqs)

    if stim_name:
        n_xplots = 2

        # get some infos
        events = mne.find_events(raw, stim_channel=stim_name, consecutive=True)

    else:
        n_xplots = 1

    fig = plt.figure('denoising', figsize=(16, 6 * n_xplots))
    gs = grd.GridSpec(n_xplots, int(len(psds_all)))

    # loop across all filenames
    for idx in range(int(len(psds_all))):

        # ---------------------------------
        # plot power spectrum
        # ---------------------------------
        p1 = plt.subplot(gs[0, idx])

        # Convert PSDs to dB
        psds = 10 * np.log10(psds_all[idx])
        psd_mean = np.mean(psds, axis=0)
        if area_mode == 'std':
            psd_std = np.std(psds, axis=0)
            hyp_limits = (psd_mean - psd_std, psd_mean + psd_std)
        elif area_mode == 'range':
            hyp_limits = (np.min(psds, axis=0), np.max(psds, axis=0))
        else:  # area_mode is None
            hyp_limits = None

        p1.plot(freqs_all[idx], psd_mean, color=color)
        if hyp_limits is not None:
            p1.fill_between(freqs_all[idx], hyp_limits[0], y2=hyp_limits[1],
                            color=color, alpha=area_alpha)

        if idx == 0:
            p1.set_title(title1)
            ylim = [np.min(psd_mean) - 10, np.max(psd_mean) + 10]
        else:
            p1.set_title(title2)

        p1.set_xlabel('Freq (Hz)')
        p1.set_ylabel('Power Spectral Density (dB/Hz)')
        p1.set_xlim(freqs_all[idx][0], freqs_all[idx][-1])
        p1.set_ylim(ylim[0], ylim[1])

        # ---------------------------------
        # plot signal around stimulus
        # onset
        # ---------------------------------
        if stim_name:
            raw = mne.io.Raw(fnraw[idx], preload=True)
            epochs = mne.Epochs(raw, events, event_id, proj=False,
                                tmin=tmin_stim, tmax=tmax_stim, picks=picks,
                                preload=True, baseline=(None, None))
            evoked = epochs.average()
            if idx == 0:
                ymin = np.min(evoked.data)
                ymax = np.max(evoked.data)

            times = evoked.times * 1e3
            p2 = plt.subplot(gs[1, idx])
            p2.plot(times, evoked.data.T, 'blue', linewidth=0.5)
            p2.set_xlim(times[0], times[len(times) - 1])
            p2.set_ylim(1.1 * ymin, 1.1 * ymax)

            if (idx == 1) and info:
                plt.text(times[0], 0.9 * ymax, '  ICs: ' + str(info))

    # save image
    if fnout:
        fig.savefig(fnout + '.png', format='png')

    # show image if requested
    if show:
        plt.show()

    plt.close('denoising')
    plt.ion()


##################################################
#
# routine to detrend the data
# update fb:
# ---> new key raw=raw
# ---> getting raw obj with jume_base cls
# ---> selecting poicks via jumeg_base cls
# ---> saving raw use jumeg_base cls
##################################################
def perform_detrending(fname_raw,raw=None,save=True):

    from mne.io import Raw
    from numpy import poly1d, polyfit
    
    raw = jumeg_base.get_raw_obj(fname_raw,raw=raw)
  # get channels
    picks = jumeg_base.pick_meg_and_ref_nobads()
    
    xval  = np.arange(raw._data.shape[1])
  # loop over all channels
    for ipick in picks:
        coeff = polyfit(xval, raw._data[ipick, :], deg=1)
        trend = poly1d(coeff)
        raw._data[ipick, :] -= trend(xval)

    # save detrended data
    if save:
       fname_out = jumeg_base.get_fif_name(raw=raw,postfix='dt')
       jume_base.apply_save_mne_data(raw,fname=fname_out,overwrite=True)

    return raw


##################################################
#
# Get indices of matching channel names from list
#
##################################################
def channel_indices_from_list(fulllist, findlist, excllist=None):
    """Get indices of matching channel names from list

    Parameters
    ----------
    fulllist: list of channel names
    findlist: list of (regexp) names to find
              regexp are resolved using mne.pick_channels_regexp()
    excllist: list of channel names to exclude,
              e.g., raw.info.get('bads')

    Returns
    -------
    chnpick: array with indices
    """
    chnpick = []
    for ir in xrange(len(findlist)):
        if findlist[ir].translate(None, ' ').isalnum():
            try:
                chnpicktmp = ([fulllist.index(findlist[ir])])
                chnpick = np.array(np.concatenate((chnpick, chnpicktmp)), dtype=int)
            except:
                print ">>>>> Channel '%s' not found." % findlist[ir]
        else:
            chnpicktmp = (mne.pick_channels_regexp(fulllist, findlist[ir]))
            if len(chnpicktmp) == 0:
                print ">>>>> '%s' does not match any channel name." % findlist[ir]
            else:
                chnpick = np.array(np.concatenate((chnpick, chnpicktmp)), dtype=int)
    if len(chnpick) > 1:
        # Remove duplicates:
        chnpick = np.sort(np.array(list(set(np.sort(chnpick)))))

    if excllist is not None and len(excllist) > 0:
        exclinds = [fulllist.index(excllist[ie]) for ie in xrange(len(excllist))]
        chnpick = list(np.setdiff1d(chnpick, exclinds))
    return chnpick


##################################################
#
# Apply noise reduction to signal channels
# using reference channels.
# !!! ONLY ONE RAW Obj Interface Version FB !!!
#
##################################################
def noise_reducer_4raw_data(fname_raw,raw=None,signals=[],noiseref=[],detrending=None,
                  tmin=None,tmax=None,reflp=None,refhp=None,refnotch=None,
                  exclude_artifacts=True,checkresults=True,
                  fif_extention="-raw.fif",fif_postfix="nr",                        
                  reject={'grad':4000e-13,'mag':4e-12,'eeg':40e-6,'eog':250e-6},
                  complementary_signal=False,fnout=None,verbose=False,save=True):

    """Apply noise reduction to signal channels using reference channels.
        
       !!! ONLY ONE RAW Obj Interface Version FB !!!
           
    Parameters
    ----------
    fname_raw : rawfile name

    raw     : fif raw object

    signals : list of string
              List of channels to compensate using noiseref.
              If empty use the meg signal channels.
    noiseref : list of string | str
              List of channels to use as noise reference.
              If empty use the magnetic reference channsls (default).
    signals and noiseref may contain regexp, which are resolved
    using mne.pick_channels_regexp(). All other channels are copied.
    tmin : lower latency bound for weight-calc [start of trace]
    tmax : upper latency bound for weight-calc [ end  of trace]
           Weights are calc'd for (tmin,tmax), but applied to entire data set
    refhp : high-pass frequency for reference signal filter [None]
    reflp :  low-pass frequency for reference signal filter [None]
            reflp < refhp: band-stop filter
            reflp > refhp: band-pass filter
            reflp is not None, refhp is None: low-pass filter
            reflp is None, refhp is not None: high-pass filter
    refnotch : (base) notch frequency for reference signal filter [None]
               use raw(ref)-notched(ref) as reference signal
    exclude_artifacts: filter signal-channels thru _is_good() [True]
                       (parameters are at present hard-coded!)
    complementary_signal : replaced signal by traces that would be subtracted [False]
                           (can be useful for debugging)
    checkresults : boolean to control internal checks and overall success [True]

    reject =  dict for rejection threshold 
              units:
              grad:    T / m (gradiometers)
              mag:     T (magnetometers)
              eeg/eog: uV (EEG channels)
              default=>{'grad':4000e-13,'mag':4e-12,'eeg':40e-6,'eog':250e-6}
              
    save : save data to fif file

    Outputfile:
    -------
    <wawa>,nr-raw.fif for input <wawa>-raw.fif

    Returns
    -------
    TBD

    Bugs
    ----
    - artifact checking is incomplete (and with arb. window of tstep=0.2s)
    - no accounting of channels used as signal/reference
    - non existing input file handled ungracefully
    """

    tc0 = time.clock()
    tw0 = time.time()

    if type(complementary_signal) != bool:
        raise ValueError("Argument complementary_signal must be of type bool")

    raw,fname_raw = jumeg_base.get_raw_obj(fname_raw,raw=raw)
    
    
    if detrending:
       raw = perform_detrending(None,raw=raw, save=False)
    
    tc1 = time.clock()
    tw1 = time.time()

    if verbose:
       print ">>> loading raw data took %.1f ms (%.2f s walltime)" % (1000. * (tc1 - tc0), (tw1 - tw0))

    # Time window selection
    # weights are calc'd based on [tmin,tmax], but applied to the entire data set.
    # tstep is used in artifact detection
    # tmin,tmax variables must not be changed here!
    if tmin is None:
        itmin = 0
    else:
        itmin = int(floor(tmin * raw.info['sfreq']))
    if tmax is None:
        itmax = raw.last_samp
    else:
        itmax = int(ceil(tmax * raw.info['sfreq']))

    if itmax - itmin < 2:
        raise ValueError("Time-window for noise compensation empty or too short")

    if verbose:
        print ">>> Set time-range to [%7.3f,%7.3f]" % \
              (raw.index_as_time(itmin)[0], raw.index_as_time(itmax)[0])

    if signals is None or len(signals) == 0:
        sigpick = jumeg_base.pick_meg_nobads(raw)
    else:
        sigpick = channel_indices_from_list(raw.info['ch_names'][:], signals,
                                            raw.info.get('bads'))
    nsig = len(sigpick)
    if nsig == 0:
        raise ValueError("No channel selected for noise compensation")

    if noiseref is None or len(noiseref) == 0:
        # References are not limited to 4D ref-chans, but can be anything,
        # incl. ECG or powerline monitor.
        if verbose:
            print ">>> Using all refchans."
            
        refexclude = "bads"      
        refpick = jumeg_base.pick_ref_nobads(raw)
    else:
        refpick = channel_indices_from_list(raw.info['ch_names'][:], noiseref,
                                            raw.info.get('bads'))
    nref = len(refpick)
    if nref == 0:
        raise ValueError("No channel selected as noise reference")

    if verbose:
        print ">>> sigpick: %3d chans, refpick: %3d chans" % (nsig, nref)

    if reflp is None and refhp is None and refnotch is None:
        use_reffilter = False
        use_refantinotch = False
    else:
        use_reffilter = True
        if verbose:
            print "########## Filter reference channels:"

        use_refantinotch = False
        if refnotch is not None:
            if reflp is None and reflp is None:
                use_refantinotch = True
                freqlast = np.min([5.01 * refnotch, 0.5 * raw.info['sfreq']])
                if verbose:
                    print ">>> notches at freq %.1f and harmonics below %.1f" % (refnotch, freqlast)
            else:
                raise ValueError("Cannot specify notch- and high-/low-pass"
                                 "reference filter together")
        else:
            if verbose:
                if reflp is not None:
                    print ">>>  low-pass with cutoff-freq %.1f" % reflp
                if refhp is not None:
                    print ">>> high-pass with cutoff-freq %.1f" % refhp

        # Adapt followg drop-chans cmd to use 'all-but-refpick'
        droplist = [raw.info['ch_names'][k] for k in xrange(raw.info['nchan']) if not k in refpick]
        tct = time.clock()
        twt = time.time()
        fltref = raw.drop_channels(droplist, copy=True)
        if use_refantinotch:
            rawref = raw.drop_channels(droplist, copy=True)
            freqlast = np.min([5.01 * refnotch, 0.5 * raw.info['sfreq']])
            fltref.notch_filter(np.arange(refnotch, freqlast, refnotch),
                                picks=np.array(xrange(nref)), method='iir')
            fltref._data = (rawref._data - fltref._data)
        else:
            fltref.filter(refhp, reflp, picks=np.array(xrange(nref)), method='iir')
        tc1 = time.clock()
        tw1 = time.time()
        if verbose:
            print ">>> filtering ref-chans  took %.1f ms (%.2f s walltime)" % (1000. * (tc1 - tct), (tw1 - twt))

    if verbose:
        print "########## Calculating sig-ref/ref-ref-channel covariances:"
    # Calculate sig-ref/ref-ref-channel covariance:
    # (there is no need to calc inter-signal-chan cov,
    #  but there seems to be no appropriat fct available)
    # Here we copy the idea from compute_raw_data_covariance()
    # and truncate it as appropriate.
    tct = time.clock()
    twt = time.time()
    # The following reject and infosig entries are only
    # used in _is_good-calls.
    # _is_good() from mne-0.9.git-py2.7.egg/mne/epochs.py seems to
    # ignore ref-channels (not covered by dict) and checks individual
    # data segments - artifacts across a buffer boundary are not found.
    
    #--- !!! FB put to kwargs    
    
    #reject = dict(grad=4000e-13, # T / m (gradiometers)
    #              mag=4e-12,     # T (magnetometers)
    #              eeg=40e-6,     # uV (EEG channels)
    #              eog=250e-6)    # uV (EOG channels)

    infosig = copy.copy(raw.info)
    infosig['chs'] = [raw.info['chs'][k] for k in sigpick]
    infosig['ch_names'] = [raw.info['ch_names'][k] for k in sigpick]
    infosig['nchan'] = len(sigpick)
    idx_by_typesig = channel_indices_by_type(infosig)

    # Read data in chunks:
    tstep = 0.2
    itstep = int(ceil(tstep * raw.info['sfreq']))
    sigmean = 0
    refmean = 0
    sscovdata = 0
    srcovdata = 0
    rrcovdata = 0
    n_samples = 0

    for first in range(itmin, itmax, itstep):
        last = first + itstep
        if last >= itmax:
            last = itmax
        raw_segmentsig, times = raw[sigpick, first:last]
        if use_reffilter:
            raw_segmentref, times = fltref[:, first:last]
        else:
            raw_segmentref, times = raw[refpick, first:last]

        if not exclude_artifacts or \
           _is_good(raw_segmentsig, infosig['ch_names'], idx_by_typesig, reject, flat=None,
                    ignore_chs=raw.info['bads']):
            sigmean += raw_segmentsig.sum(axis=1)
            refmean += raw_segmentref.sum(axis=1)
            sscovdata += (raw_segmentsig * raw_segmentsig).sum(axis=1)
            srcovdata += np.dot(raw_segmentsig, raw_segmentref.T)
            rrcovdata += np.dot(raw_segmentref, raw_segmentref.T)
            n_samples += raw_segmentsig.shape[1]
        else:
            logger.info("Artefact detected in [%d, %d]" % (first, last))
    if n_samples <= 1:
        raise ValueError('Too few samples to calculate weights')
    sigmean /= n_samples
    refmean /= n_samples
    sscovdata -= n_samples * sigmean[:] * sigmean[:]
    sscovdata /= (n_samples - 1)
    srcovdata -= n_samples * sigmean[:, None] * refmean[None, :]
    srcovdata /= (n_samples - 1)
    rrcovdata -= n_samples * refmean[:, None] * refmean[None, :]
    rrcovdata /= (n_samples - 1)
    sscovinit = np.copy(sscovdata)
    if verbose:
        print ">>> Normalize srcov..."

    rrslope = copy.copy(rrcovdata)
    for iref in xrange(nref):
        dtmp = rrcovdata[iref, iref]
        if dtmp > TINY:
            srcovdata[:, iref] /= dtmp
            rrslope[:, iref] /= dtmp
        else:
            srcovdata[:, iref] = 0.
            rrslope[:, iref] = 0.

    if verbose:
        print ">>> Number of samples used : %d" % n_samples
        tc1 = time.clock()
        tw1 = time.time()
        print ">>> sigrefchn covar-calc took %.1f ms (%.2f s walltime)" % (1000. * (tc1 - tct), (tw1 - twt))

    if checkresults:
        if verbose:
            print "########## Calculated initial signal channel covariance:"
            # Calculate initial signal channel covariance:
            # (only used as quality measure)
            print ">>> initl rt(avg sig pwr) = %12.5e" % np.sqrt(np.mean(sscovdata))
            for i in xrange(5):
                print ">>> initl signal-rms[%3d] = %12.5e" % (i, np.sqrt(sscovdata.flatten()[i]))
            print ">>>"

    U, s, V = np.linalg.svd(rrslope, full_matrices=True)
    if verbose:
        print ">>> singular values:"
        print s
        print ">>> Applying cutoff for smallest SVs:"

    dtmp = s.max() * SVD_RELCUTOFF
    s *= (abs(s) >= dtmp)
    sinv = [1. / s[k] if s[k] != 0. else 0. for k in xrange(nref)]
    if verbose:
        print ">>> singular values (after cutoff):"
        print s

    stat = np.allclose(rrslope, np.dot(U, np.dot(np.diag(s), V)))
    if verbose:
        print ">>> Testing svd-result: %s" % stat
        if not stat:
            print "    (Maybe due to SV-cutoff?)"

    # Solve for inverse coefficients:
    # Set RRinv.tr=U diag(sinv) V
    RRinv = np.transpose(np.dot(U, np.dot(np.diag(sinv), V)))
    if checkresults:
        stat = np.allclose(np.identity(nref), np.dot(RRinv, rrslope))
        if stat:
            if verbose:
                print ">>> Testing RRinv-result (should be unit-matrix): ok"
        else:
            print ">>> Testing RRinv-result (should be unit-matrix): failed"
            print np.transpose(np.dot(RRinv, rrslope))
            print ">>>"

    if verbose:
        print "########## Calc weight matrix..."

    # weights-matrix will be somewhat larger than necessary,
    # (to simplify indexing in compensation loop):
    weights = np.zeros((raw._data.shape[0], nref))
    for isig in xrange(nsig):
        for iref in xrange(nref):
            weights[sigpick[isig],iref] = np.dot(srcovdata[isig,:], RRinv[:,iref])

    if verbose:
        print "########## Compensating signal channels:"
        if complementary_signal:
            print ">>> Caveat: REPLACING signal by compensation signal"

    tct = time.clock()
    twt = time.time()

    # Work on entire data stream:
    for isl in xrange(raw._data.shape[1]):
        slice = np.take(raw._data, [isl], axis=1)
        if use_reffilter:
            refslice = np.take(fltref._data, [isl], axis=1)
            refarr = refslice[:].flatten() - refmean
            # refarr = fltres[:,isl]-refmean
        else:
            refarr = slice[refpick].flatten() - refmean
        subrefarr = np.dot(weights[:], refarr)

        if not complementary_signal:
            raw._data[:, isl] -= subrefarr
        else:
            raw._data[:, isl] = subrefarr

        if (isl % 10000 == 0) and verbose:
            print "\rProcessed slice %6d" % isl

    if verbose:
        print "\nDone."
        tc1 = time.clock()
        tw1 = time.time()
        print ">>> compensation loop took %.1f ms (%.2f s walltime)" % (1000. * (tc1 - tct), (tw1 - twt))

    if checkresults:
        if verbose:
            print "########## Calculating final signal channel covariance:"
        # Calculate final signal channel covariance:
        # (only used as quality measure)
        tct = time.clock()
        twt = time.time()
        sigmean = 0
        sscovdata = 0
        n_samples = 0
        for first in range(itmin, itmax, itstep):
            last = first + itstep
            if last >= itmax:
                last = itmax
            raw_segmentsig, times = raw[sigpick, first:last]
            # Artifacts found here will probably differ from pre-noisered artifacts!
            if not exclude_artifacts or \
               _is_good(raw_segmentsig, infosig['ch_names'], idx_by_typesig, reject,
                        flat=None, ignore_chs=raw.info['bads']):
                sigmean += raw_segmentsig.sum(axis=1)
                sscovdata += (raw_segmentsig * raw_segmentsig).sum(axis=1)
                n_samples += raw_segmentsig.shape[1]
        sigmean /= n_samples
        sscovdata -= n_samples * sigmean[:] * sigmean[:]
        sscovdata /= (n_samples - 1)
        
        if verbose:
            print ">>> no channel got worse: ", np.all(np.less_equal(sscovdata, sscovinit))
            print ">>> final rt(avg sig pwr) = %12.5e" % np.sqrt(np.mean(sscovdata))
            for i in xrange(5):
                print ">>> final signal-rms[%3d] = %12.5e" % (i, np.sqrt(sscovdata.flatten()[i]))
            tc1 = time.clock()
            tw1 = time.time()
            print ">>> signal covar-calc took %.1f ms (%.2f s walltime)" % (1000. * (tc1 - tct), (tw1 - twt))
            print ">>>"

   #--- fb update 21.07.2015     
    fname_out = jumeg_base.get_fif_name(raw=raw,postfix=fif_postfix,extention=fif_extention)                       
      
    if save:    
       jumeg_base.apply_save_mne_data(raw,fname=fname_out,overwrite=True)
     
             
    tc1 = time.clock()
    tw1 = time.time()
    if verbose:
       print ">>> Total run took %.1f ms (%.2f s walltime)" % (1000. * (tc1 - tc0), (tw1 - tw0))
        
    return raw,fname_out  

##################################################
#
# routine to test if the noise reducer is
# working properly
#
##################################################
def test_noise_reducer():

    data_path = os.environ['SUBJECTS_DIR']
    subject   = os.environ['SUBJECT']

    dname = data_path + '/' + 'empty_room_files' + '/109925_empty_room_file-raw.fif'
    subjects_dir = data_path + '/subjects'
    #
    checkresults = True
    exclart = False
    use_reffilter = True
    refflt_lpfreq = 52.
    refflt_hpfreq = 48.

    print "########## before of noisereducer call ##########"
    sigchanlist = ['MEG ..1', 'MEG ..3', 'MEG ..5', 'MEG ..7', 'MEG ..9']
    sigchanlist = None
    refchanlist = ['RFM 001', 'RFM 003', 'RFM 005', 'RFG ...']
    tmin = 15.
    noise_reducer(dname, signals=sigchanlist, noiseref=refchanlist, tmin=tmin,
                  reflp=refflt_lpfreq, refhp=refflt_hpfreq,
                  exclude_artifacts=exclart, complementary_signal=True)
    print "########## behind of noisereducer call ##########"

    print "########## Read raw data:"
    tc0 = time.clock()
    tw0 = time.time()
    raw = mne.io.Raw(dname, preload=True)
    tc1 = time.clock()
    tw1 = time.time()
    print "loading raw data  took %.1f ms (%.2f s walltime)" % (1000. * (tc1 - tc0), (tw1 - tw0))

    # Time window selection
    # weights are calc'd based on [tmin,tmax], but applied to the entire data set.
    # tstep is used in artifact detection
    tmax = raw.index_as_time(raw.last_samp)[0]
    tstep = 0.2
    itmin = int(floor(tmin * raw.info['sfreq']))
    itmax = int(ceil(tmax * raw.info['sfreq']))
    itstep = int(ceil(tstep * raw.info['sfreq']))
    print ">>> Set time-range to [%7.3f,%7.3f]" % (tmin, tmax)

    if sigchanlist is None:
        sigpick = mne.pick_types(raw.info, meg='mag', eeg=False, stim=False, eog=False, exclude='bads')
    else:
        sigpick = channel_indices_from_list(raw.info['ch_names'][:], sigchanlist)
    nsig = len(sigpick)
    print "sigpick: %3d chans" % nsig
    if nsig == 0:
        raise ValueError("No channel selected for noise compensation")

    if refchanlist is None:
        # References are not limited to 4D ref-chans, but can be anything,
        # incl. ECG or powerline monitor.
        print ">>> Using all refchans."
        refexclude = "bads"
        refpick = mne.pick_types(raw.info, ref_meg=True, meg=False, eeg=False,
                                 stim=False, eog=False, exclude=refexclude)
    else:
        refpick = channel_indices_from_list(raw.info['ch_names'][:], refchanlist)
        print "refpick = '%s'" % refpick
    nref = len(refpick)
    print "refpick: %3d chans" % nref
    if nref == 0:
        raise ValueError("No channel selected as noise reference")

    print "########## Refchan geo data:"
    # This is just for info to locate special 4D-refs.
    for iref in refpick:
        print raw.info['chs'][iref]['ch_name'], raw.info['chs'][iref]['loc'][0:3]
    print ""

    if use_reffilter:
        print "########## Filter reference channels:"
        if refflt_lpfreq is not None:
            print " low-pass with cutoff-freq %.1f" % refflt_lpfreq
        if refflt_hpfreq is not None:
            print "high-pass with cutoff-freq %.1f" % refflt_hpfreq
        # Adapt followg drop-chans cmd to use 'all-but-refpick'
        droplist = [raw.info['ch_names'][k] for k in xrange(raw.info['nchan']) if not k in refpick]
        fltref = raw.drop_channels(droplist, copy=True)
        tct = time.clock()
        twt = time.time()
        fltref.filter(refflt_hpfreq, refflt_lpfreq, picks=np.array(xrange(nref)), method='iir')
        tc1 = time.clock()
        tw1 = time.time()
        print "filtering ref-chans  took %.1f ms (%.2f s walltime)" % (1000. * (tc1 - tct), (tw1 - twt))

    print "########## Calculating sig-ref/ref-ref-channel covariances:"
    # Calculate sig-ref/ref-ref-channel covariance:
    # (there is no need to calc inter-signal-chan cov,
    #  but there seems to be no appropriat fct available)
    # Here we copy the idea from compute_raw_data_covariance()
    # and truncate it as appropriate.
    tct = time.clock()
    twt = time.time()
    # The following reject and info{sig,ref} entries are only
    # used in _is_good-calls.
    # _is_good() from mne-0.9.git-py2.7.egg/mne/epochs.py seems to
    # ignore ref-channels (not covered by dict) and checks individual
    # data segments - artifacts across a buffer boundary are not found.
    reject = dict(grad=4000e-13, # T / m (gradiometers)
                  mag=4e-12,     # T (magnetometers)
                  eeg=40e-6,     # uV (EEG channels)
                  eog=250e-6)    # uV (EOG channels)

    infosig = copy.copy(raw.info)
    infosig['chs'] = [raw.info['chs'][k] for k in sigpick]
    infosig['ch_names'] = [raw.info['ch_names'][k] for k in sigpick]
    infosig['nchan'] = len(sigpick)
    idx_by_typesig = channel_indices_by_type(infosig)

    # inforef not good w/ filtering, but anyway useless
    inforef = copy.copy(raw.info)
    inforef['chs'] = [raw.info['chs'][k] for k in refpick]
    inforef['ch_names'] = [raw.info['ch_names'][k] for k in refpick]
    inforef['nchan'] = len(refpick)
    idx_by_typeref = channel_indices_by_type(inforef)

    # Read data in chunks:
    sigmean = 0
    refmean = 0
    sscovdata = 0
    srcovdata = 0
    rrcovdata = 0
    n_samples = 0
    for first in range(itmin, itmax, itstep):
        last = first + itstep
        if last >= itmax:
            last = itmax
        raw_segmentsig, times = raw[sigpick, first:last]
        if use_reffilter:
            raw_segmentref, times = fltref[:, first:last]
        else:
            raw_segmentref, times = raw[refpick, first:last]
        # if True:
        # if _is_good(raw_segmentsig, infosig['ch_names'], idx_by_typesig, reject, flat=None,
        #            ignore_chs=raw.info['bads']) and _is_good(raw_segmentref,
        #              inforef['ch_names'], idx_by_typeref, reject, flat=None,
        #                ignore_chs=raw.info['bads']):
        if not exclart or \
           _is_good(raw_segmentsig, infosig['ch_names'], idx_by_typesig, reject,
                    flat=None, ignore_chs=raw.info['bads']):
            sigmean += raw_segmentsig.sum(axis=1)
            refmean += raw_segmentref.sum(axis=1)
            sscovdata += (raw_segmentsig * raw_segmentsig).sum(axis=1)
            srcovdata += np.dot(raw_segmentsig, raw_segmentref.T)
            rrcovdata += np.dot(raw_segmentref, raw_segmentref.T)
            n_samples += raw_segmentsig.shape[1]
        else:
            logger.info("Artefact detected in [%d, %d]" % (first, last))

    #_check_n_samples(n_samples, len(picks))
    sigmean /= n_samples
    refmean /= n_samples
    sscovdata -= n_samples * sigmean[:] * sigmean[:]
    sscovdata /= (n_samples - 1)
    srcovdata -= n_samples * sigmean[:, None] * refmean[None, :]
    srcovdata /= (n_samples - 1)
    rrcovdata -= n_samples * refmean[:, None] * refmean[None, :]
    rrcovdata /= (n_samples - 1)
    sscovinit = sscovdata
    print "Normalize srcov..."
    rrslopedata = copy.copy(rrcovdata)
    for iref in xrange(nref):
        dtmp = rrcovdata[iref][iref]
        if dtmp > TINY:
            for isig in xrange(nsig):
                srcovdata[isig][iref] /= dtmp
            for jref in xrange(nref):
                rrslopedata[jref][iref] /= dtmp
        else:
            for isig in xrange(nsig):
                srcovdata[isig][iref] = 0.
            for jref in xrange(nref):
                rrslopedata[jref][iref] = 0.
    logger.info("Number of samples used : %d" % n_samples)
    tc1 = time.clock()
    tw1 = time.time()
    print "sigrefchn covar-calc took %.1f ms (%.2f s walltime)" % (1000. * (tc1 - tct), (tw1 - twt))

    print "########## Calculating sig-ref/ref-ref-channel covariances (robust):"
    # Calculate sig-ref/ref-ref-channel covariance:
    # (usg B.P.Welford, "Note on a method for calculating corrected sums
    #                   of squares and products", Technometrics4 (1962) 419-420)
    # (there is no need to calc inter-signal-chan cov,
    #  but there seems to be no appropriat fct available)
    # Here we copy the idea from compute_raw_data_covariance()
    # and truncate it as appropriate.
    tct = time.clock()
    twt = time.time()
    # The following reject and info{sig,ref} entries are only
    # used in _is_good-calls.
    # _is_good() from mne-0.9.git-py2.7.egg/mne/epochs.py seems to
    # ignore ref-channels (not covered by dict) and checks individual
    # data segments - artifacts across a buffer boundary are not found.
    reject = dict(grad=4000e-13, # T / m (gradiometers)
                  mag=4e-12,     # T (magnetometers)
                  eeg=40e-6,     # uV (EEG channels)
                  eog=250e-6)    # uV (EOG channels)

    infosig = copy.copy(raw.info)
    infosig['chs'] = [raw.info['chs'][k] for k in sigpick]
    infosig['ch_names'] = [raw.info['ch_names'][k] for k in sigpick]
    infosig['nchan'] = len(sigpick)
    idx_by_typesig = channel_indices_by_type(infosig)

    # inforef not good w/ filtering, but anyway useless
    inforef = copy.copy(raw.info)
    inforef['chs'] = [raw.info['chs'][k] for k in refpick]
    inforef['ch_names'] = [raw.info['ch_names'][k] for k in refpick]
    inforef['nchan'] = len(refpick)
    idx_by_typeref = channel_indices_by_type(inforef)

    # Read data in chunks:
    smean = np.zeros(nsig)
    smold = np.zeros(nsig)
    rmean = np.zeros(nref)
    rmold = np.zeros(nref)
    sscov = 0
    srcov = 0
    rrcov = np.zeros((nref, nref))
    srcov = np.zeros((nsig, nref))
    n_samples = 0
    for first in range(itmin, itmax, itstep):
        last = first + itstep
        if last >= itmax:
            last = itmax
        raw_segmentsig, times = raw[sigpick, first:last]
        if use_reffilter:
            raw_segmentref, times = fltref[:, first:last]
        else:
            raw_segmentref, times = raw[refpick, first:last]
        # if True:
        # if _is_good(raw_segmentsig, infosig['ch_names'], idx_by_typesig, reject, flat=None,
        #            ignore_chs=raw.info['bads']) and _is_good(raw_segmentref,
        #              inforef['ch_names'], idx_by_typeref, reject, flat=None,
        #                ignore_chs=raw.info['bads']):
        if not exclart or \
           _is_good(raw_segmentsig, infosig['ch_names'], idx_by_typesig, reject,
                    flat=None, ignore_chs=raw.info['bads']):
            for isl in xrange(raw_segmentsig.shape[1]):
                nsl = isl + n_samples + 1
                cnslm1dnsl = float((nsl - 1)) / float(nsl)
                sslsubmean = (raw_segmentsig[:, isl] - smold)
                rslsubmean = (raw_segmentref[:, isl] - rmold)
                smean = smold + sslsubmean / nsl
                rmean = rmold + rslsubmean / nsl
                sscov += sslsubmean * (raw_segmentsig[:, isl] - smean)
                srcov += cnslm1dnsl * np.dot(sslsubmean.reshape((nsig, 1)), rslsubmean.reshape((1, nref)))
                rrcov += cnslm1dnsl * np.dot(rslsubmean.reshape((nref, 1)), rslsubmean.reshape((1, nref)))
                smold = smean
                rmold = rmean
            n_samples += raw_segmentsig.shape[1]
        else:
            logger.info("Artefact detected in [%d, %d]" % (first, last))

    #_check_n_samples(n_samples, len(picks))
    sscov /= (n_samples - 1)
    srcov /= (n_samples - 1)
    rrcov /= (n_samples - 1)
    print "Normalize srcov..."
    rrslope = copy.copy(rrcov)
    for iref in xrange(nref):
        dtmp = rrcov[iref][iref]
        if dtmp > TINY:
            srcov[:, iref] /= dtmp
            rrslope[:, iref] /= dtmp
        else:
            srcov[:, iref] = 0.
            rrslope[:, iref] = 0.
    logger.info("Number of samples used : %d" % n_samples)
    print "Compare results with 'standard' values:"
    print "cmp(sigmean,smean):", np.allclose(smean, sigmean, atol=0.)
    print "cmp(refmean,rmean):", np.allclose(rmean, refmean, atol=0.)
    print "cmp(sscovdata,sscov):", np.allclose(sscov, sscovdata, atol=0.)
    print "cmp(srcovdata,srcov):", np.allclose(srcov, srcovdata, atol=0.)
    print "cmp(rrcovdata,rrcov):", np.allclose(rrcov, rrcovdata, atol=0.)
    tc1 = time.clock()
    tw1 = time.time()
    print "sigrefchn covar-calc took %.1f ms (%.2f s walltime)" % (1000. * (tc1 - tct), (tw1 - twt))

    if checkresults:
        print "########## Calculated initial signal channel covariance:"
        # Calculate initial signal channel covariance:
        # (only used as quality measure)
        print "initl rt(avg sig pwr) = %12.5e" % np.sqrt(np.mean(sscov))
        for i in xrange(5):
            print "initl signal-rms[%3d] = %12.5e" % (i, np.sqrt(sscov.flatten()[i]))
        print " "
    if nref < 6:
        print "rrslope-entries:"
        for i in xrange(nref):
            print rrslope[i][:]

    U, s, V = np.linalg.svd(rrslope, full_matrices=True)
    print s

    print "Applying cutoff for smallest SVs:"
    dtmp = s.max() * SVD_RELCUTOFF
    sinv = np.zeros(nref)
    for i in xrange(nref):
        if abs(s[i]) >= dtmp:
            sinv[i] = 1. / s[i]
        else:
            s[i] = 0.
    # s *= (abs(s)>=dtmp)
    # sinv = ???
    print s
    stat = np.allclose(rrslope, np.dot(U, np.dot(np.diag(s), V)))
    print ">>> Testing svd-result: %s" % stat
    if not stat:
        print "    (Maybe due to SV-cutoff?)"

    # Solve for inverse coefficients:
    print ">>> Setting RRinvtr=U diag(sinv) V"
    RRinvtr = np.zeros((nref, nref))
    RRinvtr = np.dot(U, np.dot(np.diag(sinv), V))
    if checkresults:
        # print ">>> RRinvtr-result:"
        # print RRinvtr
        stat = np.allclose(np.identity(nref), np.dot(rrslope.transpose(), RRinvtr))
        if stat:
            print ">>> Testing RRinvtr-result (shld be unit-matrix): ok"
        else:
            print ">>> Testing RRinvtr-result (shld be unit-matrix): failed"
            print np.dot(rrslope.transpose(), RRinvtr)
            # np.less_equal(np.abs(np.dot(rrslope.transpose(),RRinvtr)-np.identity(nref)),0.01*np.ones((nref,nref)))
        print ""

    print "########## Calc weight matrix..."
    # weights-matrix will be somewhat larger than necessary,
    # (to simplify indexing in compensation loop):
    weights = np.zeros((raw._data.shape[0], nref))
    for isig in xrange(nsig):
        for iref in xrange(nref):
            weights[sigpick[isig]][iref] = np.dot(srcov[isig][:], RRinvtr[iref][:])

    if np.allclose(np.zeros(weights.shape), np.abs(weights), atol=1.e-8):
        print ">>> all weights are small (<=1.e-8)."
    else:
        print ">>> largest weight %12.5e" % np.max(np.abs(weights))
        wlrg = np.where(np.abs(weights) >= 0.99 * np.max(np.abs(weights)))
        for iwlrg in xrange(len(wlrg[0])):
            print ">>> weights[%3d,%2d] = %12.5e" % \
                  (wlrg[0][iwlrg], wlrg[1][iwlrg], weights[wlrg[0][iwlrg], wlrg[1][iwlrg]])

    if nref < 5:
        print "weights-entries for first sigchans:"
        for i in xrange(5):
            print 'weights[sp(%2d)][r]=[' % i + ' '.join([' %+10.7f' %
                             val for val in weights[sigpick[i]][:]]) + ']'

    print "########## Compensating signal channels:"
    tct = time.clock()
    twt = time.time()
    # data,times = raw[:,raw.time_as_index(tmin)[0]:raw.time_as_index(tmax)[0]:]
    # Work on entire data stream:
    for isl in xrange(raw._data.shape[1]):
        slice = np.take(raw._data, [isl], axis=1)
        if use_reffilter:
            refslice = np.take(fltref._data, [isl], axis=1)
            refarr = refslice[:].flatten() - rmean
            # refarr = fltres[:,isl]-rmean
        else:
            refarr = slice[refpick].flatten() - rmean
        subrefarr = np.dot(weights[:], refarr)
        # data[:,isl] -= subrefarr   will not modify raw._data?
        raw._data[:, isl] -= subrefarr
        if isl%10000 == 0:
            print "\rProcessed slice %6d" % isl
    print "\nDone."
    tc1 = time.clock()
    tw1 = time.time()
    print "compensation loop took %.1f ms (%.2f s walltime)" % (1000. * (tc1 - tct), (tw1 - twt))

    if checkresults:
        print "########## Calculating final signal channel covariance:"
        # Calculate final signal channel covariance:
        # (only used as quality measure)
        tct = time.clock()
        twt = time.time()
        sigmean = 0
        sscovdata = 0
        n_samples = 0
        for first in range(itmin, itmax, itstep):
            last = first + itstep
            if last >= itmax:
                last = itmax
            raw_segmentsig, times = raw[sigpick, first:last]
            # Artifacts found here will probably differ from pre-noisered artifacts!
            if not exclart or \
               _is_good(raw_segmentsig, infosig['ch_names'], idx_by_typesig, reject,
                        flat=None, ignore_chs=raw.info['bads']):
                sigmean += raw_segmentsig.sum(axis=1)
                sscovdata += (raw_segmentsig * raw_segmentsig).sum(axis=1)
                n_samples += raw_segmentsig.shape[1]
        sigmean /= n_samples
        sscovdata -= n_samples * sigmean[:] * sigmean[:]
        sscovdata /= (n_samples - 1)
        print ">>> no channel got worse: ", np.all(np.less_equal(sscovdata, sscovinit))
        print "final rt(avg sig pwr) = %12.5e" % np.sqrt(np.mean(sscovdata))
        for i in xrange(5):
            print "final signal-rms[%3d] = %12.5e" % (i, np.sqrt(sscovdata.flatten()[i]))
        tc1 = time.clock()
        tw1 = time.time()
        print "signal covar-calc took %.1f ms (%.2f s walltime)" % (1000. * (tc1 - tct), (tw1 - twt))
        print " "

    nrname = dname[:dname.rfind('-raw.fif')] + ',nold-raw.fif'
    print "Saving '%s'..." % nrname
    raw.save(nrname, overwrite=True)
    tc1 = time.clock()
    tw1 = time.time()
    print "Total run         took %.1f ms (%.2f s walltime)" % (1000. * (tc1 - tc0), (tw1 - tw0))
