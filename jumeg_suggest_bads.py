#!/usr/bin/env python2
# -*- coding: utf-8 -*-

'''
Contains function to identify bad channels based on time and freq domain
methods.

authors: Niko Kampel, n.kampel@gmail.com
         Praveen Sripad, pravsripad@gmail.com
'''

import numpy as np
import mne
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import euclidean_distances
from .jumeg_utils import check_read_raw


def compute_euclidean_stats(epoch, sensitivity, mode='adaptive',
                            fraction=None):
    '''
    Compute the Euclidean matrix along with necessary statistics for data
    from one single epoch.

    Function can also be used for psd. (generic function)

    Parameters
    epoch: np.array
        The data from which to compute the Euclidean matrices.
    sensitivity: float in range of [0,100]
        Percentile to compute threshold used for clustering,
        which must be between 0 and 100 inclusive.
    mode: str
        The mode in which to return the statistics results.
        Can be 'fixed' for fixed threshold or 'nearest'
        for nearest neighbour points.
        When a fixed threshold is used, a single percentile based value is
        used for all the epochs/windows of the data. If adaptive is chosen,
        a threshold value for every epoch is used.
        Note: Fixed threshold is currently incompletely implemented and
        we do not suggest using it.
    fraction: float | None
        Ratio of the number of samples to be chosen for clustering.

    Returns
    If mode is fixed returns a fixed percentile threshold.
    If mode is nearest, returns the nearest neighbour.

    #TODO doc needs to be updated
    '''
    if fraction:
        number_of_samples = int(epoch.shape[1]*fraction)
        sorted_peaks = np.sort(np.square(np.diff(epoch)), axis=1)
        # just keep 1% of the samples
        afp = sorted_peaks[:, sorted_peaks.shape[1]-number_of_samples:]
    else:
        # do not do reduced sampling fro psds
        afp = epoch  # slightly confusing, this part actually handles psd code

    mydist = euclidean_distances(afp, afp)
    # average_distances = np.average(mydist, axis=1)
    if mode == 'adaptive':
        # adaptive threshold depending on epochs
        nearest_neighbour = np.sort(mydist, axis=1)[:, 1]
        selected_threshold = np.percentile(np.tril(mydist), sensitivity)
        return afp, nearest_neighbour, selected_threshold
    elif mode == 'fixed':
        # fixed threshold for all epochs
        # not to be used
        fixed_threshold = np.percentile(np.tril(mydist), sensitivity)
        return afp, fixed_threshold
    else:
        raise RuntimeError('Mode should be one of fixed or nearest')


def clustered_afp(epochs, sensitivity_steps, fraction, mode='adaptive',
                  min_samples=1):
    '''
    Perform clustering on difference in signals from one sample to another.
    This method helps us to identify flux jumps and largespikes in the data.

    Parameters
    epochs: mne.Epochs
    sensitivity_steps: float in range of [0,100]
        Percentile to compute threshold used for clusterin
        signals,
        which must be between 0 and 100 inclusive.
    picks: list
        Picks of the channels to be used.
    min_samples: int
        Number of samples to be chosen for DBSCAN clustering.

    Returns
    afps: np.array
        Power spectral density values (n_epochs, n_chans, n_freqs)
    afp_suspects: list
        Suspected bad channels.
    afp_nearest_neighbour: list
        The nearest neighbour identified before DBSCAN clustering.
    zlimit_afp: float
        A scaling value used for plotting.

    '''
    # epochs = epochs.get_data()
    afps, afp_suspects, afp_percentiles, afp_nearest_neighbour = [], [], [], []

    # statistics for every epoch
    for epoch in epochs:
        if mode is 'adaptive':
            afp, nearest_neighbour, selected_threshold = \
                compute_euclidean_stats(epoch, sensitivity_steps, mode='adaptive')
            afp_nearest_neighbour.append(nearest_neighbour)
            afp_percentiles.append(selected_threshold)
        elif mode is 'fixed':
            #TODO complete fixed threshold computation
            # statistics and clustering for every epoch for fixed threshold
            afp, selected_threshold = compute_euclidean_stats(epoch, sensitivity_steps,
                                                      mode='fixed')
            afp_percentiles.append(selected_threshold)
        else:
            raise RuntimeError('Mode unknown.')

        # do the clustering for every epoch
        db = DBSCAN(eps=selected_threshold, min_samples=min_samples,
                    metric='euclidean').fit(afp)
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        suspect = [i for i, x in enumerate(db.labels_) if x]
        afps.append(afp)
        afp_suspects.append(suspect)

    afps = np.asarray(afps)
    afp_nearest_neighbour = np.asarray(afp_nearest_neighbour)
    # hack to get a limit for plotting (this is not supposd to be here)
    zlimit_afp = np.percentile(afp_percentiles, 50) * 4
    return afps, afp_suspects, afp_nearest_neighbour, zlimit_afp


def clustered_psd(epochs, sensitivity_psd, picks, min_samples=1):
    '''
    Perform clustering on PSDs to identify bad channels.

    Parameters
    epochs: mne.Epochs
    sensitivity_psd: float in range of [0,100]
        Percentile to compute threshold used for clustering PSDs,
        which must be between 0 and 100 inclusive.
    picks: list
        Picks of the channels to be used.
    min_samples: int
        Number of samples to be chosen for DBSCAN clustering.

    Returns
    psds: np.array
        Power spectral density values (n_epochs, n_chans, n_freqs)
    psd_suspects: list
        Suspected bad channels.
    psd_nearest_neighbour: list
        The nearest neighbour identified before DBSCAN clustering.
    zlimit_psd: float
        A scaling value used for plotting.
    '''
    psds, freqs = mne.time_frequency.psd_welch(epochs, fmin=2., fmax=200.,
                                               picks=picks)
    psd_percentiles, psd_nearest_neighbour, psd_suspects = [], [], []

    for ipsd in psds:
        psd, nearest_neighbour, selected_threshold = \
            compute_euclidean_stats(ipsd, sensitivity_psd, mode='adaptive')
        psd_nearest_neighbour.append(nearest_neighbour)
        psd_percentiles.append(selected_threshold)
        db = DBSCAN(eps=selected_threshold, min_samples=min_samples,
                    metric='euclidean').fit(psd)
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        suspect = [i for i, x in enumerate(db.labels_) if x]
        psd_suspects.append(suspect)

    psd_nearest_neighbour = np.asarray(psd_nearest_neighbour)
    zlimit_psd = np.percentile(psd_percentiles, 50) * 4
    return psds, psd_suspects, psd_nearest_neighbour, zlimit_psd


def make_minimap(picks, afp_suspects, psd_suspects):
    '''
    Make a minimap with bad channels identifed using time domain and freq
    domain methods.
    Helper function for plotting the values
    '''
    # values inside minimap are a workaround for colormap 'brg'
    minimap = np.zeros((len(picks), len(afp_suspects)))  # 0 if channel is regular

    for e in range(0, len(afp_suspects)):
        for c in afp_suspects[e]:
            minimap[c, e] = 3  # yellow if afp is unusual

    for e in range(0, len(afp_suspects)):
        for c in psd_suspects[e]:
            if minimap[c, e] == 3:
                minimap[c, e] = 2  # red if afp+psd is unusual
            else:
                minimap[c, e] = 1  # purple if psd is unusual

    # minimap marker
    # coordinates for markers
    x_afp, y_afp, x_psd, y_psd, x_both, y_both = [], [], [], [], [], []

    for e in range(0, minimap.shape[1]):
        for c in range(0, len(minimap)):
            if minimap[c, e] == 3:  # condition for afp
                x_afp.append(e)
                y_afp.append(c)
            if minimap[c, e] == 1: # condition for psd
                x_psd.append(e)
                y_psd.append(c)
            if minimap[c, e] == 2: # condition for both
                x_both.append(e)
                y_both.append(c)

    return minimap, x_afp, y_afp, x_psd, y_psd, x_both, y_both


def validation_marker(minimap, picks_bad, picks_fp):
    '''
    Helper function for plotting bad channels identified using time domain (afp)
    or freq domain (psd) methods.
    Using the validation marker helps compare already marked bad channels with
    automatically identified ones for testing purposes.
    '''
    x_miss, y_miss, x_hit, y_hit, x_fp, y_fp = [], [], [], [], [], []
    for e in range(0, minimap.shape[1]):
        for c in range(0, len(minimap)):
            if c in picks_bad and minimap[c, e] > 0:  # condition for hit
                x_hit.append(e)
                y_hit.append(c)
            if c in picks_bad and minimap[c, e] == 0:  # condition for miss
                x_miss.append(e)
                y_miss.append(c)
            if c in picks_fp and minimap[c, e] > 0:  # condition for miss
                x_fp.append(e)
                y_fp.append(c)
    return x_miss, y_miss, x_hit, y_hit, x_fp, y_fp


def plot_autosuggest_summary(afp_nearest_neighbour, psd_nearest_neighbour,
                             picks, afp_suspects, psd_suspects, picks_bad,
                             picks_fp, zlimit_afp, zlimit_psd,
                             epoch_length, marks, validation=False):
    '''
    Plot showing the automated identification of bad channels using time and
    frequency domain methods.

    #TODO Improve documentation.
    '''
    import matplotlib.pyplot as plt
    plt.style.use(['seaborn-deep'])

    # calculate data for summary_plot
    minimap, x_afp, y_afp, x_psd, y_psd, x_both, y_both = \
        make_minimap(picks, afp_suspects, psd_suspects)

    # calculate validation markers if necessary (for testing purposes only)
    if validation:
        x_miss, y_miss, x_hit, y_hit, x_fp, y_fp = \
            validation_marker(minimap, picks_bad, picks_fp)

    # do the actual plotting
    summary_plot = plt.figure(figsize=(16, 10))
    plt.subplots_adjust(hspace=0.2)
    t = np.arange(len(minimap[1]))

    # minimap
    ax1 = plt.subplot2grid((4, 1), (0, 0), rowspan=2)
    ax1.xaxis.tick_top()
    ax1.set_xticks((t))
    plt.xticks(t, (t+1)*epoch_length-epoch_length/2)  # align minimap with clusterplots
    ax1.grid(which='both')
    plt.xlim([0, len(t)-1])
    plt.ylim([len(minimap), 0])
    plt.yticks(marks, [x+1 for x in marks])  # only tick channels of interest +1 cause numpy and mne coordinates are differnt
    plt.ylabel('channel number')
    # plt.xlabel('raw_fname = '+"'"+raw_fname+"'" + ' ; marked_chn = '+str(list(marks)))
    ax1.xaxis.set_label_position('top')

    #TODO find better way to find zlimit
    # zlimit_afp = np.percentile(afp_percentiles, 50) * 4
    plt.imshow(np.clip(afp_nearest_neighbour, 0, zlimit_afp).T*-1,
               aspect='auto', interpolation='nearest', cmap='Blues')

    # mark the default points
    plt.scatter(x_afp, y_afp, s=60,  marker='o', color='gold')
    plt.scatter(x_both, y_both, s=60,  marker='o', color='red')

    # validation marker
    if validation:
        plt.scatter(x_miss, y_miss, s=10,  marker='s', color='r')
        plt.scatter(x_hit, y_hit, s=10,  marker='s', color='limegreen')
        plt.scatter(x_fp, y_fp, s=10,  marker='s', color='gold')

    # plot the AFP clustering
    ax2 = plt.subplot2grid((4, 1), (2, 0), rowspan=2)
    ax2.xaxis.tick_top()
    ax2.set_xticks((t))
    plt.xticks(t, (t+1)*epoch_length-epoch_length/2)  # align minimap with clusterplots
    ax2.grid(which='both')
    plt.xlim([0, len(t)-1])
    plt.ylim([len(minimap), 0])
    plt.yticks(marks, [x+1 for x in marks])  # only tick channels of interest +1 cause numpy and mne coordinates are differnt
    plt.ylabel('channel number')
    plt.xlabel('time')

    plt.scatter(x_psd, y_psd, s=60,  marker='o', color='purple')
    plt.scatter(x_both, y_both, s=60,  marker='o', color='red')

    # validation marker
    if validation:
        plt.scatter(x_miss, y_miss, s=20,  marker='s', color='r')
        plt.scatter(x_hit, y_hit, s=20,  marker='s', color='limegreen')
        plt.scatter(x_fp, y_fp, s=20,  marker='s', color='gold')

    #TODO find better way to find zlimit
    # zlimit_psd = np.percentile(psd_percentiles, 50) * 4
    ax2.imshow(np.clip(psd_nearest_neighbour, 0, zlimit_psd).T*-1,
               aspect='auto', interpolation='nearest', cmap='Blues')

    plt.close()
    return summary_plot


def old_summary_plot():
    '''
    old summary plot - discarded

    kept here only for archival purposes
    '''
    import matplotlib.pyplot as plt
    summary_plot = plt.figure()
    plt.subplots_adjust(hspace=0)
    t = np.arange(len(minimap[1]))

    ################## minimap ################
    ax1 = plt.subplot2grid((4,1), (0,0), rowspan=2)

    # axis setup
    ax1.xaxis.tick_top()
    ax1.set_xticks((t))
    plt.xticks(t,(t+1)*epoch_length-epoch_length/2)  # align minimap with clusterplots
    ax1.grid(which='both',color='w')
    plt.xlim([0,len(t)-1])
    plt.ylim([len(minimap),0])
    plt.yticks(marks,marks+1 )  # only tick channels of interest +1 cause numpy and mne coordinates are differnt
    plt.ylabel('channel number')
    # plt.xlabel('raw_fname = '+"'"+raw_fname+"'" + ' ; marked_chn = '+str (list(marks)))
    ax1.xaxis.set_label_position('top')
    # default marker
    if not validation:
        plt.scatter(x_afp, y_afp, s=40,  marker='o', color='yellow')
        plt.scatter(x_psd, y_psd, s=40,  marker='o', color='purple')
        plt.scatter(x_both, y_both, s=40,  marker='o', color='red')

    # validation marker
    if validation:
        plt.scatter(x_miss, y_miss, s=40,  marker='s', color='r')
        plt.scatter(x_hit, y_hit, s=40,  marker='s', color='limegreen')
        plt.scatter(x_fp, y_fp, s=40,  marker='s', color='yellow')
    # data
    plt.imshow(minimap,  aspect= 'auto',interpolation= 'nearest' ,cmap = 'brg',vmin=0, vmax=4 )

    ################## afp_clustering ################
    ax2 = plt.subplot2grid((4,1), (2,0))

    # axis setup
    ax2.set_xticks((t))
    ax2.grid(which='both')
    ylimit_afp=np.percentile(afp_percentiles,50)*6     #to provide a usfull range for clusterplot inspection; <6 for more details
    plt.ylim([np.min(afp_nearest_neighbour),ylimit_afp])
    plt.ylabel('step detection'+'\n'+'eukledian dist. '+r'[${\Delta}T^2$]')

    #topolines
    for k in range (0,len(afp_percentiles[1])-1):
        ax2.fill_between(t, 0, afp_percentiles[:,k], facecolor=(0.1, 0.2, 0.5), alpha=.09)

    # selected threshold and channel cluster points
    ax2.plot(afp_percentiles.T[len(afp_percentiles[1])-1], color='yellow', linewidth=2)
    ax2.plot(afp_nearest_neighbour, 'ro', color='b')

    # graph of marked channels
    for chanels in marks:
        plt.plot(afp_nearest_neighbour[:,chanels],alpha=.6)

    # cluster dots exceeding threshold
    for epoch in range(0,len(afp_nearest_neighbour)):
        for chn in afp_suspects[epoch]:
            if afp_nearest_neighbour[epoch,chn]> ylimit_afp:#hack to make extreme points visible inside y-limit
                ax2.plot(t[epoch],ylimit_afp*0.97, '^', color= 'yellow',markersize=8)
                ax2.annotate(str(chn+1), xy=(t[epoch]+0.15,ylimit_afp), xycoords='data',horizontalalignment='left', verticalalignment='top')
            else:
                ax2.plot(t[epoch],afp_nearest_neighbour[epoch,chn], 'ro', color= 'yellow')
                ax2.annotate(str(chn+1), xy=(t[epoch],afp_nearest_neighbour[epoch,chn]), xycoords='data')

    #################### psd_clustering #############
    ax3 = plt.subplot2grid((4,1), (3,0))

    # axis setup
    ax3.set_xticks((t))
    plt.xticks(t,(t+1)*epoch_length-epoch_length/2)
    ax3.grid(which='both')
    ylimit_psd = np.percentile(psd_percentiles,50)*6     #to provide a usfull range for clusterplot inspection; <6 for more details
    plt.ylim([np.min(psd_nearest_neighbour),ylimit_psd])
    plt.xlabel('measurement time [s]'+'\n'+'sensitivity afp = '+str(sensitivity_steps)+'      sensitivity psd = '+str(sensitivity_psd))
    plt.ylabel('frequency detection'+'\n'+'eukledian dist.[fft]')

    # topolines
    for k in range (0,len(psd_percentiles[1])-1):
        ax3.fill_between(t, 0, psd_percentiles[:,k], facecolor=(0.1, 0.2, 0.5), alpha=.09)

    ax3.plot(psd_percentiles.T[len(psd_percentiles[1])-1], color='darkviolet', linewidth=2)
    ax3.plot(psd_nearest_neighbour, 'ro', color= 'b')

    # selected threshold and channel cluster points
    for chanels in marks:
        plt.plot(psd_nearest_neighbour[:,chanels],alpha=.6)

    #cluster dots exceeding threshold
    for epoch in range(0,len(psd_nearest_neighbour)):
        for chn in psd_suspects[epoch]:
            if psd_nearest_neighbour[epoch,chn]> ylimit_psd:#hack to make extreme points visible inside y-limit
                ax3.plot(t[epoch],ylimit_psd*0.97, '^', color= 'darkviolet',markersize=8)
                ax3.annotate(str(chn+1), xy=(t[epoch]+0.15,ylimit_psd), xycoords='data', horizontalalignment='left', verticalalignment='top')
            else:
                ax3.plot(t[epoch],psd_nearest_neighbour[epoch,chn], 'ro', color= 'darkviolet')
                ax3.annotate(str(chn+1), xy=(t[epoch],psd_nearest_neighbour[epoch,chn]), xycoords='data')

    plt.tight_layout()
    # plt.show()
    # plt.close()
    return summary_plot


def suggest_bads(raw, sensitivity_steps=97, sensitivity_psd=95,
                 fraction=0.001, epoch_length=None, summary_plot=False,
                 show_raw=False, validation=True):
    '''
    Function to suggest bad channels. The bad channels are identified using
    time domain methods looking for sharp jumps in short windows of data and
    in the frequency domain looking for channels with unusual power
    spectral densities.

    Note: This function is still in the development stage and contains a lot of
    hard coded values.

    Parameters
    ----------
    raw: str | mne.io.Raw
        Filename or the raw object.
    epoch_length: int | None
        Length of the window to apply methods on.
    summary_plot: bool
        Set True to generate a summary plot showing suggested bads.

    # parameters for step detection (AFP)
    # in %, 0 marks all chanels 100 marks none; percentile of

    # parameter for frequency analysis
    # in %, 0 marks all chanels 100 marks none; percentile of

    Returns
    -------
    suggest_bads: list
        List of suggested bad channels.
    raw: mne.io.Raw
        Raw object updated with suggested bad channels.
    '''

    raw = check_read_raw(raw, preload=False)
    picks = mne.pick_types(raw.info, meg=True, eeg=False, eog=False,
                           ecg=False, exclude=[])
    # if epoch length is not provided, chose a suitable length
    if not epoch_length:
        epoch_length = int(raw.n_times/(raw.info['sfreq'] * 20))
    print 'epoch_length of %d chosen' % epoch_length
    # add 0.01 to avoid 'dropping' of first epoch
    events = mne.make_fixed_length_events(raw, 42, start=0.01,
                                          duration=epoch_length)
    epochs = mne.Epochs(raw, events, event_id=42, tmin=-epoch_length/2,
                        tmax=epoch_length/2, picks=picks)
    picks_bad = [raw.ch_names.index(l) for l in raw.info['bads']]

    # compute differences in time domain to identify abrupt jumps in the data
    afps, afp_suspects, afp_nearest_neighbour, zlimit_afp = \
        clustered_afp(epochs, sensitivity_steps, fraction)

    # compute the psds and do the clustering to identify unusual channels
    psds, psd_suspects, psd_nearest_neighbour, zlimit_psd = \
        clustered_psd(epochs, sensitivity_psd, picks)

    # if any of the channels' psds are all zeros, mark as suspect
    zero_suspects = [ind for ind in range(psds.shape[1]) if not np.any(psds[:, ind, :])]

    # reduce lists of marked epochs to lists of bad channels
    picks_autodetect = \
        list(set().union([item for sublist in psd_suspects for item in sublist],
                         [item for sublist in afp_suspects for item in sublist]))

    # get the bads suggested but not previosuly marked
    picks_fp = [x for x in set(picks_autodetect) if x not in set(picks_bad)]

    #  marks are all channels of interest, including premarked bad channels
    # and zero channels (channel indices)

    jumps = list(set([item for sublist in afp_suspects for item in sublist]))
    jumps_ch_names = [raw.ch_names[i] for i in jumps]
    unusual = list(set([item for sublist in psd_suspects for item in sublist]))
    unusual_ch_names = [raw.ch_names[i] for i in unusual]
    dead_ch_names = [raw.ch_names[i] for i in zero_suspects]

    print "Suggested bads [jumps]:", jumps_ch_names
    print "Suggested bads [unusual]:", unusual_ch_names
    print "Suggested bads [dead]:", dead_ch_names

    marks = list(set(picks_autodetect) | set(picks_bad) | set(zero_suspects))

    # show summary plot for enhanced manual inspection
    #TODO zero suspects do not have any colour coding for the moment
    if summary_plot:
        fig = \
            plot_autosuggest_summary(afp_nearest_neighbour, psd_nearest_neighbour,
                                     picks, afp_suspects, psd_suspects, picks_bad,
                                     picks_fp, zlimit_afp, zlimit_psd,
                                     epoch_length, marks,
                                     validation=False)
        fig.show()

    # channel names in str
    marked = [raw.ch_names[i] for i in marks]
    # add suggested channels to the raw.info
    raw.info['bads'] = marked
    print 'Suggested bad channels: ', marked

    if show_raw:
        raw.plot(block=True)
        marked = raw.info['bads']
        marked.sort()

    return marked, raw
