#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 10:14:56 2017

@author: nkampel
"""

import numpy as np
import mne
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import euclidean_distances


def epoch_from_raw(raw_fname):
    '''
    # load data and do epoching
    '''
    raw = mne.io.Raw(raw_fname, preload=False)
    picks = mne.pick_types(raw.info, meg=True, eeg=False, eog=False, ecg=False, exclude=[])
    # to many epochs can produce memory lag;
    # to do: investigate detection rate in dependece of epoch length
    if auto_epoch:
        _, times = raw[picks, :]
        epoch_length = int(times[-1]/20)
    # add 0.01 to avoid 'dropping' of first epoch
    events = mne.make_fixed_length_events(raw, 42, start=epoch_length/2+0.01,
                                          duration=epoch_length)
    epochs = mne.Epochs(raw, events, event_id=42, tmin=-epoch_length/2,
                        tmax=epoch_length/2, picks=picks)
    n_epochs = len(events)
    bads = raw.info['bads']
    picks_bad = np.array([raw.ch_names.index(l) for l in bads])
    return epochs, picks, n_epochs, picks_bad, epoch_length


def compute_euclidean_stats(data, sensitivity, mode='fixed'):
    '''
    Compute the Euclidean matrix along with necessary statistics.

    Parameters
    data: np.array
        The data from which to compute the Euclidean matrices.
    mode: str
        The mode in which to return the statistics results.
        Can be 'fixed' for fixed threshold or 'nearest'
        for nearest neighbour points.

    Returns
    If mode is fixed returns a fixed percentile threshold.
    If mode is nearest, returns the nearest neighbour.
    '''
    mydist = euclidean_distances(data, data)
    average_distances = np.average(mydist, axis=1)
    if mode == 'nearest':
        nearest_neighbour = np.sort(mydist, axis=1)[:, 1]
        return nearest_neighbour
    elif mode == 'fixed':
        fixed_threshold = np.percentile(average_distances, sensitivity_steps)
        return fixed_threshold
    else:
        raise RuntimeError('Mode should be one of fixed or nearest')


def clustered_afp(epochs, sensitivity_steps, fraction, adpt_t_afp):
    '''
    # stepdetection
    '''
    epochs = epochs.get_data()
    afps, afp_suspects, afp_percentiles, afp_nearest_neighbour = [], [], [], []
    # adpt_t_afp = False
    # precalculation for fixed threshold

    if not adpt_t_afp:
        print 'adapt_false'
        fixed_threshold = []
        for k in range(0, len(epochs)):
            epoch = epochs[k]
            number_of_samples = int(len(epoch[k])*fraction)
            # arranged fluctuation plots for step detection
            sorted_peaks = np.sort(np.square(np.diff(epoch)), axis=1)
            # just keep 1%(fraction) of the samples
            afp = np.delete(sorted_peaks, np.s_[0:len(sorted_peaks[1])-number_of_samples:1], 1)
            # full eucledian distance matrix to calculate statistics and fixed threshold
            dist = euclidean_distances(afp, afp)
            average_distances_afp = np.average(dist, axis=1)
            fixed_threshold = np.append(fixed_threshold,
                                        np.percentile(average_distances_afp,
                                        sensitivity_steps))
        fixed_threshold = np.mean(fixed_threshold)
        # based on the stats of full distant matrix (not only nearest neighbours)
        # increased calculation time,

    # statistics for every epoch
    for k in range(0, len(epochs)):
        epoch = epochs[k]
        number_of_samples = int(len(epoch[k])*fraction)
        sorted_peaks = np.sort(np.square(np.diff(epoch)), axis=1)
        afp = np.delete(sorted_peaks, np.s_[0:len(sorted_peaks[1])-number_of_samples:1], 1)
        # full eucledian distance matrix to calculate statistics and adaptive threshold
        dist = euclidean_distances(afp, afp)
        average_distances_afp = np.average(dist, axis=1)
        nearest_neighbour = np.sort(dist, axis=1)[:, 1]
        afp_nearest_neighbour.append(nearest_neighbour)

        if adpt_t_afp:
            selected_threshold = np.percentile(average_distances_afp, sensitivity_steps)
            stats = []  # below are the given sensitivities for creating topo lines
            for topocurve in np.array([50,75,87.5,93.75,96.875,98.4375,99.21875, sensitivity_steps]):
                stats.append(np.percentile(average_distances_afp, topocurve))
            afp_percentiles.append(stats)
            # clustering for every epoch
            db = DBSCAN(eps=selected_threshold, min_samples=neighbours_steps,
                        metric='euclidean').fit(afp)
            core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
            core_samples_mask[db.core_sample_indices_] = True
            suspect = [i for i, x in enumerate(db.labels_) if x]
            afps.append(afp)
            afp_suspects.append(suspect)
        else:
            # statistics and clustering for every epoch for fixed threshold
            selected_threshold = fixed_threshold
            stats = []  # below are the given sensitivities for creating topo lines
            for topocurve in np.array([50,75,87.5,93.75,96.875,98.4375,99.21875]):
                stats.append(np.percentile(average_distances_afp, topocurve))
                stats.append(selected_threshold)
            afp_percentiles.append(stats)
            # clustering for every epoch for fixed threshold
            db = DBSCAN(eps=selected_threshold, min_samples=neighbours_steps,
                        metric='euclidean').fit(afp)
            core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
            core_samples_mask[db.core_sample_indices_] = True
            suspect = [i for i, x in enumerate(db.labels_) if x]
            afps.append(afp)
            afp_suspects.append(suspect)

    # contains arranged fluctuation plots for every epoch not really needed as
    # output (safe memory?), maybe for further analysis
    afps = np.asarray(afps)
    afp_percentiles = np.asanyarray(afp_percentiles)
    afp_nearest_neighbour = np.asarray(afp_nearest_neighbour)
    return afps, afp_suspects, afp_percentiles, afp_nearest_neighbour


def clustered_psd(epochs, sensitivity_psd, picks):
    '''
    # frequency analysis
    '''
    psds, freqs = mne.time_frequency.psd_welch(epochs, fmin=2., fmax=200.,
                                               picks=picks)
    psd_percentiles, psd_nearest_neighbour = [], []

    # psds = np.log10(psds)
    psd_suspects = []

    # if any of the channels' psds are all zeros, mark as suspect
    # psd_suspects = [ind for ind in range(psds.shape[1]) if not np.any(psds[:, ind, :])]
    print psd_suspects

    for k in range(0, len(epochs)):
        psd = psds[k]
        # full eucledian distance matrix to calculate statistics and adaptive threshold
        dist_psd = euclidean_distances(psd, psd)
        average_distances_psd = np.average(dist_psd, axis=1)
        nearest_neighbour = np.sort(dist_psd, axis=1)[:, 1]
        psd_nearest_neighbour.append(nearest_neighbour)
        selected_percentile = np.percentile(average_distances_psd, sensitivity_psd)
        stats = []  # those are the given sensitivities for creating topo lines
        for k in np.array([50,75,87.5,93.75,96.875,98.4375,99.21875, sensitivity_psd]):
            stats.append(np.percentile(average_distances_psd, k))
        psd_percentiles.append(stats)
        # percentile = np.percentile(selected_percentile,sensitivity_psd)
        db = DBSCAN(eps=selected_percentile, min_samples=neighbours_psd,
                    metric='euclidean').fit(psd)
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        suspect = [i for i, x in enumerate(db.labels_) if x]
        psd_suspects.append(suspect)

    psd_percentiles = np.asanyarray(psd_percentiles)
    psd_nearest_neighbour = np.asarray(psd_nearest_neighbour)
    # psd not really needed as output, (save memory?)
    return psds, psd_suspects, psd_percentiles, psd_nearest_neighbour


def minimap(picks, afp_suspects, psd_suspects):
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

    for e in range(0, len(minimap[1])):
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


def validation_marker(minimap,picks_bad,picks_fp):
    '''
    # validation marker
    # coordinates for validation markers
    '''
    x_miss, y_miss, x_hit, y_hit, x_fp, y_fp = [], [], [], [], [], []
    for e in range(0, len(minimap[1])):
        for c in range(0, len(minimap)):
            if c in picks_bad and minimap[c, e] > 0:  # condition for hit
                x_hit.append(e)
                y_hit.append(c)
            if c in picks_bad and minimap[c, e] == 0: # condition for miss
                x_miss.append(e)
                y_miss.append(c)
            if c in picks_fp and minimap[c, e] > 0:  # condition for miss
                x_fp.append(e)
                y_fp.append(c)
    return x_miss, y_miss, x_hit, y_hit, x_fp, y_fp


def summary_plot():
    '''
    summary plot
    '''
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

# Main Script

# provide the path of the filename:
raw_fname = '/Users/psripad/fzj_sciebo/noisy_channel_detection_2016/jul017/jul017_BadChTst-2_16-12-06@11:50_1_c,rfDC-raw.fif'

# enhanced manual inspection
visual_inspection = True
automark = True  # mark all suspicious channels
show_raw = True  # open MNE-raw browser for checking

# provide validation markers on minimap, use with bcc.raw files
validation =False

# epoching:chose smaller the epochs for better time resolution
auto_epoch = True
number_of_epochs = 20
# or manual epoching
epoch_length = 30

# parameter for step detection
adpt_t_afp = True              # avoid the marking of muscle artifacts, change to False when muscle artifacts should be marked
sensitivity_steps = 97         # in %, 0 marks all chanels 100 marks none; percentile of eucledian distance distribution(channel-afp)
neighbours_steps = 1           # for tuning DBSCAN-clustering
fraction = 0.001               # percentage of samples used for afp calculation

# parameter for frequency analysis
adaptive_threshold_freq = True  # not implemented yet
sensitivity_psd = 95            # in %, 0 marks all chanels 100 marks none; percentile of eucledian distance distribution(channel-psd)
neighbours_psd = 1              # for tuning DBSCAN-clustering


# loading -> epoching -> clustering
epochs, picks, n_epochs, picks_bad, epoch_length = epoch_from_raw(raw_fname)
afps, afp_suspects, afp_percentiles, afp_nearest_neighbour = clustered_afp(epochs, sensitivity_steps, fraction, adpt_t_afp)
psds, psd_suspects, psd_percentiles, psd_nearest_neighbour = clustered_psd(epochs, sensitivity_psd, picks)

# reduce lists of marked epochs to lists of bad channels
picks_autodetect = list(set().union([item for sublist in psd_suspects for item in sublist],
                                    [item for sublist in afp_suspects for item in sublist]))

picks_autodetect = np.asarray(picks_autodetect)
picks_bad = np.asarray(picks_bad)
picks_fp = np.setdiff1d(picks_autodetect, picks_bad)

# marks are all channels of interest, including premarked bad channels
marks=np.union1d(picks_autodetect,picks_bad)

# calculate data for summary_plot
minimap,x_afp,y_afp,x_psd,y_psd,x_both,y_both=minimap(picks,afp_suspects,psd_suspects)

# calculate validation markers
if validation:
    x_miss,y_miss,x_hit,y_hit,x_fp,y_fp = validation_marker(minimap,picks_bad,picks_fp)

# show summary plot for enhanced manual inspection(execute above code before)
# or put other methods inside to run function without script???
if visual_inspection:
    summary_plot = summary_plot()
    summary_plot.show()

# suspicious channels are marked in the mne dataset
if automark:
    raw = mne.io.Raw(raw_fname, preload=True)
    marked = list()
    for chns in marks:
        marked.append('MEG '+"%03d" % (chns+1))
    raw.info['bads'] = marked
    # show raw data in mne browser
    if show_raw:
        raw.plot(block=True)

# text output of the script
print 'possible bad channels: ' + str(marks+1)

# TODO Save the bad channels permanently after adding bcc prefix
