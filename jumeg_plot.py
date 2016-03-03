'''
Plotting functions for jumeg.
'''
import os
import numpy as np
import matplotlib.pylab as pl
import matplotlib.ticker as ticker
import mne
from mpl_toolkits.axes_grid import make_axes_locatable
from jumeg.jumeg_utils import (get_files_from_list, thresholded_arr,
                               triu_indices)
from jumeg.jumeg_base import jumeg_base
from jumeg_math import (calc_performance,
                        calc_frequency_correlation)


def plot_powerspectrum(fname, raw=None, picks=None, dir_plots="plots",
                       tmin=None, tmax=None, fmin=0.0, fmax=450.0, n_fft=4096):
        '''

        '''
        import os
        import matplotlib.pyplot as pl
        import mne
        from distutils.dir_util import mkpath

        if raw is None:
            assert os.path.isfile(fname), 'ERROR: file not found: ' + fname
            raw = mne.io.Raw(fname, preload=True)

        if picks is None:
            picks = jumeg_base.pick_meg_nobads(raw)

        dir_plots = os.path.join(os.path.dirname(fname), dir_plots)
        base_fname = os.path.basename(fname).strip('.fif')

        mkpath(dir_plots)

        file_name = fname.split('/')[-1]
        fnfig = dir_plots + '/' + base_fname + '-psds.png'

        pl.figure()
        pl.title('PSDS ' + file_name)
        ax = pl.axes()
        fig = raw.plot_psds(fmin=fmin, fmax=fmax, n_fft=n_fft, n_jobs=1, proj=False, ax=ax,
                            color=(0, 0, 1), picks=picks, area_mode='range')
        pl.ioff()
        # pl.ion()
        fig.savefig(fnfig)
        pl.close()

        return fname


def plot_average(filenames, save_plot=True, show_plot=False, dpi=100):

    ''' Plot Signal average from a list of averaged files. '''

    fname = get_files_from_list(filenames)

    # plot averages
    pl.ioff()  # switch off (interactive) plot visualisation
    factor = 1e15
    for fnavg in fname:
        name = fnavg[0:len(fnavg) - 4]
        basename = os.path.splitext(os.path.basename(name))[0]
        print fnavg
        # mne.read_evokeds provides a list or a single evoked based on condition.
        # here we assume only one evoked is returned (requires further handling)
        avg = mne.read_evokeds(fnavg)[0]
        ymin, ymax = avg.data.min(), avg.data.max()
        ymin *= factor * 1.1
        ymax *= factor * 1.1
        fig = pl.figure(basename, figsize=(10, 8), dpi=100)
        pl.clf()
        pl.ylim([ymin, ymax])
        pl.xlim([avg.times.min(), avg.times.max()])
        pl.plot(avg.times, avg.data.T * factor, color='black')
        pl.title(basename)

        # save figure
        fnfig = os.path.splitext(fnavg)[0] + '.png'
        pl.savefig(fnfig, dpi=dpi)

    pl.ion()  # switch on (interactive) plot visualisation


def plot_performance_artifact_rejection(meg_raw, ica, fnout_fig,
                                        meg_clean=None, show=False,
                                        proj=False, verbose=False,
                                        name_ecg='ECG 001', name_eog='EOG 002'):
    '''
    Creates a performance image of the data before
    and after the cleaning process.
    '''

    from mne.preprocessing import find_ecg_events, find_eog_events
    from jumeg import jumeg_math as jmath

    # name_ecg = 'ECG 001'
    # name_eog_hor = 'EOG 001'
    # name_eog_ver = 'EOG 002'
    event_id_ecg = 999
    event_id_eog = 998
    tmin_ecg = -0.4
    tmax_ecg = 0.4
    tmin_eog = -0.4
    tmax_eog = 0.4

    picks = mne.pick_types(meg_raw.info, meg=True, ref_meg=False,
                           exclude='bads')
    # as we defined x% of the explained variance as noise (e.g. 5%)
    # we will remove this noise from the data
    if meg_clean:
        meg_clean_given = True
    else:
        meg_clean_given = False
        meg_clean = ica.apply(meg_raw, exclude=ica.exclude,
                              n_pca_components=ica.n_components_,
                              copy=True)

    # plotting parameter
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    # check if ECG and EOG was recorded in addition
    # to the MEG data
    ch_names = meg_raw.info['ch_names']

    # ECG
    if name_ecg in ch_names:
        nstart = 0
        nrange = 1
    else:
        nstart = 1
        nrange = 1

    # EOG
    if name_eog in ch_names:
        nrange = 2

    y_figsize = 6 * nrange
    perf_art_rej = np.zeros(2)

    # ToDo:  How can we avoid popping up the window if show=False ?
    pl.ioff()
    pl.figure('performance image', figsize=(12, y_figsize))
    pl.clf()

    # ECG, EOG:  loop over all artifact events
    for i in range(nstart, nrange):
        # get event indices
        if i == 0:
            baseline = (None, None)
            event_id = event_id_ecg
            idx_event, _, _ = find_ecg_events(meg_raw, event_id,
                                              ch_name=name_ecg,
                                              verbose=verbose)
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
            idx_event = find_eog_events(meg_raw, event_id, ch_name=name_eog,
                                        verbose=verbose)
            idx_ref_chan = meg_raw.ch_names.index(name_eog)
            tmin = tmin_eog
            tmax = tmax_eog
            pl1 = nrange * 100 + 21 + (nrange - nstart - 1) * 2
            pl2 = nrange * 100 + 22 + (nrange - nstart - 1) * 2
            text1 = "OA: original data"
            text2 = "OA: cleaned data"

        # average the signals
        raw_epochs = mne.Epochs(meg_raw, idx_event, event_id, tmin, tmax,
                                picks=picks, baseline=baseline, proj=proj,
                                verbose=verbose)
        cleaned_epochs = mne.Epochs(meg_clean, idx_event, event_id, tmin, tmax,
                                    picks=picks, baseline=baseline, proj=proj,
                                    verbose=verbose)
        ref_epochs = mne.Epochs(meg_raw, idx_event, event_id, tmin, tmax,
                                picks=[idx_ref_chan], baseline=baseline,
                                proj=proj, verbose=verbose)

        raw_epochs_avg = raw_epochs.average()
        cleaned_epochs_avg = cleaned_epochs.average()
        ref_epochs_avg = np.average(ref_epochs.get_data(), axis=0).flatten() * -1.0
        times = raw_epochs_avg.times * 1e3
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
        pl.xlim(times[0], times[len(times) - 1])
        pl.ylim(1.1 * ymin, 1.1 * ymax)
        # print some info
        textstr1 = 'num_events=%d\nEpochs: tmin, tmax = %0.1f, %0.1f' \
                   % (len(idx_event), tmin, tmax)
        pl.text(times[10], 1.09 * ymax, textstr1, fontsize=10,
                verticalalignment='top', bbox=props)

        # plotting data after cleaning
        pl.subplot(pl2)
        pl.plot(times, cleaned_epochs_avg.data.T * factor, 'k')
        pl.title(text2)
        # plotting reference signal again
        pl.plot(times, jmath.rescale(ref_epochs_avg, ymin, ymax), 'r')
        pl.xlim(times[0], times[len(times) - 1])
        pl.ylim(1.1 * ymin, 1.1 * ymax)
        # print some info
        perf_art_rej[i] = calc_performance(raw_epochs_avg, cleaned_epochs_avg)
        # ToDo: would be nice to add info about ica.excluded
        if meg_clean_given:
            textstr1 = 'Performance: %d\nFrequency Correlation: %d'\
                       % (perf_art_rej[i],
                          calc_frequency_correlation(raw_epochs_avg, cleaned_epochs_avg))
        else:
            textstr1 = 'Performance: %d\nFrequency Correlation: %d\n# ICs: %d\nExplained Var.: %d'\
                       % (perf_art_rej[i],
                          calc_frequency_correlation(raw_epochs_avg, cleaned_epochs_avg),
                          ica.n_components_, ica.n_components * 100)

        pl.text(times[10], 1.09 * ymax, textstr1, fontsize=10,
                verticalalignment='top', bbox=props)

    if show:
        pl.show()

    # save image
    pl.savefig(fnout_fig + '.png', format='png')
    pl.close('performance image')
    pl.ion()

    return perf_art_rej


def plot_compare_brain_responses(fname_orig, fname_new, event_id=1,
                                 tmin=-0.2, tmax=0.5, stim_name=None,
                                 proj=False, show=False):

    '''
    Function showing performance of signal with brain responses from
    selected components only. Plots the evoked (avg) signal of original
    data and brain responses only data along with difference between them.

    fname_orig, fname_new: str
    stim_ch: str (default STI 014)
    show: bool (default False)
    '''

    pl.ioff()
    if show:
        pl.ion()

    # Get the stimulus channel for special event from the fname_new
    # make a judgment, whether this raw data include more than one kind of event.
    # if True, use the first event as the start point of the epoches.
    # Adjust the size of the time window based on different connditions
    basename = fname_new.split('-raw.fif')[0]

    # if stim_name is given we assume that the input data are raw and
    # cleaned data ('cleaned' means data were cardiac and ocular artifacts
    # were rejected)
    if stim_name:
        fnout_fig = basename + '-' + stim_name + '.png'
    else:
        stim_name = fname_new.rsplit(',ctpsbr')[0].rsplit('ar,')[1]
        # Construct file names.
        fnout_fig = basename + '.png'

    if ',' in stim_name:
        stim_ch = 'STI 014'
    elif stim_name == 'trigger':
        stim_ch = 'STI 014'
    elif stim_name == 'response':
        stim_ch = 'STI 013'

    # Read raw, calculate events, epochs, and evoked.
    raw_orig = mne.io.Raw(fname_orig, preload=True)
    raw_br = mne.io.Raw(fname_new, preload=True)

    events = mne.find_events(raw_orig, stim_channel=stim_ch, consecutive=True)
    events = mne.find_events(raw_br, stim_channel=stim_ch, consecutive=True)

    picks_orig = mne.pick_types(raw_orig.info, meg=True, exclude='bads')
    picks_br = mne.pick_types(raw_br.info, meg=True, exclude='bads')

    epochs_orig = mne.Epochs(raw_orig, events, event_id, proj=proj,
                             tmin=tmin, tmax=tmax, picks=picks_orig,
                             preload=True)
    epochs_br = mne.Epochs(raw_br, events, event_id, proj=proj,
                           tmin=tmin, tmax=tmax, picks=picks_br, preload=True)

    evoked_orig = epochs_orig.average()
    evoked_br = epochs_br.average()

    times = evoked_orig.times * 1e3
    if np.max(evoked_orig.data) < 1:
        factor = 1e15
    else:
        factor = 1
    ymin = np.min(evoked_orig.data) * factor
    ymax = np.max(evoked_orig.data) * factor

    # Make the comparison plot.
    pl.figure('Compare raw data', figsize=(14, 5))
    pl.subplot(1, 2, 1)
    pl.plot(times, evoked_orig.data.T * factor, 'k', linewidth=0.5)
    pl.plot(times, evoked_br.data.T * factor, 'r', linewidth=0.5)
    pl.title('Signal before (black) and after (red) cleaning')
    pl.xlim(times[0], times[len(times) - 1])
    pl.ylim(1.1 * ymin, 1.1 * ymax)

    # print out some information
    textstr1 = 'Performance: %d\nFrequency Correlation: %d'\
               % (calc_performance(evoked_orig, evoked_br),
                  calc_frequency_correlation(evoked_orig, evoked_br))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    pl.text(times[10], 1.09 * ymax, textstr1, fontsize=10,
            verticalalignment='top', bbox=props)

    pl.subplot(1, 2, 2)
    evoked_diff = evoked_orig - evoked_br
    pl.plot(times, evoked_diff.data.T * factor, 'k', linewidth=0.5)
    pl.title('Difference signal')
    pl.xlim(times[0], times[len(times) - 1])
    pl.ylim(1.1 * ymin, 1.1 * ymax)

    pl.savefig(fnout_fig, format='png')
    pl.close('Compare raw data')
    pl.ion()


###########################################################
#
# These functions copied from NIPY (http://nipy.org/nitime)
#
###########################################################
def drawmatrix_channels(in_m, channel_names=None, fig=None, x_tick_rot=0,
                        size=None, cmap=pl.cm.RdBu_r, colorbar=True,
                        color_anchor=None, title=None):
    r"""Creates a lower-triangle of the matrix of an nxn set of values. This is
    the typical format to show a symmetrical bivariate quantity (such as
    correlation or coherence between two different ROIs).

    Parameters
    ----------

    in_m: nxn array with values of relationships between two sets of rois or
    channels

    channel_names (optional): list of strings with the labels to be applied to
    the channels in the input. Defaults to '0','1','2', etc.

    fig (optional): a matplotlib figure

    cmap (optional): a matplotlib colormap to be used for displaying the values
    of the connections on the graph

    title (optional): string to title the figure (can be like '$\alpha$')

    color_anchor (optional): determine the mapping from values to colormap
        if None, min and max of colormap correspond to min and max of in_m
        if 0, min and max of colormap correspond to max of abs(in_m)
        if (a,b), min and max of colormap correspond to (a,b)

    Returns
    -------

    fig: a figure object

    """
    N = in_m.shape[0]
    ind = np.arange(N)  # the evenly spaced plot indices

    def channel_formatter(x, pos=None):
        thisind = np.clip(int(x), 0, N - 1)
        return channel_names[thisind]

    if fig is None:
        fig = pl.figure()

    if size is not None:

        fig.set_figwidth(size[0])
        fig.set_figheight(size[1])

    w = fig.get_figwidth()
    h = fig.get_figheight()

    ax_im = fig.add_subplot(1, 1, 1)

    #If you want to draw the colorbar:
    if colorbar:
        divider = make_axes_locatable(ax_im)
        ax_cb = divider.new_vertical(size="10%", pad=0.1, pack_start=True)
        fig.add_axes(ax_cb)

    #Make a copy of the input, so that you don't make changes to the original
    #data provided
    m = in_m.copy()

    #Null the upper triangle, so that you don't get the redundant and the
    #diagonal values:
    idx_null = triu_indices(m.shape[0])
    m[idx_null] = np.nan

    #Extract the minimum and maximum values for scaling of the
    #colormap/colorbar:
    max_val = np.nanmax(m)
    min_val = np.nanmin(m)

    if color_anchor is None:
        color_min = min_val
        color_max = max_val
    elif color_anchor == 0:
        bound = max(abs(max_val), abs(min_val))
        color_min = -bound
        color_max = bound
    else:
        color_min = color_anchor[0]
        color_max = color_anchor[1]

    #The call to imshow produces the matrix plot:
    im = ax_im.imshow(m, origin='upper', interpolation='nearest',
                      vmin=color_min, vmax=color_max, cmap=cmap)

    #Formatting:
    ax = ax_im
    ax.grid(True)
    #Label each of the cells with the row and the column:
    if channel_names is not None:
        for i in range(0, m.shape[0]):
            if i < (m.shape[0] - 1):
                ax.text(i - 0.3, i, channel_names[i], rotation=x_tick_rot)
            if i > 0:
                ax.text(-1, i + 0.3, channel_names[i],
                        horizontalalignment='right')

        ax.set_axis_off()
        ax.set_xticks(np.arange(N))
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(channel_formatter))
        fig.autofmt_xdate(rotation=x_tick_rot)
        ax.set_yticks(np.arange(N))
        ax.set_yticklabels(channel_names)
        ax.set_ybound([-0.5, N - 0.5])
        ax.set_xbound([-0.5, N - 1.5])

    #Make the tick-marks invisible:
    for line in ax.xaxis.get_ticklines():
        line.set_markeredgewidth(0)

    for line in ax.yaxis.get_ticklines():
        line.set_markeredgewidth(0)

    ax.set_axis_off()

    if title is not None:
        ax.set_title(title)

    #The following produces the colorbar and sets the ticks
    if colorbar:
        #Set the ticks - if 0 is in the interval of values, set that, as well
        #as the maximal and minimal values:
        if min_val < 0:
            ticks = [color_min, min_val, 0, max_val, color_max]
        #Otherwise - only set the minimal and maximal value:
        else:
            ticks = [color_min, min_val, max_val, color_max]

        #This makes the colorbar:
        cb = fig.colorbar(im, cax=ax_cb, orientation='horizontal',
                          cmap=cmap,
                          norm=im.norm,
                          boundaries=np.linspace(color_min, color_max, 256),
                          ticks=ticks,
                          format='%.2f')

    # Set the current figure active axis to be the top-one, which is the one
    # most likely to be operated on by users later on
    fig.sca(ax)

    return fig


def draw_matrix(mat, th1=None, th2=None, clim=None, cmap=None):
    """Draw a matrix, optionally thresholding it.
    """
    if th1 is not None:
        m2 = thresholded_arr(mat, th1, th2)
    else:
        m2 = mat
    ax = pl.matshow(m2, cmap=cmap)
    if clim is not None:
        ax.set_clim(*clim)
    pl.colorbar()
    return ax


def plot_intersection_matrix(mylabels):
    '''
    Plots matrix showing intersections/ overlaps between labels
    in the same hemisphere, all the labels are unique
    this means that no labels reduction is possible.
    '''
    import matplotlib.pyplot as pl
    import itertools

    length = len(mylabels)
    intersection_matrix = np.zeros((length, length))
    for i, j in itertools.product(range(length), range(length)):
        if mylabels[i].hemi == mylabels[j].hemi:
            intersection_matrix[i][j] = np.intersect1d(mylabels[i].vertices,
                                                       mylabels[j].vertices).size
        else:
            intersection_matrix[i][j] = 0
    pl.spy(intersection_matrix)
    pl.show()
    return intersection_matrix


def plot_matrix_with_values(mat, cmap='seismic', colorbar=True):
    '''
    Show a matrix with text inside showing the values of the matrix
    may be useful for showing connectivity maps.
    '''
    import matplotlib.pyplot as pl
    fig, ax = pl.subplots()
    im = ax.matshow(mat, cmap=cmap)
    if colorbar:
        pl.colorbar(im)
    for (a, b), z in np.ndenumerate(mat):
        ax.text(b, a, z, ha='center', va='center')
    pl.show()
