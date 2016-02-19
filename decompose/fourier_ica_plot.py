# Authors: Lukas Breuer <l.breuer@fz-juelich.de>

"""
----------------------------------------------------------------------
--- jumeg.decompose.fourier_ica_plot ---------------------------------
----------------------------------------------------------------------
 autor      : Lukas Breuer
 email      : l.breuer@fz-juelich.de
 last update: 27.11.2015
 version    : 1.1

----------------------------------------------------------------------
 This is a simple implementation to plot the results achieved by
 applying FourierICA
----------------------------------------------------------------------
"""

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++
#              import necessary modules
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++
import numpy as np


#######################################################
#                                                     #
#          plotting functions for FourierICA          #
#                                                     #
#######################################################
def adjust_spines(ax, spines, labelsize=10):

    """
    Simple function to adjust axis in plots
    """

    for loc, spine in ax.spines.items():
        if loc in spines:
            spine.set_position(('outward', 4))  # outward by 4 points
            spine.set_smart_bounds(True)
        else:
            spine.set_color('none')  # don't draw spine

    # turn off ticks where there is no spine
    if 'left' in spines:
        ax.yaxis.set_ticks_position('left')
    else:
        # no yaxis ticks
        ax.yaxis.set_ticks([])

    if 'bottom' in spines:
        ax.xaxis.set_ticks_position('bottom')
    else:
        # no xaxis ticks
        ax.xaxis.set_ticks([])

    ax.tick_params(axis='x', labelsize=labelsize)
    ax.tick_params(axis='y', labelsize=labelsize)



# +++++++++++++++++++++++++++++++++++++++++++++++++++++++
# plot averaged independent components
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++
def plot_avg_ics_time_domain(fourier_ica_obj, meg_data, W_orig,
                             fnout=None, show=True):

    """
    Plot averaged FourierICA components in the time-domain

        Parameters
        ----------
        fourier_ica_obj: FourierICA object
        meg_data: array of data to be decomposed [nchan, ntsl].
        W_orig: estimated de-mixing matrix
        fnout: output name for the result image. If not set, the
            image won't be saved. Note, the ending '.png' is
            automatically added
            default: fnout=None
        show: if set plotting results are shown
            default: show=True
    """

    # ------------------------------------------
    # import necessary modules
    # ------------------------------------------
    from matplotlib import pyplot as plt
    from matplotlib import gridspec as grd


    # ------------------------------------------
    # generate sources for plotting
    # ------------------------------------------
    temporal_envelope, pk_values = fourier_ica_obj.get_temporal_envelope(meg_data, W_orig)


    # ------------------------------------------
    # collect some general information
    # ------------------------------------------
    tpost = fourier_ica_obj.tpre + fourier_ica_obj.win_length_sec
    ncomp, ntsl = temporal_envelope.shape

    # define axis/positions for plots
    xaxis_time = np.arange(ntsl)/fourier_ica_obj.sfreq + fourier_ica_obj.tpre
    ylim_act = [np.min(temporal_envelope), np.max(temporal_envelope)]

    # ------------------------------------------
    # loop over all activations
    # ------------------------------------------
    plt.ioff()
    plt.figure('activations', figsize=(5, 14))
    nplot = np.min([10, ncomp])

    gs = grd.GridSpec(nplot, 1)
    # loop over all components
    for icomp in range(nplot):

        if icomp == nplot-1:
            spines = ['bottom']
        else:
            spines = []

        # ----------------------------------------------
        # plot activations in time domain
        # ----------------------------------------------
        p1 = plt.subplot(gs[icomp, 0])
        plt.xlim(fourier_ica_obj.tpre, tpost)
        plt.ylim(ylim_act)
        adjust_spines(p1, spines, labelsize=13)
        if icomp == nplot-1:
            plt.xlabel('time [s]')
        elif icomp == 0:
            p1.set_title("activations [time domain]")
        p1.plot(xaxis_time, temporal_envelope[icomp, :])

        # add some information
        if pk_values.any():
            info = 'pk-value: %0.2f' % pk_values[icomp]
            p1.text(0.97*fourier_ica_obj.tpre+0.03*tpost, 0.8*ylim_act[1] + 0.1*ylim_act[0],
                    info, color='r')

        IC_number = 'IC#%d' % (icomp+1)
        p1.text(1.1*fourier_ica_obj.tpre-0.1*tpost, 0.4*ylim_act[1] + 0.6*ylim_act[0],
                IC_number, color='black', rotation=90)

    # save image
    if fnout:
        plt.savefig(fnout + '.png', format='png')

    # show image if requested
    if show:
        plt.show()

    plt.close('activations')
    plt.ion()



# +++++++++++++++++++++++++++++++++++++++++++++++++++++++
# plot independent components separately back-transformed
# to MEG space
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++
def plot_avg_ics_meg_space(fourier_ica_obj, meg_data, W_orig,
                           A_orig, fnout=None, show=True):

    """
    Plot averaged FourierICA components in the sensor-space.

        Parameters
        ----------
        fourier_ica_obj: FourierICA object
        meg_data: array of data to be decomposed [nchan, ntsl].
        W_orig: estimated de-mixing matrix
        A_orig: estimated mixing matrix
        fnout: output name for the result image. If not set, the
            image won't be saved. Note, the ending '.png' is
            automatically added
            default: fnout=None
        show: if set plotting results are shown
            default: show=True
    """

    # ------------------------------------------
    # import necessary modules
    # ------------------------------------------
    from matplotlib import pyplot as plt
    from matplotlib import gridspec as grd


    # ------------------------------------------
    # generate sources for plotting
    # ------------------------------------------
    rec_signal_avg, orig_avg = fourier_ica_obj.get_reconstructed_signal(meg_data, W_orig, A_orig)


    # ------------------------------------------
    # collect some general information
    # ------------------------------------------
    tpost = fourier_ica_obj.tpre + fourier_ica_obj.win_length_sec
    ncomp, nchan, ntsl = rec_signal_avg.shape

    # define axis/positions for plots
    xaxis_time = np.arange(ntsl)/fourier_ica_obj.sfreq + fourier_ica_obj.tpre
    ylim_meg = [np.min(orig_avg), np.max(orig_avg)]


    # ------------------------------------------
    # loop over all activations
    # ------------------------------------------
    plt.ioff()
    plt.figure('averaged MEG signals', figsize=(5, 14))
    nplot = np.min([10, ncomp])

    gs = grd.GridSpec(nplot, 1)
    for icomp in range(nplot):

        if icomp == nplot-1:
            spines = ['bottom']
        else:
            spines = []

        # ----------------------------------------------
        # plot back-transformed signals
        # ----------------------------------------------
        p1 = plt.subplot(gs[icomp, 0])
        plt.xlim(fourier_ica_obj.tpre, tpost)
        plt.ylim(ylim_meg)
        adjust_spines(p1, spines, labelsize=13)
        if icomp == nplot-1:
            plt.xlabel('time [s]')
        elif icomp == 0:
            p1.set_title("reconstructed MEG-signals")
        p1.plot(xaxis_time, orig_avg.T, 'b', linewidth=0.5)
        p1.plot(xaxis_time, rec_signal_avg[icomp, :, :].T, 'r', linewidth=0.5)

        # add some information
        IC_number = 'IC#%d' % (icomp+1)
        p1.text(1.1*fourier_ica_obj.tpre-0.1*tpost, 0.4*ylim_meg[1] + 0.6*ylim_meg[0],
                IC_number, color='black', rotation=90)

    # save image
    if fnout:
        plt.savefig(fnout + '.png', format='png')

    # show image if requested
    if show:
        plt.show()

    plt.close('averaged MEG signals')
    plt.ion()



# +++++++++++++++++++++++++++++++++++++++++++++++++++++++
# plot fourier amplitude
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++
def plot_fourier_ampl(fourier_ica_obj, meg_data, W_orig,
                      fnout=None, show=True):


    """
    Plot Fourier amplitude of each component
    estimated using FourierICA

        Parameters
        ----------
        fourier_ica_obj: FourierICA object
        meg_data: array of data to be decomposed [nchan, ntsl].
        W_orig: estimated de-mixing matrix
        fnout: output name for the result image. If not set, the
            image won't be saved. Note, the ending '.png' is
            automatically added
            default: fnout=None
        show: if set plotting results are shown
            default: show=True
    """


    # ------------------------------------------
    # import necessary modules
    # ------------------------------------------
    from matplotlib import pyplot as plt
    from matplotlib import gridspec as grd


    # ------------------------------------------
    # generate sources for plotting
    # ------------------------------------------
    fourier_ampl = fourier_ica_obj.get_fourier_ampl(meg_data, W_orig)


    # ------------------------------------------
    # collect some general information
    # ------------------------------------------
    ncomp = fourier_ampl.shape[0]
    nbins = fourier_ampl.shape[1]
    sfreq_bins = nbins/(fourier_ica_obj.fhigh - fourier_ica_obj.flow)

    # define axis/positions for plots
    xaxis_fourier = np.arange(nbins)/sfreq_bins + fourier_ica_obj.flow


    # ------------------------------------------
    # loop over all activations
    # ------------------------------------------
    plt.ioff()
    plt.figure('Fourier amplitude', figsize=(5, 14))
    nplot = np.min([10, ncomp])

    gs = grd.GridSpec(nplot, 1)
    for icomp in range(nplot):

        if icomp == nplot-1:
            spines = ['bottom']
        else:
            spines = []

        # ----------------------------------------------
        # plot Fourier amplitudes
        # ----------------------------------------------
        p1 = plt.subplot(gs[icomp, 0])
        plt.xlim(fourier_ica_obj.flow, fourier_ica_obj.fhigh)
        plt.ylim(0.0, 1.0)
        adjust_spines(p1, spines, labelsize=13)
        if icomp == nplot-1:
            plt.xlabel('freq [Hz]')
        elif icomp == 0:
            p1.set_title("Fourier amplitude (arbitrary units)")

        p1.bar(xaxis_fourier, fourier_ampl[icomp, :], 0.8, color='b', )

        # add some information
        IC_number = 'IC#%d' % (icomp+1)
        p1.text(fourier_ica_obj.flow-5, 0.4, IC_number, color='black', rotation=90)

    # save image
    if fnout:
        plt.savefig(fnout + '.png', format='png')

    # show image if requested
    if show:
        plt.show()

    plt.close('Fourier amplitude')
    plt.ion()



# +++++++++++++++++++++++++++++++++++++++++++++++++++++++
# generate topoplot for each IC
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++
def ICs_topoplot(A_orig, info, picks, fnout=None, show=True):

    """
    Generate topoplots from the demixing matrix recieved
    by applying FourierICA.

        Parameters
        ----------
        fourier_ica_obj: FourierICA object
        info: instance of mne.io.meas_info.Info
            Measurement info.
        picks: Channel indices to generate topomap coords for.
        fnout: output name for the result image. If not set, the
            image won't be saved. Note, the ending '.png' is
            automatically added
            default: fnout=None
        show: if set plotting results are shown
            default: show=True
    """

    # ------------------------------------------
    # import necessary modules
    # ------------------------------------------
    from matplotlib import pyplot as plt
    from matplotlib import gridspec as grd
    from mne.viz import plot_topomap
    from mne.channels.layout import _find_topomap_coords
    import types

    # ------------------------------------------
    # collect some general information
    # ------------------------------------------
    nchan, ncomp = A_orig.shape

    # define axis/positions for plots
    pos = _find_topomap_coords(info, picks)

    plt.ioff()
    plt.figure('Topoplots', figsize=(5, 14))
    nplot = np.min([10, ncomp])

    if isinstance(A_orig[0, 0], types.ComplexType):
        nplots_per_comp = 2
    else:
        nplots_per_comp = 1

    gs = grd.GridSpec(nplot, nplots_per_comp)

    # ------------------------------------------
    # loop over all activations
    # ------------------------------------------
    for icomp in range(nplot):

        # ----------------------------------------------
        # (topoplots (magnitude / phase difference)
        # ----------------------------------------------
        if isinstance(A_orig[0, icomp], types.ComplexType):
            magnitude = np.abs(A_orig[:, icomp])
            magnitude = (2 * magnitude/np.max(magnitude)) - 1
            p1 = plt.subplot(gs[icomp, 0])
            im, _ = plot_topomap(magnitude, pos, res=200, vmin=-1, vmax=1, contours=0)
            if icomp == 0:
                p1.set_title("Magnitude")
            if icomp == nplot-1:
                cbar = plt.colorbar(im, ticks=[-1, 0, 1], orientation='horizontal', shrink=0.8)
                cbar.ax.set_yticklabels(['-1.0', '0.0', '1.0'])

            phase_diff = (np.angle(A_orig[:, icomp]) + np.pi) / (2 * np.pi)
            p2 = plt.subplot(gs[icomp, 1])
            im, _ = plot_topomap(phase_diff, pos, res=200, vmin=0, vmax=1, contours=0)
            if icomp == 0:
                p2.set_title("Phase differences")
            if icomp == nplot-1:
                cbar = plt.colorbar(im, ticks=[-1, 0, 1], orientation='horizontal', shrink=0.8)
                cbar.ax.set_yticklabels(['0.0', '0.5', '1.0'])

        else:
            p1 = plt.subplot(gs[icomp, 0:2])
            magnitude = A_orig[:, icomp]
            magnitude = (2 * magnitude/np.max(magnitude)) - 1
            plot_topomap(magnitude, pos, res=200, vmin=-1, vmax=1, contours=0)
            if icomp == 0:
                p1.set_title("Magnitude distribution")
            if icomp == nplot-1:
                cbar = plt.colorbar(im, ticks=[-1, 0, 1], orientation='horizontal', shrink=0.8)
                cbar.ax.set_yticklabels(['-1.0', '0.0', '1.0'])

    # save image
    if fnout:
        plt.savefig(fnout + '.png', format='png')

    # show image if requested
    if show:
        plt.show()

    plt.close('Topoplots')
    plt.ion()



# +++++++++++++++++++++++++++++++++++++++++++++++++++++++
# plot results
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++
def plot_results(fourier_ica_obj, meg_data,
                 W_orig, A_orig, info, picks,
                 cluster_quality=[], fnout=None,
                 show=True, plot_all=True,
                 layout=None):

    """
    Generate plot containing all results achieved by
    applying FourierICA, i.e., plot activations in
    time- and source-space, as well as fourier
    amplitudes and topoplots.

        Parameters
        ----------
        fourier_ica_obj: FourierICA object
        meg_data: array of data to be decomposed [nchan, ntsl].
        W_orig: estimated de-mixing matrix
        A_orig: estimated mixing matrix
        info: instance of mne.io.meas_info.Info
            Measurement info.
        picks: Channel indices to generate topomap coords for.
        cluster_quality: if set cluster quality is added as text
            info on the plot. Cluster quality is of interest when
            FourierICA combined with ICASSO was applied.
            default: cluster_quality=[]
        fnout: output name for the result image. If not set, the
            image won't be saved. Note, the ending '.png' is
            automatically added
            default: fnout=None
        show: if set plotting results are shown
            default: show=True
        plot_all: if set results for all components are plotted.
            Otherwise only the first 10 components are plotted.
            default: plot_all=True
    """



    # ------------------------------------------
    # import necessary modules
    # ------------------------------------------
    from matplotlib import pyplot as plt
    from matplotlib import gridspec as grd
    from mne.viz import plot_topomap
    from mne.channels.layout import _find_topomap_coords
    import types

    # ------------------------------------------
    # generate sources for plotting
    # ------------------------------------------
    temporal_envelope, pk_values = fourier_ica_obj.get_temporal_envelope(meg_data, W_orig)
    rec_signal_avg, orig_avg = fourier_ica_obj.get_reconstructed_signal(meg_data, W_orig, A_orig)
    fourier_ampl = fourier_ica_obj.get_fourier_ampl(meg_data, W_orig)

    # ------------------------------------------
    # collect some general information
    # ------------------------------------------
    ntsl = int(np.floor(fourier_ica_obj.sfreq*fourier_ica_obj.win_length_sec))
    tpost = fourier_ica_obj.tpre + fourier_ica_obj.win_length_sec
    nchan, ncomp = A_orig.shape
    nbins = fourier_ampl.shape[1]
    sfreq_bins = (1.0 * nbins)/(fourier_ica_obj.fhigh - fourier_ica_obj.flow)

    # define axis/positions for plots
    xaxis_time = np.arange(ntsl)/(1.0*fourier_ica_obj.sfreq) + fourier_ica_obj.tpre
    xaxis_fourier = np.arange(nbins)/sfreq_bins + fourier_ica_obj.flow

    ylim_act = [np.min(temporal_envelope), np.max(temporal_envelope)]
    ylim_meg = [np.min(orig_avg), np.max(orig_avg)]
    pos = _find_topomap_coords(info, picks, layout=layout)

    # ------------------------------------------
    # loop over all activations
    # ------------------------------------------
    plt.ioff()
    if plot_all:
        nimg = int(np.ceil(ncomp /10.0))
    else:
        nimg = 1

    if isinstance(A_orig[0, 0], types.ComplexType):
        nplots_per_comp = 8
    else:
        nplots_per_comp = 7


    # loop over all images
    for iimg in range(nimg):

        fig = plt.figure('FourierICA plots', figsize=(18, 14))

        # estimate how many plots on current image
        istart_plot = int(10*iimg)
        nplot = np.min([10*(iimg+1), ncomp])
        gs = grd.GridSpec(10, nplots_per_comp)

        for icomp in range(istart_plot, nplot):

            if icomp == nplot-1:
                spines = ['bottom']
            else:
                spines = []

            # ----------------------------------------------
            # (1.) plot activations in time domain
            # ----------------------------------------------
            p1 = plt.subplot(gs[icomp-istart_plot, 0:2])
            plt.xlim(fourier_ica_obj.tpre, tpost)
            plt.ylim(ylim_act)
            adjust_spines(p1, spines, labelsize=13)
            if icomp == nplot-1:
                plt.xlabel('time [s]')
            elif icomp == istart_plot:
                p1.set_title("activations [time domain]")
            p1.plot(xaxis_time, temporal_envelope[icomp, :])

            # add some information
            txt_info = 'cluster qual.: %0.2f; ' % cluster_quality[icomp] if np.any(cluster_quality) else ''

            if pk_values.any():
                txt_info += 'pk: %0.2f' % pk_values[icomp]
                p1.text(0.97*fourier_ica_obj.tpre+0.03*tpost, 0.8*ylim_act[1] + 0.1*ylim_act[0],
                        txt_info, color='r')


            IC_number = 'IC#%d' % (icomp+1)
            p1.text(1.1*fourier_ica_obj.tpre-0.1*tpost, 0.4*ylim_act[1] + 0.6*ylim_act[0],
                    IC_number, color='black', rotation=90)

            # ----------------------------------------------
            # (2.) plot back-transformed signals
            # ----------------------------------------------
            p2 = plt.subplot(gs[icomp-istart_plot, 2:4])
            plt.xlim(fourier_ica_obj.tpre, tpost)
            plt.ylim(ylim_meg)
            adjust_spines(p2, spines, labelsize=13)
            if icomp == nplot-1:
                plt.xlabel('time [s]')
            elif icomp == istart_plot:
                p2.set_title("reconstructed MEG-signals")
            p2.plot(xaxis_time, orig_avg.T, 'b', linewidth=0.5)
            p2.plot(xaxis_time, rec_signal_avg[icomp, :, :].T, 'r', linewidth=0.5)

            # ----------------------------------------------
            # (3.) plot Fourier amplitudes
            # ----------------------------------------------
            p3 = plt.subplot(gs[icomp-istart_plot, 4:6])
            plt.xlim(fourier_ica_obj.flow, fourier_ica_obj.fhigh)
            plt.ylim(0.0, 1.0)
            adjust_spines(p3, spines, labelsize=13)
            if icomp == nplot-1:
                plt.xlabel('freq [Hz]')
            elif icomp == istart_plot:
                p3.set_title("Fourier amplitude (arbitrary units)")

            p3.bar(xaxis_fourier, fourier_ampl[icomp, :], 0.8, color='b')

            # ----------------------------------------------
            # (4.) topoplots (magnitude / phase difference)
            # ----------------------------------------------
            if isinstance(A_orig[0, icomp], types.ComplexType):
                magnitude = np.abs(A_orig[:, icomp])
                magnitude = (2 * magnitude/np.max(magnitude)) - 1
                p4 = plt.subplot(gs[icomp-istart_plot, 6])
                im, _ = plot_topomap(magnitude, pos, res=200, vmin=-1, vmax=1, contours=0)
                if icomp == istart_plot:
                    p4.set_title("Magnitude")
                if icomp == nplot-1:
                    cbar = plt.colorbar(im, ticks=[-1, 0, 1], orientation='horizontal', shrink=0.8,
                                        fraction=0.04, pad=0.04)
                    cbar.ax.set_yticklabels(['-1.0', '0.0', '1.0'])

                phase_diff = (np.angle(A_orig[:, icomp]) + 2.0 * np.pi) % (2.0 * np.pi) / (2 * np.pi)
                p5 = plt.subplot(gs[icomp-istart_plot, 7])
                im, _ = plot_topomap(phase_diff, pos, res=200, vmin=0, vmax=1, contours=0)
                if icomp == istart_plot:
                    p5.set_title("Phase differences")
                if icomp == nplot-1:
                    cbar = plt.colorbar(im, ticks=[-1, 0, 1], orientation='horizontal', shrink=0.9,
                                        fraction=0.04, pad=0.04)
                    cbar.ax.set_yticklabels(['0.0', '0.5', '1.0'])

            else:
                from jumeg.jumeg_math import rescale
                p4 = plt.subplot(gs[icomp-istart_plot, 6])
                magnitude = A_orig[:, icomp]
                magnitude = rescale(magnitude, -1, 1)
                im, _ = plot_topomap(magnitude, pos, res=200, vmin=-1, vmax=1, contours=0)
                if icomp == istart_plot:
                    p4.set_title("Magnitude distribution")
                if icomp == nplot-1:
                    cbar = plt.colorbar(im, ticks=[-1, 0, 1], orientation='horizontal', shrink=0.9,
                                        fraction=0.04, pad=0.04)
                    cbar.ax.set_yticklabels(['-1.0', '0.0', '1.0'])

        # save image
        if fnout:
            if plot_all:
                fnout_complete = '%s%02d.png' % (fnout, iimg+1)
            else:
                fnout_complete = '%s.png' % fnout

            plt.savefig(fnout_complete, format='png')

        # show image if requested
        if show:
            plt.show()

        plt.close('FourierICA plots')

    plt.ion()

    return pk_values



# +++++++++++++++++++++++++++++++++++++++++++++++++++++++
# plot results when Fourier ICA was applied in the
# source space
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++
def plot_results_src_space(fourier_ica_obj, W_orig, A_orig,
                           src_loc_data, vertno,
                           sfreq=None, flow=None, fhigh=None,
                           tpre=None, win_length_sec=None,
                           fnout=None, tICA=False,
                           stim_name=[], n_jobs=4,
                           morph2fsaverage=False,
                           subject='fsaverage',
                           subjects_dir=None,
                           temporal_envelope=[],
                           time_range=[None, None],
                           global_scaling=False,
                           classification={},
                           percentile=97, show=True):

    """
    Generate plot containing all results achieved by
    applying FourierICA in source space, i.e., plot
    spatial and spectral profiles.

        Parameters
        ----------

    """

    # ------------------------------------------
    # import necessary modules
    # ------------------------------------------
    from matplotlib import pyplot as plt
    from matplotlib import gridspec as grd
    from mayavi import mlab
    from mne.baseline import rescale
    from mne.source_estimate import _make_stc
    from mne.time_frequency._stockwell import _induced_power_stockwell
    from os import environ, makedirs, rmdir, remove
    from os.path import exists, join
    from scipy import fftpack, misc
    import types


    # -------------------------------------------
    # check input parameter
    # -------------------------------------------
    if tpre == None:
        tpre = fourier_ica_obj.tpre
    if flow == None:
        flow = fourier_ica_obj.flow
    if not fhigh:
        fhigh = fourier_ica_obj.fhigh
    if not sfreq:
        sfreq = fourier_ica_obj.sfreq
    if not win_length_sec:
        win_length_sec = fourier_ica_obj.win_length_sec


    win_ntsl = int(np.floor(sfreq * win_length_sec))
    startfftind = int(np.floor(flow * win_length_sec))
    ncomp, nvoxel = W_orig.shape
    nfreq, nepochs, nvoxel = src_loc_data.shape

    if time_range == [None, None]:
        time_range = [tpre, tpre + win_length_sec]


    # -------------------------------------------
    # generate spatial profiles
    # (using magnitude and phase)
    # -------------------------------------------
    if not subjects_dir:
        subjects_dir = environ.get('SUBJECTS_DIR')

    if isinstance(A_orig[0, 0], types.ComplexType):
        A_orig_mag = np.abs(A_orig)
    else:
        A_orig_mag = A_orig


    # create temporary directory to save plots
    # of spatial profiles
    temp_plot_dir = join(subjects_dir, subject, 'temp_plots')
    if not exists(temp_plot_dir):
        makedirs(temp_plot_dir)


    # -------------------------------------------
    # check if temporal envelope is already
    # given or should be estimated
    # -------------------------------------------
    if not np.any(temporal_envelope):
        # -------------------------------------------
        # get independent components
        # -------------------------------------------
        act = np.zeros((ncomp, nepochs, nfreq), dtype=np.complex)
        if tICA:
            win_ntsl = nfreq

        temporal_envelope = np.zeros((nepochs, ncomp, win_ntsl))
        fft_act = np.zeros((ncomp, win_ntsl), dtype=np.complex)

        for iepoch in range(nepochs):
            src_loc_zero_mean = (src_loc_data[:, iepoch, :] - np.dot(np.ones((nfreq, 1)), fourier_ica_obj.dmean)) / \
                                np.dot(np.ones((nfreq, 1)), fourier_ica_obj.dstd)

            act[:ncomp, iepoch, :] = np.dot(W_orig, src_loc_zero_mean.transpose())
            act[ncomp:, iepoch, :] = np.dot(W_orig, src_loc_zero_mean.transpose())

            if tICA:
                temporal_envelope[iepoch, :, :] = act[:, iepoch, :].real

            else:
                # -------------------------------------------
                # generate temporal profiles
                # -------------------------------------------
                # apply inverse STFT to get temporal envelope
                fft_act[:, startfftind:(startfftind+nfreq)] = act[:, iepoch, :]
                temporal_envelope[iepoch, :, :] = fftpack.ifft(fft_act, n=win_ntsl, axis=1).real


    # check if classsification was done
    key_borders = []
    if np.any(classification):
        idx_sort = []
        keys = classification.keys()
        for key in classification:
            idx_sort += classification[key]
            key_borders.append(len(classification[key]))

        key_borders = np.insert(key_borders, 0, 1)
        key_borders = np.cumsum(key_borders)[:-1]

    else:
        idx_sort = np.arange(ncomp)


    # average temporal envelope
    if not isinstance(temporal_envelope, list):
        temporal_envelope = [[temporal_envelope]]

    ntemp = len(temporal_envelope)
    temporal_envelope_mean = np.empty((ntemp, 0)).tolist()

    for itemp in range(ntemp):
        temporal_envelope_mean[itemp].append(np.mean(temporal_envelope[itemp][0], axis=0)[:, 5:-5])

    # scale temporal envelope between 0 and 1
    min_val = np.min(temporal_envelope_mean)
    max_val = np.max(temporal_envelope_mean)
    scale_fact = 1.0 / (max_val - min_val)

    for itemp in range(ntemp):
        temporal_envelope_mean[itemp][0] = np.clip(scale_fact * temporal_envelope_mean[itemp][0] - scale_fact * min_val, 0., 1.)

    ylim_temp = [-0.05, 1.05]


    # -------------------------------------------
    # loop over all components to generate
    # spatial profiles
    # Note: This will take a while
    # -------------------------------------------
    for icomp in range(ncomp):

        # generate stc-object from current component
        A_cur = A_orig_mag[:, icomp]

        src_loc = _make_stc(A_cur[:, np.newaxis], vertices=vertno, tmin=0, tstep=1,
                            subject=subject)

        # define current range (Xth percentile)
        fmin = np.percentile(A_cur, percentile)
        fmax = np.max(A_cur)
        fmid = 0.5 * (fmin + fmax)
        clim = {'kind': 'value',
                'lims': [fmin, fmid, fmax]}


        # plot spatial profiles
        brain = src_loc.plot(surface='inflated', hemi='split', subjects_dir=subjects_dir,
                             config_opts={'cortex': 'bone'}, views=['lateral', 'medial'],
                             time_label=' ', colorbar=False, clim=clim)

        # save results
        fn_base = "IC%02d_spatial_profile.png" % (icomp+1)
        fnout_img = join(temp_plot_dir, fn_base)
        brain.save_image(fnout_img)


    # close mlab figure
    mlab.close(all=True)


    # -------------------------------------------
    # loop over all components to generate
    # spectral profiles
    # -------------------------------------------
    average_power_all = np.empty((ntemp, 0)).tolist()
    vmin = np.zeros(ncomp)
    vmax = np.zeros(ncomp)

    for itemp in range(ntemp):
        for icomp in range(ncomp):

            nepochs = temporal_envelope[itemp][0].shape[0]
            times = np.arange(win_ntsl)/sfreq + tpre
            idx_start = np.argmin(np.abs(times - time_range[0]))
            idx_end = np.argmin(np.abs(times - time_range[1]))

            data_stockwell = temporal_envelope[itemp][0][:, icomp, idx_start:idx_end].\
                reshape((nepochs, 1, idx_end-idx_start))


            power_data, _, freqs = _induced_power_stockwell(data_stockwell, sfreq=sfreq, fmin=flow,
                                                            fmax=fhigh, width=1.0, decim=1,
                                                            return_itc=False, n_jobs=4)



            # perform baseline correction
            if time_range[0] < 0:
                power_data = rescale(power_data, times[idx_start:idx_end], (None, 0), 'mean')
                imax = np.argmin(np.abs(times[idx_start:idx_end]))
                power_data /= np.sqrt(np.std(power_data[..., :imax], axis=-1)[..., None])

            average_power = power_data.reshape((power_data.shape[1], power_data.shape[2]))
            average_power_all[itemp].append(average_power)

            # define thresholds
            if time_range[0] < 0:
                # vmax[icomp] = np.max((np.nanmax(average_power), vmax[icomp]))  # np.percentile(average_power_all, 99.9)
                # vmin[icomp] = np.min((np.nanmin(average_power), vmin[icomp]))  # np.percentile(average_power_all, 0.1)
                vmax[icomp] = np.max((np.percentile(average_power, 99), vmax[icomp]))  # np.percentile(average_power_all, 99.9)
                vmin[icomp] = np.min((np.percentile(average_power, 1), vmin[icomp]))  # np.percentile(average_power_all, 0.1)


                if np.abs(vmax[icomp]) > np.abs(vmin[icomp]):
                    vmin[icomp] = - np.abs(vmax[icomp])
                else:
                    vmax[icomp] = np.abs(vmin[icomp])

            else:
                vmin[icomp] = None
                vmax[icomp] = None


    # ------------------------------------------
    # loop over all activations
    # ------------------------------------------
    plt.ioff()
    nimg = 1

    # loop over all images
    for iimg in range(nimg):

        fig = plt.figure('FourierICA plots', figsize=(11 + ntemp*10, 60))
        idx_class = 0

        # estimate how many plots on current image
        istart_plot = int(ncomp*iimg)
        nplot = [ncomp]
        gs = grd.GridSpec(ncomp*20+len(key_borders)*10, (ntemp+1)*10, wspace=0.1, hspace=0.05,
                          left=0.04, right=0.96, bottom=0.04, top=0.96)

        for icomp in range(istart_plot, nplot[iimg]):


            if (icomp + 1) in key_borders:
                p_text = fig.add_subplot(gs[20*(icomp-istart_plot)+idx_class*10:20*(icomp-istart_plot)+8+idx_class*10, 0:10])
                idx_class += 1
                p_text.text(0, 0, keys[idx_class-1], fontsize=25)
                adjust_spines(p_text, [])


            # ----------------------------------------------
            # plot spatial profiles (magnitude)
            # ----------------------------------------------
            # spatial profile
            fn_base = "IC%02d_spatial_profile.png" % (idx_sort[icomp]+1)
            fnin_img = join(temp_plot_dir, fn_base)
            spat_tmp = misc.imread(fnin_img)
            remove(fnin_img)

            # rearrange image
            x_size, y_size, _ = spat_tmp.shape
            x_half, y_half = x_size/2, y_size/2
            x_frame = int(0.15*x_half)
            y_frame = int(0.05*y_half)
            spatial_profile = np.concatenate((spat_tmp[x_frame:(x_half-x_frame), y_frame:(y_half-y_frame), :],
                                              spat_tmp[(x_half+x_frame):-x_frame, y_frame:(y_half-y_frame), :],
                                              spat_tmp[(x_half+x_frame):-x_frame, (y_half+y_frame):-y_frame, :],
                                              spat_tmp[x_frame:(x_half-x_frame), (y_half+y_frame):-y_frame, :]), axis=1)


            p1 = fig.add_subplot(gs[20*(icomp-istart_plot)+idx_class*10:20*(icomp-istart_plot)+15+idx_class*10, 0:10])
            p1.imshow(spatial_profile)
            p1.yaxis.set_ticks([])
            p1.xaxis.set_ticks([])
            y_name = "IC#%02d" % (idx_sort[icomp]+1)
            p1.set_ylabel(y_name)


            # ----------------------------------------------
            # temporal/spectral profile
            # ----------------------------------------------
            for itemp in range(ntemp):

                if icomp == 0 and len(stim_name):
                    p_text = fig.add_subplot(gs[20*(icomp-istart_plot)+(idx_class-1)*10: \
                        20*(icomp-istart_plot)+8+(idx_class-1)*10, (itemp+1)*10+4:(itemp+2)*10-1])
                    p_text.text(0, 0, "  " + stim_name[itemp], fontsize=30)
                    adjust_spines(p_text, [])


                times = (np.arange(win_ntsl)/sfreq + tpre)[5:-5]
                idx_start = np.argmin(np.abs(times - time_range[0]))
                idx_end = np.argmin(np.abs(times - time_range[1]))
                average_power = average_power_all[itemp][idx_sort[icomp]]
                extent = (times[idx_start], times[idx_end], freqs[0], freqs[-1])
                p2 = plt.subplot(gs[20*(icomp-istart_plot)+idx_class*10:20*(icomp-istart_plot)+15+idx_class*10,
                                 (itemp+1)*10+1:(itemp+2)*10-1])

                if global_scaling:
                    vmin_cur, vmax_cur = np.min(vmin), np.max(vmax)
                else:
                    vmin_cur, vmax_cur = vmin[icomp], vmax[icomp]

                p2.imshow(average_power, extent=extent, aspect="auto", origin="lower",
                          picker=False, cmap='RdBu_r', vmin=vmin_cur, vmax=vmax_cur)    # cmap='RdBu', vmin=vmin, vmax=vmax)
                p2.set_xlabel("time [s]")
                p2.set_ylabel("freq. [Hz]")
                ax = p2.twinx()
                ax.set_xlim(times[idx_start], times[idx_end])
                ax.set_ylim(ylim_temp)
                ax.set_ylabel("ampl. [a.u.]")
                ax.plot(times[idx_start:idx_end], temporal_envelope_mean[itemp][0][idx_sort[icomp], idx_start:idx_end],
                        color='black', linewidth=3.0)


        # save image
        if fnout:
            fnout_complete = '%s%02d.png' % (fnout, iimg+1)
            plt.savefig(fnout_complete, format='png', dpi=300)

        # show image if requested
        if show:
            plt.show()

        plt.close('FourierICA plots')

    # remove temporary directory for
    # spatial profile plots
    if exists(temp_plot_dir):
        rmdir(temp_plot_dir)

    plt.ion()


