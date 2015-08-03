# Authors: Lukas Breuer <l.breuer@fz-juelich.de>

"""
----------------------------------------------------------------------
--- jumeg.decompose.fourier_ica_plot ---------------------------------
----------------------------------------------------------------------
 autor      : Lukas Breuer
 email      : l.breuer@fz-juelich.de
 last update: 24.02.2015
 version    : 1.0

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
                 show=True, plot_all=True):

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
    sfreq_bins = nbins/(fourier_ica_obj.fhigh - fourier_ica_obj.flow)

    # define axis/positions for plots
    xaxis_time = np.arange(ntsl)/fourier_ica_obj.sfreq + fourier_ica_obj.tpre
    xaxis_fourier = np.arange(nbins)/sfreq_bins + fourier_ica_obj.flow
    ylim_act = [np.min(temporal_envelope), np.max(temporal_envelope)]
    ylim_meg = [np.min(orig_avg), np.max(orig_avg)]
    pos = _find_topomap_coords(info, picks)

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
            txt_info = 'cluster qual.: %0.2f; ' % cluster_quality[icomp] if cluster_quality.any() else ''

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

                phase_diff = (np.angle(A_orig[:, icomp]) + np.pi) / (2 * np.pi)
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
                fnout_complete = '%s%2d.png' % (fnout, iimg+1)
            else:
                fnout_complete = '%s.png' % fnout

            plt.savefig(fnout_complete, format='png')

        # show image if requested
        if show:
            plt.show()

        plt.close('FourierICA plots')

    plt.ion()

    return pk_values


