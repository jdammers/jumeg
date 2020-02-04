from os import path as op
import numpy as np
from matplotlib import pyplot as plt

import time as time2
from time import strftime, localtime

from nibabel.affines import apply_affine
from nilearn import plotting
from nilearn.image import index_img

from nilearn.plotting.img_plotting import _MNI152Template
MNI152TEMPLATE = _MNI152Template()


def plot_vstc(vstc, vsrc, tstep, subjects_dir, time_sample=None, coords=None,
              figure=None, axes=None, cmap='magma', symmetric_cbar=False,
              threshold='min', save=False, fname_save=None):
    """ Plot a volume source space estimation.

    Parameters
    ----------
    vstc : VolSourceEstimate
        The volume source estimate.
    vsrc : instance of SourceSpaces
        The source space of the subject equivalent to the
        subject.
    tstep : scalar
        Time step between successive samples in data.
    subjects_dir : str
        The path to the subjects directory.
    time_sample : int, float | None
        None is default for finding the time sample with the voxel with global
        maximal amplitude. If int, float the given time sample is selected and
        plotted.
    coords : arr | None
        None is default for finding the coordinates with the maximal amplitude
        for the given or automatically found time sample
    figure : integer | matplotlib.figure | None
        Specify the figure container to plot in or its number. If None is
        given, a new figure is created.
    axes : matplotlib.figure.axes | None
        Specify the axes of the given figure to plot in. Only necessary if
        a figure is passed.
    threshold : a number, None, 'auto', or 'min'
        If None is given, the image is not thresholded.
        If a number is given, it is used to threshold the image:
        values below the threshold (in absolute value) are plotted
        as transparent. If auto is given, the threshold is determined
        magically by analysis of the image.
    cmap : matplotlib colormap, optional
        The colormap for specified image. The ccolormap *must* be
        symmetrical.
    symmetric_cbar : boolean or 'auto', optional, default 'auto'
        Specifies whether the colorbar should range from -vmax to vmax
        or from vmin to vmax. Setting to 'auto' will select the latter if
        the range of the whole image is either positive or negative.
        Note: The colormap will always be set to range from -vmax to vmax.
    save : bool | None
        Default is False. If True the plot is forced to close and written to disk
        at fname_save location
    fname_save : string
        The path where to save the plot.

    Returns
    -------
    Figure : matplotlib.figure
          VolSourceEstimation plotted for given or 'auto' coordinates at given
          or 'auto' timepoint.
    """
    vstcdata = vstc.data
    img = vstc.as_volume(vsrc, dest='mri', mri_resolution=False)
    subject = vsrc[0]['subject_his_id']
    if vstc == 0:
        if tstep is not None:
            img = _make_image(vstc, vsrc, tstep, dest='mri', mri_resolution=False)
        else:
            print('    Please provide the tstep value !')
    img_data = img.get_data()
    aff = img.affine
    if time_sample is None:
        # global maximum amp in time
        t = int(np.where(np.sum(vstcdata, axis=0) == np.max(np.sum(vstcdata, axis=0)))[0])
    else:
        print('    Time slice', time_sample)
        t = time_sample
    t_in_ms = vstc.times[t] * 1e3
    print('    Found time slice: ', t_in_ms, 'ms')
    if coords is None:
        cut_coords = np.where(img_data == img_data[:, :, :, t].max())
        max_try = np.concatenate((np.array([cut_coords[0][0]]),
                                  np.array([cut_coords[1][0]]),
                                  np.array([cut_coords[2][0]])))
        cut_coords = apply_affine(aff, max_try)
    else:
        cut_coords = coords
    slice_x, slice_y = int(cut_coords[0]), int(cut_coords[1])
    slice_z = int(cut_coords[2])
    print(('    Coords [mri-space]:'
           + 'X: ', slice_x, 'Y: ', slice_y, 'Z: ', slice_z))
    temp_t1_fname = op.join(subjects_dir, subject, 'mri', 'T1.mgz')

    if threshold == 'min':
        threshold = vstcdata.min()

    vstc_plt = plotting.plot_stat_map(index_img(img, t), temp_t1_fname,
                                      figure=figure, axes=axes,
                                      display_mode='ortho',
                                      threshold=threshold,
                                      annotate=True,
                                      title='%s | t=%.2f ms'
                                            % (subject, t_in_ms),
                                      cut_coords=(slice_x, slice_y, slice_z),
                                      cmap=cmap, symmetric_cbar=symmetric_cbar)
    if save:
        if fname_save is None:
            print('please provide an filepath to save .png')
        else:
            plt.savefig(fname_save)
            plt.close()

    return vstc_plt


def plot_vstc_sliced_grid(subjects_dir, vstc, vsrc, title, cut_coords,
                          time=None, display_mode='x', cmap='magma',
                          threshold='min', cbar_range=None, grid=None,
                          res_save=None, fn_image='plot.png',
                          overwrite=False):
    """

    Parameters:
    -----------
    subjects_dir : str
        The path to the subjects directory.
    vstc : VolSourceEstimate
        The volume source estimate.
    vsrc : instance of SourceSpaces
        The source space of the subject equivalent to the
        subject.
    title : str
        Title for the plot.
    cut_coords : list
        The MNI coordinates of the points where the cuts are performed
        For display_mode == 'x', 'y', or 'z', then these are the
        coordinates of each cut in the corresponding direction.
        len(cut_coords) has to match grid[0]*grid[1].
    time : float
        Time point for which the image will be created.
    display_mode : 'x', 'y', 'z'
        Direction in which the brain is sliced.
    cmap : str
        Name of the matplotlib color map to use.
        See https://matplotlib.org/examples/color/colormaps_reference.html
    threshold : a number, None, 'auto', or 'min'
        If None is given, the image is not thresholded.
        If a number is given, it is used to threshold the image:
        values below the threshold (in absolute value) are plotted
        as transparent. If auto is given, the threshold is determined
        magically by analysis of the image.
    cbar_range : None, 2-tuple
        Color range of the plot.
    grid : None | 2-tuple
        Specifies how many images per row and column are to be depicted.
        If grid is None it defaults to [4, 6]
    res_save : None | 2-tuple
        Resolution of the saved image in pixel.
        If res_save is None it defaults to [1920, 1080]
    fn_image : str
        File name for the saved image.
    overwrite : bool
        Overwrite an existing image.

    Returns:
    --------
    None
    """
    if grid is None:
        grid = [4, 6]

    if res_save is None:
        res_save = [1920, 1080]

    if display_mode not in {'x', 'y', 'z'}:
        raise ValueError("display_mode must be one of 'x', 'y', or 'z'.")

    if len(cut_coords) != grid[0]*grid[1]:
        raise ValueError("len(cut_coords) has to match the size of the grid (length must be grid[0]*grid[1]=%d)"
                         % grid[0] * grid[1])

    if not op.exists(fn_image) or overwrite:

        start_time = time2.time()

        print(strftime('Start at %H:%M:%S on the %d.%m.%Y \n', localtime()))

        figure, axes = plt.subplots(grid[0], grid[1])

        axes = axes.flatten()

        params_plot_img_with_bg = get_params_for_grid_slice(vstc, vsrc, vstc.tstep, subjects_dir,
                                                            cbar_range=cbar_range)

        for i, (ax, z) in enumerate(zip(axes, cut_coords)):

            # to get a single slice in plot_vstc_grid_sliced this has to be a list of a single float
            cut_coords_slice = [z]

            colorbar = False
            if grid[1] - 1 == i:
                colorbar = True

            vstc_plot = plot_vstc_grid_slice(vstc=vstc, params_plot_img_with_bg=params_plot_img_with_bg, time=time,
                                             cut_coords=cut_coords_slice, display_mode=display_mode,
                                             figure=figure, axes=ax, colorbar=colorbar, cmap=cmap,
                                             threshold=threshold)

        plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95,
                            wspace=0, hspace=0)

        if title is not None:
            plt.suptitle(title)

        DPI = figure.get_dpi()
        figure.set_size_inches(res_save[0] / float(DPI), res_save[1] / float(DPI))
        # bbox_inches='tight' not useful for images for videos, see:
        # https://github.com/matplotlib/matplotlib/issues/8543#issuecomment-400679840

        frmt = fn_image.split('.')[-1]
        print(DPI, figure.get_size_inches())
        plt.savefig(fn_image, format=frmt, dpi=DPI)

        plt.close()

        end_time = time2.time()

        print(strftime('End at %H:%M:%S on the %d.%m.%Y \n', localtime()))
        minutes = (end_time - start_time) / 60
        seconds = (end_time - start_time) % 60
        print("Calculation took %d minutes and %d seconds" % (minutes, seconds))
        print("")
    else:
        print("File %s exists." % fn_image)


def get_params_for_grid_slice(vstc, vsrc, tstep, subjects_dir, cbar_range=None, **kwargs):
    """
    Makes calculations that would be executed repeatedly every time a slice is
    computed and saves the results in a dictionary which is then read by
    plot_vstc_grid_slice().

    Parameters:
    -----------
    vstc : mne.VolSourceEstimate
        The volume source estimate.
    vsrc : mne.SourceSpaces
        The source space of the subject equivalent to the
    tstep : int
        Time step between successive samples in data.
    subjects_dir:
        Path to the subject directory.
    cbar_range : None, 2-tuple
        Color range of the plot.

    Returns:
    --------
    params_plot_img_with_bg : dict
        Dictionary containing the parameters for plotting.
    """

    img = vstc.as_volume(vsrc, dest='mri', mri_resolution=False)

    # TODO: why should vstc ever be 0?
    if vstc == 0:
        # TODO: how would _make_image work if vstc is zero anyways?
        if tstep is not None:
            img = _make_image(vstc, vsrc, tstep, dest='mri', mri_resolution=False)
        else:
            print('    Please provide the tstep value !')

    subject = vsrc[0]['subject_his_id']
    temp_t1_fname = op.join(subjects_dir, subject, 'mri', 'T1.mgz')
    bg_img = temp_t1_fname
    dim = 'auto'
    black_bg = 'auto'
    vmax = None
    symmetric_cbar = False

    from nilearn.plotting.img_plotting import _load_anat, _get_colorbar_and_data_ranges
    from nilearn._utils import check_niimg_4d
    from nilearn._utils.niimg_conversions import _safe_get_data

    bg_img, black_bg, bg_vmin, bg_vmax = _load_anat(bg_img, dim=dim, black_bg=black_bg)

    stat_map_img = check_niimg_4d(img, dtype='auto')

    cbar_vmin, cbar_vmax, vmin, vmax = _get_colorbar_and_data_ranges(
        _safe_get_data(stat_map_img, ensure_finite=True), vmax, symmetric_cbar, kwargs)

    if cbar_range is not None:
        cbar_vmin = cbar_range[0]
        cbar_vmax = cbar_range[1]
        vmin = cbar_range[0]
        vmax = cbar_range[1]

    params_plot_img_with_bg = dict()
    params_plot_img_with_bg['bg_img'] = bg_img
    params_plot_img_with_bg['black_bg'] = black_bg
    params_plot_img_with_bg['bg_vmin'] = bg_vmin
    params_plot_img_with_bg['bg_vmax'] = bg_vmax
    params_plot_img_with_bg['stat_map_img'] = stat_map_img
    params_plot_img_with_bg['cbar_vmin'] = cbar_vmin
    params_plot_img_with_bg['cbar_vmax'] = cbar_vmax
    params_plot_img_with_bg['vmin'] = vmin
    params_plot_img_with_bg['vmax'] = vmax

    return params_plot_img_with_bg


def plot_vstc_grid_slice(vstc, params_plot_img_with_bg, time=None, cut_coords=6, display_mode='z',
                         figure=None, axes=None, colorbar=False, cmap='magma', threshold='min',
                         **kwargs):
    """
    Plot a volume source space estimation for one slice in the grid in
    plot_vstc_sliced_grid.

    Parameters:
    -----------
    vstc : VolSourceEstimate
        The volume source estimate.
    time : int, float | None
        None is default for finding the time sample with the voxel with global
        maximal amplitude. If int, float the given time point is selected and
        plotted.
    cut_coords : list of a single float
        The MNI coordinates of the point where the cut is performed
        For display_mode == 'x', 'y', or 'z' this is the
        coordinate of the cut in the corresponding direction.
    display_mode : 'x', 'y', 'z'
        Direction in which the brain is sliced.
    figure : matplotlib.figure | None
        Specify the figure container to plot in. If None, a new
        matplotlib.figure is created
    axes : matplotlib.figure.axes | None
        Specify the axes of the given figure to plot in. Only necessary if
        a figure is passed.
    colorbar : bool
        Show the colorbar.
    cmap : matplotlib colormap, optional
        The colormap for specified image. The colormap *must* be
        symmetrical.
    threshold : a number, None, 'auto', or 'min'
        If None is given, the image is not thresholded.
        If a number is given, it is used to threshold the image:
        values below the threshold (in absolute value) are plotted
        as transparent. If auto is given, the threshold is determined
        magically by analysis of the image.

    Returns:
    --------
    Figure : matplotlib.figure
          VolSourceEstimation plotted for given or 'auto' coordinates at given
          or 'auto' timepoint.
    """

    vstcdata = vstc.data

    if time is None:
        # global maximum amp in time
        t = int(np.where(np.sum(vstcdata, axis=0) == np.max(np.sum(vstcdata, axis=0)))[0])
        t_in_ms = vstc.times[t] * 1e3

    else:
        t = np.argmin(np.fabs(vstc.times - time))
        t_in_ms = vstc.times[t] * 1e3

    print('    Found time slice: ', t_in_ms, 'ms')

    if threshold == 'min':
        threshold = vstcdata.min()

    # jumeg_plot_stat_map starts here
    output_file = None
    title = None
    annotate = True
    draw_cross = True
    resampling_interpolation = 'continuous'

    bg_img = params_plot_img_with_bg['bg_img']
    black_bg = params_plot_img_with_bg['black_bg']
    bg_vmin = params_plot_img_with_bg['bg_vmin']
    bg_vmax = params_plot_img_with_bg['bg_vmax']
    stat_map_img = params_plot_img_with_bg['stat_map_img']
    cbar_vmin = params_plot_img_with_bg['cbar_vmin']
    cbar_vmax = params_plot_img_with_bg['cbar_vmax']
    vmin = params_plot_img_with_bg['vmin']
    vmax = params_plot_img_with_bg['vmax']

    # noqa: E501
    # dim the background
    from nilearn.plotting.img_plotting import _plot_img_with_bg
    from nilearn._utils import check_niimg_3d

    stat_map_img_at_time_t = index_img(stat_map_img, t)
    stat_map_img_at_time_t = check_niimg_3d(stat_map_img_at_time_t, dtype='auto')

    display = _plot_img_with_bg(
        img=stat_map_img_at_time_t, bg_img=bg_img, cut_coords=cut_coords,
        output_file=output_file, display_mode=display_mode,
        figure=figure, axes=axes, title=title, annotate=annotate,
        draw_cross=draw_cross, black_bg=black_bg, threshold=threshold,
        bg_vmin=bg_vmin, bg_vmax=bg_vmax, cmap=cmap, vmin=vmin, vmax=vmax,
        colorbar=colorbar, cbar_vmin=cbar_vmin, cbar_vmax=cbar_vmax,
        resampling_interpolation=resampling_interpolation, **kwargs)

    vstc_plt = display

    return vstc_plt


def plot_vstc_sliced_old(vstc, vsrc, tstep, subjects_dir, time=None, title=None, cut_coords=6,
                         display_mode='z', figure=None, axes=None, colorbar=False, cmap='magma',
                         symmetric_cbar=False, threshold='min', cbar_range=None,
                         save=False, fname_save=None, verbose=False):
    """
    Plot a volume source space estimation.

    Parameters
    ----------
    vstc : VolSourceEstimate
        The volume source estimate.
    vsrc : instance of SourceSpaces
        The source space of the subject equivalent to the
        subject.
    tstep : scalar
        Time step between successive samples in data.
    subjects_dir : str
        The path to the subjects directory.
    time : int, float | None
        None is default for finding the time sample with the voxel with global
        maximal amplitude. If int, float the given time point is selected and
        plotted.
    title : string, optional
        The title displayed on the figure.
    display_mode : 'x', 'y', 'z'
        Direction in which the brain is sliced.
    cut_coords : None, a tuple of floats, or an integer
        The MNI coordinates of the point where the cut is performed
        If display_mode is 'ortho', this should be a 3-tuple: (x, y, z)
        For display_mode == 'x', 'y', or 'z', then these are the
        coordinates of each cut in the corresponding direction.
        If None is given, the cuts is calculated automaticaly.
        If display_mode is 'x', 'y' or 'z', cut_coords can be an integer,
        in which case it specifies the number of cuts to perform
    figure : matplotlib.figure | None
        Specify the figure container to plot in. If None, a new
        matplotlib.figure is created
    axes : matplotlib.figure.axes | None
        Specify the axes of the given figure to plot in. Only necessary if
        a figure is passed.
    threshold : a number, None, 'auto', or 'min'
        If None is given, the image is not thresholded.
        If a number is given, it is used to threshold the image:
        values below the threshold (in absolute value) are plotted
        as transparent. If auto is given, the threshold is determined
        magically by analysis of the image.
    colorbar : bool
        Show the colorbar.
    cmap : matplotlib colormap, optional
        The colormap for specified image. The colormap *must* be
        symmetrical.
    symmetric_cbar : boolean or 'auto', optional, default 'auto'
        Specifies whether the colorbar should range from -vmax to vmax
        or from vmin to vmax. Setting to 'auto' will select the latter if
        the range of the whole image is either positive or negative.
        Note: The colormap will always be set to range from -vmax to vmax.
    cbar_range : None, 2-tuple
        Color range of the plot.
    save : bool | None
        Default is False. If True the plot is forced to close and written to disk
        at fname_save location
    fname_save : string
        The path where to save the plot.
    verbose : bool
        Print additional information.

    Returns
    -------
    Figure : matplotlib.figure
          VolSourceEstimation plotted for given or 'auto' coordinates at given
          or 'auto' timepoint.
    """
    vstcdata = vstc.data
    img = vstc.as_volume(vsrc, dest='mri', mri_resolution=False)
    subject = vsrc[0]['subject_his_id']
    if vstc == 0:
        if tstep is not None:
            img = _make_image(vstc, vsrc, tstep, dest='mri', mri_resolution=False)
        else:
            print('    Please provide the tstep value !')

    if time is None:
        # global maximum amp in time
        t = int(np.where(np.sum(vstcdata, axis=0) == np.max(np.sum(vstcdata, axis=0)))[0])
        t_in_ms = vstc.times[t] * 1e3

    else:
        t = np.argmin(np.fabs(vstc.times - time))
        t_in_ms = vstc.times[t] * 1e3

    if verbose:
        print('    Found time slice: ', t_in_ms, 'ms')

    # img_data = img.get_data()
    # aff = img.affine
    # if cut_coords is None:
    #     cut_coords = np.where(img_data == img_data[:, :, :, t].max())
    #     max_try = np.concatenate((np.array([cut_coords[0][0]]),
    #                               np.array([cut_coords[1][0]]),
    #                               np.array([cut_coords[2][0]])))
    #     cut_coords = apply_affine(aff, max_try)

    temp_t1_fname = op.join(subjects_dir, subject, 'mri', 'T1.mgz')

    if threshold == 'min':
        threshold = vstcdata.min()

    vstc_plt = jumeg_plot_stat_map(stat_map_img=img, t=t,
                                   bg_img=temp_t1_fname,
                                   figure=figure, axes=axes,
                                   display_mode=display_mode,
                                   threshold=threshold,
                                   annotate=True, title=title,
                                   cut_coords=cut_coords,
                                   cmap=cmap, colorbar=colorbar,
                                   symmetric_cbar=symmetric_cbar,
                                   cbar_range=cbar_range)
    if save:
        if fname_save is None:
            print('please provide an filepath to save .png')
        else:
            plt.savefig(fname_save)
            plt.close()

    return vstc_plt


def plot_vstc_sliced(vstc, vsrc, tstep, subjects_dir, img=None, time=None, cut_coords=6,
                     display_mode='z', figure=None, axes=None, colorbar=False, cmap='magma',
                     symmetric_cbar=False, cbar_range=None, threshold='min', save=False,
                     fname_save=None):
    """ Plot a volume source space estimation.

    Parameters
    ----------
    vstc : VolSourceEstimate
        The volume source estimate.
    vsrc : instance of SourceSpaces
        The source space of the subject equivalent to the
        subject.
    tstep : scalar
        Time step between successive samples in data.
    subjects_dir : str
        The path to the subjects directory.
    img : Nifti1Image | None
        Pre-computed vstc.as_volume(vsrc, dest='mri', mri_resolution=False).
    time : int, float | None
        None is default for finding the time sample with the voxel with global
        maximal amplitude. If int, float the given time point is selected and
        plotted.
    display_mode : 'x', 'y', 'z'
        Direction in which the brain is sliced.
    cut_coords : None, a tuple of floats, or an integer
        The MNI coordinates of the point where the cut is performed
        If display_mode is 'ortho', this should be a 3-tuple: (x, y, z)
        For display_mode == 'x', 'y', or 'z', then these are the
        coordinates of each cut in the corresponding direction.
        If None is given, the cuts is calculated automaticaly.
        If display_mode is 'x', 'y' or 'z', cut_coords can be an integer,
        in which case it specifies the number of cuts to perform
    figure : matplotlib.figure | None
        Specify the figure container to plot in. If None, a new
        matplotlib.figure is created
    axes : matplotlib.figure.axes | None
        Specify the axes of the given figure to plot in. Only necessary if
        a figure is passed.
    threshold : a number, None, 'auto', or 'min'
        If None is given, the image is not thresholded.
        If a number is given, it is used to threshold the image:
        values below the threshold (in absolute value) are plotted
        as transparent. If auto is given, the threshold is determined
        magically by analysis of the image.
    colorbar : bool
        Show the colorbar.
    cmap : matplotlib colormap, optional
        The colormap for specified image. The colormap *must* be
        symmetrical.
    symmetric_cbar : boolean or 'auto', optional, default 'auto'
        Specifies whether the colorbar should range from -vmax to vmax
        or from vmin to vmax. Setting to 'auto' will select the latter if
        the range of the whole image is either positive or negative.
        Note: The colormap will always be set to range from -vmax to vmax.
    cbar_range : None, 2-tuple
        Color range of the plot.
    save : bool | None
        Default is False. If True the plot is forced to close and written to disk
        at fname_save location
    fname_save : string
        The path where to save the plot.

    Returns
    -------
    Figure : matplotlib.figure
          VolSourceEstimation plotted for given or 'auto' coordinates at given
          or 'auto' timepoint.
    """
    vstcdata = vstc.data
    if img is None:
        img = vstc.as_volume(vsrc, dest='mri', mri_resolution=False)
        if vstc == 0:
            if tstep is not None:
                img = _make_image(vstc, vsrc, tstep, dest='mri', mri_resolution=False)
            else:
                print('    Please provide the tstep value !')
    subject = vsrc[0]['subject_his_id']

    if time is None:
        # global maximum amp in time
        t = int(np.where(np.sum(vstcdata, axis=0) == np.max(np.sum(vstcdata, axis=0)))[0])
    else:
        t = np.argmin(np.fabs(vstc.times - time))

    t_in_ms = vstc.times[t] * 1e3
    print('    Found time slice: ', t_in_ms, 'ms')

    # img_data = img.get_data()
    # aff = img.affine
    # if cut_coords is None:
    #     cut_coords = np.where(img_data == img_data[:, :, :, t].max())
    #     max_try = np.concatenate((np.array([cut_coords[0][0]]),
    #                               np.array([cut_coords[1][0]]),
    #                               np.array([cut_coords[2][0]])))
    #     cut_coords = apply_affine(aff, max_try)

    temp_t1_fname = op.join(subjects_dir, subject, 'mri', 'T1.mgz')

    if threshold == 'min':
        threshold = vstcdata.min()

    vstc_plt = jumeg_plot_stat_map(stat_map_img=img, t=t,
                                   bg_img=temp_t1_fname,
                                   figure=figure, axes=axes,
                                   display_mode=display_mode,
                                   threshold=threshold,
                                   annotate=True, title=None,
                                   cut_coords=cut_coords,
                                   cmap=cmap, colorbar=colorbar,
                                   symmetric_cbar=symmetric_cbar,
                                   cbar_range=cbar_range)
    if save:
        if fname_save is None:
            print('please provide an filepath to save .png')
        else:
            plt.savefig(fname_save)
            plt.close()

    return vstc_plt


def jumeg_plot_stat_map(stat_map_img, t, bg_img=MNI152TEMPLATE, cut_coords=None,
                        output_file=None, display_mode='ortho', colorbar=True,
                        figure=None, axes=None, title=None, threshold=1e-6,
                        annotate=True, draw_cross=True, black_bg='auto',
                        cmap='magma', symmetric_cbar="auto", cbar_range=None,
                        dim='auto', vmax=None, resampling_interpolation='continuous',
                        **kwargs):
    """
    Plot cuts of an ROI/mask image (by default 3 cuts: Frontal, Axial, and
    Lateral)

    This is based on nilearn.plotting.plot_stat_map
    Parameters
    ----------
    stat_map_img : Niimg-like object
        See http://nilearn.github.io/manipulating_images/input_output.html
        The statistical map image
    t : int
        Plot activity at time point given by time t.
    bg_img : Niimg-like object
        See http://nilearn.github.io/manipulating_images/input_output.html
        The background image that the ROI/mask will be plotted on top of.
        If nothing is specified, the MNI152 template will be used.
        To turn off background image, just pass "bg_img=False".
    cut_coords : None, a tuple of floats, or an integer
        The MNI coordinates of the point where the cut is performed
        If display_mode is 'ortho', this should be a 3-tuple: (x, y, z)
        For display_mode == 'x', 'y', or 'z', then these are the
        coordinates of each cut in the corresponding direction.
        If None is given, the cuts is calculated automaticaly.
        If display_mode is 'x', 'y' or 'z', cut_coords can be an integer,
        in which case it specifies the number of cuts to perform
    output_file : string, or None, optional
        The name of an image file to export the plot to. Valid extensions
        are .png, .pdf, .svg. If output_file is not None, the plot
        is saved to a file, and the display is closed.
    display_mode : {'ortho', 'x', 'y', 'z', 'yx', 'xz', 'yz'}
        Choose the direction of the cuts: 'x' - sagittal, 'y' - coronal,
        'z' - axial, 'ortho' - three cuts are performed in orthogonal
        directions.
    colorbar : boolean, optional
        If True, display a colorbar on the right of the plots.
    figure : integer or matplotlib figure, optional
        Matplotlib figure used or its number. If None is given, a
        new figure is created.
    axes : matplotlib axes or 4 tuple of float: (xmin, ymin, width, height), optional
        The axes, or the coordinates, in matplotlib figure space,
        of the axes used to display the plot. If None, the complete
        figure is used.
    title : string, optional
        The title displayed on the figure.
    threshold : a number, None, or 'auto'
        If None is given, the image is not thresholded.
        If a number is given, it is used to threshold the image:
        values below the threshold (in absolute value) are plotted
        as transparent. If auto is given, the threshold is determined
        magically by analysis of the image.
    annotate : boolean, optional
        If annotate is True, positions and left/right annotation
        are added to the plot.
    draw_cross : boolean, optional
        If draw_cross is True, a cross is drawn on the plot to
        indicate the cut plosition.
    black_bg : boolean, optional
        If True, the background of the image is set to be black. If
        you wish to save figures with a black background, you
        will need to pass "facecolor='k', edgecolor='k'"
        to matplotlib.pyplot.savefig.
    cmap : matplotlib colormap, optional
        The colormap for specified image. The ccolormap *must* be
        symmetrical.
    symmetric_cbar : boolean or 'auto', optional, default 'auto'
        Specifies whether the colorbar should range from -vmax to vmax
        or from vmin to vmax. Setting to 'auto' will select the latter if
        the range of the whole image is either positive or negative.
        Note: The colormap will always be set to range from -vmax to vmax.
    cbar_range : None, 2-tuple
        Color range of the plot.
    dim : float, 'auto' (by default), optional
        Dimming factor applied to background image. By default, automatic
        heuristics are applied based upon the background image intensity.
        Accepted float values, where a typical scan is between -2 and 2
        (-2 = increase constrast; 2 = decrease contrast), but larger values
        can be used for a more pronounced effect. 0 means no dimming.
    vmax : float
        Upper bound for plotting, passed to matplotlib.pyplot.imshow
    resampling_interpolation : str
        Interpolation to use when resampling the image to the destination
        space. Can be "continuous" (default) to use 3rd-order spline
        interpolation, or "nearest" to use nearest-neighbor mapping.
        "nearest" is faster but can be noisier in some cases.

    Notes
    -----
    Arrays should be passed in numpy convention: (x, y, z)
    ordered.

    For visualization, non-finite values found in passed 'stat_map_img' or
    'bg_img' are set to zero.

    See Also
    --------

    nilearn.plotting.plot_anat : To simply plot anatomical images
    nilearn.plotting.plot_epi : To simply plot raw EPI images
    nilearn.plotting.plot_glass_brain : To plot maps in a glass brain

    """
    # noqa: E501
    # dim the background
    from nilearn.plotting.img_plotting import _load_anat, _plot_img_with_bg, _get_colorbar_and_data_ranges
    from nilearn._utils import check_niimg_3d, check_niimg_4d
    from nilearn._utils.niimg_conversions import _safe_get_data

    bg_img, black_bg, bg_vmin, bg_vmax = _load_anat(bg_img, dim=dim,
                                                    black_bg=black_bg)

    stat_map_img = check_niimg_4d(stat_map_img, dtype='auto')

    cbar_vmin, cbar_vmax, vmin, vmax = _get_colorbar_and_data_ranges(_safe_get_data(stat_map_img, ensure_finite=True),
                                                                     vmax, symmetric_cbar, kwargs)

    if cbar_range is not None:
        cbar_vmin = cbar_range[0]
        cbar_vmax = cbar_range[1]
        vmin = cbar_range[0]
        vmax = cbar_range[1]

    stat_map_img_at_time_t = index_img(stat_map_img, t)
    stat_map_img_at_time_t = check_niimg_3d(stat_map_img_at_time_t, dtype='auto')

    display = _plot_img_with_bg(
        img=stat_map_img_at_time_t, bg_img=bg_img, cut_coords=cut_coords,
        output_file=output_file, display_mode=display_mode,
        figure=figure, axes=axes, title=title, annotate=annotate,
        draw_cross=draw_cross, black_bg=black_bg, threshold=threshold,
        bg_vmin=bg_vmin, bg_vmax=bg_vmax, cmap=cmap, vmin=vmin, vmax=vmax,
        colorbar=colorbar, cbar_vmin=cbar_vmin, cbar_vmax=cbar_vmax,
        resampling_interpolation=resampling_interpolation, **kwargs)

    return display


def _make_image(vstc, vsrc, tstep, label_inds=None, dest='mri',
                mri_resolution=False):
    # TODO: remove tstep if not necessary (use vstc.tstep)
    """
     TODO: improve function description
    Make a volume source estimation in a NIfTI file.

    Parameters
    ----------
    vstc : VolSourceEstimate
        The volume source estimate.
    vsrc : instance of VolSourceSpaces
        The source space of the subject equivalent to the
        subject.
    dest : 'mri' | 'surf'
        If 'mri' the volume is defined in the coordinate system of
        the original T1 image. If 'surf' the coordinate system
        of the FreeSurfer surface is used (Surface RAS).
    mri_resolution: bool
        It True the image is saved in MRI resolution.
        WARNING: if you have many time points the file produced can be
        huge.

    Returns
    -------
    img : instance Nifti1Image
        The image object.
    """
    n_times = vstc.shape[1]
    shape = vsrc[0]['shape']
    shape3d = (shape[2], shape[1], shape[0])
    shape = (n_times, shape[2], shape[1], shape[0])
    vol = np.zeros(shape)

    if label_inds is not None:
        inuse_replace = np.zeros(vsrc[0]['inuse'].shape, dtype=int)
        for i, idx in enumerate(label_inds):
            inuse_replace[idx] = 1
        mask3d = inuse_replace.reshape(shape3d).astype(np.bool)
    else:
        mask3d = vsrc[0]['inuse'].reshape(shape3d).astype(np.bool)

    if mri_resolution:
        mri_shape3d = (vsrc[0]['mri_height'], vsrc[0]['mri_depth'],
                       vsrc[0]['mri_width'])
        mri_shape = (n_times, vsrc[0]['mri_height'], vsrc[0]['mri_depth'],
                     vsrc[0]['mri_width'])
        mri_vol = np.zeros(mri_shape)
        interpolator = vsrc[0]['interpolator']

    for k, v in enumerate(vol):
        v[mask3d] = vstc[:, k]
        if mri_resolution:
            mri_vol[k] = (interpolator * v.ravel()).reshape(mri_shape3d)

    if mri_resolution:
        vol = mri_vol

    vol = vol.T

    if mri_resolution:
        affine = vsrc[0]['vox_mri_t']['trans'].copy()
    else:
        affine = vsrc[0]['src_mri_t']['trans'].copy()
    if dest == 'mri':
        affine = np.dot(vsrc[0]['mri_ras_t']['trans'], affine)
    affine[:3] *= 1e3

    try:
        import nibabel as nib  # lazy import to avoid dependency
    except ImportError:
        raise ImportError("nibabel is required to save volume images.")

    header = nib.nifti1.Nifti1Header()
    header.set_xyzt_units('mm', 'msec')
    header['pixdim'][4] = 1e3 * tstep
    img = nib.Nifti1Image(vol, affine, header=header)

    return img


def plot_VSTCPT(vstc, vsrc, tstep, subjects_dir, time_sample=None, coords=None,
                figure=None, axes=None, title=None, save=False, fname_save=None):
    """ Plotting the cluster permutation test results based on a
        volume source estimation.

    Parameters
    ----------
    fname_stc_orig : String
          Filename
    subject_from : list of Labels
          Filename
    Returns
    -------
    new-data : dictionary of one or more new stc
          The generated source time courses.
    """

    print('\n#### Attempting to plot volume stc from file..')
    print('    Creating 3D image from stc..')
    vstcdata = vstc.data
    img = vstc.as_volume(vsrc, dest='mri', mri_resolution=False)
    subject = vsrc[0]['subject_his_id']
    if vstc == 0:
        if tstep is not None:
            img = _make_image(vstc, vsrc, tstep, dest='mri', mri_resolution=False)
        else:
            print('    Please provide the tstep value !')
    img_data = img.get_data()
    aff = img.affine
    if time_sample is None:
        print('    Searching for maximal Activation..')
        t = int(np.where(np.sum(vstcdata, axis=0) == np.max(np.sum(vstcdata, axis=0)))[0])  # maximum amp in time
    else:
        print('    Using Cluster No.', time_sample)
        t = time_sample
    if title is None:
        title = 'Cluster No. %i' % t
        if t == 0:
            title = 'All Cluster'  # |sig.%i'%vstc.times.shape[0]-1
    if coords is None:
        cut_coords = np.where(img_data == img_data[:, :, :, t].max())
        max_try = np.concatenate((
            np.array([cut_coords[0][0]]),
            np.array([cut_coords[1][0]]),
            np.array([cut_coords[2][0]])))
        cut_coords = apply_affine(aff, max_try)
    else:
        cut_coords = coords
    slice_x, slice_y = int(cut_coords[0]), int(cut_coords[1])
    slice_z = int(cut_coords[2])
    print('    Respective Space Coords [mri-space]:')
    print('    X: ', slice_x, '    Y: ', slice_y, '    Z: ', slice_z)
    temp_t1_fname = op.join(subjects_dir, subject, 'mri', 'T1.mgz')
    VSTCPT_plot = plotting.plot_stat_map(index_img(img, t), temp_t1_fname,
                                         figure=figure, axes=axes,
                                         display_mode='ortho',
                                         threshold=vstcdata.min(),
                                         annotate=True,
                                         title=title,
                                         cut_coords=None,
                                         cmap='magma')
    if save:
        if fname_save is None:
            print('please provide an filepath to save .png')
        else:
            plt.savefig(fname_save)
            plt.close()

    return VSTCPT_plot