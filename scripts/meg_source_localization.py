'''
Created on 18.12.2013

@author: lbreuer
'''
#!/usr/bin/env python
"""Perform source localization for MEG data"""


# =========================================
# import necessary modules
# =========================================
import mne, os, sys, pdb
import numpy as np
import pylab as pl
import matplotlib.pyplot as plt
import mne.minimum_norm as min_norm
import mne.beamformer as beam
from matplotlib.backends.backend_pdf import PdfPages
from mayavi import mlab
from mne.time_frequency import compute_epochs_csd


# *******************************************
# * Function creates the noise-covariance   *
# * matrix from an empty room file          *
# *******************************************
def create_noise_covariance_matrix(fname_empty_room, fname_out=None, verbose=None):
    """Creates the noise covariance matrix from an empty room file"""

    print ">>>> estimate noise covariance matrix from empty room file..."

    # read in data
    raw_empty = mne.fiff.Raw(fname_empty_room,
                             verbose=verbose)
    # filter data

    # pick only MEG channels
    picks = mne.fiff.pick_types(raw_empty.info,
                                meg=True,
                                exclude='bads')

    # calculate noise-covariance matrix
    noise_cov_mat = mne.compute_raw_data_covariance(raw_empty,
                                                    picks=picks,
                                                    verbose=verbose)
    # write noise-covariance matrix to disk
    if fname_out is not None:
        mne.write_cov(fname_out, noise_cov_mat)

    return noise_cov_mat




# *******************************************
# * Function to plot noise-covariance matrix*
# *******************************************
def plot_noise_covariance_matrix(noise_cov_mat, info, fnout_noise_cov=None, show=None):
    """"Function to plot the noise-covariance matrix."""

    fig_cov, fig_svd = mne.viz.plot_cov(noise_cov_mat,
                                        info,
                                        show=show)

    # save image if desired
    if fnout_noise_cov is not None:
        pp = PdfPages(fnout_noise_cov + '.pdf')
        pp.savefig(fig_cov)
        pp.savefig(fig_svd)
        pp.close()




# *******************************************
# * Function to check BEM surfaces          *
# *******************************************
def plot_BEM_surface(subject, fnout_img=None):
    """"Reads in and plots the BEM surface."""

    # get name of BEM-file
    subjects_dir = os.environ.get('SUBJECTS_DIR')
    fn_bem = os.path.join(subjects_dir,
                          subject,
                          'bem',
                          subject + "-5120-5120-5120-bem-sol.fif")

    surfaces = mne.read_bem_surfaces(fn_bem, add_geom=True)

    print "Number of surfaces : %d" % len(surfaces)

    # Show result
    head_col = (0.95, 0.83, 0.83)  # light pink
    skull_col = (0.91, 0.89, 0.67)
    brain_col = (0.67, 0.89, 0.91)  # light blue
    colors = [head_col, skull_col, brain_col]

    # create figure and plot results
    fig_BEM = mlab.figure(size=(1200, 1200),
                          bgcolor=(0, 0, 0))
    for c, surf in zip(colors, surfaces):
        points = surf['rr']
        faces = surf['tris']
        mlab.triangular_mesh(points[:, 0],
                             points[:, 1],
                             points[:, 2],
                             faces,
                             color=c,
                             opacity=0.3)

    # save result
    if fnout_img is not None:
        mlab.savefig(fnout_img,
                     figure=fig_BEM,
                     size=(1200, 1200))

    mlab.close(all=True)




# *******************************************
# * Function to check the forward solution  *
# *******************************************
def plot_forward_operator(subject, fnout_img=None):
    """"Reads in and plots the forward solution."""

    # get name of inverse solution-file
    subjects_dir = os.environ.get('SUBJECTS_DIR')
    fname_dir_fwd = os.path.join(subjects_dir,
                                 subject)

    for files in os.listdir(fname_dir_fwd):
        if files.endswith(",cleaned_epochs_avg-7-src-fwd.fif"):
            fname_fwd = os.path.join(fname_dir_fwd,
                                     files)
            break

    try:
        fname_fwd
    except NameError:
        print "ERROR: No forward solution found!"
        sys.exit()


    # read inverse solution
    add_geom = True # include high resolution source space
    src = mne.read_source_spaces(fname_fwd,
                                 add_geom=add_geom)

    # 3D source space (high sampling)
    lh_points = src[0]['rr']
    lh_faces = src[0]['tris']
    rh_points = src[1]['rr']
    rh_faces = src[1]['tris']

    # create figure and plot results
    fig_forward = mlab.figure(size=(1200, 1200),
                              bgcolor=(0, 0, 0))
    mlab.triangular_mesh(lh_points[:, 0],
                         lh_points[:, 1],
                         lh_points[:, 2],
                         lh_faces)

    mlab.triangular_mesh(rh_points[:, 0],
                         rh_points[:, 1],
                         rh_points[:, 2],
                         rh_faces)

    # save image if desired
    if fnout_img is not None:
        mlab.savefig(fnout_img,
                     figure=fig_forward,
                     size=(1200, 1200))

    mlab.close(all=True)




# *******************************************
# * Function to check the inverse solution  *
# *******************************************
def plot_inverse_operator(subject, fnout_img=None, verbose=None):
    """"Reads in and plots the inverse solution."""

    # get name of inverse solution-file
    subjects_dir = os.environ.get('SUBJECTS_DIR')
    fname_dir_inv = os.path.join(subjects_dir,
                                 subject)

    for files in os.listdir(fname_dir_inv):
        if files.endswith(",cleaned_epochs_avg-7-src-meg-inv.fif"):
            fname_inv = os.path.join(fname_dir_inv,
                                     files)
            break

    try:
        fname_inv
    except NameError:
        print "ERROR: No forward solution found!"
        sys.exit()


    # read inverse solution
    inv = min_norm.read_inverse_operator(fname_inv)

    # print some information if desired
    if verbose is not None:
        print "Method: %s" % inv['methods']
        print "fMRI prior: %s" % inv['fmri_prior']
        print "Number of sources: %s" % inv['nsource']
        print "Number of channels: %s" % inv['nchan']


    # show 3D source space
    lh_points = inv['src'][0]['rr']
    lh_faces = inv['src'][0]['tris']
    rh_points = inv['src'][1]['rr']
    rh_faces = inv['src'][1]['tris']

    # create figure and plot results
    fig_inverse = mlab.figure(size=(1200, 1200),
                              bgcolor=(0, 0, 0))

    mlab.triangular_mesh(lh_points[:, 0],
                         lh_points[:, 1],
                         lh_points[:, 2],
                         lh_faces)

    mlab.triangular_mesh(rh_points[:, 0],
                         rh_points[:, 1],
                         rh_points[:, 2],
                         rh_faces)

    # save result
    if fnout_img is not None:
        mlab.savefig(fnout_img,
                     figure=fig_inverse,
                     size=(1200, 1200))

    mlab.close(all=True)




# *******************************************
# * Function to check the forward solution  *
# *******************************************
def prepare_data_for_source_localization(meg_data,
                                         event_chan,
                                         fnout_prepared_data=None,
                                         fnout_img_prepared_data=None,
                                         event_id=1,
                                         use_baseline=None,      # use baseline while averaging?
                                         tmin=-0.2,              # time border prior event for averaging (in s)
                                         tmax=0.4,               # time border after event for averaging (in s)
                                         texp_resp=0.1,          # time of expected brain response
                                         reject=dict(mag=4e-12), # define border for rejecting epochs
                                         verbose=None):
    """Prepare data for source localization, i.e. average data around stimulus event and save average."""

    # get events
    events = mne.find_events(meg_data,
                             stim_channel=event_chan,
                             verbose=verbose)
    # pick channels
    picks = mne.fiff.pick_types(meg_data.info,
                                meg=True,
                                eeg=False,
                                eog=False,
                                exclude='bads')

    if use_baseline:
        baseline = (None, 0)
    else:
        baseline = (None, None)


    # create epochs
    epochs = mne.Epochs(meg_data,
                        events,
                        event_id,
                        tmin,
                        tmax,
                        proj=True,
                        picks=picks,
                        baseline=baseline,
                        reject=reject,
                        verbose=verbose)

    epochs_avg = epochs.average()

    # and save epochs
    if fnout_prepared_data is not None:
        epochs_avg.save(fnout_prepared_data)


    # -----------------------------------
    # plot the evoked response in sensor
    # space
    # -----------------------------------
    if (fnout_img_prepared_data is not None):
        evoked = epochs.average()
        evoked.plot(show=None)
        fig1 = pl.gcf()
        pl.close()

        times = np.linspace(texp_resp-0.1,
                            texp_resp+0.1, 5)
        evoked.plot_topomap(times=times,
                            ch_type='mag',
                            show=None)
        fig2 = pl.gcf()
        pl.close()

        pp = PdfPages(fnout_img_prepared_data + '.pdf')
        pp.savefig(fig1)
        pp.savefig(fig2)
        pp.close()

    return epochs





# *******************************************
# * Function to estimate forward solution   *
# *******************************************
def estimate_forward_solution(fname_data,
                              subject,
                              fname_fwd=None):
    """"Estimates forward solution for the given data set."""

    # define some parameter
    pattern_trans = "-trans.fif"
    pattern_src = "-7-src.fif"
    pattern_bem = "-5120-5120-5120-bem-sol.fif"

    print ">>>> estimate forward operator..."
    # first we have to search for the necessary files
    subjects_dir = os.environ.get('SUBJECTS_DIR')
    fname_sub_dir = os.path.join(subjects_dir,
                                 subject)

    # (a) the transformation file
    #     --> it should be located in the main directory of the subject
    fname_trans = os.path.join(fname_sub_dir,
                               subject + pattern_trans)

    try:
        fname_trans
    except NameError:
        print "ERROR: Coregistration file not found!"
        sys.exit()


    # (b) mne source space file
    #     --> it should be located in the "bem" folder in
    #         the main directory of the subject
    fname_src = os.path.join(fname_sub_dir,
                             "bem",
                             subject + pattern_src)

    try:
        fname_src
    except NameError:
        print "ERROR: Source space file not found!"
        sys.exit()


    # (c) BEM file
    #     --> it should be located in the "bem" folder in
    #         the main directory of the subject
    fname_bem = os.path.join(fname_sub_dir,
                             "bem",
                             subject + pattern_bem)

    try:
        fname_bem
    except NameError:
        print "ERROR: BEM file not found!"
        sys.exit()

    # now estimate the forward solution
    fwd_sol = mne.make_forward_solution(fname_data,
                                        mri=fname_trans,
                                        src=fname_src,
                                        bem=fname_bem,
                                        fname=fname_fwd,
                                        meg=True,         # include MEG channels
                                        eeg=False,        # exclude EEG channels
                                        mindist=5.0,      # ignore sources <= 5mm from inner skull
                                        n_jobs=2,         # number of jobs to run in parallel
                                        overwrite=True)   # if file exist overwrite it

    # at least convert to surface orientation for
    # cortically constrained inverse modeling
    fwd_sol = mne.convert_forward_solution(fwd_sol, surf_ori=True)

    return fwd_sol





# *******************************************
# * Function to perform source localization *
# *******************************************
def plot_sensitivity_map(fwd_sol,
                         subject,
                         fname_leadfield_plot,
                         fname_sensitvity_plot):
    """Estimates and plots sensitivity map of forward solution."""

    # estimate lead field
    leadfield = fwd_sol['sol']['data']

    pp = PdfPages(fname_leadfield_plot)

    # plot leadfield
    plt.matshow(leadfield[:, :500])
    plt.xlabel('sources')
    plt.ylabel('sensors')
    plt.title('Lead field matrix (500 dipoles only)')
    pp.savefig()
    plt.close()


    # estimate sensitivity map for magnetometer
    mag_map = mne.sensitivity_map(fwd_sol, ch_type='mag', mode='fixed')

    # plot histogram of sensitivity
    plt.hist(mag_map.data.ravel(),
                   bins=20,
                   label='Magnetometers')
    plt.legend()
    plt.title('Normal orientation sensitivity')
    pp.savefig()
    plt.close()
    pp.close()

    subjects_dir = os.environ.get('SUBJECTS_DIR')
    brain = mag_map.plot(subject=subject,
                         time_label='Magnetometer sensitivity',
                         subjects_dir=subjects_dir,
                         fmin=0.1,
                         fmid=0.5,
                         fmax=0.9,
                         smoothing_steps=7)

    brain.save_montage(fname_sensitvity_plot)
    brain.close()





# *******************************************
# * Function to estimate inverse solution   *
# *******************************************
def estimate_inverse_solution(info,
                              noise_cov_mat,
                              fwd_sol=None,
                              fname_fwd=None,
                              fname_inv=None):
    """"Estimates inverse solution for the given data set."""

    if fwd_sol is not None:
        pass
    elif fname_fwd is not None:
        fwd_sol = mne.read_forward_solution(fname_fwd,
                                            surf_ori=True)
    else:
        print "ERROR: Neither a forward solution given nor the filename of one!"
        sys.exit()

    # restrict forward solution as necessary for MEG
    fwd = mne.fiff.pick_types_forward(fwd_sol,
                                      meg=True,
                                      eeg=False)


    # # regularize noise covariance
    # # --> not necessary as the data set to estimate the
    # #     noise-covariance matrix is quiet long, i.e.
    # #     the noise-covariance matrix is robust
    # noise_cov_mat = mne.cov.regularize(noise_cov_mat,
    #                                    info,
    #                                    mag=0.1,
    #                                    proj=True,
    #                                    verbose=verbose)

    # create the MEG inverse operators
    print ">>>> estimate inverse operator..."
    inverse_operator = min_norm.make_inverse_operator(info,
                                                      fwd,
                                                      noise_cov_mat,
                                                      loose=0.2,
                                                      depth=0.8)

    if fname_inv is not None:
        min_norm.write_inverse_operator(fname_inv, inverse_operator)

    return inverse_operator




# *******************************************
# * Function to estimate source localization*
# *******************************************
def estimate_source_localization(fname_data,
                                 fname_inv,
                                 subject,
                                 snr=3.0,
                                 method="dSPM",
                                 fnout_src_loc=None,
                                 fnout_mov=None,
                                 verbose=None,
                                 show=True):
    """"Estimates source localization for the given data set and visualize results."""


    print ">>>> performing source localization using " + method + "..."


    # estimate lambda square
    lambda2 = 1. / snr ** 2

    # read in inverse operator
    inverse_operator = min_norm.read_inverse_operator(fname_inv)

    # read in data set for source localization
    data = mne.fiff.Evoked(fname_data)

    # perform real source localization
    src_loc = min_norm.apply_inverse(data,
                                     inverse_operator,
                                     lambda2,
                                     method=method,
                                     pick_ori=None)

    # save results if desired
    if fnout_src_loc is not None:
        src_loc.save(fnout_src_loc)


    # show results if desired
    if show is not None:
        subjects_dir = os.environ.get('SUBJECTS_DIR')
        brain = src_loc.plot(surface='inflated',
                             hemi='both',
                             subjects_dir=subjects_dir,
                             config_opts={'cortex':'bone'},
                             time_viewer=True)


    # create movie of activation
    if fnout_mov is not None:
        print ">>>> create movie of activations..."

        # check which method should be used
        if method == "dSPM":
            method_mov = " --spm"
        elif method == "sLORETA":
            method_mov = " --sLORETA"
        else:
            method_mov = ""

        os.system("mne_make_movie " + \
                  "--inv " + fname_inv + \
                  " --meas " + fname_data + \
                  " --subject " + subject + \
                  " --mov " + fnout_mov + \
                  method_mov + \
                  " --smooth 10 --surface inflated" + \
                  " --tmin 0 --tmax 150" + \
                  " --width 1920 --height 1200 --noscalebar"
                  " --alpha 0.5 --view lat" + \
                  " --fthresh 5 --fmid 10 --fmax 15")





# *******************************************
# * Function to estimate beamformer source  *
# * localization                            *
# *******************************************
def estimate_beamformer(epochs,
                        fname_forward_sol,
                        noise_cov_mat=None,
                        method="DICS",
                        show=True):
    """"Estimates source localization for the given data set using beamformer."""

    # read forward operator
    forward_sol = mne.read_forward_solution(fname_forward_sol, surf_ori=True)

    if method == "DICS":
        print ">>>> performing source localization using DICS beamformer..."
        # Computing the data and noise cross-spectral density matrices
        data_csds = compute_epochs_csd(epochs,
                                       mode='fourier',
                                       tmin=0.04,
                                       tmax=0.15,
                                       fmin=5,
                                       fmax=20)

        noise_csds = compute_epochs_csd(epochs,
                                        mode='fourier',
                                        tmin=-0.11,
                                        tmax=-0.0,
                                        fmin=5,
                                        fmax=20)
        pdb.set_trace()
        # Compute DICS spatial filter and estimate source power
        src_loc = beam.dics(epochs.average(),
                            forward_sol,
                            noise_csds,
                            data_csds)

        # for showing results
        fmin = 0.5; fmid = 3.0; fmax = 5.0

    else:
        print ">>>> performing source localization using LCMV beamformer..."

        if noise_cov_mat is None:
            print "To use LCMV beamformer the noise-covariance matrix keyword must be set."
            sys.exit()

        evoked = epochs.average()
        noise_cov_mat = mne.cov.regularize(noise_cov_mat,
                                           evoked.info,
                                           mag=0.05,
                                           proj=True)

        data_cov = mne.compute_covariance(epochs,
                                          tmin=0.04,
                                          tmax=0.15)

        src_loc = beam.lcmv(evoked,
                            forward_sol,
                            noise_cov_mat,
                            data_cov,
                            reg=0.01,
                            pick_ori=None)
        # for showing results
        fmin = 0.01; fmid = 0.5; fmax = 1.0


    # show results if desired
    if show is not None:
        subjects_dir = os.environ.get('SUBJECTS_DIR')
        brain = src_loc.plot(surface='inflated',
                             hemi='both',
                             subjects_dir=subjects_dir,
                             config_opts={'cortex':'bone'},
                             time_viewer=True,
                             fmin=fmin,
                             fmid=fmid,
                             fmax=fmax)





