#!/usr/bin/env python

"""
Perform MNE-dSPM source localization in surface space space on CAU data.
"""

import os
import os.path as op
import time

from utils import find_files

import mne
from mne.minimum_norm import (make_inverse_operator, write_inverse_operator,
                              read_inverse_operator, apply_inverse, apply_inverse_epochs)

import yaml

with open('config_file.yaml', 'r') as f:
    config = yaml.load(f)

###############################################################################
# Get settings from config
###############################################################################

# directories
basedir = config['basedir']
subjects_dir = op.join(basedir, config['subjects_dir'])
recordings_dir = op.join(basedir, config['recordings_dir'])

os.environ['SUBJECTS_DIR'] = subjects_dir

# subject list
subjects = config['subjects']

n_jobs = config['n_jobs']
pre_proc_ext = config['pre_proc_ext']

# inverse solution
method = config['inv_method']
snr = config['snr']
lambda2 = 1.0 / snr ** 2
do_inv_epo = False

# plotting stc
plot_stc = True
time_viewer = True

###############################################################################
# Calculate forward and inverse solution
###############################################################################

for subj in subjects:

    for cond in config['conditions']:

        # if you have multiple runs with different head
        # positions you have to create a fwd and inv
        # operator for each run
        # if you have multiple conditions in the same run
        # you can use the same fwd operator for these
        # conditions

        dirname_rec = op.join(recordings_dir, subj)
        dirname_subj = op.join(subjects_dir, subj)

        # select epochs
        pattern = subj + '_*' + pre_proc_ext + ',ar,' + cond + '-epo.fif'
        fn_epo_list = find_files(dirname_rec, pattern=pattern)
        fn_epo = fn_epo_list[0]

        fn_fwd = op.join(dirname_rec, subj + '_' + 'ico-4-' + cond + '-fwd.fif')
        fn_src = op.join(dirname_subj, 'bem', subj + '-ico-4-src.fif')
        fn_bem = op.join(dirname_subj, 'bem', subj + '-5120-5120-5120-bem-sol.fif')
        fn_trans = op.join(dirname_rec, subj + '-trans.fif')

        # load epochs
        epochs = mne.read_epochs(fn_epo, preload=False)

        if not op.exists(fn_fwd):

            # create forward solution
            # similar to mne_do_foward_solution (uses MNE-C binaries)
            fwd = mne.make_forward_solution(epochs.info, trans=fn_trans, src=fn_src,
                                            bem=fn_bem, mindist=5, eeg=False)

            # https://martinos.org/mne/stable/generated/mne.convert_forward_solution.html
            # for loose constraint, depth weighted inverse
            # solution set surf_ofi to True and force_fixed
            # to False
            # see https://martinos.org/mne/stable/generated/mne.minimum_norm.make_inverse_operator.html

            fwd = mne.convert_forward_solution(fwd, surf_ori=True, force_fixed=False)

            mne.write_forward_solution(fname=fn_fwd, fwd=fwd, overwrite=True)

        else:
            fwd = mne.read_forward_solution(fn_fwd)

        # for evoked data use noise cov based on epochs, however,
        # if there is no baseline, e.g., in resting state data
        # there is no stimulus, use empty room noise cov
        fn_cov = fn_epo.split('-epo.fif')[0] + '-cov.fif'
        fn_inv = fn_epo.rsplit('-epo.fif')[0] + '-inv.fif'

        # to read forward solution and noise_cov

        noise_cov = mne.read_cov(fn_cov)

        if not op.isfile(fn_inv):
            # compute the inverse operator (mne_do_inverse_operator)
            inv = make_inverse_operator(epochs.info, fwd, noise_cov, loose=0.2,
                                        depth=0.8, limit_depth_chs=False)
            write_inverse_operator(fn_inv, inv)
        else:
            inv = read_inverse_operator(fn_inv)

        evoked = epochs.average()

        # Compute inverse solution for evoked data
        fn_stc = fn_epo.rsplit('-epo.fif')[0] + ',ave'

        if not op.isfile(fn_stc + '-lh.stc'):
            stc = apply_inverse(evoked, inv, lambda2, method, pick_ori=None)
            stc.save(fn_stc)
            print('Saved...', fn_stc + '-lh.stc')

        if plot_stc:

            stc = mne.read_source_estimate(fn_stc + '-lh.stc')
            pos, t_peak = stc.get_peak(hemi=None, tmin=0., tmax=0.5,
                                       mode='abs')
            brain = stc.plot(subject=subj, surface='inflated', hemi='both',
                             colormap='auto', time_label='auto',
                             subjects_dir=subjects_dir, figure=None,
                             colorbar=True, clim='auto', initial_time=t_peak,
                             time_viewer=time_viewer)
            stc_plot_fname = op.join(basedir, 'plots', op.basename(fn_stc) + ',plot.png')
            time.sleep(1)

            if not time_viewer:
                # works only if time viewer is disabled
                brain.save_montage(stc_plot_fname, order=['lat', 'dor', 'med'])
                brain.close()
                time.sleep(1)

        # compute inverse solution for epochs

        if do_inv_epo:

            # Compute inverse solution for epoch data
            fn_stc = fn_epo.rsplit('-epo.fif')[0] + ',epo'

            # get a list of one stc per epoch
            # usually it is best to create the
            # stcs when you need them instead
            # of saving them
            stc = apply_inverse_epochs(epochs, inv, lambda2, method, pick_ori=None)

