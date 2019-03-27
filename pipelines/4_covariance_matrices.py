#!/usr/bin/env python2

"""
Compute the covariance matrix and plot_white and plot evoked for the data.
"""

import os.path as op
import mne
from utils import find_files, set_directory

import yaml

with open('config_file.yaml', 'r') as f:
    config = yaml.load(f)

###############################################################################
# Get settings from config
###############################################################################

# directories
basedir = config['basedir']
recordings_dir = op.join(basedir, config['recordings_dir'])
subjects_dir = op.join(basedir, config['subjects_dir'])
plot_dir = op.join(basedir, config['plot_dir'])

# subject list
subjects = config['subjects']
l_freq, h_freq = config['l_freq'], config['h_freq']
pre_proc_ext = config['pre_proc_ext']

compute_empty_cov = config['cov_matrices']['compute_empty_cov']
compute_epo_cov = config['cov_matrices']['compute_epo_cov']
unfiltered = config['cov_matrices']['unfiltered']

###############################################################################
# Compute noise covariance matrix based on empty room measurements
###############################################################################

if compute_empty_cov:

    for subj in subjects:

        pattern = subj + '_*' + pre_proc_ext + '-empty.fif'
        empty_fnames = find_files(recordings_dir, pattern=pattern)

        if unfiltered:
            pattern_unfilt = pattern.replace(',fibp', '')
            empty_fnames_unfilt = find_files(recordings_dir, pattern=pattern_unfilt)
            empty_fnames += empty_fnames_unfilt

        for empty_fpath in empty_fnames:

            empty_fname = op.basename(empty_fpath)
            raw = mne.io.Raw(empty_fpath)

            # create and save covariance matrix
            cov = mne.compute_raw_covariance(raw)
            cov_fpath = empty_fpath.rsplit('-empty.fif')[0] + ',empty-cov.fif'
            cov.save(cov_fpath)

            # create plot
            fig, _ = cov.plot(raw.info, show_svd=False, show=False)

            set_directory(op.join(plot_dir, 'cov', subj))
            covplot_fpath = op.join(plot_dir, 'cov', subj, empty_fname.split('-empty.fif')[0] + ',empty-cov.png')
            fig.savefig(covplot_fpath)


###############################################################################
# Compute noise covariance matrix based on the baseline (pre-stim) of epoch data
###############################################################################

if compute_epo_cov:
    for cond in config['conditions']:
        print(cond)

        for subj in subjects:
            pattern = subj + '_*' + pre_proc_ext + ',ar,' + cond + '-epo.fif'
            fn_list = find_files(recordings_dir, pattern=pattern)

            if unfiltered:
                pattern_unfilt = pattern.replace(',fibp', '')
                fn_list_unfilt = find_files(recordings_dir, pattern=pattern_unfilt)
                fn_list += fn_list_unfilt

            for fpath in fn_list:

                epochs_fname = op.basename(fpath)
                subj = epochs_fname.split('_')[0]
                epochs_fpath = op.join(recordings_dir, subj, epochs_fname)

                epochs = mne.read_epochs(epochs_fpath)
                noise_cov_reg = mne.compute_covariance(epochs, tmin=None, tmax=0., method='empirical', n_jobs=2)
                cov_fpath = epochs_fpath.split('-epo.fif')[0] + '-cov.fif'
                noise_cov_reg.save(cov_fpath)

                evoked = epochs.average()
                figwhite = evoked.plot_white(noise_cov_reg, show=False)

                set_directory(op.join(plot_dir, 'cov', subj))
                white_plot_fpath = op.join(plot_dir, 'cov', subj, epochs_fname.split('-epo.fif')[0] + ',white.png')
                figwhite.savefig(white_plot_fpath)
