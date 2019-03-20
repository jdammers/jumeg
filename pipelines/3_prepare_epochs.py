#!/usr/bin/env python2
# author : pravsripad@gmail.com

"""
Make epochs for different conditions and save them.
"""

import os.path as op
import mne
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

# subject list
subjects = config['subjects']

###############################################################################
# Create epochs
###############################################################################

for cond in config['conditions']:
    print(cond)

    # you may want to use glob etc. to avoid having to write down all file names in the config
    for fname in config[cond]['raw_fnames']:
        subject = fname.split('_')[0]
        raw_fname = op.join(recordings_dir, subject, fname)
        print(raw_fname)

        raw = mne.io.Raw(raw_fname, preload=True)

        # select only MEG channels for filtering
        picks = mne.pick_types(raw.info, meg=True, exclude='bads')

        # get channel with events
        stim_channel = config[cond]['ch_name']
        event_id = config['event_id']

        # get tmin, tmax for epoch creation
        tmin, tmax = config[cond]['tmin'], config[cond]['tmax']

        events = mne.find_events(raw, stim_channel=stim_channel, output='onset')
        epochs = mne.Epochs(raw, events, event_id, tmin, tmax, baseline=(None, 0.),
                            picks=picks, proj=False)

        epochs_fname = raw_fname.split('-raw.fif')[0] + ',%s-epo.fif' % cond
        print('saving ', epochs_fname)
        epochs.save(epochs_fname)

        evoked = epochs.average()
        evoked_fname = epochs_fname.split('-epo.fif')[0] + '-ave.fif'
        print('saving ', evoked_fname)
        evoked.save(evoked_fname)
