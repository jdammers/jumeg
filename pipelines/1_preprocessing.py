import os
import os.path as op
import mne

from utils import noise_reduction
from utils import interpolate_bads_batch
from utils import save_state_space_file

import yaml

with open('config_file.yaml', 'r') as f:
    config = yaml.load(f)

###############################################################################
# Get settings from config
###############################################################################

# directories
basedir = config['basedir']
recordings_dir = op.join(basedir, config['recordings_dir'])

# subject list
subjects = config['subjects']

# noise reducer
refnotch = config['noise_reducer']['refnotch']

# filtering
flow = config['filtering']['l_freq']
fhigh = config['filtering']['h_freq']
unfiltered = config['filtering']['unfiltered']
fi_method = config['filtering']['method']
fir_design = config['filtering']['fir_design']
phase = config['filtering']['phase']

# resampling
rsfreq = config['resampling']['rsfreq']

# TODO WIP: the state space file saves the parameters for the creation of each file
state_space_fname = '_state_space_dict.pkl'

###############################################################################
# Noise reduction
###############################################################################

for subj in subjects:
    print("Noise reduction for subject %s" % subj)
    dirname = op.join(recordings_dir, subj)
    sub_file_list = os.listdir(dirname)

    for raw_fname in sub_file_list:

        if raw_fname.endswith('meeg-raw.fif') or raw_fname.endswith('rfDC-empty.fif'):

            if raw_fname.endswith('-raw.fif'):
                denoised_fname = raw_fname.rsplit('-raw.fif')[0] + ',nr-raw.fif'
            else:
                denoised_fname = raw_fname.rsplit('-empty.fif')[0] + ',nr-empty.fif'

            if not op.isfile(op.join(dirname, denoised_fname)):
                noise_reduction(dirname, raw_fname, denoised_fname, refnotch,
                                state_space_fname)

###############################################################################
# Identify bad channels
###############################################################################

interpolate_bads_batch(subjects, recordings_dir, state_space_fname)

###############################################################################
# Filter
###############################################################################

for subj in subjects:
    print("Filtering for subject %s" % subj)

    dirname = op.join(recordings_dir, subj)
    sub_file_list = os.listdir(dirname)

    ss_dict_fname = op.join(dirname, subj + state_space_fname)

    for raw_fname in sub_file_list:

        if raw_fname.endswith('bcc-raw.fif') or raw_fname.endswith('bcc-empty.fif'):

            raw = mne.io.Raw(op.join(dirname, raw_fname), preload=True)

            raw_filt = raw.filter(flow, fhigh, method=fi_method, n_jobs=1,
                                  fir_design=fir_design, phase=phase)

            if raw_fname.endswith('-raw.fif'):
                raw_filt_fname = raw_fname.rsplit('-raw.fif')[0] + ',fibp-raw.fif'
            else:
                raw_filt_fname = raw_fname.rsplit('-empty.fif')[0] + ',fibp-empty.fif'

            fi_dict = dict()
            fi_dict['flow'] = flow
            fi_dict['fhigh'] = fhigh
            fi_dict['method'] = fi_method
            fi_dict['fir_design'] = fir_design
            fi_dict['phase'] = phase
            fi_dict['output_file'] = raw_filt_fname

            save_state_space_file(ss_dict_fname, process='filtering',
                                  input_fname=raw_fname, process_config_dict=fi_dict)

            raw_filt.save(op.join(dirname, raw_filt_fname))

            raw_filt.close()

###############################################################################
# Resampling
###############################################################################

for subj in subjects:
    print("Filtering for subject %s" % subj)

    dirname = op.join(recordings_dir, subj)
    sub_file_list = os.listdir(dirname)

    ss_dict_fname = op.join(dirname, subj + state_space_fname)

    for raw_fname in sub_file_list:

        valid_file = raw_fname.endswith('fibp-raw.fif') or raw_fname.endswith('fibp-empty.fif')

        # perform resampling on unfiltered data
        if unfiltered:
            valid_file = valid_file or raw_fname.endswith('bcc-raw.fif') or raw_fname.endswith('bcc-empty.fif')

        # resample valid data
        if valid_file:

            raw = mne.io.Raw(op.join(dirname, raw_fname), preload=True)

            npad = 'auto'
            raw_rs = raw.resample(rsfreq, npad=npad)

            if raw_fname.endswith('-raw.fif'):
                raw_rs_fname = raw_fname.rsplit('-raw.fif')[0] + ',rs-raw.fif'
            else:
                raw_rs_fname = raw_fname.rsplit('-empty.fif')[0] + ',rs-empty.fif'

            rs_dict = dict()
            rs_dict['rsfreq'] = rsfreq
            rs_dict['npad'] = npad
            rs_dict['output_file'] = raw_rs_fname

            save_state_space_file(ss_dict_fname, process='filtering',
                                  input_fname=raw_fname, process_config_dict=rs_dict)

            raw_rs.save(op.join(dirname, raw_rs_fname))

            raw_rs.close()
