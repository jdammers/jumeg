import os
import os.path as op
import mne

from utils import noise_reduction
from utils import interpolate_bads_batch

subject_list = ['subj']
subjects_dir = 'test'

# noise reducer
refnotch = [50., 100., 150., 200., 250., 300., 350., 400., 60., 120., 180., 240., 360.]

# filtering
flow = 1.
fhigh = 45.

# resampling
rsfreq = 250

###############################################################################
# Noise reduction
###############################################################################

for subj in subject_list:
    print("Noise reduction for subject %s" % subj)
    dirname = os.path.join(subjects_dir, subj)
    sub_file_list = os.listdir(dirname)

    for raw_fname in sub_file_list:

        if raw_fname.endswith('meeg-raw.fif') or raw_fname.endswith('meeg-empty.fif'):

            if raw_fname.endswith('-raw.fif'):
                denoised_fname = raw_fname.rsplit('-raw.fif')[0] + ',nr-raw.fif'
            else:
                denoised_fname = raw_fname.rsplit('-empty.fif')[0] + ',nr-empty.fif'

            if not op.isfile(op.join(dirname, denoised_fname)):
                noise_reduction(dirname, raw_fname, denoised_fname, refnotch)

###############################################################################
# Identify bad channels
###############################################################################

bads_dict_fname = 'bad_channels.pkl'
interpolate_bads_batch(subject_list, subjects_dir, bads_dict_fname)

###############################################################################
# Filter
###############################################################################

for subj in subject_list:
    print("Filtering for subject %s" % subj)

    sub_file_list = os.listdir(dirname)

    for raw_fname in sub_file_list:

        if raw_fname.endswith('bcc-raw.fif') or raw_fname.endswith('bcc-empty.fif'):

            raw = mne.io.Raw(raw_fname, preload=True)

            raw_filt = raw.filter(flow, fhigh, method='fir', n_jobs=2,
                                  fir_design='firwin', phase='zero')

            if raw_fname.endswith('-raw.fif'):
                raw_filt_fname = raw_fname.rsplit('-raw.fif')[0] + ',fibp-raw.fif'
            else:
                raw_filt_fname = raw_fname.rsplit('-empty.fif')[0] + ',fibp-empty.fif'

            raw_filt.save(raw_filt_fname)

            raw_filt.close()

###############################################################################
# Resampling
###############################################################################

for subj in subject_list:
    print("Filtering for subject %s" % subj)

    sub_file_list = os.listdir(dirname)

    for raw_fname in sub_file_list:

        # resample filtered and unfiltered data
        if raw_fname.endswith('bcc-raw.fif') or raw_fname.endswith('bcc-empty.fif') or \
                raw_fname.endswith('fibp-raw.fif') or raw_fname.endswith('fibp-empty.fif'):

            raw = mne.io.Raw(raw_fname, preload=True)

            raw_rs = raw.resample(100, npad="auto")

            if raw_fname.endswith('-raw.fif'):
                raw_rs_fname = raw_fname.rsplit('-raw.fif')[0] + ',rs-raw.fif'
            else:
                raw_rs_fname = raw_fname.rsplit('-empty.fif')[0] + ',rs-empty.fif'

            raw_rs.save(raw_rs_fname)

            raw_rs.close()
