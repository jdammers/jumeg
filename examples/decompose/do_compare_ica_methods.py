"""
Compute the ica object on filtered data based on the mne and on the
jumeg method.
Compare pca_mean_ and pre_whitener_ for:
mne & filtered data, jumeg & filtered data, jumeg & unfiltered data
"""

import os.path as op
import mne
from mne.preprocessing.ica import ICA as ICA_mne
from jumeg.decompose.ica_replace_mean_std import ICA as ICA_jumeg
from jumeg.decompose.ica_replace_mean_std import apply_ica_replace_mean_std
from mne.datasets import sample

flow = 1.
fhigh = 45.

reject = {'mag': 5e-12}

data_path = sample.data_path()
subjects_dir = op.join(data_path, 'subjects')

fname_raw = op.join(data_path, 'MEG/sample/sample_audvis_raw.fif')

raw = mne.io.Raw(fname_raw, preload=True)
raw_filt = raw.copy().filter(flow, fhigh, method='fir', n_jobs=2, fir_design='firwin', phase='zero')

# use 60s of data
raw_filt.crop(0, 60)
raw.crop(0, 60)
raw_unfilt = raw.copy()

picks = mne.pick_types(raw.info, meg=True, exclude='bads')

ica_mne = ICA_mne(method='fastica', n_components=60, random_state=42,
                  max_pca_components=None, max_iter=5000, verbose=False)

# fit ica object from mne to filtered data
ica_mne.fit(raw_filt, picks=picks, reject=reject, verbose=True)

# save mean and standard deviation of filtered MEG channels for the standard mne routine
pca_mean_filt_mne = ica_mne.pca_mean_.copy()
pca_pre_whitener_filt_mne = ica_mne.pre_whitener_.copy()  # this is the standard deviation of MEG channels


ica_jumeg = ICA_jumeg(method='fastica', n_components=60, random_state=42,
                      max_pca_components=None, max_iter=5000, verbose=False)

# fit ica object from jumeg to filtered data
ica_jumeg.fit(raw_filt, picks=picks, reject=reject, verbose=True)

# save mean and standard deviation of filtered MEG channels for the standard mne routine
pca_mean_filt_jumeg = ica_jumeg.pca_mean_.copy()
pca_pre_whitener_filt_jumeg = ica_jumeg.pre_whitener_.copy()  # this is the standard deviation of MEG channels

# use the same arguments for apply_ica_replace_mean_std as when you are initializing the ICA
# object and when you are fitting it to the data
# the ica object is modified in place!!

# apply ica object from jumeg to unfiltered data while replacing the mean and std
raw_clean = apply_ica_replace_mean_std(raw_unfilt, ica_jumeg, picks=picks, reject=reject, exclude=ica_mne.exclude,
                                       n_pca_components=None)

# save mean and standard deviation of unfiltered MEG channels
pca_mean_replaced_unfilt_jumeg = ica_jumeg.pca_mean_
pca_pre_whitener_replaced_unfilt_jumeg = ica_jumeg.pre_whitener_

# compare methods for filtered and unfiltered data
for idx in range(0, len(pca_mean_filt_mne)):
    print('%10.6f\t%10.6f\t%10.6f' % (pca_mean_filt_mne[idx], pca_mean_filt_jumeg[idx],
                                      pca_mean_replaced_unfilt_jumeg[idx]))
    if idx >= 9:
        break

for idx in range(0, len(pca_pre_whitener_filt_mne)):
    print(pca_pre_whitener_filt_mne[idx], pca_pre_whitener_filt_jumeg[idx],\
        pca_pre_whitener_replaced_unfilt_jumeg[idx])
    if idx >= 9:
        break
