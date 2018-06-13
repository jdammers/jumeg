import mne
from jumeg.decompose.ica_replace_mean_std import apply_ica_replace_mean_std
from mne.datasets import sample
from mne.preprocessing import ICA

flow = 1.
fhigh = 45.

reject = {'mag': 5e-12}

data_path = sample.data_path()
subjects_dir = data_path + '/subjects'

fname_raw = data_path + '/MEG/sample/sample_audvis_raw.fif'

raw = mne.io.Raw(fname_raw, preload=True)
raw_filt = raw.copy().filter(flow, fhigh, method='fir', n_jobs=2, fir_design='firwin', phase='zero')

# use 60s of data
raw_filt.crop(0, 60)
raw.crop(0, 60)
raw_unfilt = raw.copy()

picks = mne.pick_types(raw.info, meg=True, exclude='bads')

ica = ICA(method='fastica', n_components=60, random_state=None,
          max_pca_components=None, max_iter=1500, verbose=False)

# fit ica object to filtered data
ica.fit(raw_filt, picks=picks, reject=reject, verbose=True)

# save mean and standard deviation of filtered MEG channels for the standard mne routine
pca_mean_filt = ica.pca_mean_.copy()
pca_pre_whitener_filt = ica._pre_whitener.copy()  # this is the standard deviation of MEG channels

# use the same arguments for apply_ica_replace_mean_std as when you are initializing the ICA
# object and when you are fitting it to the data
# the ica object is modified in place!!

raw_filt_clean = apply_ica_replace_mean_std(raw_filt, ica, picks=picks, reject=reject, exclude=ica.exclude,
                                            n_pca_components=None)

# save mean and standard deviation of filtered MEG channels with mean and std replaced
pca_mean_replaced_filt = ica.pca_mean_.copy()
pca_pre_whitener_replaced_filt = ica._pre_whitener.copy()

raw_clean = apply_ica_replace_mean_std(raw_unfilt, ica, picks=picks, reject=reject, exclude=ica.exclude,
                                       n_pca_components=None)

# save mean and standard deviation of unfiltered MEG channels
pca_mean_replaced_unfilt = ica.pca_mean_
pca_pre_whitener_replaced_unfilt = ica._pre_whitener

# compare filtered and unfiltered data
for idx in range(0, len(pca_mean_filt)):
    print '%10.6f\t%10.6f\t%10.6f' % (pca_mean_filt[idx], pca_mean_replaced_filt[idx], pca_mean_replaced_unfilt[idx])
    if idx >= 9:
        break

for idx in range(0, len(pca_pre_whitener_filt)):
    print pca_pre_whitener_filt[idx], pca_pre_whitener_replaced_filt[idx], pca_pre_whitener_replaced_unfilt[idx]
    if idx >= 9:
        break
