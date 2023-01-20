"""
Compute infomax ICA on raw data.
"""

import os.path as op
import mne
from mne.datasets import sample
from jumeg.decompose.ica import infomax

data_path = sample.data_path()

fname_raw = op.join(data_path, 'MEG/sample/sample_audvis_filt-0-40_raw.fif')

raw = mne.io.Raw(fname_raw, preload=True)

# use 60s of data
raw.crop(0, 60)

picks = mne.pick_types(raw.info, meg=True, exclude='bads')
data = raw.get_data()[:10, :].transpose(1, 0)
print(data.shape)

umixing_matrix =  infomax(data, weights=None, l_rate=None, block=None, w_change=1e-12,
                          anneal_deg=60., anneal_step=0.9, extended=False, n_subgauss=1,
                          kurt_size=6000, ext_blocks=1, max_iter=20,
                          fixed_random_state=37, verbose=True)

print(umixing_matrix.shape)
