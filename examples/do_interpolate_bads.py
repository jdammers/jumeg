import mne
from mne.datasets import sample
from jumeg import suggest_bads
from jumeg_interpolate_bads import interpolate_bads

data_path = sample.data_path()
subjects_dir = data_path + '/subjects'

fname_raw = data_path + '/MEG/sample/sample_audvis_raw.fif'

raw = mne.io.Raw(fname_raw, preload=True)
mybads, raw = suggest_bads(raw, show_raw=False, summary_plot=False)

interpolate_bads(raw, reset_bads=True)
