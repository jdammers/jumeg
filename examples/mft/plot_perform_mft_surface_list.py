"""
====================
Jumeg MFT example.

Perform MFT on a surface based forward solution.
"""

import numpy as np
import mne
from mne.datasets import sample
from jumeg.mft import (apply_mft, jumeg_mft_plot)

data_path = sample.data_path()
subject = 'sample'
subjects_dir = data_path + '/subjects'
fwdname = data_path + '/MEG/sample/sample_audvis-meg-oct-6-fwd.fif'
lblname = 'aparc'
evoname = data_path + '/MEG/sample/sample_audvis-ave.fif'
evocondition = 'Left Auditory'
rawname = data_path + '/MEG/sample/sample_audvis_10s-raw.fif'
t1_fname = subjects_dir + '/' + 'sample/mri/T1.mgz'

# Set up pick list: MEG - bad channels
want_meg = 'mag'
want_ref = False
want_eeg = False
want_stim = False
exclude = 'bads'
include = []

print("########## MFT parameters:")
mftpar = {'prbfct': 'Gauss',
          'prbcnt': np.array([[0.039, 0.013, 0.062], [-0.039, 0.013, 0.062]]),
          'prbhw': np.array([[0.040, 0.040, 0.040], [0.040, 0.040, 0.040]])}
# mftpar = {'prbfct': 'uniform',
#           'prbcnt': None,
#           'prbhw': None}
mftpar.update({'iter': 2, 'currexp': 1.0})
mftpar.update({'regtype': 'PzetaE', 'zetareg': 1.00})
# mftpar.update({ 'regtype':'classic', 'zetareg':1.0})
mftpar.update({'solver': 'lu', 'svrelcut': 5.e-4})

print("mftpar['prbcnt'  ] = ", mftpar['prbcnt'])
print("mftpar['prbhw'   ] = ", mftpar['prbhw'])
print("mftpar['iter'    ] = ", mftpar['iter'])
print("mftpar['regtype' ] = ", mftpar['regtype'])
print("mftpar['zetareg' ] = ", mftpar['zetareg'])
print("mftpar['solver'  ] = ", mftpar['solver'])
print("mftpar['svrelcut'] = ", mftpar['svrelcut'])
cdmcut = 0.10
print("cdmcut = ", cdmcut)

print("########## get labels:")
if lblname is not None:
    labels = mne.read_labels_from_annot(subject, parc=lblname,
                                        subjects_dir=subjects_dir)
else:
    labels = None

print("##########################")
print("##### Calling apply_mft()")
print("##########################")
iterlist = [0, 1, 2]
fwdmag, qualmft, stc_mftl = apply_mft(fwdname, evoname, evocondition=evocondition,
                                      subject=subject, meg=want_meg, iterlist=iterlist,
                                      calccdm='all', cdmcut=cdmcut, cdmlabels=labels,
                                      mftpar=mftpar, verbose='verbose')

evo = mne.read_evokeds(evoname, condition=evocondition, baseline=(None, 0))
tstep = 1. / evo.info['sfreq']

if iterlist == None:
    stc_mft = stc_mftl
else:
    stc_mft = stc_mftl[-1]
stcdata = stc_mft.data
tmin = stc_mft.tmin

print(" ")
print("########## Some geo-numbers:")
lhinds = np.where(fwdmag['source_rr'][:, 0] <= 0.)
rhinds = np.where(fwdmag['source_rr'][:, 0] > 0.)
print("> Discriminating lh/rh by sign of fwdmag['source_rr'][:,0]:")
print("> lhinds[0].shape[0] = ", lhinds[0].shape[0], " rhinds[0].shape[0] = ", rhinds[0].shape[0])
invmri_head_t = mne.transforms.invert_transform(fwdmag['info']['mri_head_t'])
mrsrc = np.zeros(fwdmag['source_rr'].shape)
mrsrc = mne.transforms.apply_trans(invmri_head_t['trans'], fwdmag['source_rr'], move=True)
lhmrinds = np.where(mrsrc[:, 0] <= 0.)
rhmrinds = np.where(mrsrc[:, 0] > 0.)
print("> Discriminating lh/rh by sign of fwdmag['source_rr'][:,0] in MR coords:")
print("> lhmrinds[0].shape[0] = ", lhmrinds[0].shape[0], " rhmrinds[0].shape[0] = ", rhmrinds[0].shape[0])

# plotting routines
jumeg_mft_plot.plot_global_cdv_dist(stcdata)
jumeg_mft_plot.plot_visualize_mft_sources(fwdmag, stcdata, tmin=tmin, tstep=tstep,
                                          subject=subject, subjects_dir=subjects_dir)
jumeg_mft_plot.plot_cdv_distribution(fwdmag, stcdata)
jumeg_mft_plot.plot_max_amplitude_data(fwdmag, stcdata, tmin=tmin, tstep=tstep,
                                       subject=subject)
jumeg_mft_plot.plot_max_cdv_data(stc_mft, lhmrinds, rhmrinds)
jumeg_mft_plot.plot_cdvsum_data(stc_mft, lhmrinds, rhmrinds)
jumeg_mft_plot.plot_quality_data(qualmft, stc_mft)
jumeg_mft_plot.plot_cdm_data(qualmft, stc_mft, cdmlabels=labels)
jumeg_mft_plot.plot_jlong_labeldata(qualmft, stc_mft, labels)
jumeg_mft_plot.plot_jtotal_labeldata(qualmft, stc_mft, labels)

jumeg_mft_plot.plot_jlong_data(qualmft, stc_mft)

print_transforms = False
if print_transforms:
    print("##### Transforms:")
    print("fwdmag['info']['mri_head_t']:")
    print(fwdmag['info']['mri_head_t'])
    invmri_head_t = mne.transforms.invert_transform(fwdmag['info']['mri_head_t'])
    print("Inverse of fwdmag['info']['mri_head_t']:")
    print(invmri_head_t)

write_tab_files = True
if write_tab_files:
    tabfilenam = 'testtab_stc.dat'
    time_idx = np.argmax(np.max(stcdata, axis=0))
    print("##### Creating %s with |cdv(time_idx=%d)|" % (tabfilenam, time_idx))
    tabfile = open(tabfilenam, mode='w')
    cdvnmax = np.max(stcdata[:, time_idx])
    n_loc = stcdata.shape[0]
    tabfile.write("# time_idx = %d\n" % time_idx)
    tabfile.write("# max amplitude = %11.4e\n" % cdvnmax)
    tabfile.write("#  x/mm    y/mm    z/mm     |cdv|   index\n")
    for ipnt in range(int(n_loc / 3)):
        copnt = 1000. * fwdmag['source_rr'][ipnt]
        tabfile.write(" %7.2f %7.2f %7.2f %11.4e %5d\n" % \
                      (copnt[0], copnt[1], copnt[2], stcdata[ipnt, time_idx], ipnt))
    tabfile.close()

print("Done.")
