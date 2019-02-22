"""
====================
Jumeg MFT example.
====================
"""

import numpy as np
import mne
from mne.datasets import sample
from jumeg.mft import (apply_mft, jumeg_mft_plot)

data_path = sample.data_path()
subject = 'sample'
subjects_dir = data_path + '/subjects'
fwdname = data_path + '/MEG/sample/sample_audvis-meg-vol-7-fwd.fif'
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

print("########## MFT parameter:")
mftpar = {'prbfct': 'Gauss',
          'prbcnt': np.array([[-1.039, 0.013, 0.062], [-0.039, 0.013, 0.062]]),
          'prbhw': np.array([[0.040, 0.040, 0.040], [0.040, 0.040, 0.040]])}
# mftpar = { 'prbfct': 'uniform',
#            'prbcnt': None,
#            'prbhw': None}
mftpar.update({'iter': 4, 'currexp': 1.0})
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

print("##########################")
print("##### Read fwd-soln:")
print("##########################")
fwd = mne.read_forward_solution(fwdname, verbose=True)

# Select magnetometer channels:
fwdmag = mne.io.pick.pick_types_forward(fwd, meg=want_meg, ref_meg=False,
                                        eeg=False, exclude=exclude)

print(" ")
print("########## Some geo-numbers:")
lhinds = np.where(fwdmag['source_rr'][:, 0] <= 0.)
rhinds = np.where(fwdmag['source_rr'][:, 0] > 0.)
ypinds = np.where(fwdmag['source_rr'][:, 1] > 0.)
yninds = np.where(fwdmag['source_rr'][:, 1] <= 0.)
lhfinds = np.array(np.setdiff1d(lhinds[0], ypinds[0]))
lhbinds = np.array(np.setdiff1d(lhinds[0], yninds[0]))
rhfinds = np.array(np.setdiff1d(rhinds[0], ypinds[0]))
rhbinds = np.array(np.setdiff1d(rhinds[0], yninds[0]))
print("> Discriminating lh/rh by sign of fwdmag['source_rr'][:,0]:")
print("> lhinds[0].shape[0] = ", lhinds[0].shape[0], " rhinds[0].shape[0] = ", rhinds[0].shape[0])
invmri_head_t = mne.transforms.invert_transform(fwdmag['info']['mri_head_t'])
mrsrc = np.zeros(fwdmag['source_rr'].shape)
mrsrc = mne.transforms.apply_trans(invmri_head_t['trans'], fwdmag['source_rr'], move=True)
lhmrinds = np.where(mrsrc[:, 0] <= 0.)
rhmrinds = np.where(mrsrc[:, 0] > 0.)
print("> Discriminating lh/rh by sign of fwdmag['source_rr'][:,0] in MR coords:")
print("> lhmrinds[0].shape[0] = ", lhmrinds[0].shape[0], " rhmrinds[0].shape[0] = ", rhmrinds[0].shape[0])

print("########## prepare to put labels into label-files:")
mylablist = []
# mylab = mne.Label(lhinds[0],pos=fwdmag['source_rr'][lhinds[0],:],hemi='lh',name='test_lh', verbose=True)
# mylablist.append(mylab)
# mylab = mne.Label(rhinds[0],pos=fwdmag['source_rr'][rhinds[0],:],hemi='rh',name='test_rh', verbose=True)
# mylablist.append(mylab)

# mylab = mne.Label(lhfinds,pos=fwdmag['source_rr'][lhfinds,:],hemi='lhf',name='test_lhf', verbose=True)
# mylablist.append(mylab)
# mylab = mne.Label(lhbinds,pos=fwdmag['source_rr'][lhbinds,:],hemi='lhb',name='test_lhb', verbose=True)
# mylablist.append(mylab)
# mylab = mne.Label(rhfinds,pos=fwdmag['source_rr'][rhfinds,:],hemi='rhf',name='test_rhf', verbose=True)
# mylablist.append(mylab)
# mylab = mne.Label(rhbinds,pos=fwdmag['source_rr'][rhbinds,:],hemi='rhb',name='test_rhb', verbose=True)
# mylablist.append(mylab)

print("##################################")
print("##### Calling apply_mft(mylablist)")
print("##################################")
fwdmag, qualmft, stc_mft = apply_mft(fwdname, evoname, evocondition=evocondition,
                                     subject=subject, meg=want_meg,
                                     calccdm='all', cdmcut=cdmcut, cdmlabels=mylablist,
                                     mftpar=mftpar, verbose='verbose')

evo = mne.read_evokeds(evoname, condition=evocondition, baseline=(None, 0))
tmin = -0.2
tstep = 1. / evo.info['sfreq']
stcdata = stc_mft.data

# plotting routines
jumeg_mft_plot.plot_global_cdv_dist(stcdata)
# print ">>>>> skipping jumeg_mft_plot.plot_visualize_mft_sources() (fails due to old PySurfer-version)"
jumeg_mft_plot.plot_visualize_mft_sources(fwdmag, stcdata, tmin=tmin, tstep=tstep,
                                          subject=subject, subjects_dir=subjects_dir)
jumeg_mft_plot.plot_cdv_distribution(fwdmag, stcdata)
jumeg_mft_plot.plot_max_amplitude_data(fwdmag, stcdata, tmin=tmin, tstep=tstep,
                                       subject=subject)
jumeg_mft_plot.plot_max_cdv_data(stc_mft, lhmrinds, rhmrinds)
jumeg_mft_plot.plot_cdvsum_data(stc_mft, lhmrinds, rhmrinds)
jumeg_mft_plot.plot_quality_data(qualmft, stc_mft)
jumeg_mft_plot.plot_cdm_data(qualmft, stc_mft)

jumeg_mft_plot.plot_cdm_data(qualmft, stc_mft, cdmlabels=mylablist)
jumeg_mft_plot.plot_jlong_labeldata(qualmft, stc_mft, mylablist)
jumeg_mft_plot.plot_jtotal_labeldata(qualmft, stc_mft, mylablist)

jumeg_mft_plot.plot_cdm_data(qualmft, stc_mft)
jumeg_mft_plot.plot_jlong_data(qualmft, stc_mft)

print("Done.")
