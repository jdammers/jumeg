#The version 2, morph the common RefROI into individual space, then compute
#phase lock indices and connectivity under the individual space. Then morph
#the phase lock estimates into the common space. 
import numpy as np
import mne, sys, os
from mne.datasets import sample
from mne.fiff import Raw, pick_types
from mne.minimum_norm import (apply_inverse, apply_inverse_epochs,
                              read_inverse_operator)
from mne.connectivity import seed_target_indices, spectral_connectivity
from array import array
try:
    subject = sys.argv[1]
    trigger = sys.argv[2]#Get the trigger is stim or resp
except:
    print "Please run with input file provided. Exiting"
    sys.exit()

res_ch_name = 'STI 013'
sti_ch_name = 'STI 014'

#Load cleaned raw data based on trigger information 
subjects_dir = '/home/qdong/freesurfer/subjects/'
subject_path = subjects_dir + subject#Set the data path of the subject
raw_fname = subject_path + '/MEG/raw_%s_audi_cued-raw_cle_%s.fif' %(subject, trigger)
raw_basename = os.path.splitext(os.path.basename(raw_fname))[0]
raw = Raw(raw_fname, preload=True)
#Make events based on trigger information
if trigger == 'resp':
    tmin, tmax = -0.3, 0.3
    events = mne.find_events(raw, stim_channel=res_ch_name)
elif trigger == 'stim':
    tmin, tmax = -0.2, 0.4
    events = mne.find_events(raw, stim_channel=sti_ch_name)
picks = mne.fiff.pick_types(raw.info, meg=True, exclude='bads')
epochs = mne.Epochs(raw, events, 1, tmin, tmax, proj=False, picks=picks, preload=True, reject=None)

mri = subject_path + '/MEG/' + subject + '-trans.fif'#Set the path of coordinates trans data
src = subject_path + '/bem/' + subject + '-ico-4-src.fif'#Set the path of src including dipole locations and orientations
bem = subject_path + '/bem/' + subject + '-5120-5120-5120-bem-sol.fif'#Set the path of file including the triangulation and conducivity\
#information together with the BEM
fname_cov = subject_path + '/MEG/' + subject + '_emptyroom_cov.fif'#Empty room noise covariance
noise_cov = mne.read_cov(fname_cov)

#Forward solution
fwd = mne.make_forward_solution(epochs.info, mri=mri, src=src, bem=bem,
                                fname=None, meg=True, eeg=False, mindist=5.0,
                                n_jobs=2, overwrite=True)
fwd = mne.convert_forward_solution(fwd, surf_ori=True)

#Make Inverse operator
noise_cov = mne.cov.regularize(noise_cov, epochs.info,
                               mag=0.05, proj=True)
forward_meg = mne.fiff.pick_types_forward(fwd, meg=True, eeg=False)
inverse_operator = mne.minimum_norm.make_inverse_operator(epochs.info, forward_meg, noise_cov,
                                             loose=0.2, depth=0.8)

#Load the reference ROI
label_fname = '/home/qdong/freesurfer/subjects/fsaverage/label/lh.RefROI1.label'
label = mne.read_label(label_fname)
label.values.fill(1.0)
label_morph = label.morph(subject_from='fsaverage', subject_to=subject, smooth=5, 
                                 n_jobs=1, copy=True)
                                 
# First, we find the most active vertex in the left auditory cortex, which
# we will later use as seed for the connectivity computation
snr = 3.0
method = "dSPM"
lambda2 = 1.0 / snr ** 2
evoked = epochs.average()
stc = apply_inverse(evoked, inverse_operator, lambda2, method,
                    pick_ori="normal")
#For defining REF_ROI
stc_morph = mne.morph_data(subject, 'fsaverage', stc, 5, smooth=5)
stc_morph.save(subject_path+'/Ref_fsaverage'+subject+'_' +trigger, ftype='stc')
# Restrict the source estimate to the label in the left auditory cortex 
#(reference ROI)
stc_label = stc.in_label(label_morph)

# Find number and index of vertex with most power
src_pow = np.sum(stc_label.data ** 2, axis=1)
seed_vertno = stc_label.vertno[0][np.argmax(src_pow)]
seed_idx = np.searchsorted(stc.vertno[0], seed_vertno)  # index in original stc

# Generate index parameter for seed-based connectivity analysis
n_sources = stc.data.shape[0]
indices = seed_target_indices([seed_idx], np.arange(n_sources))

# Compute inverse solution and for each epoch. By using "return_generator=True"
# stcs will be a generator object instead of a list. This allows us so to
# compute the coherence without having to keep all source estimates in memory.
snr = 1.0  # use lower SNR for single epochs
lambda2 = 1.0 / snr ** 2
stcs = apply_inverse_epochs(epochs, inverse_operator, lambda2, method,
                            pick_ori="normal", return_generator=True)

# Now we are ready to compute the coherence in the gamma band.
# fmin and fmax specify the lower and upper freq. for each band, resp.
sfreq = raw.info['sfreq']  # the sampling frequency

cwt_frequencies = np.array([40])
coh, freqs, times, n_epochs, n_tapers = spectral_connectivity(stcs,
    method='plv', mode='cwt_morlet', indices=indices,
    sfreq=sfreq, cwt_frequencies=cwt_frequencies, cwt_n_cycles=7, faverage=True, n_jobs=2)

#tmin = np.mean(freqs[0])
#tstep = np.mean(freqs[1]) - tmin
coh_new=coh.squeeze()
coh_stc = mne.SourceEstimate(coh_new, vertices=stc.vertno, tmin=tmin,#tmin=1e-3 * tmin,
                             tstep=1e-3*1, subject=subject)
                             #tstep=1e-3 * tstep, subject=subject)
coh_stc.save(subject_path+'/ROI_'+subject+'_' +trigger, ftype='stc')

#Some commands to call freesurfer and MNE
#tksurfer -annot aparc subjectid rh inflated
#mne_make_movie --stcin ROI_101611_stim-lh.stc --tmin 20 --tmax 120 --morph fsaverage --smooth 5 --mov first_cwt --subject 101611 --rh  --fthresh 0.15 --fmid 0.3 --fmax 0.55
#mne_make_movie --stcin Ref_fsaverage101611_stim-lh.stc --tmin 20 --tmax 120 --mov first_dSPM --subject fsaverage --smooth 5 --lh --spm --fthresh 5 --fmid 8 --fmax 12
