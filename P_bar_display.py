#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
License: BSD (3-clause)
@author: imenb101<dongqunxi@gmail.com>
Input clean MEG data file under the path <subject path>/MEG/subject_audi_cued-raw_cle.fif
Output the bar of ctps' p value between '35-45Hz'
"""

import numpy as np
import matplotlib.pylab as pl
import mne, sys, os
from mne.viz import tight_layout
from mne.fiff import Raw
from mne.preprocessing import ICA
from ctps import compute_ctps
from ctps import plot_ctps_panel

try:
    subject = sys.argv[1]
    trigger = sys.argv[2]#Get the trigger is stim or resp
except:
    print "Please run with input file provided. Exiting"
    sys.exit()

res_ch_name = 'STI 013'
sti_ch_name = 'STI 014'
n_components=1.0
n_pca_components=1.0
max_pca_components=None

subjects_dir = '/home/qdong/data/'
subject_path = subjects_dir + subject#Set the data path of the subject
#raw_fname = subject_path + '/MEG/ssp_cleaned_%s_audi_cued-raw_cle.fif' %subject
raw_fname = subject_path + '/MEG/%s_audi_cued-raw_cle.fif' %subject
raw_basename = os.path.splitext(os.path.basename(raw_fname))[0]
raw = Raw(raw_fname, preload=True)
picks = mne.fiff.pick_types(raw.info, meg=True, eeg=False, eog=False, 
                            stim=False, exclude='bads')
#Only focus on the gamma bandwidth
flow, fhigh = 35.0, 45.0
filter_type = 'butter'
filter_order = 4
n_jobs = 4

raw.filter(flow, fhigh, picks=picks, n_jobs=n_jobs, method='iir',
           iir_params={'ftype': filter_type, 'order': filter_order})
           
ica = ICA(n_components=n_components, n_pca_components=n_pca_components,  max_pca_components=max_pca_components, random_state=0)
ica.decompose_raw(raw, picks=picks, decim=3)
if trigger == 'resp':#'1' represents the response channel
    add_from_raw = mne.fiff.pick_types(raw.info, meg=False, resp=True, exclude='bads')
    sources_add = ica.sources_as_raw(raw, picks=add_from_raw)
    events = mne.find_events(sources_add, stim_channel=res_ch_name)
    raw_basename += '_resp'
elif trigger == 'stim':#'0' represents the stimuli channel
    add_from_raw = mne.fiff.pick_types(raw.info, meg=False, stim=True, exclude='bads')
    sources_add = ica.sources_as_raw(raw, picks=add_from_raw)
    events = mne.find_events(sources_add, stim_channel=sti_ch_name)
    raw_basename += '_stim'
else:
    print "Please select the triger channel 'resp' for response channel or 'stim' for stimilus channel."
    sys.exit()
picks = mne.fiff.pick_types(sources_add.info, meg=False, misc=True, exclude='bads')

l_f = 35 # make the lower limits for CTPs
Brain_idx1=[]#The index of ICA related with trigger channels
axes_band = [221, 222, 223, 224]
ax_index = 0
for i in [37, 39, 41, 43]:
    h_f = i
    get_ylim = True
    if l_f != 35:
      get_ylim = False
      sources_add = ica.sources_as_raw(raw, picks=add_from_raw)
    #sources_add.filter(l_freq=l_f, h_freq=h_f, method='iir', n_jobs=4)
    sources_add.filter(l_freq=l_f, h_freq=h_f, n_jobs=4, method='iir')
    this_band = '%i-%iHz' % (l_f, h_f)
    temp = l_f
    l_f = h_f
    # Epochs at R peak onset, from stim_eve.
    ica_epochs_events = mne.Epochs(sources_add, events, event_id=1, tmin=-0.3, tmax=0.3,
                        picks=picks, preload=True, proj=False)
    x_length = len(ica_epochs_events.ch_names)
    _ , pk_dyn_stim, phase_trial = compute_ctps(ica_epochs_events.get_data())
    # Get kuiper maxima
    pk_max = pk_dyn_stim.max(axis=1)
    Brain_sources = pk_max > 0.1  # bool array, get the prominient components related with trigger
    Brain_ind = np.where(Brain_sources)[0].tolist() # indices
    #skip the null idx related with response
    Brain_idx1 += (Brain_ind)#Get the obvious sources related
    pl.subplot(axes_band[ax_index])
    x_bar = np.arange(x_length)
    pl.bar(x_bar, pk_max)
    for x in Brain_ind:
        pl.bar(x, pk_max[x], facecolor='r')
    pl.axhline(0.1, color='k', label='threshod')
    pl.xlabel('%s' %this_band)
    pl.ylim(0, 0.5)
    ax_index += 1
pl.tight_layout()
pl.show()
pl.savefig(subject_path+'/MEG/ctps_distribution_%s_%s.png'%(subject, trigger))  
Brain_idx = list(set(Brain_idx1))
print '%s has been identified as trigger components' %(Brain_idx)