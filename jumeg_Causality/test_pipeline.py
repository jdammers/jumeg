# -*- coding: utf-8 -*-
import os, glob
from Causality_pipeline import apply_inverse_ave, apply_merge, apply_norm,\
                               apply_comSTC,  focal_ROIs
subjects_dir = os.environ['SUBJECTS_DIR']
evt_list = ['LLst', 'LRst', 'RRst', 'RLst']

###################################################
# ROIs definition
##################################################

#Inverse individual evoked data into the source space, and nomalize them
for evt_st in evt_list:
    evo_list = glob.glob(subjects_dir+'/*/MEG/*evtW_%s_bc-ave.fif' %evt_st)
    apply_inverse_ave(evo_list, event=evt_st, baseline=True)
    fn_stcs = glob.glob(subjects_dir + '/fsaverage/dSPM_ROIs/*/*fibp1-45,evtW_%s_bc-lh.stc' %evt_st)
    apply_norm(fn_stcs, event=evt_st, ref_event=evt_st)
#Average all the evoked STCs to form the common STC    
stcs_path = subjects_dir+'/fsaverage/dSPM_ROIs/'
apply_comSTC(stcs_path, evt_list)
#Concentrate the avaliable interest anatomical labels from Freesurfer. The initial
#labels are stored in path '*/fsaverage/dSPM_ROIs/anno_ROIs/ini', they are named
#by the cortical regions they belong. 'ana_label_list' contains the indices of 
#each anatomical labels.
fn_avg = subjects_dir+'/fsaverage/dSPM_ROIs/common-lh.stc'
ana_list_file = subjects_dir+'/fsaverage/dSPM_ROIs/anno_ROIs/ana_label_list.txt'
focal_ROIs(fn_avg, radius=10.0, tmin=0.0, tmax=0.6, fn_ana_list=ana_list_file)
#Merge the overlapped ROIs
labels_path = subjects_dir+'/fsaverage/dSPM_ROIs/anno_ROIs'
apply_merge(labels_path)
'''
   If want to view the coordinates of each ROI‘s centroid， you can run function
   ’plot_coor.py'. 
'''

###################################################
# Causality analysis
##################################################
from Causality_pipeline import apply_inverse_epo, cal_labelts, normalize_data,\
                               model_estimation, causal_analysis, sig_thresh,\
                               group_causality, diff_mat
method = 'MNE'
sfreq=678
freqs = [(8, 12), (30, 40)]

for evt_st in evt_list:
    #Inverse epochs into the source space
    fn_epo = glob.glob(subjects_dir+'/*/MEG/*evtW_%s_bc-epo.fif' %evt_st)
    apply_inverse_epo(fn_epo,method=method, event=evt_st)
    #Calculate the representative STCs(rSTCs) for each ROI.
    stcs_path = glob.glob(subjects_dir+'/fsaverage/stcs/*[0-9]/%s/' %evt_st)
    cal_labelts(stcs_path, condition=evt_st, min_subject='fsaverage')
    #Normalize rSTCs
    fn_ts = glob.glob(subjects_dir+'/fsaverage/stcs/*[0-9]/%s_labels_ts.npy' %evt_st)
    normalize_data(fn_ts) 
    #MVAR model construction and evaluation, individual causality analysis for
    #each condition 
    fn_norm = glob.glob(subjects_dir+'/fsaverage/stcs/*[0-9]/%s_labels_ts,1-norm.npy' %evt_st)
    model_estimation(fn_norm)
    causal_analysis(fn_norm)
    #Estimate the significance of each causal matrix.
    fn_cau = glob.glob(subjects_dir+'/fsaverage/stcs/*[0-9]/%s_labels_ts,1-norm,cau.npy' %evt_st)
    sig_thresh(cau_list=fn_cau, condition=evt_st, sfreq=sfreq, freqs=freqs)
    #Group causality analysis 
    fn_sig = glob.glob(subjects_dir+'/fsaverage/stcs/*[0-9]/sig_con/%s_sig_con_band.npy' %evt_st)
    group_causality(fn_sig, evt_st, freqs=freqs, submount=11)
    
#Difference between incongruent and congruent tasks
incon_list = ['LRst', 'RLst']
fmin, fmax = 8, 12
for incon_event in incon_list: 
    diff_mat(incon_event, fmin, fmax)