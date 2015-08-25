import glob
from dirs_manage import reset_directory
from MNE_ROIs_Definition01 import apply_create_noise_covariance, apply_inverse
from MNE_ROIs_Definition01 import apply_rois, merge_rois, stan_rois, group_rois
evt_st, evt_rt = 'LLst', 'LLrt'
stmin, stmax = 0.0, 0.4
rtmin, rtmax = -0.4, 0.00 
method = 'dSPM'
#calculate noise cov from empty room file
#emp_list = glob.glob('/home/qdong/18subjects/*/MEG/*empty-raw.fif')
#apply_create_noise_covariance(emp_list)
#inverse epochs into the source space
epo_st_list = glob.glob('/home/qdong/18subjects/*/MEG/*evtW_%s_bc-epo.fif' %evt_st)
epo_rt_list = glob.glob('/home/qdong/18subjects/*/MEG/*evtW_%s_bc-epo.fif' %evt_rt)
apply_inverse(epo_st_list[:], method=method, event=evt_st)
apply_inverse(epo_rt_list[:], method=method, event=evt_rt)
#make ROIs for special event
stc_st_list = glob.glob('/home/qdong/18subjects/fsaverage/%s_ROIs/*/*,evtW_%s_bc-lh.stc' % (method, evt_st))
stc_rt_list = glob.glob('/home/qdong/18subjects/fsaverage/%s_ROIs/*/*,evtW_%s_bc-lh.stc' % (method, evt_rt))
apply_rois(stc_st_list, event=evt_st, tmin=stmin, tmax=stmax)
apply_rois(stc_rt_list, event=evt_rt, tmin=rtmin, tmax=rtmax)
#merge kinds of ROIs together for each subject
labels_path = glob.glob('/home/qdong/18subjects/fsaverage/%s_ROIs/*[0-9]' %method) 
merge_rois(labels_path, group=False, evelist=['LLst','LLrt'])
#standardize the size of ROIs and interegrate all the subjects ROIs
stan_path = '/home/qdong/18subjects/fsaverage/%s_ROIs/standard/' %method
reset_directory(stan_path)
stan_rois(stc_st_list, stan_path, size=8.0)
#merge ROIs across subjects, and select the common ROIs
labels_path = '/home/qdong/18subjects/fsaverage/%s_ROIs/' %method
merge_rois(labels_path, group=True)
mer_path = '/home/qdong/18subjects/fsaverage/%s_ROIs/merged/' %method
com_path = '/home/qdong/18subjects/fsaverage/%s_ROIs/common/' %method
group_rois(am_sub=8, com_path=com_path, mer_path=mer_path)
