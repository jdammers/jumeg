import numpy as np
import mne
'''
----------------------------------------------------------------------
--- JuMEG Filter_MNE            --------------------------------------
---------------------------------------------------------------------- 
 autor      : Frank Boers 
 email      : f.boers@fz-juelich.de
 last update: 17.09.2014
 version    : 0.0113
---------------------------------------------------------------------- 
 jumeg oo filter interface to the MNE filter types
 mne.low_pass_filter
 mne.high_pass_filter
 mne.band_pass_filter
 mne.band_stop_filter
 mne.notch_filter
----------------------------------------------------------------------

----------------------------------------------------------------------

'''
from jumeg.filter.jumeg_filter_base import JuMEG_Filter_Base

class JuMEG_Filter_MNE(JuMEG_Filter_Base):
     def __init__ (self,filter_type='bp',njobs=4,fcut1=1.0,fcut2=200.0,remove_dcoffset=True,sampling_frequency=1017.25, 
                   mne_filter_method='fft',mne_filter_length='10s',trans_bandwith=0.5,notch=np.array([]),notch_width=None):
                   
         super(JuMEG_Filter_MNE, self).__init__()
         
         self._jumeg_filter_ws_version     = 0.0113
         
         self._mne_filter_method           = mne_filter_method  #fft
         self._mne_filter_length           = mne_filter_length
         self._mne_njobs                   = njobs
         self._mne_trans_bandwith          = trans_bandwith
         
         self._sampling_frequency          = sampling_frequency
         self._filter_type                 = filter_type #lp, bp, hp
         self._fcut1                       = fcut1
         self._fcut2                       = fcut2
#---
         self._filter_notch                = notch
         self._filter_notch_width          = notch_width

#---
#         self._filter_attenuation_factor   = 1  # 1, 2
#         self._filter_window               = filter_window # hamming, blackmann, kaiser
 
#---
#         self._filter_kernel_length_factor = kernel_length_factor
#         self._settling_time_factor        = settling_time_factor
#--               
         self._remove_dcoffset             = remove_dcoffset
         
#--- version
     def _get_version(self):  
         return self._jumeg_filter_ws_version
       
     version = property(_get_version)


#--- MNE filter values

#--- MNE number of parallel jobs    
     def _set_mne_njobs(self,value):
         self._mne_njobs = value
       
     def _get_mne_njobs(self):
         return self._mne_njobs
       
     mne_njobs = property(_get_mne_njobs,_set_mne_njobs)
     
#--- MNE filter_method  fft   
     def _set_mne_filter_method(self,value):
         self._mne_filter_method = value
       
     def _get_mne_filter_method(self):
         return self._mne_filter_method
    
     mne_filter_method = property(_get_mne_filter_method,_set_mne_filter_method)

#--- MNE filter_length e.g. '10s' None     
     def _set_mne_filter_length(self,value):
         self._mne_filter_length = value
       
     def _get_mne_filter_length(self):
         return self._mne_filter_length
    
     mne_filter_length = property(_get_mne_filter_length,_set_mne_filter_length)

#--- MNE trans_bandwith  0.5     
     def _set_mne_trans_bandwith(self,value):
         self._mne_trans_bandwith = value
       
     def _get_mne_trans_bandwith(self):
         return self._mne_trans_bandwith
    
     mne_trans_bandwith = property(_get_mne_trans_bandwith,_set_mne_trans_bandwith)

#---------------------------------------------------------# 
#--- apply_filter MNE           -------------------------#
#---------------------------------------------------------# 
     def apply_filter(self,data):
       """apply mne filter """
       
       self.data = data 
     
       dmean = self.calc_remove_dcoffset(data)
      
       Fs    = self.sampling_frequency
       njobs = self.mne_njobs        
       fcut1 = self.fcut1
       fcut2 = self.fcut2
       fl    = self.mne_filter_length
       tbw   = self.mne_trans_bandwith
       method = self.mne_filter_method
       v      = self.verbose

       if self.verbose :
          t0 = time.time()
          print"===> Start apply mne filter"
       
       if   self.filter_type =='lp' :
            mne.filter.low_pass_filter(data,Fs,fcut1,filter_length = fl,trans_bandwidth = tbw,method = method,
                                       iir_params = None,picks = None,n_jobs = njobs,copy = False,verbose = v)
                    
       elif self.filter_type =='hp' :
            mne.filter.high_pass_filter(data,Fs,fcut1,filter_length = fl,trans_bandwidth = tbw,method = method,
                                        iir_params = None,picks = None,n_jobs = njobs,copy = False,verbose = v)
      
       elif self.filter_type =='bp' :
            mne.filter.band_pass_filter(data,Fs,fcut1,fcut2,filter_length = fl, l_trans_bandwidth = tbw, h_trans_bandwidth = tbw,method = method,
                                        iir_params = None,picks = None,n_jobs = njobs,copy = False,verbose = v)
       
                             
       elif self.filter_type =='bs' :
            mne.filter.band_stop_filter(data,Fs,fcut1,fcut2,filter_length = fl,trans_bandwidth = tbw,method = method,
                                        iir_params = None,picks = None,n_jobs = njobs,copy = False,verbose = v)
    
       elif self.filter_type == 'notch' :      
            mne.filter.notch_filter(data,Fs,self.filter_notch,filter_length = fl,notch_widths = self.filter_notch_width,trans_bandwidth=1,method = method,
                                    iir_params = None,picks = None,n_jobs = njobs,copy = False,verbose = v,mt_bandwidth = None,p_value = 0.05)                       

#--- retain dc offset       
       if ( self.remove_dcoffset == False) : 
            if dmean.size == 1 :
                  data += dmean
            else :
                  data += dmean[:, np.newaxis]
             
       if self.verbose :
           print"===> Done apply mne filter %d" %( time.time() -t0 )
  
       return data

