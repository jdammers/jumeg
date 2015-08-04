import numpy as np
import mne
'''
----------------------------------------------------------------------
--- JuMEG Filter_MNE            --------------------------------------
---------------------------------------------------------------------- 
 autor      : Frank Boers 
 email      : f.boers@fz-juelich.de
 last update: 28.05.2015
 version    : 0.03141
---------------------------------------------------------------------- 
 jumeg obj filter interface to the MNE filter types
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
         
         self.__jumeg_filter_mne_version    = 0.03141
         self.__filter_method               = 'mne'
         
         self.__mne_filter_method           = mne_filter_method  #fft
         self.__mne_filter_length           = mne_filter_length
         self.__mne_njobs                   = njobs
         self.__mne_trans_bandwith          = trans_bandwith
         
         self.sampling_frequency          = sampling_frequency
         self.filter_type                 = filter_type #lp, hp, bp, br,bs, notch
         self.fcut1                       = fcut1
         self.fcut2                       = fcut2
#---
         self.filter_notch                = notch
         self.filter_notch_width          = notch_width

         self.remove_dcoffset             = remove_dcoffset


#--- filter method bw,ws,mne
     def __get_filter_method(self):
         return self.__filter_method
     filter_method = property(__get_filter_method)

#--- version
     def __get_version(self):  
         return self.__jumeg_filter_ws_version
       
     version = property(__get_version)

#--- MNE filter values

#--- MNE number of parallel jobs    
     def __set_mne_njobs(self,value):
         self.__mne_njobs = value
       
     def __get_mne_njobs(self):
         return self.__mne_njobs
       
     mne_njobs = property(__get_mne_njobs,__set_mne_njobs)
     
#--- MNE filter_method  fft   
     def __set_mne_filter_method(self,value):
         self.__mne_filter_method = value
       
     def __get_mne_filter_method(self):
         return self.__mne_filter_method
    
     mne_filter_method = property(__get_mne_filter_method,__set_mne_filter_method)

#--- MNE filter_length e.g. '10s' None     
     def __set_mne_filter_length(self,value):
         self.__mne_filter_length = value
       
     def __get_mne_filter_length(self):
         return self.__mne_filter_length
    
     mne_filter_length = property(__get_mne_filter_length,__set_mne_filter_length)

#--- MNE trans_bandwith  0.5     
     def __set_mne_trans_bandwith(self,value):
         self.__mne_trans_bandwith = value
       
     def __get_mne_trans_bandwith(self):
         return self.__mne_trans_bandwith
    
     mne_trans_bandwith = property(__get_mne_trans_bandwith,__set_mne_trans_bandwith)

#---------------------------------------------------------# 
#--- apply_filter MNE           -------------------------#
#---------------------------------------------------------# 
     def apply_filter(self,data,picks=None):
       """apply mne filter """
       
       if picks is None:
          picks = np.arange(data.shape[0])
  
       # self.data = data
       dmean  = self.calc_remove_dcoffset(data[picks, :])
       Fs     = self.sampling_frequency
       njobs  = self.mne_njobs        
       fcut1  = self.fcut1
       fcut2  = self.fcut2
       fl     = self.mne_filter_length
       tbw    = self.mne_trans_bandwith
       
       method = self.mne_filter_method
       v      = self.verbose

       if self.verbose :
          import time
          t0 = time.time()
          print"===> Start apply mne filter"
          
    #--- apply notches
       if (self.filter_notch.size) or (self.filter_type == 'notch'):
           data[:,: ] = mne.filter.notch_filter(data,Fs,self.filter_notch,filter_length = fl,notch_widths = self.filter_notch_width,trans_bandwidth=1,method = method,
                                    iir_params = None,picks = picks,n_jobs = njobs,copy = False,verbose = v,mt_bandwidth = None,p_value = 0.05) 
      
    #---- filter lp hp bp bs br  
       if   self.filter_type =='lp' :
            data[:,:] = mne.filter.low_pass_filter(data,Fs,fcut1,filter_length = fl,trans_bandwidth = tbw,method = method,
                                       iir_params = None,picks = picks,n_jobs = njobs,copy = False,verbose = v)
       
       elif self.filter_type =='hp' :
            data[:, :] = mne.filter.high_pass_filter(data,Fs,fcut1,filter_length = fl,trans_bandwidth = tbw,method = method,
                                        iir_params = None,picks = picks,n_jobs = njobs,copy = False,verbose = v)
            
       elif self.filter_type =='bp' :
            data[:,:] = mne.filter.band_pass_filter(data,Fs,fcut1,fcut2,filter_length = fl, l_trans_bandwidth = tbw, h_trans_bandwidth = tbw,
                                                    method = method,iir_params = None,picks = picks,n_jobs = njobs,copy = False,verbose = v)
                                     
       
       elif self.filter_type in ['bs','br'] :
            data[:,:] = mne.filter.band_stop_filter(data,Fs,fcut1,fcut2,filter_length = fl,trans_bandwidth = tbw,method = method,
                                        iir_params = None,picks = picks,n_jobs = njobs,copy = False,verbose = v)

                                    
    
      #data = data.astype(data_type_orig)
       
#--- retain dc offset       
      
       if ( self.remove_dcoffset == False) : 
            if dmean.size == 1 :
                  data[picks, :] += dmean
            else :
                  data[picks, :] += dmean[:, np.newaxis]
             
       if self.verbose :
           print"===> Done apply mne filter %d" %( time.time() -t0 )
  
       return data

