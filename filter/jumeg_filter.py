import numpy as np

'''
----------------------------------------------------------------------
--- jumeg_filter          --------------------------------------------
---------------------------------------------------------------------- 
 autor      : Frank Boers 
 email      : f.boers@fz-juelich.de
 last update: 04.09.2014
 version    : 0.0213
---------------------------------------------------------------------- 
 Window Sinc Filter are taken from:
 The Scientist and Engineer's Guide to Digital Signal Processing
 By Steven W. Smith, Ph.D.
 Chapter 16: Window-Sinc Filter
 http://www.dspguide.com/ch16.htm
----------------------------------------------------------------------
 Butterworth filter design from  KD
----------------------------------------------------------------------
 Dependency:
  numpy
  scipy
----------------------------------------------------------------------
 How to use the jumeg filter
---------------------------------------------------------------------- 

from jumeg.filter.jumeg_filter import jumeg_filter

===> make a window sinc bp 1-45 Hz with dc offset correction
  filter_type = "bp"
  fcut1       = 1.0
  fcut2       = 45.0
  srate       = 1017.25
  fiws_bp = jumeg_filter( filter_method="ws",filter_type=filter_type,fcut1=fcut1, fcut2=fcut2, remove_dcoffset=True, sampling_frequency=srate)

---> apply changes to default parameter  
  fiws_bp.filter_window               = "hamming"
  fiws_bp.filter_kernel_length_factor = 8.0
  fiws_bp.settling_time_factor        = 3.0
  fiws_bp.sampling_frequency          = 1017.25
 
===> make a window sinc  hp 5 Hz
  filter_type = "hp"
  fcut1       =  5.0
  fiws_hp     = jumeg_filter( filter_method="ws",filter_type=filter_type,fcut1=fcut1,remove_dcoffset=True, sampling_frequency=srate)
  
===> make a window sinc  lp 25 Hz
  filter_type = "lp"
  fcut1       = 25.0
  fiws_lp     = jumeg_filter( filter_method="ws",filter_type=filter_type,fcut1=fcut1,remove_dcoffset=True, sampling_frequency=srate)
  
  
===> apply the filter  !!! works inplace;loop over all channels!!!
  data_bp = data_channel.copy()
  data_lp = data_channel.copy()
  data_hp = data_channel.copy()
 
  fiws_bp.apply_filter( data_bp )
  fiws_lp.apply_filter( data_lp )
  fiws_hp.apply_filter( data_hp )
 
   
===> make a butter  bp 1-45 Hz with dc offset correction and notches at 50,100,150,200 Hz
  filter_type = "bp"
  fcut1       = 1.0
  fcut2       = 45.0
  srate       = 1017.25
  
  notch_start = 50
  notch_end   = 200
  notch       = numpy.arange(notch_start,notch_end,notch_start) 
  
  fibw_bp = jumeg_filter( filter_method="bw",filter_type=filter_type,fcut1=fcut1, fcut2=fcut2, remove_dcoffset=True, sampling_frequency=srate,notch=notch)
  
  or instead of notch =  <np.array> 
  fibw_bp.calc_notches(50) # => 50,100 ... samplig frequency /2
  
  
  fibw_bp.apply_filter( data_channels )
  
----------------------------------------------------------------------
 ende gut alles gut

'''

def jumeg_filter(filter_method="bw",filter_type='bp', fcut1=1.0, fcut2=45.0, remove_dcoffset=True, sampling_frequency=1017.25, filter_window='blackmann', notch=np.array([]), notch_width=1.0, order=4):

    if filter_method.lower() == "bw"  :
       from jumeg.filter.jumeg_filter_bw import JuMEG_Filter_Bw
       return JuMEG_Filter_Bw(filter_type=filter_type,fcut1=fcut1, fcut2=fcut2, remove_dcoffset=remove_dcoffset, sampling_frequency=sampling_frequency,notch=notch, notch_width=notch_width,order=order)   	  
    else:
       from jumeg.filter.jumeg_filter_ws import JuMEG_Filter_Ws
       return JuMEG_Filter_Ws(filter_type=filter_type,fcut1=fcut1, fcut2=fcut2, remove_dcoffset=remove_dcoffset, sampling_frequency=sampling_frequency, filter_window=filter_window)
       #, notch=notch, notch_width=notch_width)  
                
  
