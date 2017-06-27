import numpy as np

'''
----------------------------------------------------------------------
--- jumeg_filter          --------------------------------------------
---------------------------------------------------------------------- 
 autor      : Frank Boers 
 email      : f.boers@fz-juelich.de
 last update: 09.11.2016
 version    : 0.03142
---------------------------------------------------------------------- 
 Butterworth filter design from  KD,JD
 Oppenheim, Schafer, "Discrete-Time Signal Processing"
----------------------------------------------------------------------
 OBJ interface to MNE filter functions
----------------------------------------------------------------------
excluded:   
 Window Sinc Filter are taken from:
 The Scientist and Engineer's Guide to Digital Signal Processing
 By Steven W. Smith, Ph.D.
 Chapter 16: Window-Sinc Filter
 http://www.dspguide.com/ch16.htm
----------------------------------------------------------------------

 Dependency:
  numpy
  scipy
  mne
----------------------------------------------------------------------
 How to use the jumeg filter
---------------------------------------------------------------------- 

from jumeg.filter import jumeg_filter

#===> set some global values
ftype = "bp"
fcut1 =  1.0
fcut2 = 45.0
srate = 1017.25 # sampling rate in Hz

#---> make some notches
notch_start = 50
notch_end   = 200
notch       = np.arange(notch_start,notch_end+1,notch_start) 

#===> make an MNE FIR FFT filter, bp1-45 OOcall to the MNE filter class
fi_mne_bp = jumeg_filter(filter_method="mne",filter_type=ftype,fcut1=fcut1,fcut2=fcut2,remove_dcoffset=True,sampling_frequency=srate,notch=notch)

#---> or apply notches from 50 100 150 250 300 ... 450 
fi_mne_bp.calc_notches(50)

#---> or apply notches from 50 100 150
fi_mne_bp.calc_notches(50,150)

#---> apply filter works inpalce !!!
fi_mne_bp.apply_filter(raw._data,picks)


#===> make a butter bp1-45 Hz with dc offset correction and notches at 50,100,150,200 Hz
fi_bw_bp = jumeg_filter( filter_method="bw",filter_type=ftype,fcut1=fcut1,fcut2=fcut2,remove_dcoffset=True,sampling_frequency=srate,notch=notch)

#---> or apply notches from 50 100 150 250 300 ... 450 
fi_bw_bp.calc_notches(50)

#---> apply filter works inpalce !!!   
fi_bw_bp.apply_filter(raw._data,picks)


#=== make some filter objects
fi_bw_obj = []
for i in range(0,2):
  fi_bw_obj.append = jumeg_filter( filter_method="bw",filter_type=ftype,fcut1=fcut1,fcut2=fcut2,remove_dcoffset=True,sampling_frequency=srate,notch=notch)

#---> change the Obj filter parameter
#- obj1 => low-pass 35Hz
fi_bw_obj[0].fcut1      = 35.0
fi_bw_obj[0].filter_type='lp'

#- obj2 => high-pass 10Hz
fi_bw_obj[1].fcut1      = 10.0
fi_bw_obj[1].filter_type='hp'

#- obj3 => band-pass 10-30Hz
fi_bw_obj[2].fcut1      = 10.0
fi_bw_obj[2].fcut2      = 30.0
fi_bw_obj[3].filter_type='bp'

#--->apply the filter to your data , !!! works inplace !!!
for i in range(0,2):
    fi_bw_obj[i].apply_filter(data[i,:])

#--->finaly get the obj related filter name postfix e.g. to save the filterd data
fi_bw_obj[0].filter_name_postfix
filp35Hz
#----------------------------------------------------------------------
 ende gut alles gut

'''

def jumeg_filter(filter_method="bw",filter_type='bp',fcut1=1.0,fcut2=45.0,remove_dcoffset=True,sampling_frequency=1017.25,
                 filter_window='blackmann',notch=np.array([]),notch_width=1.0,order=4,njobs=4,
                 mne_filter_method='fft',mne_filter_length='10s',trans_bandwith=0.5):
    
    if filter_method.lower() == "bw"  :
       from jumeg.filter.jumeg_filter_bw import JuMEG_Filter_Bw
       return JuMEG_Filter_Bw(filter_type=filter_type,fcut1=fcut1, fcut2=fcut2, remove_dcoffset=remove_dcoffset, sampling_frequency=sampling_frequency,notch=notch, notch_width=notch_width,order=order)   	  
    else : 
       from jumeg.filter.jumeg_filter_mne import JuMEG_Filter_MNE
       return JuMEG_Filter_MNE(filter_type=filter_type,njobs=njobs,fcut1=fcut1,fcut2=fcut2,remove_dcoffset=True,sampling_frequency=sampling_frequency,
                               mne_filter_method=mne_filter_method,mne_filter_length=mne_filter_length,trans_bandwith=trans_bandwith,notch=notch,notch_width=notch_width)
    #elif filter_method.lower() == "ws"  :
    #   from jumeg.filter.jumeg_filter_ws import JuMEG_Filter_Ws
    #   return JuMEG_Filter_Ws(filter_type=filter_type,fcut1=fcut1, fcut2=fcut2, remove_dcoffset=remove_dcoffset, sampling_frequency=sampling_frequency, filter_window=filter_window)
    #   #, notch=notch, notch_width=notch_width)
  
  
if __name__ == "__main__":
     jumeg_filter(**kwargv)
