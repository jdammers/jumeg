'''
----------------------------------------------------------------------
--- JuMEG Filter_Base  Class    --------------------------------------
---------------------------------------------------------------------- 
 autor      : Frank Boers 
 email      : f.boers@fz-juelich.de
 update: 16.12.2014
 update: 21.12.2016
  --> add function calc_lowpass_value
 update: 17.12.2018
  --> filter_name_postfix use print() with format(),
      print fcut as float 0.001-200.0Hz

 version    : 0.031415
---------------------------------------------------------------------- 
 Window sinc filter taken from:
 The Scientist and Engineer's Guide to Digital Signal Processing
 By Steven W. Smith, Ph.D.
 Chapter 16: Window-Sinc Filter
 http://www.dspguide.com/ch16.htm
----------------------------------------------------------------------
 Butterworth filter design from  KD
---------------------------------------------------------------------- 
2019.04.03 update logger
----------------------------------------------------------------------

'''

import numpy as np
import time,logging

from jumeg.base.jumeg_base import JuMEG_Base_Basic

logger = logging.getLogger('jumeg')
__version__= '2019.05.14.001'

class JuMEG_Filter_Base(JuMEG_Base_Basic):
    """
    JuMEG Filter Basic Class for FIR FFT filtering
    """
    def __init__ (self):
        super(JuMEG_Filter_Base,self).__init__()  
        
        self.__sampling_frequency          = None # 678.17
        self.__default_sampling_frequency  = 678.17 # JuMEG 4D-srate
        
        self.__filter_type                 = "bp" #lp,hp,bp,notch
        self.__filter_method               = None

        self.__fcut1                       = 1.0
        self.__fcut2                       = 45.0
        self.__filter_order                = None 

        self.__filter_notch                = np.array([])
        self.__filter_notch_width          = 1.0
         
        self.___settling_time_factor        = 5.0
        self.__filter_kernel_length_factor = 16.0

        self.__filter_attenuation_factor   = 1  # 1, 2
        self.__filter_window               = 'blackmann' # blackmann, kaiser, hamming
        self.__kaiser_beta                 = 16 #  for kaiser window  
       #--- flags    
        self.__remove_dcoffset             = True
        self.__filter_kernel_isinit        = False       
       #--- data      
        self.__data                        = np.array([])
        self.__data_mean                   = 0.0
        self.__data_plane_isinit           = False
        self.__data_plane_data_in          = np.array([])

        self._lp_for_srate = {'678': 200.0, '1017': 400.0}
     
#--- version
    @property       
    def version(self): return __version__
     
#--- filter method bw,ws,mne
    @property       
    def filter_method(self): return self.__filter_method 
    
#--- filter_kernel_isinit check for call init fct.
    @property       
    def filter_kernel_isinit(self):   return self.__filter_kernel_isinit
    @filter_kernel_isinit.setter
    def filter_kernel_isinit(self,v): self.__filter_kernel_isinit = v 
    
#--- data_plane_isinit check for call init fct.
    @property       
    def data_plane_isinit(self):   return self.__data_plane_isinit
    @data_plane_isinit.setter
    def data_plane_isinit(self,v): self.__data_plane_isinit = v       
                   
#--- remove dcoffset 
    @property       
    def remove_dcoffset(self):   return self.__remove_dcoffset
    @remove_dcoffset.setter
    def remove_dcoffset(self,v): self.__remove_dcoffset = v
    
#---  settling_time_factor
    @property       
    def settling_time_factor(self): return self.__settling_time_factor
    @settling_time_factor.setter
    def settling_time_factor(self,v):
        self.filter_kernel_isinit  = False
        self.__settling_time_factor = v   
        
#--- sampling_frequency 
    @property       
    def sampling_frequency(self): return self.__sampling_frequency
    @sampling_frequency.setter
    def sampling_frequency(self,v):
        self.__sampling_frequency = v
        self.filter_kernel_isinit  = False

#--- default_sampling_frequency 
    @property       
    def default_sampling_frequency(self): return self.__default_sampling_frequency
        
#--- alias  MNE sfreq sampling_frquency
    @property
    def sfreq(self):   return self.__sampling_frequency
    @sfreq.setter
    def sfreq(self,v): self.__sampling_frequency=v     
    
#--- alias  MEG srate sampling_frquency
    @property
    def srate(self):   return self.__sampling_frequency
    @srate.setter
    def sfrate(self,v): self.__sampling_frequency=v     
    
#--- filter_kernel_length_factor
    @property       
    def filter_kernel_length_factor(self):  return self.__filter_kernel_length_factor
    @filter_kernel_length_factor.setter
    def filter_kernel_length_factor(self,v):
        self.__filter_kernel_length_factor = v
        self.filter_kernel_isinit  = False
        
#--- filter_kernel_length
    @property       
    def filter_kernel_length(self):
        try:
           self.__filter_kernel_length = np.ceil( self.sampling_frequency ) * self.filter_kernel_length_factor
        finally: 
           return self.__filter_kernel_length
       
#--- filter_type    
    @property       
    def filter_type(self):  return self.__filter_type
    @filter_type.setter
    def filter_type(self,v):
        self.__filter_type = v
        self.filter_kernel_isinit  = False
        
#--- fcut1    
    @property       
    def fcut1(self):  return self.__fcut1
    @fcut1.setter
    def fcut1(self,v):
        self.__fcut1 = v
        self.filter_kernel_isinit  = False
        
#--- fcut2    
    @property       
    def fcut2(self):  return self.__fcut2
    @fcut2.setter
    def fcut2(self,v):
        self.__fcut2 = v
        self.filter_kernel_isinit  = False
        
#--- filter_window   
    @property       
    def filter_window(self):  return self.__filter_window
    @filter_window.setter
    def filter_window(self,v):
        self.__filter_window = v
        self.filter_kernel_isinit = False
        
#--- kaiser_beta beat value for kaiser window e.g. 8.6 9.5 14 ...    
    @property       
    def kaiser_beta(self):       return self.__kaiser_beta
    @kaiser_beta.setter
    def kaiser_beta(self,v): self.__kaiser_beta=v  
    
#--- filter_kernel_data
    @property       
    def filter_kernel_data(self):   return self.__filter_kernel_data
    @filter_kernel_data.setter
    def filter_kernel_data(self,v): self.__filter_kernel_data = v

#---- filter_data_length
    @property
    def filter_data_length(self):  return self.__filter_data_length 
    
#---- notch
    @property
    def filter_notch(self):    return self.__filter_notch
    @filter_notch.setter
    def filter_notch(self,v):
        if isinstance(v,(np.ndarray)):
           self.__filter_notch = v
        elif v:
           self.__filter_notch = np.array([v],dtype=np.float)
        else:   
           self.__filter_notch = np.array([],dtype=np.float)        
        self.filter_kernel_isinit = False
    
    @property
    def notch(self):    return self.__filter_notch
    @notch.setter
    def notch(self,v): 
        self.filter_notch(v)

#---- notch width e.g. window sinc
    @property
    def filter_notch_width(self):    return self.__filter_notch_width
    @filter_notch_width.setter
    def filter_notch_width (self,v): self.__filter_notch_width = v
    
#--- data_mean    
    @property
    def data_mean(self): return self.__data_mean
    @data_mean.setter
    def data_mean(self,v):   self.__data_mean = v
    
#--- data_std    
    @property
    def data_std(self):   return self.__data_std
    @data_std.setter
    def data_std(self,v): self.__data_std = v    
    
#--- data_plane dummy data container to filter data
    @property
    def data_plane(self):   return self.__data_plane
    @data_plane.setter
    def data_plane(self,v): self.__data_plane = v    
    
#--- data_plane_in dummy data container to filter data
    @property
    def data_plane_data_in(self):   return self.__data_plane_data_in
    @data_plane_data_in.setter
    def data_plane_data_in(self,v): self.__data_plane_data_in = v 
    
#--- data_plane_data_in_length
    @property
    def data_plane_data_in_length(self): return self.data_plane_data_in.shape[-1]  
    
#--- data_plane_out dummy data container to filter data
    @property
    def data_plane_data_out(self):   return self.__data_plane_data_out
    @data_plane_data_out.setter   
    def data_plane_data_out(self,v): self.__data_plane_data_out = v   
    
#--- data_plane_pre dummy data container reduce filter onset artefact
    @property
    def data_plane_data_pre(self):   return self.__data_plane_data_pre
    @data_plane_data_pre.setter
    def data_plane_data_pre(self,v): self.__data_plane_data_pre = v    
    
#--- data_plane_post dummy data container reduce filter offset artefact
    @property
    def data_plane_data_post(self):   return self.__data_plane_data_post
    @data_plane_data_post.setter
    def data_plane_data_post(self,v): self.__data_plane_data_post = v
    
#--- data to filter
    @property
    def data(self):        return self.__data
    @data.setter
    def data(self,v):
        self.filter_kernel_isinit = False
        self.data_plane_isinit    = False
        self.__data = v
        
#--- data_length
    @property
    def data_length(self): return self.data.shape[-1]
 
#--- data_plane_cplx 
    @property
    def data_plane_cplx(self):    return self.__data_plane_cplx
    @data_plane_cplx.setter 
    def data_plane_cplx(self,v): self.__data_plane_cplx = v

#--- filter_name_postfix
    @property
    def filter_name_postfix(self):
        """return string with filter parameters for file name postfix"""

        self.__filter_name_extension = "fi" + self.filter_type
        if self.filter_type == 'bp':
            self.__filter_name_extension += "{}".format(self.fcut1).rstrip("0").rstrip(".")
            self.__filter_name_extension += "-{}".format(self.fcut2).rstrip("0").rstrip(".")
           
        elif self.filter_type != 'notch':
            self.__filter_name_extension += "{}".format(self.fcut1).rstrip("0").rstrip(".")

        if self.filter_notch.size:
            self.__filter_name_extension += "n%d" % (self.filter_notch.size)

        return self.__filter_name_extension

#--- filter_info
    @property
    def filter_info(self):
        """return info string with filter parameters """ 
        self.__filter_info_string = self.filter_method +" ---> "+ self.filter_type

        if self.filter_type == 'bp' :
           self.__filter_info_string += "{}-{} Hz".format(self.fcut1,self.fcut2)
        elif self.filter_type != "notch":
           self.__filter_info_string += "{} Hz".format(self.fcut1)
       
        if self.filter_notch.size :
           self.__filter_info_string += ",apply notch"
           print( self.filter_notch)
         
        if ( self.remove_dcoffset ):
           self.__filter_info_string +=",remove DC offset"
         
        return self.__filter_info_string
     
#--- filter_info short string
    @property 
    def filter_info_short_string(self):
        """return info string with filter parameters """ 
        self.__filter_info_short_string = self.filter_method +"/"+ self.filter_type+"/"

        if self.filter_type == 'bp' :
           self.__filter_info_short_string += "%0.3f-%0.1f Hz/" % (self.fcut1,self.fcut2)
        else:
           self.__filter_info_short_string += "%0.3f Hz/" % (self.fcut1)
         
        if self.__filter_order:
           self.__filter_info_short_string += "O%d/" % ( self.__filter_order )
         
        if self.filter_notch.size :
           self.__filter_info_short_string += "n/"
         
        if ( self.remove_dcoffset ):
           self.__filter_info_short_string +="DC/"
         
        return self.__filter_info_short_string
         
    def update_info_filter_settings(self,raw):
        """ 
        update raw info filter settings low-,highpass
        store
         
        Parameters
        -----------
         raw obj
        """
        if self.filter_type == 'bp' :
          if raw.info.get('highwpass'):
             raw.info['highpass'] = self.fcut2
          else:
             raw.info['highpass']=self.fcut2
              
        if self.filter_type in ('lp','bp') :
           if raw.info.get('lowpass'):
              raw.info['lowpass'] =self.fcut1
           else:
              raw.info['lowpass']=self.fcut1
        elif self.filter_type == 'hp' :
             if raw.info.get('highpass'):
                raw.info['highpass'] += self.fcut1
             else:
                raw.info['highpass']=self.fcut1
       

    def calc_lowpass_value(self,sf):
        """
         extimates fcut1 for lowpass with respect 4D aquisition settings: sampling-rates & bandwidth
         678  -> 200.0
         1017 -> 400.0
          
        Parameters
        -----------
         sf: sampling frequency default obj.sampling_frequency
         
        Results
        -------
         value to set the fcut1 for lp
         must be set in main pgr !!!
        """
        sfreq = np.int64( self.sampling_frequency )
        if sf:
           sfreq = np.int64(sf)
        if str(sfreq) in self._lp_for_srate:
           return self._lp_for_srate[str(sfreq)]
        else:
           return sfreq/3.0
   
    def calc_filter_data_length(self,dl):
        """ 
        calc length of data array to filter via fft
        with zero padding and settling-time-factor for start and end range
         
        Parameter
        ---------
         data lenth
         
        Result
        ------
         filter length
        """
        self.__filter_data_length = self.calc_zero_padding( dl + self.filter_kernel_data.size * self.settling_time_factor * 2 +1)
        return self.__filter_data_length 

    def calc_notches(self,nfreq,nfreq_max=0.0):
        """
        calc notch array
        
        if sampling rate is not defined use default srate
        for 4D-system (678.17 Hz)
        
        
        Parameters
        ----------
         nfreq    : notch frequency, string e.g 50,100 or list or np.array [50,100.0]
         nfreq_max: maximum harmonic notch frequency
                    if defined will extend the notches from first notch in steps of first notch to maximum 
                    default = <0.0>
       
        Results
        --------
         numpy array with notches and their harmonics
        
        Examples
        --------
        from jumeg.filter.jumeg_filter_base import JuMEG_Filter_Base as JFB
        jfb=JFB()
        jfb.verbose=True
        
        
        jfb.calc_notches("50")  
        >>calculate Notches: [50.]
        
        jfb.calc_notches("50,60,100")
        >>calculate Notches: [ 50.  60. 100.]
        
        jfb.calc_notches([50,60,70])
        >>calculate Notches: [50. 60. 70.]
        
        n=np.array([10,20,30])
        jfb.calc_notches(n)
        >>calculate Notches: [10 20 30]

        now use <nfreq_max> to extend notches in steps of first notch      
        jfb.calc_notches("50,60,111,150",nfreq_max=200)
        >>calculate Notches: [ 50.  60. 100. 111. 150. 200.]
        
        """
       
        if self.sampling_frequency is None :
           srate = self.__default_sampling_frequency # JuMEG 4D-srate
        else :
           srate = self.sampling_frequency
        
        notches           = np.array([])
        self.filter_notch = notches  
        
        if self.isNotEmpty( nfreq ): # ck is not empty string
           if isinstance( nfreq,(int,float,list,tuple) ):
              notches = np.array([nfreq],dtype=np.float)
           elif self.isNotEmptyString(nfreq): # string
                notches = np.array( [nfreq.replace(","," ").split() ],dtype=np.float)
        elif isinstance(nfreq,np.ndarray): # np.array
             notches = nfreq 
        else: return
           
        notches = notches.flatten() # just in case xdim
        self.filter_notch = notches
        
      #--- ck for max notches to calc 
        if nfreq_max: 
           notch_max=float(nfreq_max) 
           if notch_max and notches.size:
              nf_max = float(srate)/ 2.5 -1.0
          #--- ck max notch via srate lookup
              if notch_max > nf_max: notch_max = nf_max # not optimal
              if notch_max > notches[0]:
                 self.filter_notch = np.unique( np.append(notches,np.arange(notches[0],notch_max+1,notches[0])) )
       
        if self.verbose:
           logger.info("calculate Notches: {}".format(self.filter_notch))
           
        return self.filter_notch  
    
    def calc_zero_padding(self,nr):
        """
         return number of elements to get padded
         
         Parameter
         ---------
          int => e.g. 10 out=>16
         
         Result
         ------
          int
        """
        return 2 ** ( np.ceil( np.log(nr) / np.log(2) ) )  
 
    def calc_data_mean_std(self,data):
        """
        cal data mean
         
        Parameter
        ---------
         numpy array 1d
        
        Results
        --------
        mean and std
        """  
        self.data_mean = np.mean(data)
        d              =  data - self.data_mean 
        # variance      = np.dot(d,d)/data.size
        self.data_std  = np.sqrt( np.dot(d,d)/data.size ) # speed up np.std      

        return (self.data_mean,self.data_std)  

    def calc_data_mean(self,data):
        """
        calc mean from data use last axis
        
        Parameter
        ---------
         numpy array 1d
        
        Results
        --------
        mean and std
        """  
        self.data_mean = np.mean(data, axis = -1)
        return (self.data_mean)  
         
    def calc_remove_dcoffset(self,data):
        """
        remove dcoffset remove zero mean  
        
        Parameter
        ---------
         numpy array 1d
        
        Results
        --------
        data with zero mean
         
        """ 
        self.calc_data_mean(data)
        if self.data_mean.size == 1 :
           data -= self.data_mean
        else :
           data -= self.data_mean[:, np.newaxis] 
     
        return self.data_mean  

    
    def init_filter(self):  
         """
         setting complex filter function/kernel and dataplane in memory
         
         Result
         ------
         True/False
         """    
         self.init_filter_kernel()
         self.init_filter_data_plane()
         
         if self.verbose :
            logger.info( "---> DONE init filter %s ---> calculating filter function/kernel and adjusting data plane" % (self.filter_type) )
            
         return (self.filter_kernel_isinit and self.data_plane_isinit)

    def filter_isinit(self):  
        """
        check if filter parameter and data plane is initialize
        
        Result
        ------
         True/False
        """
        if self.verbose :
           msg=["---> Check if filter is init",
                "  -> filter     is init: %r" % (self.filter_kernel_isinit) ,
                "  -> data plane is init: %r" % (self.data_plane_isinit)]
           
           res = ( self.filter_kernel_isinit and self.data_plane_isinit and ( self.data_length == self.data_plane_data_in_length ))
           
           msg.append("  -> result     is init: %r" % (res))
           msg.append("  -> size data: %d plane %d "% ( self.data_length, self.data_plane_data_in_length ))
           logger.info( "\n".join(msg))
        return ( self.filter_kernel_isinit and self.data_plane_isinit and (self.data_length == self.data_plane_data_in_length )) 
  
    def apply_filter(self,data, picks=None):
       """
       apply filter
       
       Parameters
       ----------
       data as np.array [ch,timepoints]
       picks: np.array of data index to process <None>
       
       !!! filter works inplace!!!
       input data will be overwriten
       
       Results
       -------
       None
       """
       
       self.data = data 
     # from joblib import Parallel, delayed

       if self.verbose :
           t0 = time.time()
           self.line()
           logger.info("===> Start apply filter")
       
       if not( self.filter_isinit() ): self.init_filter()
           
       if data.ndim > 1 :
          if picks is None :
             picks = np.arange( self.data.shape[0] )
          for ichan in picks:
              self.do_apply_filter( self.data[ichan,:] )
              if self.verbose : 
                 logger.info("  => filter channel idx : {} dc correction: {}".format(ichan,self.remove_dcoffset) )
       else:
            self.do_apply_filter( data )  
       
       if self.verbose :
            logger.info(" ==> Done apply filter %d" %( time.time() -t0 ) )
            
            
    def reset(self):
        """ rest all """
        self.filter_kernel_isinit = False
        self.data_plane_isinit    = False
         
        self.__data_plane                   = None
        self.__data_plane_data_in           = None
        self.__data_plane_data_out          = None
        self.__data_plane_pre               = None
        self.__data_plane_post              = None
        self.__data_plane_cplx              = None
        self.__data                         = None
         
        self.__filter_kernel_data_cplx      = None
        self.__filter_kernel_data_cplx_sqrt = None
        self.__filter_kernel_data           = None
        return True

    
