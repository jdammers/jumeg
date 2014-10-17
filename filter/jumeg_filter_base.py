import numpy as np
import time
'''
----------------------------------------------------------------------
--- JuMEG Filter_Base  Class    --------------------------------------
---------------------------------------------------------------------- 
 autor      : Frank Boers 
 email      : f.boers@fz-juelich.de
 last update: 30.09.2014
 version    : 0.0314
---------------------------------------------------------------------- 
 Window sinc filter taken from:
 The Scientist and Engineer's Guide to Digital Signal Processing
 By Steven W. Smith, Ph.D.
 Chapter 16: Window-Sinc Filter
 http://www.dspguide.com/ch16.htm
----------------------------------------------------------------------
 Butterworth filter design from  KD
---------------------------------------------------------------------- 

----------------------------------------------------------------------

'''

class JuMEG_Filter_Base(object):
     def __init__ (self):
        self._jumeg_filter_base_version   = 0.0314
         
        self._sampling_frequency          = 1017.25
        self._filter_type                 = "bp" #lp,hp,bp,notch
        self._filter_method               = None
        
        self._fcut1                       = 1.0
        self._fcut2                       = 45.0
         
#---
        self._filter_notch                = np.array([])
        self._filter_notch_width          = 1.0
         
        self._settling_time_factor        = 5.0
        self._filter_kernel_length_factor = 16.0
#---
        self._filter_attenuation_factor   = 1  # 1, 2
        self._filter_window               = 'blackmann' # blackmann, kaiser, hamming
        self._kaiser_beta                 = 16 #  for kaiser window  
#---   flags    
        self._remove_dcoffset             = True
        self._filter_kernel_isinit        = False
        self._verbose                     = False
#---         
        
#---   data      
        self._data                        = np.array([])
        self._data_mean                   = 0.0
        self._data_plane_isinit           = False
        self._data_plane_data_in          = np.array([])
        
         
#--- version
     def _get_version(self):  
         return self._jumeg_filter_base_version
       
     version = property(_get_version)

#--- verbose    
     def _set_verbose(self,value):
         self._verbose = value

     def _get_verbose(self):
         return self._verbose
       
     verbose = property(_get_verbose, _set_verbose)
     
#--- filter method bw,ws,mne
     def _get_filter_method(self):
         return self._filter_method
     
     filter_method = property(_get_filter_method)
     
#--- filter_kernel_isinit check for call init fct.
     def _set_filter_kernel_isinit(self,value):
        self._filter_kernel_isinit = value
         
     def _get_filter_kernel_isinit(self):
        return self._filter_kernel_isinit
        
     filter_kernel_isinit = property(_get_filter_kernel_isinit, _set_filter_kernel_isinit)
 
#--- data_plane_isinit check for call init fct.
     def _set_data_plane_isinit(self,value):
        self._data_plane_isinit = value
         
     def _get_data_plane_isinit(self):
        return self._data_plane_isinit
        
     data_plane_isinit = property(_get_data_plane_isinit, _set_data_plane_isinit)
                          
#--- remove dcoffset    
     def _set_remove_dcoffset(self,value):
         self._remove_dcoffset = value

     def _get_remove_dcoffset(self):
         return self._remove_dcoffset
       
     remove_dcoffset = property(_get_remove_dcoffset, _set_remove_dcoffset)
                       
#---  settling_time_factor   
     def _set_settling_time_factor(self,value):
         self.filter_kernel_isinit  = False
         self._settling_time_factor = value
      
         
     def _get_settling_time_factor(self):
         return self._settling_time_factor

     settling_time_factor = property(_get_settling_time_factor,_set_settling_time_factor)

#--- sampling_frequency  
     def _set_sampling_frequency(self, value):
         self._sampling_frequency = value
         self.filter_kernel_isinit  = False

     def _get_sampling_frequency(self):
         return self._sampling_frequency

     sampling_frequency = property(_get_sampling_frequency,_set_sampling_frequency)

#--- alias  MNE sfreq sampling_frquency
     sfreq = property(_get_sampling_frequency,_set_sampling_frequency)

#--- alias  MEG srate sampling_frquency
     srate = property(_get_sampling_frequency,_set_sampling_frequency)

#--- filter_kernel_length_factor    
     def _set_filter_kernel_length_factor(self, value):
         self._filter_kernel_length_factor = value
         self.filter_kernel_isinit  = False
         
     def _get_filter_kernel_length_factor(self):
         return self._filter_kernel_length_factor

     filter_kernel_length_factor = property(_get_filter_kernel_length_factor,_set_filter_kernel_length_factor)

#--- filter_kernel_length
     def _get_filter_kernel_length(self):
         try:
            self._filter_kernel_length = self.sampling_frequency * self.filter_kernel_length_factor
         finally: 
               return self._filter_kernel_length

     filter_kernel_length = property(_get_filter_kernel_length)

#--- filter_type    
     def _set_filter_type(self,value):
         self._filter_type = value
         self.filter_kernel_isinit  = False
     def _get_filter_type(self):
         return self._filter_type
       
     filter_type = property(_get_filter_type, _set_filter_type)

#--- fcut1    
     def _set_fcut1(self,value):
         self._fcut1 = value
         self.filter_kernel_isinit  = False
         
     def _get_fcut1(self):
         return self._fcut1
    
     fcut1 = property(_get_fcut1, _set_fcut1)

#--- fcut2    
     def _set_fcut2(self,value):
         self._fcut2 = value
         self.filter_kernel_isinit  = False
         
     def _get_fcut2(self):
         return self._fcut2
    
     fcut2 = property(_get_fcut2, _set_fcut2)

#--- filter_window   
     def _set_filter_window(self,value):
         self._filter_window = value
         self.filter_kernel_isinit = False
 
     def _get_filter_window(self):
         return self._filter_window
   
     filter_window = property(_get_filter_window,_set_filter_window)

#--- kaiser_beta beat value for kaiser window e.g. 8.6 9.5 14 ...    
     def _set_kaiser_beta(self,value):
         self._kaiser_beta=value
       
     def _get_kaiser_beta(self):
         return self._kaiser_beta
       
     kaiser_beta = property(_get_kaiser_beta,_set_kaiser_beta)
     
#--- filter_kernel_data   
     def _set_filter_kernel_data(self,d):
         self._filter_kernel_data = d
     
     def _get_filter_kernel_data(self):
         return self._filter_kernel_data
   
     filter_kernel_data = property(_get_filter_kernel_data,_set_filter_kernel_data)

#---- filter_data_length
     def calc_filter_data_length(self,dl):
         self._filter_data_length = self.calc_zero_padding( dl + self.filter_kernel_data.size * self.settling_time_factor * 2 +1)
         return self._filter_data_length 

     def _get_filter_data_length(self):
         return self._filter_data_length  
    
     filter_data_length = property(_get_filter_data_length)

#---- notch
     def _set_filter_notch (self,value):
         self._filter_notch = value
         self.filter_kernel_isinit = False
         
     def _get_filter_notch(self):
         return self._filter_notch

     filter_notch = property(_get_filter_notch,_set_filter_notch)
     notch        = property(_get_filter_notch,_set_filter_notch)
   
#---- notch width e.g. window sinc
     def _set_filter_notch_width (self,value):
         self._filter_notch_width = value

     def _get_filter_notch_width(self):
         return self._filter_notch_width

     filter_notch_width = property(_get_filter_notch_width,_set_filter_notch_width)

#--- data_mean    
     def _set_data_mean(self,value):
         self._data_mean = value
    
     def _get_data_mean(self):
         return self._data_mean
    
     data_mean = property(_get_data_mean, _set_data_mean)

#--- data_std    
     def _set_data_std(self,value):
         self._data_std = value
    
     def _get_data_std(self):
         return self._data_std
    
     data_std = property(_get_data_std, _set_data_std)

#--- data_plane dummy data container to filter data
     def _set_data_plane(self,value):
         self._data_plane = value
    
     def _get_data_plane(self):
         return self._data_plane
    
     data_plane = property(_get_data_plane, _set_data_plane)    
    
#--- data_plane_in dummy data container to filter data
     def _set_data_plane_data_in(self,value):
         self._data_plane_data_in = value
    
     def _get_data_plane_data_in(self):
         return self._data_plane_data_in
    
     data_plane_data_in = property(_get_data_plane_data_in, _set_data_plane_data_in)
 
#--- data_plane_data_in_lenght 
     def _get_data_plane_data_in_lenght(self):
         return self.data_plane_data_in.shape[-1]
         
     data_plane_data_in_length = property(_get_data_plane_data_in_lenght)
    
#--- data_plane_out dummy data container to filter data
     def _set_data_plane_data_out(self,value):
         self._data_plane_data_out = value
    
     def _get_data_plane_data_out(self):
         return self._data_plane_data_out
    
     data_plane_data_out = property(_get_data_plane_data_out, _set_data_plane_data_out)

#--- data_plane_pre dummy data container reduce filter onset artefact
     def _set_data_plane_data_pre(self,value):
         self._data_plane_data_pre = value
    
     def _get_data_plane_data_pre(self):
         return self._data_plane_data_pre
    
     data_plane_data_pre = property(_get_data_plane_data_pre, _set_data_plane_data_pre)

#--- data_plane_post dummy data container reduce filter offset artefact
     def _set_data_plane_data_post(self,value):
         self._data_plane_data_post = value
    
     def _get_data_plane_data_post(self):
         return self._data_plane_data_post
    
     data_plane_data_post = property(_get_data_plane_data_post, _set_data_plane_data_post)

#--- data to filter
     def _set_data(self,value):
         self.filter_kernel_isinit = False
         self.data_plane_isinit    = False
         self._data = value

     def _get_data(self):
         return self._data
         
     data = property(_get_data,_set_data)
         
#--- data_length
     def _get_data_length(self):
         return self.data.shape[-1]
         
     data_length = property(_get_data_length)    
 
#--- data_plane_cplx 
     def _get_data_plane_cplx(self):
         return self._data_plane_cplx
     
     def _set_data_plane_cplx(self,value):
         self._data_plane_cplx = value
         
     data_plane_cplx = property(_get_data_plane_cplx,_set_data_plane_cplx)            
    
#--- filter_name_postfix
     def _get_filter_name_postfix(self):
         """return string with filter parameters for file name postfix"""
         self._filter_name_extention  = "fi" + self.filter_type 
         
         if self.filter_type == 'bp' :
              self._filter_name_extention += "%d-%d" % (self.fcut1,self.fcut2)
         else:
              self._filter_name_extention += "%d" % (self.fcut1)

         #if ( self.filter_attenuation_factor != 1 ):
         #     self._filter_name_extention += "att%d" % (self.filter_attenuation_factor)
        
         if self.filter_notch.size :
              self._filter_name_extention += "n"
         
         #if ( self.dcoffset ):
         #    self._filter_name_extention += "o"
         
         return self._filter_name_extention
        
     filter_name_postfix = property(_get_filter_name_postfix)    

#--- filter_info
     def _get_filter_info_string(self):
         """return info string with filter parameters """ 
         self._filter_info_string = self.filter_method +" ---> "+self.filter_type
         
         if self.filter_type == 'bp ' :
            self._filter_info_string += "%0.1f-%0.1f Hz" % (self.fcut1,self.fcut2)
         else:
            self._filter_info_string += "%0.1f Hz" % (self.fcut1)

         if self.filter_notch.size :
            self._filter_info_string += ",apply notch"
         
         if ( self.remove_dcoffset ):
          self._filter_info_string +=",remove DC offset"
         
         return self._filter_info_string
     
     filter_info = property(_get_filter_info_string)  
 
#---------------------------------------------------------# 
#--- calc_notches                -------------------------#
#---------------------------------------------------------# 
     def calc_notches(self,nfreq,nfreq_max=0.0):
         """
             return numpy array with notches and their harmoincs
             
             input: notch frequency , maximum harmonic frequency
         """
         
         if self.sampling_frequency == None :
             srate = 1000
         else :
             srate = self.sampling_frequency
             
         if  not nfreq_max :
            nfreq_max = srate/ 2.5 -1
            
         self.filter_notch = np.array([])
         self.filter_notch = np.arange(nfreq,nfreq_max+1,nfreq) 
  
         return self.filter_notch  


#---------------------------------------------------------# 
#--- calc_zero_padding           -------------------------#
#---------------------------------------------------------# 
     def calc_zero_padding(self,nr):
         """
             return number of elements to get padded
             
             input nr => e.g. 10 out=>16
         """
         
         return 2 ** ( np.ceil( np.log(nr) / np.log(2) ) )  

#---------------------------------------------------------# 
#---- calc_data_mean_std         -------------------------#
#---------------------------------------------------------# 
     def calc_data_mean_std(self,data):
         """
            return mean and std from data
           
            input: data => signal of one channel
         """  
         
         self.data_mean = np.mean(data)
         d              =  data - self.data_mean 
         # variance      = np.dot(d,d)/data.size
         self.data_std  = np.sqrt( np.dot(d,d)/data.size ) # speed up np.std      

         return (self.data_mean,self.data_std)  

#---------------------------------------------------------# 
#---- calc_data_mean             -------------------------#
#---------------------------------------------------------# 
     def calc_data_mean(self,data):
         """
            return mean from data use last axis
           
            input: data => signal of one channel
         """  
         
         self.data_mean = np.mean(data, axis = -1)
         #self.data_mean = np.mean(data)
        
         return (self.data_mean)  
         
#---------------------------------------------------------# 
#---- calc_remove_dcoffset       -------------------------#
#---------------------------------------------------------# 
     def calc_remove_dcoffset(self,data):
         """
            return data with zero mean
         
            input: data => signal of one channel
         """ 
         
         self.calc_data_mean(data)
         if self.data_mean.size == 1 :
              data -= self.data_mean
         else :
              data -= self.data_mean[:, np.newaxis] 
     
         return self.data_mean  

 
#---------------------------------------------------------# 
#--- init_filter                 -------------------------#
#---------------------------------------------------------# 
     def init_filter(self):  
         """
            setting complex filter function/kernel and dataplane in memory
           
            input: number_of_samples => will be padded 
         """    
         self.init_filter_kernel()
         self.init_filter_data_plane()
         
         if self.verbose :
            print "---> DONE init filter %s ---> calculating filter function/kernel and adjusting data plane" % (self.filter_type)
            
         return (self.filter_kernel_isinit and self.data_plane_isinit)

#---------------------------------------------------------# 
#--- filter_isinit               -------------------------#
#---------------------------------------------------------# 
     def filter_isinit(self):  
         """check if filter parameter and data plane is initialize"""
        
         if self.verbose :
             print"====== Check filter is init ================================="                 
             print"---> filter     is init: %r" % (self.filter_kernel_isinit)
             print"---> data plane is init: %r" % (self.data_plane_isinit) 
             res = ( self.filter_kernel_isinit and self.data_plane_isinit and ( self.data_length == self.data_plane_data_in_length )) 
             print"result is init         : %r" % (res) 
             print"---> size   data: %d  plane %d " % ( self.data_length, self.data_plane_data_in_length ) 
                        
         return ( self.filter_kernel_isinit and self.data_plane_isinit and (self.data_length == self.data_plane_data_in_length )) 

#---------------------------------------------------------# 
#--- apply_filter                -------------------------#
#---------------------------------------------------------# 
     def apply_filter(self,data, picks=None):
       """apply filter """
       
       self.data = data 
       
       if self.verbose :
           t0 = time.time()
           print"===> Start apply filter"
       
       if not( self.filter_isinit() ): 
            self.init_filter()
           
       if data.ndim > 1 :
           if picks == None :
              picks = np.arange( self.data.shape[0] )
           for ichan in picks:
               self.do_apply_filter( self.data[ichan,:] )
               if self.verbose :
                    print"===> ch %d" %(ichan)    
       else:
            self.do_apply_filter( data )  
       
       if self.verbose :
            print"===> Done apply filter %d" %( time.time() -t0 )

#---------------------------------------------------------# 
#--- reset                       -------------------------#
#---------------------------------------------------------# 
     def reset(self):
         self.filter_kernel_isinit = False
         self.data_plane_isinit    = False
         
         self._data_plane                   = None
         self._data_plane_data_in           = None
         self._data_plane_data_out          = None
         self._data_plane_pre               = None
         self._data_plane_post              = None
         self._data_plane_cplx              = None
         self._data                         = None
         
         self._filter_kernel_data_cplx      = None
         self._filter_kernel_data_cplx_sqrt = None
         self._filter_kernel_data           = None
         return 1
        
#---------------------------------------------------------# 
#--- destroy                     -------------------------#
#---------------------------------------------------------# 
     def destroy(self):
         self.filter_kernel_isinit = False
         self.data_plane_isinit    = False
         
         self._data_plane                   = None
         self._data_plane_data_in           = None
         self._data_plane_data_out          = None
         self._data_plane_pre               = None
         self._data_plane_post              = None
         self._data_plane_cplx              = None
         self._data                         = None
         
         self._filter_kernel_data_cplx      = None
         self._filter_kernel_data_cplx_sqrt = None
         self._filter_kernel_data           = None
         
         # d = np.array([])
         del self._data_plane
         del self._data_plane_data_in 
         del self._data_plane_data_out 
         del self._filter_kernel_data_cplx 
         del self._filter_kernel_data_cplx_sqrt 
         del self._filter_kernel_data 
         #print self.data_plane

         return 1
        
