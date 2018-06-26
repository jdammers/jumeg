import numpy as np
import time
'''
----------------------------------------------------------------------
--- JuMEG Filter_Base  Class    --------------------------------------
---------------------------------------------------------------------- 
 autor      : Frank Boers 
 email      : f.boers@fz-juelich.de
 update: 16.12.2014
 update: 21.12.2016
  --> add function calc_lowpass_value

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

----------------------------------------------------------------------

'''

class JuMEG_Filter_Base(object):
     def __init__ (self):
        self.__jumeg_filter_base_version   = 0.0314
         
        self.__sampling_frequency          = None # 678.17
        self.__filter_type                 = "bp" #lp,hp,bp,notch
        self.__filter_method               = None

        self.__fcut1                       = 1.0
        self.__fcut2                       = 45.0
        self.__filter_order                = None 
#---
        self.__filter_notch                = np.array([])
        self.__filter_notch_width          = 1.0
         
        self.___settling_time_factor        = 5.0
        self.__filter_kernel_length_factor = 16.0
#---
        self.__filter_attenuation_factor   = 1  # 1, 2
        self.__filter_window               = 'blackmann' # blackmann, kaiser, hamming
        self.__kaiser_beta                 = 16 #  for kaiser window  
#---   flags    
        self.__remove_dcoffset             = True
        self.__filter_kernel_isinit        = False
        self.__verbose                     = False
#---         
        
#---   data      
        self.__data                        = np.array([])
        self.__data_mean                   = 0.0
        self.__data_plane_isinit           = False
        self.__data_plane_data_in          = np.array([])

        self._lp_for_srate = {'678': 200.0, '1017': 400.0}

     def calc_lowpass_value(self,sf):
         '''
          extimates fcut1 for lowpass with respect 4D aquisition settings: sampling-rates & bandwidth
          678  -> 200.0
          1017 -> 400.0
         :param
          sf: sampling frequency default obj.sampling_frequency
         :return:
          value to set the fcut1 for lp
          must be set in main pgr !!!
         '''
         sfreq = np.int64( self.sampling_frequency )
         if sf:
            sfreq = np.int64(sf)
         if str(sfreq) in self._lp_for_srate:
            return self._lp_for_srate[str(sfreq)]
         else:
            return sfreq/3.0

#--- version
     def __get_version(self):
         return self.__jumeg_filter_base_version
       
     version = property(__get_version)

#--- verbose    
     def __set_verbose(self,value):
         self.__verbose = value

     def __get_verbose(self):
         return self.__verbose
       
     verbose = property(__get_verbose, __set_verbose)
     
#--- filter method bw,ws,mne
     def __get_filter_method(self):
         return self.__filter_method
     filter_method = property(__get_filter_method)
     
#--- filter_kernel_isinit check for call init fct.
     def __set_filter_kernel_isinit(self,value):
        self.__filter_kernel_isinit = value
         
     def __get_filter_kernel_isinit(self):
        return self.__filter_kernel_isinit
        
     filter_kernel_isinit = property(__get_filter_kernel_isinit, __set_filter_kernel_isinit)
 
#--- data_plane_isinit check for call init fct.
     def __set_data_plane_isinit(self,value):
        self.__data_plane_isinit = value
         
     def __get_data_plane_isinit(self):
        return self.__data_plane_isinit
        
     data_plane_isinit = property(__get_data_plane_isinit, __set_data_plane_isinit)
                          
#--- remove dcoffset    
     def __set_remove_dcoffset(self,value):
         self.__remove_dcoffset = value

     def __get_remove_dcoffset(self):
         return self.__remove_dcoffset
       
     remove_dcoffset = property(__get_remove_dcoffset, __set_remove_dcoffset)
                       
#---  settling_time_factor   
     def __set_settling_time_factor(self,value):
         self.filter_kernel_isinit  = False
         self.__settling_time_factor = value
      
         
     def __get_settling_time_factor(self):
         return self.__settling_time_factor

     settling_time_factor = property(__get_settling_time_factor,__set_settling_time_factor)

#--- sampling_frequency  
     def __set_sampling_frequency(self, value):
         self.__sampling_frequency = value
         self.filter_kernel_isinit  = False

     def __get_sampling_frequency(self):
         return self.__sampling_frequency

     sampling_frequency = property(__get_sampling_frequency,__set_sampling_frequency)

#--- alias  MNE sfreq sampling_frquency
     sfreq = property(__get_sampling_frequency,__set_sampling_frequency)

#--- alias  MEG srate sampling_frquency
     srate = property(__get_sampling_frequency,__set_sampling_frequency)

#--- filter_kernel_length_factor    
     def __set_filter_kernel_length_factor(self, value):
         self.__filter_kernel_length_factor = value
         self.filter_kernel_isinit  = False
         
     def __get_filter_kernel_length_factor(self):
         return self.__filter_kernel_length_factor

     filter_kernel_length_factor = property(__get_filter_kernel_length_factor,__set_filter_kernel_length_factor)

#--- filter_kernel_length
     def __get_filter_kernel_length(self):
         try:
            self.__filter_kernel_length = np.ceil( self.sampling_frequency ) * self.filter_kernel_length_factor
            
         finally: 
               return self.__filter_kernel_length

     filter_kernel_length = property(__get_filter_kernel_length)

#--- filter_type    
     def __set_filter_type(self,value):
         self.__filter_type = value
         self.filter_kernel_isinit  = False
     def __get_filter_type(self):
         return self.__filter_type
       
     filter_type = property(__get_filter_type, __set_filter_type)

#--- fcut1    
     def __set_fcut1(self,value):
         self.__fcut1 = value
         self.filter_kernel_isinit  = False
         
     def __get_fcut1(self):
         return self.__fcut1
    
     fcut1 = property(__get_fcut1, __set_fcut1)

#--- fcut2    
     def __set_fcut2(self,value):
         self.__fcut2 = value
         self.filter_kernel_isinit  = False
         
     def __get_fcut2(self):
         return self.__fcut2
    
     fcut2 = property(__get_fcut2, __set_fcut2)

#--- filter_window   
     def __set_filter_window(self,value):
         self.__filter_window = value
         self.filter_kernel_isinit = False
 
     def __get_filter_window(self):
         return self.__filter_window
   
     filter_window = property(__get_filter_window,__set_filter_window)

#--- kaiser_beta beat value for kaiser window e.g. 8.6 9.5 14 ...    
     def __set_kaiser_beta(self,value):
         self.__kaiser_beta=value
       
     def __get_kaiser_beta(self):
         return self.__kaiser_beta
       
     kaiser_beta = property(__get_kaiser_beta,__set_kaiser_beta)
     
#--- filter_kernel_data   
     def __set_filter_kernel_data(self,d):
         self.__filter_kernel_data = d
     
     def __get_filter_kernel_data(self):
         return self.__filter_kernel_data
   
     filter_kernel_data = property(__get_filter_kernel_data,__set_filter_kernel_data)

#---- filter_data_length
     def calc_filter_data_length(self,dl):
         self.__filter_data_length = self.calc_zero_padding( dl + self.filter_kernel_data.size * self.settling_time_factor * 2 +1)
         return self.__filter_data_length 

     def __get_filter_data_length(self):
         return self.__filter_data_length  
    
     filter_data_length = property(__get_filter_data_length)

#---- notch
     def __set_filter_notch (self,value):
         self.__filter_notch = np.array([])
         if isinstance(value, (np.ndarray)) :
           self.__filter_notch = value
         else :
            self.__filter_notch = np.array([value])
            
         self.filter_kernel_isinit = False
         
     def __get_filter_notch(self):
         return self.__filter_notch

     filter_notch = property(__get_filter_notch,__set_filter_notch)
     notch        = property(__get_filter_notch,__set_filter_notch)
   
#---- notch width e.g. window sinc
     def __set_filter_notch_width (self,value):
         self.__filter_notch_width = value

     def __get_filter_notch_width(self):
         return self.__filter_notch_width

     filter_notch_width = property(__get_filter_notch_width,__set_filter_notch_width)

#--- data_mean    
     def __set_data_mean(self,value):
         self.__data_mean = value
    
     def __get_data_mean(self):
         return self.__data_mean
    
     data_mean = property(__get_data_mean, __set_data_mean)

#--- data_std    
     def __set_data_std(self,value):
         self.__data_std = value
    
     def __get_data_std(self):
         return self.__data_std
    
     data_std = property(__get_data_std, __set_data_std)

#--- data_plane dummy data container to filter data
     def __set_data_plane(self,value):
         self.__data_plane = value
    
     def __get_data_plane(self):
         return self.__data_plane
    
     data_plane = property(__get_data_plane, __set_data_plane)
    
#--- data_plane_in dummy data container to filter data
     def __set_data_plane_data_in(self,value):
         self.__data_plane_data_in = value
    
     def __get_data_plane_data_in(self):
         return self.__data_plane_data_in
    
     data_plane_data_in = property(__get_data_plane_data_in, __set_data_plane_data_in)
 
#--- data_plane_data_in_lenght 
     def __get_data_plane_data_in_lenght(self):
         return self.data_plane_data_in.shape[-1]
         
     data_plane_data_in_length = property(__get_data_plane_data_in_lenght)
    
#--- data_plane_out dummy data container to filter data
     def __set_data_plane_data_out(self,value):
         self.__data_plane_data_out = value
    
     def __get_data_plane_data_out(self):
         return self.__data_plane_data_out
    
     data_plane_data_out = property(__get_data_plane_data_out, __set_data_plane_data_out)

#--- data_plane_pre dummy data container reduce filter onset artefact
     def __set_data_plane_data_pre(self,value):
         self.__data_plane_data_pre = value
    
     def __get_data_plane_data_pre(self):
         return self.__data_plane_data_pre
    
     data_plane_data_pre = property(__get_data_plane_data_pre, __set_data_plane_data_pre)

#--- data_plane_post dummy data container reduce filter offset artefact
     def __set_data_plane_data_post(self,value):
         self.__data_plane_data_post = value
    
     def __get_data_plane_data_post(self):
         return self.__data_plane_data_post
    
     data_plane_data_post = property(__get_data_plane_data_post, __set_data_plane_data_post)

#--- data to filter
     def __set_data(self,value):
         self.filter_kernel_isinit = False
         self.data_plane_isinit    = False
         self.__data = value

     def __get_data(self):
         return self.__data
         
     data = property(__get_data,__set_data)
         
#--- data_length
     def __get_data_length(self):
         return self.data.shape[-1]
         
     data_length = property(__get_data_length)
 
#--- data_plane_cplx 
     def __get_data_plane_cplx(self):
         return self.__data_plane_cplx
     
     def __set_data_plane_cplx(self,value):
         self.__data_plane_cplx = value
         
     data_plane_cplx = property(__get_data_plane_cplx,__set_data_plane_cplx)
    
#--- filter_name_postfix
     def __get_filter_name_postfix(self):
         """return string with filter parameters for file name postfix"""
         self.__filter_name_extention  = "fi" + self.filter_type 
         
         if self.filter_type == 'bp' :
              self.__filter_name_extention += "%d-%d" % (self.fcut1,self.fcut2)
         else:
              self.__filter_name_extention += "%d" % (self.fcut1)

         #if ( self.filter_attenuation_factor != 1 ):
         #     self.__filter_name_extention += "att%d" % (self.filter_attenuation_factor)
        
         if self.filter_notch.size :
              self.__filter_name_extention += "n"
         
         #if ( self.dcoffset ):
         #    self.__filter_name_extention += "o"
         
         return self.__filter_name_extention
        
     filter_name_postfix = property(__get_filter_name_postfix)

#--- filter_info
     def __get_filter_info_string(self):
         """return info string with filter parameters """ 
         self.__filter_info_string = self.filter_method +" ---> "+ self.filter_type


         if self.filter_type == 'bp' :
            self.__filter_info_string += "%0.3f-%0.1f Hz" % (self.fcut1,self.fcut2)
         else:
            self.__filter_info_string += "%0.3f Hz" % (self.fcut1)

         if self.filter_notch.size :
            self.__filter_info_string += ",apply notch"
            print self.filter_notch
         
         if ( self.remove_dcoffset ):
          self.__filter_info_string +=",remove DC offset"
         
         return self.__filter_info_string
     
     filter_info = property(__get_filter_info_string)
     
#--- filter_info short string
     def __get_filter_info_short_string(self):
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
     
     filter_info_short = property(__get_filter_info_short_string)
          
     def update_info_filter_settings(self,raw):
         """ update raw info filter settings low-,highpass
             input: raw obj
         """
         if self.filter_type == 'bp' :
            raw.info['lowpass'] = self.fcut1
            raw.info['highpass']= self.fcut2
         elif self.filter_type == 'lp' :
            raw.info['lowpass'] = self.fcut1
         elif self.filter_type == 'hp' :
            raw.info['highpass']= self.fcut1
           

#---------------------------------------------------------# 
#--- calc_notches                -------------------------#
#---------------------------------------------------------# 
     def calc_notches(self,nfreq,nfreq_max=0.0):
         """
             return numpy array with notches and their harmoincs
             
             input: notch frequency , maximum harmonic frequency
         """
         
         if self.sampling_frequency is None :
             srate = 678.17 # JuMEG 4D-srate
         else :
             srate = self.sampling_frequency

         freq_max_nyq = srate/ 2.5 -1

         if nfreq_max > freq_max_nyq :
            nfreq_max = freq_max_nyq
            
         self.filter_notch = np.array([])
         if nfreq_max :
            self.filter_notch = np.arange(nfreq,nfreq_max+1,nfreq)
         elif nfreq :
            self.filter_notch = np.array( [nfreq] )

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

      # from joblib import Parallel, delayed

       if self.verbose :
           t0 = time.time()
           print"===> Start apply filter"
       
       if not( self.filter_isinit() ): 
            self.init_filter()
           
       if data.ndim > 1 :
           if picks is None :
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
         return 1
        
#---------------------------------------------------------# 
#--- destroy                     -------------------------#
#---------------------------------------------------------# 
     def destroy(self):
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
         
         # d = np.array([])
         del self.__data_plane
         del self.__data_plane_data_in 
         del self.__data_plane_data_out 
         del self.__filter_kernel_data_cplx 
         del self.__filter_kernel_data_cplx_sqrt 
         del self.__filter_kernel_data 
         #print self.data_plane

         return 1
        
