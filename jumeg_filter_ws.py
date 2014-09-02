import numpy as np
'''
#----------------------------------------------------------------------# 
#--- Filter_WS             --------------------------------------------#
#----------------------------------------------------------------------# 
# autor      : Frank Boers 
# email      : f.boers@fz-juelich.de
# last update: 27.05.2014
# version    : 0.006
#----------------------------------------------------------------------# 
# Taken from:
# The Scientist and Engineer's Guide to Digital Signal Processing
# By Steven W. Smith, Ph.D.
# Chapter 16: Window-Sinc Filter
# http://www.dspguide.com/ch16.htm
#----------------------------------------------------------------------#
# the functions 
# calc_window_hamming, calc_window_blackmann and calc_sinc_filter_data
# are copied from
# https://github.com/
# mubeta06/python/blob/master/signal_processing/sp/firwin.py
#
#----------------------------------------------------------------------# 
#
# from filter_ws import Filter_WS
#
# fiws = Filter_WS()
#
# fiws = Filter_WS(filter_type='bp',fcut1=11.0,fcut2=35.0,dcoffset=True,sampling_frequency=1234)
#
#--- get/set parameter 
# fiws.sampling_frequency = 1017.25 # e.g. MEG 4D Neuroimaging
# fiws.filter_type        = 'bp'    # bp hp lp notch
# fiws.fcut1              = 1
# fiws.fcut2              = 45
# fiws.dcoffset           = True    # True for DC correction or False
#
#--- init functions 
#
# fiws.init_filter_kernel(signal.size)
# fiws.init_filter(signal.size)
#
#--- call to apply the filter
# 
#   fiws.apply_filter(signal)
#--> this works inplace => input data will be change
#
#----------------------------------------------------------------------#
# have fun
#
#'''


class Filter_WS:
     def __init__ (self,filter_type='bp', fcut1=1.0, fcut2=45.0, dcoffset=True, sampling_frequency=1017.25 ):
		 
         self._sampling_frequency          = sampling_frequency
         self._filter_type                 = filter_type #lp, bp, hp
         self._filter_attenuation_factor   = 1  # 1, 2
         self._filter_window               = 'blackmann' # hamming, blackmann
         self._fcut1                       = fcut1
         self._fcut2                       = fcut2
         self._dcoffset                    = dcoffset
#---
         self._filter_kernel_length_factor = 16.0
         self._settling_time_factor        = 5.0
         self._data_mean                   = 0.0
  


#--- dcoffset    
     def _set_dcoffset(self,value):
         self._dcoffset = value

     def _get_dcoffset(self):
         return self._dcoffset
       
     dcoffset = property(_get_dcoffset, _set_dcoffset)
                       
#---  settling_time_factor   
     def _set_settling_time_factor(self,value):
         self._settling_time_factor = value

     def _get_settling_time_factor(self):
         return self._settling_time_factor

     settling_time_factor = property(_get_settling_time_factor,_set_settling_time_factor)

#--- sampling_frequency  
     def _set_sampling_frequency(self, value):
         self._sampling_frequency = value

     def _get_sampling_frequency(self):
         return self._sampling_frequency

     sampling_frequency = property(_get_sampling_frequency,_set_sampling_frequency)

#--- filter_kernel_length_factor    
     def _set_filter_kernel_length_factor(self, value):
         self._filter_kernel_length_factor = value

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

     def _get_filter_type(self):
         return self._filter_type
       
     filter_type = property(_get_filter_type, _set_filter_type)

#--- fcut1    
     def _set_fcut1(self,value):
         self._fcut1 = value
    
     def _get_fcut1(self):
         return self._fcut1
    
     fcut1 = property(_get_fcut1, _set_fcut1)

#--- fcut2    
     def _set_fcut2(self,value):
         self._fcut2 = value
    
     def _get_fcut2(self):
         return self._fcut2
    
     fcut2 = property(_get_fcut2, _set_fcut2)

#--- filter_window   
     def _set_filter_window(self,value):
         self._filter_window = value
 
     def _get_filter_window(self):
         return self._filter_window
   
     filter_window = property(_get_filter_window,_set_filter_window)
     
#--- filter_kernel_data   
     def _set_filter_kernel_data(self,d):
         self._filter_kernel_data = d
     
     def _get_filter_kernel_data(self):
         return self._filter_kernel_data
   
     filter_kernel_data = property(_get_filter_kernel_data,_set_filter_kernel_data)

#--- filter_kernel_data_cplx     
     def _get_filter_kernel_data_cplx(self):
         return self._filter_kernel_data_cplx

     filter_kernel_data_cplx = property(_get_filter_kernel_data_cplx)

#---- filter_data_length
     def calc_filter_data_length(self,dl):
         self._filter_data_length = self.calc_zero_padding( dl + self.filter_kernel_data.size * self.settling_time_factor * 2 +1)
         return self._filter_data_length 

     def _get_filter_data_length(self):
         return self._filter_data_length  
    
     filter_data_length = property(_get_filter_data_length)

#---- filter_attenuation_factor
     def _set_filter_attenuation_factor(self,value):
         if value < 2 :
            self._filter_attenuation_factor = 1
         else :
            self._filter_attenuation_factor = 2
           
     def _get_filter_attenuation_factor(self):
         return self._filter_attenuation_factor 
  
     filter_attenuation_factor = property(_get_filter_attenuation_factor,_set_filter_attenuation_factor)
  
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
    
#--- data_plane_out dummy data container to filter data
     def _set_data_plane_data_out(self,value):
         self._data_plane_data_out = value
    
     def _get_data_plane_data_out(self):
         return self._data_plane_data_out
    
     data_plane_data_out = property(_get_data_plane_data_out, _set_data_plane_data_out)

#---------------------------------------------------------# 
#--- filter_name_extention       -------------------------#
#---------------------------------------------------------# 
     def filter_name_extention(self):
         """Return string with filter parameters for file name extention e.g. postfix."""
         dstr = "fiws" + self.filter_type + "%d-%dord%d" % (self.fcut1, self.fcut2,self.filter_kernel_length_factor)
         if ( self.filter_attenuation_factor != 1 ):
              dstr += "att%d" % (self.filter_attenuation_factor)
        
         #if ( self.notches ):
         #    dstr += "n%d" %( self.number_of_notches )
   
         if ( self.dcoffset ):
             dstr += "dc"
         print dstr
         return dstr
         
#---------------------------------------------------------# 
#--- window_hamming              -------------------------#
#---------------------------------------------------------# 
#---------------------------------------------------------# 
     def calc_window_hamming(self,M):
         """Return an M + 1 point symmetric hamming window."""
         if M%2:
                raise Exception('M must be even')
            
         return (0.54 - 0.46*np.cos(2*np.pi*np.arange( M + 1)/M))
         
#---------------------------------------------------------# 
#--- window_blackman             -------------------------#
#---------------------------------------------------------# 
     def calc_window_blackman(self,M):
         """Return an M + 1 point symmetric point hamming window."""
         if M%2:
                raise Exception('M must be even')
         return (0.42 - 0.5*np.cos(2*np.pi*np.arange(M + 1)/M) + 0.08*np.cos(4*np.pi*np.arange(M + 1)/M))
   
#---------------------------------------------------------# 
#--- calc_sinc_filter_data       -------------------------#
#---------------------------------------------------------#
     def calc_sinc_filter_data(self,M,fc):
         """Return an M + 1 point symmetric point sinc kernel with normalised cut-off frequency fc 0->0.5."""
         if M%2:
                raise Exception('M must be even')
         return np.sinc(2*fc*(np.arange(M + 1) - M/2))
   
#---------------------------------------------------------# 
#--- calc_sinc_filter_function       ---------------------#
#---------------------------------------------------------# 
     def calc_sinc_filter_function(self,M,fc):
         """Return normialized window sinc data; windowed with balckman or hamming; unity gain at DC """
         if self.filter_window == 'hamming' :
            d = self.calc_sinc_filter_data(M,fc) * self.calc_window_hamming(M)
         else:
	        d = self.calc_sinc_filter_data(M,fc) * self.calc_window_blackman(M)
    
         return d/d.sum()

#---------------------------------------------------------# 
#---- calc_data_mean_std         -------------------------#
#---------------------------------------------------------# 
     def calc_data_mean_std(self,data):
         """Return mean and std from data"""
         self.data_mean = np.mean(data)
         d =  data - self.data_mean 
         self.data_std  = np.sqrt( np.dot(d,d)/data.size ) # speed up np.std
         return (self.data_mean,self.data_std)  
         
#---------------------------------------------------------# 
#---- calc_remove_dcoffset       -------------------------#
#---------------------------------------------------------# 
     def calc_remove_dcoffset(self,data):
         """Return data with zero mean."""
         self.calc_data_mean_std(data);
         data -= self.data_mean # ptr to data
         return (self.data_mean,self.data_std)  

#---------------------------------------------------------# 
#--- calc_zero_padding           -------------------------#
#---------------------------------------------------------# 
     def calc_zero_padding(self,nr):
         """calc number of elements to get padde  array length: in=>10 out=>16"""
         return 2 ** ( np.ceil( np.log(nr) / np.log(2) ) )

#---------------------------------------------------------# 
#---  calc_filter_kernel         -------------------------#
#---------------------------------------------------------# 
     def calc_filter_kernel(self,M):
         """calc filter kernel type: bp,lp,hp,notch"""
         M       = int(M)
         Midx    = int( M / 2 );
         
         dfcut1  = self.fcut1 / self.sampling_frequency
         dfcut2  = self.fcut2 / self.sampling_frequency
                  
         #--- M has to bee even !!!
         self.filter_kernel_data       = np.zeros( M + 1 ,np.float64)
         
         if self.filter_type == 'lp':
           #print "lp"  
           self.filter_kernel_data[:] = self.calc_sinc_filter_function(M, dfcut1)
         elif self.filter_type == 'hp':
             #print "hp"
             if self.fcut1 is None:
                dfcut1= self.fcut2 / self.sampling_frequency

             self.filter_kernel_data -= self.calc_sinc_filter_function(M, dfcut1)
             self.filter_kernel_data[Midx] += 1.0
 
         elif self.filter_type == 'bp':
             #print "bp"
             if self.fcut1 > self.fcut2:
               dfcut2 = self.fcut1 / self.sampling_frequency
               dfcut1 = self.fcut2 / self.sampling_frequency
          #-- lp
             self.filter_kernel_data[:] = self.calc_sinc_filter_function(M,dfcut1)
          #-- hp
             fbp_fcut2       = np.zeros( M + 1 , np.float64)
             fbp_fcut2      -= self.calc_sinc_filter_function(M, dfcut2)
             fbp_fcut2[Midx] += 1.0 
          #--- bp
             self.filter_kernel_data       += fbp_fcut2
             self.filter_kernel_data       *=-1 # now make notch to bp => invert spectrum!!!
             self.filter_kernel_data[Midx] += 1 
      
         elif self.filter_type == 'notch':
             #print "notch"
             if self.fcut1 > self.fcut2:
               dfcut2 = self.fcut1 / self.sampling_frequency
               dfcut1 = self.fcut2 / self.sampling_frequency
          #--- lp
             self.filter_kernel_data[:] = self.calc_sinc_filter_function(M, dfcut1)
          #-- hp
             fbp_fcut2       = np.zeros( M + 1 , np.float64)
             fbp_fcut2[:]    = self.calc_sinc_filter_function(M, dfcut2)
             fbp_fcut2[Midx] += 1 
          #--- now make notch
             self.filter_kernel_data += fbp_fcut2

         return self.filter_kernel_data

#---------------------------------------------------------# 
#---  calc_filter_kernel_cplx    -------------------------#
#---------------------------------------------------------# 
     def calc_filter_kernel_cplx(self,data_size):
         self.calc_filter_data_length(data_size)
         fkd                                    = np.zeros( self.filter_data_length,np.float64 )
         fkd[0:self.filter_kernel_data.size ]   = self.filter_kernel_data.copy()   
         
        #--- factor 2 => -148 dB 
         if self.filter_attenuation_factor == 2 :
           fkd_cplx = np.fft.rfft( fkd )
           self.filter_kernel_data_cplx = fkd_cplx * fkd_cplx
         else :
        #--- factor 1 =>  -74 dB 
           self.filter_kernel_data_cplx = np.fft.rfft( fkd )
                     
         return self.filter_kernel_data_cplx

#---------------------------------------------------------# 
#--- init_filter_kernel          -------------------------#
#---------------------------------------------------------# 
     def init_filter_kernel(self,data_size):  
         self.calc_filter_kernel( self.filter_kernel_length )
         self.calc_filter_kernel_cplx(data_size)
         return 1
        
#---------------------------------------------------------# 
#--- init_filter                 -------------------------#
#---------------------------------------------------------# 
     def init_filter(self,number_of_samples):  
         #print number_of_samples
         
       #--- init data array for filter
         data_length          = self.calc_filter_data_length( number_of_samples )
         self.data_plane      = np.zeros( data_length,np.float64 )
         
       #--- init part of data array for input data => pointer to part of data_plane        
         data_tsl_start_in       = int( self.filter_kernel_data.size * self.settling_time_factor )
         self.data_plane_data_in = self.data_plane[data_tsl_start_in:data_tsl_start_in+number_of_samples]    
      
       #--- init part of data array for output data due to filter kernel shift  => pointer to part of data_plane        
         if (self.filter_attenuation_factor == 1) :
            data_tsl_start_out   = int( data_tsl_start_in + np.floor( self.filter_kernel_data.size/2 ) )
         else :
			data_tsl_start_out   = int( data_tsl_start_in + np.floor( self.filter_kernel_data.size/2 + self.filter_kernel_data.size / 2 * ( self.filter_attenuation_factor-1 ) ) ) 
        
         self.data_plane_data_out = self.data_plane[data_tsl_start_out:data_tsl_start_out+number_of_samples]    
         
         #  self.data_plane_post  = self.data_plane[data_tsl_start+number_of_samples,:] 
               
         return self.data_plane

#---------------------------------------------------------# 
#--- apply_filter                -------------------------#
#---------------------------------------------------------# 
     def apply_filter(self,data):  
             
         # data substract dc offset works inplace 
         (dmean,dstd) = self.calc_remove_dcoffset(data) 
                 
         self.data_plane.fill(self.data_std)

    #--- copy data at right place in data plane array            
         self.data_plane_data_in[:] = data
    
    #--- apply fft to data_plane; multiply with cplx filter kernel; apply ifft to get back to signal space
         self.data_plane[:] = np.fft.irfft( np.fft.rfft( self.data_plane ) * self.filter_kernel_data_cplx )
   
    #--- copy filtered data back at right place in data array !!! M-1 kernel shift !!!
         data[:] = self.data_plane_data_out

    #--- retain dc offset       
         if ( self.dcoffset == False) : 
            data += dmean
     
         return data

#---------------------------------------------------------# 
#--- destroy                     -------------------------#
#---------------------------------------------------------# 
     def destroy(self):
         # print self.data_plane

         del self._data_plane
         del self._data_plane_data_in 
         del self._data_plane_data_out 
         del self._filter_kernel_data_cplx 
         del self._filter_kernel_data 
         #print self.data_plane

         return 1
        
