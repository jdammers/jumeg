import numpy as np
'''
#---------------------------------------------------------# 
#--- Filter_WS             -------------------------#
#---------------------------------------------------------# 
# autor      : Frank Boers 
# last update: 26.05.2014
# version    : 0.005
#---------------------------------------------------------# 
# Taken from:
# The Scientist and Engineer's Guide to Digital Signal Processing
# By Steven W. Smith, Ph.D.
# Chapter 16: Window-Sinc Filter
# http://www.dspguide.com/ch16.htm
#---------------------------------------------------------# 
#
# from filter_ws import Filter_WS
#
# filter_type = 'bp'
# fcut1       = 1.0
# fcut2       = 45.0
# sampling_frequency = 1017.25
#
# fiws = Filter_WS()
#--- parameter
# fiws.sampling_frequency = sampling_frequency
# fiws.filter_type        = filter_type  # bp hp lp notch
# fiws.fcut1              = fcut1
# fiws.fcut2              = fcut2
#
# fiws.init_filter_kernel(signal.size)
# fiws.init_filter(signal.size)
# fiws.dcoffset           = True or False
#
#--- filter run 
#   fiws.apply_filter(signal)
#--> this works inplace input data will be change
'''


class Filter_WS:
     def __init__ (self):
         self._sampling_frequency          = 1017.25
         self._filter_type                 = 'bp' #lp, bp, hp
         self._filter_attenuation_factor   = 1.0  # 1, 2
         self._filter_window               = 'blackmann' # hamming, blackmann
         self._fcut1                       = 1
         self._fcut2                       = 200
#---
         self._filter_kernel_length_factor = 16.0
         self._settling_time_factor        = 5.0
         self._data_mean                   = 0.0
         self._dcoffset                    = 0


#--- dcoffset    
     def set_dcoffset(self,value):
         self._dcoffset = value

     def get_dcoffset(self):
         return self._dcoffset
       
     dcoffset = property(get_dcoffset, set_dcoffset)
                       
#---  settling_time_factor   
     def set_settling_time_factor(self,value):
         self._settling_time_factor = value

     def get_settling_time_factor(self):
         return self._settling_time_factor

     settling_time_factor = property(get_settling_time_factor, set_settling_time_factor)

#--- sampling_frequency  
     def set_sampling_frequency(self, value):
         self._sampling_frequency = value

     def get_sampling_frequency(self):
         return self._sampling_frequency

     sampling_frequency = property(get_sampling_frequency, set_sampling_frequency)

#--- filter_kernel_length_factor    
     def set_filter_kernel_length_factor(self, value):
         self._filter_kernel_length_factor = value

     def get_filter_kernel_length_factor(self):
         return self._filter_kernel_length_factor

     filter_kernel_length_factor = property(get_filter_kernel_length_factor, set_filter_kernel_length_factor)

#--- filter_kernel_length
     def get_filter_kernel_length(self):
         try:
            self._filter_kernel_length = self.sampling_frequency * self.filter_kernel_length_factor
         finally: 
               return self._filter_kernel_length

     filter_kernel_length = property(get_filter_kernel_length)

#--- filter_type    
     def set_filter_type(self,value):
         self._filter_type = value

     def get_filter_type(self):
         return self._filter_type
       
     filter_type = property(get_filter_type, set_filter_type)

#--- fcut1    
     def set_fcut1(self,value):
         self._fcut1 = value
    
     def get_fcut1(self):
         return self._fcut1
    
     fcut1 = property(get_fcut1, set_fcut1)

#--- fcut2    
     def set_fcut2(self,value):
         self._fcut2 = value
    
     def get_fcut2(self):
         return self._fcut2
    
     fcut1 = property(get_fcut2, set_fcut2)

#--- filter_window   
     def set_filter_window(self,value):
         self._filter_window = value
 
     def get_filter_window(self):
         return self._filter_window
   
     filter_window = property(get_filter_window,set_filter_window)
     
#--- filter_kernel_data   
     def set_filter_kernel_data(self,d):
         self._filter_kernel_data = d
     
     def get_filter_kernel_data(self):
         return self._filter_kernel_data
   
     filter_kernel_data = property(get_filter_kernel_data,set_filter_kernel_data)

#--- filter_kernel_data_cplx     
     def get_filter_kernel_data_cplx(self):
         return self._filter_kernel_data_cplx

     filter_kernel_data_cplx = property(get_filter_kernel_data_cplx)

#---- filter_data_length
     def calc_filter_data_length(self,dl):
         self._filter_data_length = self.calc_zero_padding( dl + self.filter_kernel_data.size * self.settling_time_factor * 2 +1)
         return self._filter_data_length 

     def get_filter_data_length(self):
         return self._filter_data_length  
    
     filter_data_length = property(get_filter_data_length)

#---- filter_attenuation_factor
     def set_filter_attenuation_factor(self,value):
         if value < 2 :
            self._filter_attenuation_factor = 1
         else :
            self._filter_attenuation_factor = 2
           
     def get_filter_attenuation_factor(self):
         return self._filter_attenuation_factor 
  
     filter_attenuation_factor = property(get_filter_attenuation_factor,set_filter_attenuation_factor)
  
#--- data_mean    
     def set_data_mean(self,value):
         self._data_mean = value
    
     def get_data_mean(self):
         return self._data_mean
    
     data_mean = property(get_data_mean, set_data_mean)

#--- data_std    
     def set_data_std(self,value):
         self._data_std = value
    
     def get_data_std(self):
         return self._data_std
    
     data_std = property(get_data_std, set_data_std)

#--- data_plane dummy data container to filter data
     def set_data_plane(self,value):
         self._data_plane = value
    
     def get_data_plane(self):
         return self._data_plane
    
     data_plane = property(get_data_plane, set_data_plane)    
    
#--- data_plane_in dummy data container to filter data
     def set_data_plane_data_in(self,value):
         self._data_plane_data_in = value
    
     def get_data_plane_data_in(self):
         return self._data_plane_data_in
    
     data_plane_data_in = property(get_data_plane_data_in, set_data_plane_data_in)
    
#--- data_plane_out dummy data container to filter data
     def set_data_plane_data_out(self,value):
         self._data_plane_data_out = value
    
     def get_data_plane_data_out(self):
         return self._data_plane_data_out
    
     data_plane_data_out = property(get_data_plane_data_out, set_data_plane_data_out)
    
     
#---------------------------------------------------------# 
#--- window_hamming              -------------------------#
#---------------------------------------------------------# 
     def calc_window_hamming(self,M):
         """Return an M + 1 point symmetric hamming window."""
         if M%2:
                raise Exception('M must be even')
         return (0.54 - 0.46*np.cos(2*np.pi*np.arange(M + 1)/M))
         
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
         #return (data - self.data_mean,self.data_mean,self.data_std)  
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

         Midx    = int( M / 2 );
         dfcut1  = self.fcut1 / self.sampling_frequency
         dfcut2  = self.fcut2 / self.sampling_frequency
         
         #--- M has to bee even !!!
         self.filter_kernel_data       = np.zeros( M + 1 ,np.float64)
         #self.filter_kernel_data.dtype = np.double
         
         if self.filter_type == 'lp':
           print "lp"  
           self.filter_kernel_data[:] = self.calc_sinc_filter_function(M, dfcut1)
         elif self.filter_type == 'hp':
             print "hp"
             if self.fcut1 is None:
                dfcut1= self.fcut2 / self.sampling_frequency

             self.filter_kernel_data -= self.calc_sinc_filter_function(M, dfcut1)
             self.filter_kernel_data[Midx] += 1.0
 
         elif self.filter_type == 'bp':
             print "bp"
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
         fkd                                     = np.zeros( self.filter_data_length ) #*1.0
         fkd.dtype                               = np.double
         fkd[0: self.filter_kernel_data.size ]   = self.filter_kernel_data.copy()   
         
        #--- factor 2 will give -148 dB 
         if self.filter_attenuation_factor == 2 :
           fkd_cplx = np.fft.rfft( fkd )
           self.filter_kernel_data_cplx = fkd_cplx * fkd_cplx
         else :
        #--- factor 1 will give -74 dB 
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
         self.data_plane      = np.zeros( data_length )
         self.data_plane.dtype= np.double
         
       #--- init part of data array for input data => pointer to part of data_plane        
         data_tsl_start_in       = int( self.filter_kernel_data.size * self.settling_time_factor )
         self.data_plane_data_in = self.data_plane[data_tsl_start_in:data_tsl_start_in+number_of_samples]    
      
       #--- init part of data array for output data due to filter kernel shift  => pointer to part of data_plane        
         data_tsl_start_out       = int( data_tsl_start_in + np.floor( self.filter_kernel_data.size)/2 ) * self.filter_attenuation_factor
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

    #--- cp data at right place in data plane array
              
         self.data_plane_data_in[:] = data
               
         self.data_plane[:] = np.fft.irfft( np.fft.rfft( self.data_plane ) * self.filter_kernel_data_cplx )
   
    #--- cp filtered data at right place in data array !!! M-1 kernel shift !!!
         data[:] = self.data_plane_data_out
      
         if ( self.dcoffset ) : 
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


          # data_tsl_start = int( self.filter_kernel_data.size * self.settling_time_factor )
          # data_tsl_end   = data_tsl_start + data.size 
          # data_idx_a     = int( data_tsl_start + np.floor(self.filter_kernel_data.size/2 ) )
          # data_length    = self.filter_data_length
          #--- prepare data  
          # data_dummy         = np.zeros(data_length) #,float64)
          # data_dummy.dtype   = np.double
          # data_dummy[data_tsl_start:data_tsl_start+data.size] += data.copy() 

          # data_dummy[0:data_tsl_start] = data[range(data_tsl_start,0,-1)].copy() 
          #--- prepare post data  
          # data_tsl_end_start = data.size-2
          # data_tsl_end_end   = data_tsl_start # - data_tsl_start
          # data_offset        = data_dummy[data_tsl_start+data.size:] 
          # data_offset        = data[range(data_tsl_end_start, data_tsl_end_start - data_offset.size ,-1)].copy() 
          ## data_dummy         = np.fft.irfft( np.fft.rfft(data_dummy) * self.filter_kernel_data_cplx ).copy()

          #data_dummy =scipy.signal.fftconvolve(data_dummy,self.filter_kernel_data)

          #data              *= 0.0
          #data              += dmean *1.0
          #data              += data_dummy[data_idx_a:data_idx_a + data.size] 


