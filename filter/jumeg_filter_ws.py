import numpy as np
'''
----------------------------------------------------------------------
--- JuMEG Filter_Ws             --------------------------------------
---------------------------------------------------------------------- 
 autor      : Frank Boers 
 email      : f.boers@fz-juelich.de
 last update: 11.12.2014
 version    : 0.03141
---------------------------------------------------------------------- 
 Taken from:
 The Scientist and Engineer's Guide to Digital Signal Processing
 By Steven W. Smith, Ph.D.
 Chapter 16: Window-Sinc Filter
 http://www.dspguide.com/ch16.htm
----------------------------------------------------------------------
 Butterworth filter design from  KD,JD
 Oppenheim, Schafer, "Discrete-Time Signal Processing"
----------------------------------------------------------------------

'''
from jumeg.filter.jumeg_filter_base import JuMEG_Filter_Base

class JuMEG_Filter_Ws(JuMEG_Filter_Base):
     def __init__ (self,filter_type='bp',fcut1=1.0,fcut2=200.0,remove_dcoffset=True,sampling_frequency=1017.25,filter_window='blackmann',
                   kernel_length_factor=16.0,settling_time_factor=5.0): #, notch=np.array([]),notch_width=1.0):
         super(JuMEG_Filter_Ws, self).__init__()
         
         self.__jumeg_filter_ws_version   = 0.0314
         self.__filter_method             = 'ws'

         self.sampling_frequency          = sampling_frequency
         self.filter_type                 = filter_type #lp, bp, hp
         self.fcut1                       = fcut1
         self.fcut2                       = fcut2
#---      
         self.__filter_kernel_data_rfft   = np.array([])
         self.__filter_kernel_length_factor = kernel_length_factor
         self.filter_attenuation_factor   = 1
         self.settling_time_factor        = settling_time_factor
#--               
         self.remove_dcoffset             = remove_dcoffset

#--- filter method bw,ws,mne
     def __get_filter_method(self):
         return self.__filter_method
     filter_method = property(__get_filter_method)

#--- version
     def __get_version(self):  
         return self.__jumeg_filter_ws_version
       
     version = property(__get_version)
                             

#--- filter_kernel_data_rfft     
     def __get_filter_kernel_data_rfft(self):
         return self.__filter_kernel_data_rfft

     filter_kernel_data_rfft = property(__get_filter_kernel_data_rfft)

#---- filter_attenuation_factor
     def __set_filter_attenuation_factor(self,value):
         if value < 2 :
            self.__filter_attenuation_factor = 1
         else :
            self.__filter_attenuation_factor = 2
         self.filter_kernel_isinit = False 
         
     def __get_filter_attenuation_factor(self):
         return self.__filter_attenuation_factor 
  
     filter_attenuation_factor = property(__get_filter_attenuation_factor,__set_filter_attenuation_factor)
  
#---------------------------------------------------------# 
#--- calc_sinc_filter_data       -------------------------#
#---------------------------------------------------------#
     def calc_sinc_filter_data(self,M,fc):
         """
            return an M + 1 point symmetric point sinc kernel with normalised cut-off frequency
         
            input:  M  => has to be even !!!
                    fc => normalised cut-off frequency fc between 0 <> 0.5 times sampling frequency 
         
         """
         if M%2:
                raise Exception('M must be even')
                
         x    = np.arange(M + 1) - M/2.0
         ws   = np.zeros(M + 1,np.float64)
         midx = int( M / 2 )
         ws[0:midx - 1] = np.sin(2.0 *np.pi*fc*x[0:midx - 1])/ x[0:midx - 1]
         ws[midx + 1:]  = np.sin(2.0 *np.pi*fc*x[midx + 1: ])/ x[midx + 1 :]
         ws[ midx ]     = 2.0*np.pi*fc
         return ws
   
#---------------------------------------------------------# 
#--- calc_sinc_filter_function       ---------------------#
#---------------------------------------------------------# 
     def calc_sinc_filter_function(self,M,fc,w=None):
         """
            return normialized window sinc data; windowed with balckman or hamming; unity gain at DC
            
            input:  M  => has to be even !!!
                    fc => normalised cut-off frequency fc between 0 <> 0.5 times sampling frequency  
                    w  => blackmann or hamming if none then default will be used       
         """
         if M%2:
                raise Exception('M must be even')
         
         if w is None:
             w = self.filter_window
       
         if w == 'hamming' :
            d = self.calc_sinc_filter_data(M,fc) * np.hamming(M+1)  
            d[ int(M/2) ] = 2.0 * np.pi *fc
         elif w == 'kaiser' :
            d = self.calc_sinc_filter_data(M,fc) * np.kaiser(M+1,self.kaiser_beta)
         else :
            d = self.calc_sinc_filter_data(M,fc) * np.blackman(M+1) 
            d[int(M/2)] = 2.0 * np.pi *fc
         return d/d.sum()
         
#---------------------------------------------------------# 
#---  calc_filter_kernel         -------------------------#
#---------------------------------------------------------# 
     def calc_filter_kernel(self):
         """ 
            return filter kernel 
            
            input: M  => has to be even !!!
                   using global/default parameter: filter_type,fcut1,fcut2,sampling_frequency                
         """
        
         M = self.filter_kernel_length 
        
         if M%2:
                raise Exception('M must be even')
       
         if self.fcut1 is None:
             self.fcut1 = self.fcut2
             print "WARNING JuMEG_Filter_WS.calc_filter_kernel value for fcut1 is not defined using fcut2 => %f" %(self.fcut2)

         M       = int(M)
         Midx    = int( M / 2 );
                   
         #--- M has to bee even !!!
         self.filter_kernel_data       = np.zeros( M + 1 ,np.float64)
         
         if self.filter_type == 'lp':
             self.filter_kernel_data[:] = self.calc_sinc_filter_function(M, self.fcut1 / self.sampling_frequency)
         
         elif self.filter_type == 'hp':
             self.filter_kernel_data       -= self.calc_sinc_filter_function(M, self.fcut1 / self.sampling_frequency  )
             self.filter_kernel_data[Midx] += 1.0
 
         elif self.filter_type == 'bp':			 
             if (self.fcut1 > self.fcut2) :
                dfcut1     = self.fcut1
                self.fcut1 = self.fcut2
                self.fcut2 = dfcut1
             
          #-- bp => make lp
             self.filter_kernel_data[:] = self.calc_sinc_filter_function(M, self.fcut1 / self.sampling_frequency )
          #-- bp => make hp
             fbp_fcut2       = np.zeros( M + 1 , np.float64)
             fbp_fcut2      -= self.calc_sinc_filter_function(M, self.fcut2 / self.sampling_frequency )
             fbp_fcut2[Midx] += 1.0 
          #--- bp
             self.filter_kernel_data       += fbp_fcut2
             self.filter_kernel_data       *=-1.0 # now make notch to bp => invert spectrum!!!
             self.filter_kernel_data[Midx] += 1.0 
      
         elif self.filter_type == 'notch':
             if (self.fcut1 > self.fcut2):
                 dfcut1     = self.fcut1
                 self.fcut1 = self.fcut2
                 self.fcut2 = dfcut1
          #--- notch => make lp
             self.filter_kernel_data[:] = self.calc_sinc_filter_function(M, self.fcut1 / self.sampling_frequency)
          #--- notch => make hp
             fbp_fcut2       = np.zeros( M + 1 , np.float64)
             fbp_fcut2[:]   -= self.calc_sinc_filter_function(M, self.fcut2 / self.sampling_frequency)
             fbp_fcut2[Midx] += 1.0 
          #--- now make notch
             self.filter_kernel_data += fbp_fcut2
             
         return self.filter_kernel_data

        
#---------------------------------------------------------# 
#---  calc_filter_kernel_rfft    -------------------------#
#---------------------------------------------------------# 
     def calc_filter_kernel_rfft(self):
         """
            return complex filter kernel
            input real data
            data_size => will be padded
         """
         
         self.calc_filter_data_length( self.data_length )
         fkd = np.zeros( self.filter_data_length,np.float64 )
      #   self.filter_kernel_data.astype( np.complex64 )
         fkd[0:self.filter_kernel_data.size ] = self.filter_kernel_data.copy()   
    
        #--- factor 2 => ws  ~ -148 dB depends on window
         if self.filter_attenuation_factor == 2 :
            fkd_cplx = np.fft.rfft( fkd )
            self.__filter_kernel_data_rfft = fkd_cplx * fkd_cplx
         else :
        #--- factor 1 => ws  ~ -74 dB depends on window
            self.__filter_kernel_data_rfft = np.fft.rfft( fkd )              
                       
         return self.filter_kernel_data_rfft
 
         
#---------------------------------------------------------# 
#--- init_filter_kernel          -------------------------#
#---------------------------------------------------------# 
     def init_filter_kernel(self):  
         """
            calculating complex filter kernel to filter data
           
            input: data_size => will be padded 
         """ 
         self.data_plane_isinit    = False
         self.filter_kernel_isinit = False
         
         self.calc_filter_kernel()
         self.calc_filter_kernel_rfft()
         
         self.filter_kernel_isinit = True
         
         return self.filter_kernel_isinit
 
#---------------------------------------------------------# 
#--- init_filter_data_plane      -------------------------#
#---------------------------------------------------------# 
     def init_filter_data_plane(self):  
         """
            setting up data plane in memory to filter data 
           
            input: number_of_samples => will be padded 
         """  
       
         self.data_plane_isinit = False 
          
       #--- init data array for filter
         number_of_input_samples = self.data_length
         data_length             = self.calc_filter_data_length( number_of_input_samples )
         self.data_plane         = np.zeros( data_length,np.float64 )
         
       #--- init part of data array for input data => pointer to part of data_plane        
         data_tsl_start_in       = int( self.filter_kernel_data.size * self.settling_time_factor ) 
         self.data_plane_data_in = self.data_plane[ data_tsl_start_in:data_tsl_start_in + number_of_input_samples ]    
      
       #--- init part of data array for output data due to filter kernel shift  => pointer to part of data_plane        
         if (self.filter_attenuation_factor == 1) :
            data_tsl_start_out   = int( data_tsl_start_in + np.floor( self.filter_kernel_data.size/2 ) )
         else :
            data_tsl_start_out   = int( data_tsl_start_in + np.floor( self.filter_kernel_data.size/2 + self.filter_kernel_data.size / 2 * ( self.filter_attenuation_factor-1 ) ) ) 
        
         self.data_plane_data_out = self.data_plane[ data_tsl_start_out:data_tsl_start_out + number_of_input_samples ]    
       
       #--- init pre part of data array, till onset  index [1 ... end]
         idx0 = 0 
         idx1 = data_tsl_start_in
         if ( number_of_input_samples < data_tsl_start_in ) :
             idx0 = idx1 - number_of_input_samples + 2 # !!! for post range 
         self.data_plane_data_pre = self.data_plane[idx0:idx1]    
        
       #--- init post part of data array, start at offset  
         idx0 = data_tsl_start_in + number_of_input_samples
         idx1 = idx0 + self.data_plane_data_pre.size
         self.data_plane_data_post = self.data_plane[idx0:idx1]    
      
         self.data_plane_isinit = True
                  
         #return self.filter_data_plane_isinit
       
#---------------------------------------------------------# 
#--- do_apply_filter             -------------------------#
#---------------------------------------------------------# 
     def do_apply_filter(self,data):  
             
                                
    #--- data substract dc offset works inplace 
         dmean = self.calc_remove_dcoffset(data) 
                  
    #--- mirror data at pre and post in data plane array to reduce transient oscillation filter artefact  
         self.data_plane_data_pre[:] = data[self.data_plane_data_pre.size :0:-1 ]    
         self.data_plane_data_post[:]= data[data.size -2: data.size -2 - self.data_plane_data_post.size :-1]
                 
    #--- copy data at right place in data plane array            
         self.data_plane_data_in[:] = data
    
    #--- filter : apply real-fft to data_plane; fft-convolution with filter kernel; irfft to transform back into time domain
         self.data_plane[:] = np.fft.irfft( np.fft.rfft( self.data_plane ) * self.filter_kernel_data_rfft )
                     
    #--- copy filtered data back at right place in data array !!! M-1 kernel shift !!!
         data[:] = self.data_plane_data_out

    #--- retain dc offset       
         if ( self.remove_dcoffset == False) : 
            data += dmean
             
     
         return data

