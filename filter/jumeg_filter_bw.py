import numpy as np
import scipy.fftpack as scipack # import fft,ifft

'''
----------------------------------------------------------------------
--- JuMEG Filter_Bw             --------------------------------------
---------------------------------------------------------------------- 
 autor      : Frank Boers 
 email      : f.boers@fz-juelich.de
 update: 11.12.2014

 update: 21.12.2014
  --> change float index to np.int64
   -> VisibleDeprecationWarning: using a non-integer number instead of an integer will result in an error in the future

 version    : 0.03142
---------------------------------------------------------------------- 
 Butterworth filter design from  KD,JD
 Oppenheim, Schafer, "Discrete-Time Signal Processing"
----------------------------------------------------------------------

'''
from jumeg.filter.jumeg_filter_base import JuMEG_Filter_Base

class JuMEG_Filter_Bw(JuMEG_Filter_Base):
     
     def __init__ (self,filter_type='bp',fcut1=1.0,fcut2=200.0,remove_dcoffset=True,sampling_frequency=1017.25,
                        notch=np.array([]),notch_width=2.0,order=4.0,settling_time_factor=5.0):
         super(JuMEG_Filter_Bw, self).__init__()
         self._jumeg_filter_bw_version     = 0.03142
         self.__filter_method              = 'bw'
         
         self.sampling_frequency              = sampling_frequency
         self.filter_type                     = filter_type #lp, bp, hp
         self.fcut1                           = fcut1
         self.fcut2                           = fcut2
         self.filter_notch                    = notch
         self.filter_notch_width              = notch_width  
         self.settling_time_factor            = settling_time_factor

         self.__filter_order                    = order       
         self.__filter_kernel_data_cplx_sqrt    = np.array([])

         self.__settling_time_factor_timeslices = 10000
        
         self.remove_dcoffset                 = remove_dcoffset

#--- filter method bw,ws,mne
     def __get_filter_method(self):
         return self.__filter_method
     filter_method = property(__get_filter_method)

#--- version
     def __get_version(self):  
         return self.__jumeg_filter_ws_version
       
     version = property(__get_version)

#--- filter_order
     def __set_filter_order(self, value):  
         self.__filter_order = value
         self.filter_kernel_isinit = False
        
     def __get_filter_order(self):  
         return self.__filter_order
       
     filter_order = property(__get_filter_order, __set_filter_order)

#--- filter_kernel_data_cplx_sqrt     
     def __get_filter_kernel_data_cplx_sqrt(self):
         return self.__filter_kernel_data_cplx_sqrt
     def __set_filter_kernel_data_cplx_sqrt(self, value):
         self.__filter_kernel_data_cplx_sqrt = value

     filter_kernel_data_cplx_sqrt = property(__get_filter_kernel_data_cplx_sqrt,__set_filter_kernel_data_cplx_sqrt)
        
#--- settling time factor timeslices        
     def __get_settling_time_factor_timeslices(self):
        tau = self.fcut1 * 2.0 * np.pi
        if tau is None:
           tau = 0.01 * 2.0 * np.pi
        #---  ~calc 5 times tau   
        self.__settling_time_factor_timeslices = np.ceil( ( 1.0 / tau ) * self.settling_time_factor * self.sampling_frequency )
        
        return self.__settling_time_factor_timeslices

     settling_time_factor_timeslices = property(__get_settling_time_factor_timeslices)
    
#---------------------------------------------------------# 
#---  calc_filter_kernel         -------------------------#
#---------------------------------------------------------#         
     def calc_filter_kernel(self):        
         """ 
            return butterworth filter function 
            
            input: M  => data lenth will be padded to an even number
                   using global/default parameter: filter_type,fcut1,fcut2,sampling_frequency                
         """
         
         M = self.data_length
         
         if self.fcut1 is None :
             self.fcut1 = self.fcut2
             print "WARNING JuMEG_Filter_Bw.calc_filter_kernel value for fcut1 is not defined using fcut2 => %f" %(self.fcut2)
         
         #t0 = time.time()  
         
         fcut1 = self.fcut1
         fcut2 = self.fcut2
         order = self.filter_order
#----------         
         pad = np.ceil( self.sampling_frequency / 2.0 / self.fcut1 )
         len = self.calc_zero_padding( M + pad  + 2 * self.settling_time_factor_timeslices)
#-----------      
         nyq    = len / 2.0 + 1.0
         f_n    = self.sampling_frequency / 2.0
         f_res  = f_n  / nyq
#-----------
         nyq_idx = np.int64(nyq)
         darray                  = np.arange( nyq_idx -1   ,dtype=np.float64 ) + 1.0
         self.filter_kernel_data = np.zeros( 2 * (nyq_idx -1),dtype=np.float64)
         fkd_part1               = self.filter_kernel_data[1:nyq_idx]
#-----------  
 
         if   self.filter_type == 'lp' :
                   omega       = darray * f_res / fcut1 
                   fkd_part1[:]= ( 1 / ( omega**( 2.0 * order ) +1.0 ) )
                   self.filter_kernel_data[0] = 1
                  
         elif self.filter_type == 'hp' :
                   omega       = fcut1 / ( darray * f_res ); 
                   fkd_part1[:]= ( 1 / ( omega**( 2.0 * order ) +1.0 ) )
    
         elif self.filter_type == 'bp' :
                   if (self.fcut1 > self.fcut2) :
                      fcut2      = self.fcut1
                      fcut1      = self.fcut2
                      self.fcut1 = fcut2
                      self.fcut2 = fcut1
                      
                   norm        = ( 2.0 * fcut2 * fcut1 ) / ( np.sqrt( fcut1 * fcut2 ) * ( fcut2 - fcut1 ) ) 
                   norm        = 1 + norm**( 2.0 * order )
                   omega       = (( darray * f_res )**2.0 + fcut2 * fcut1 )/ ( darray * f_res *( fcut2 - fcut1 ) )
                   fkd_part1[:]= norm / ( 1.0 + omega**( 2.0 * order ) ) 
        
         else: # flat only notches 
                   fkd_part1[:]=  1

#--- add notches
         if self.filter_notch.size :
            n_fac = np.sqrt( np.log(2.0) )
            for idx in range( self.filter_notch.shape[0] ):     
            
                 omega = self.filter_notch[idx]
                 
                 # print "NOTCH %d  %f" %(idx,omega) 
                 
                 sigma = n_fac * self.filter_notch_width / 2.0
                 freq  = np.arange(nyq,dtype=np.float64) * f_res - omega 
                 min_idx  = np.int64( np.argmin( np.abs( freq ) ) )
                 self.filter_kernel_data[min_idx] = 0.0
                 if ( min_idx > 1 ):
                      self.filter_kernel_data[1:min_idx] *= np.exp( - ( sigma / freq[1:min_idx] )**2 )
                 else : 
                      self.filter_kernel_data[1] *= np.exp( - ( sigma / freq[1] )**2 )
                 
                 self.filter_kernel_data[min_idx+1:nyq_idx] = self.filter_kernel_data[min_idx+1:nyq_idx] * np.exp( - ( sigma/freq[min_idx+1:] )**2.0 )
   
#--- construct the negative frequencies in filter kernel data
         self.filter_kernel_data[nyq_idx :] = fkd_part1[-1:0:-1]
         
         if self.verbose :
             print"======================================="                    
             #print "Time to calc filter kernel data: % f " % (time.time() - t0 )       
             print "FKD shape" 
             print self.filter_kernel_data.shape
             print "===>DONE calc butterworth filter function"     
        
         return self.filter_kernel_data

#---------------------------------------------------------# 
#---  calc_filter_kernel_cplx_sqrt  ----------------------#
#---------------------------------------------------------# 
     def calc_filter_kernel_cplx_sqrt(self):
         """
            return sqrt of complex filter function/kernel!!!
         """ 
         self.filter_kernel_data_cplx_sqrt = np.sqrt( self.filter_kernel_data.astype( np.complex64 ) )
         
         return self.filter_kernel_data_cplx_sqrt

#---------------------------------------------------------# 
#--- init_filter_kernel          -------------------------#
#---------------------------------------------------------# 
     def init_filter_kernel(self):  
         """
            calculating complex butterworth filter function to filter data
           
            input: data_size => will be padded 
         """ 
         self.data_plane_isinit    = False     
         self.filter_kernel_isinit = False 
         
         self.calc_filter_kernel()         
         self.calc_filter_kernel_cplx_sqrt()
         
         self.filter_kernel_isinit = True
      
         return self.filter_kernel_isinit
                 
#---------------------------------------------------------# 
#--- init_filter_data_plane      -------------------------#
#---------------------------------------------------------# 
     def init_filter_data_plane(self):  
         """
            setting up data plane in memory to filter data 
                      
            defining pre and post parts in data plane to
            avoid filter artefact due to transient oscillation
            data_length => will be padded 
         """  
        
         self.data_plane_isinit = False 
          
       #--- init data array for filter
         number_of_input_samples = self.data_length
         data_length             = self.filter_kernel_data.size
         self.data_plane_cplx    = np.zeros( data_length ,np.complex64 )
                  
       #--- init part of data array for input data => pointer to part of data_plane        
         data_tsl_start_in        = np.int64( self.settling_time_factor_timeslices )
         self.data_plane_data_in  = self.data_plane_cplx[data_tsl_start_in:data_tsl_start_in + number_of_input_samples].real    
     
       #--- init pre part of data array, till data onset 
         idx0 = 0 
         idx1 = data_tsl_start_in
         
         if ( self.data_length < self.settling_time_factor_timeslices ) :
              idx0 = self.settling_time_factor_timeslices - number_of_input_samples +1
                       
         self.data_plane_data_pre = self.data_plane_cplx[idx0:idx1].real    
         #print "pre idx0: %d idx1: %d  size: %d" %(idx0, idx1, self.data_plane_data_pre.size )
         
       #--- init post part of data array, start at data offset  
         idx0 = data_tsl_start_in + number_of_input_samples
         idx1 = idx0 + self.data_plane_data_pre.size -1
         self.data_plane_data_post = self.data_plane_cplx[idx0:idx1].real    
         
         #print "size data %d pre %d post %d" % (number_of_samples, self.data_plane_data_pre.size, self.data_plane_data_post.size)
         
         self.data_plane_isinit = True
                  
         return self.data_plane_isinit

#---------------------------------------------------------# 
#--- do_apply_filter             -------------------------#
#---------------------------------------------------------# 
     def do_apply_filter(self,data):  
           
         dmean = self.calc_remove_dcoffset(data) 
  
         self.data_plane_cplx.fill(0.0) 
         
    #--- mirror data at pre and post in data plane array to reduce transient oscillation filter artefact
         self.data_plane_data_pre[:] = data[self.data_plane_data_pre.size :0:-1 ]    
         self.data_plane_data_post[:]= data[data.size -2: data.size -2 - self.data_plane_data_post.size :-1]
         
    #--- copy data at right place in data plane array       
         self.data_plane_data_in[:] = data
    
    #--- filter : apply fft to complex data_plane; fft-convolution with filter kernel; ifft to transform back into time domain     
         self.data_plane_cplx[:]    = scipack.ifft( scipack.fft( self.data_plane_cplx ) * self.filter_kernel_data_cplx_sqrt )
         
    #--- copy filtered real part data back 
         data[:] = self.data_plane_data_in

    #--- retain dc offset       
         if (self.remove_dcoffset == False ): 
            data += dmean
            if self.verbose:
               print " ---> NO DC Offset correction !!!"
         return data
