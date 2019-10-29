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
2019.04.03 update logger
----------------------------------------------------------------------
'''
import logging
import numpy as np
import scipy.fftpack as scipack # import fft,ifft

from jumeg.base.jumeg_base          import jumeg_base_basic as jb
from jumeg.filter.jumeg_filter_base import JuMEG_Filter_Base


logger = logging.getLogger('jumeg')
__version__= '2019.05.14.001'

class JuMEG_Filter_Bw(JuMEG_Filter_Base):
    """  Filter FIR FFT butterworth implementation""" 
    
    def __init__ (self,filter_type='bp',fcut1=1.0,fcut2=200.0,remove_dcoffset=True,sampling_frequency=1017.25,verbose=False,
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
        self.verbose                         = verbose
        
#--- filter method bw,ws,mne
    @property
    def filter_method(self):   return self.__filter_method

    @property
    def filter_order(self):    return self.__filter_order
    @filter_order.setter
    def filter_order(self,v):  
        self.__filter_order = v
        self.filter_kernel_isinit = False
    
    @property
    def filter_kernel_data_cplx_sqrt(self):  return self.__filter_kernel_data_cplx_sqrt
    @filter_kernel_data_cplx_sqrt.setter
    def filter_kernel_data_cplx_sqrt(self,v):self.__filter_kernel_data_cplx_sqrt=v

#--- settling time factor timeslices        
    @property
    def settling_time_factor_timeslices(self):
        tau = None
        if self.fcut1:
           tau = self.fcut1 * 2.0 * np.pi
        elif self.fcut2:
           tau = self.fcut2 * 2.0 * np.pi
        if tau is None:
           tau = 0.01 * 2.0 * np.pi
        #---  ~calc 5 times tau   
        self.__settling_time_factor_timeslices = np.ceil( ( 1.0 / tau ) * self.settling_time_factor * self.sampling_frequency )
        
        return self.__settling_time_factor_timeslices

    def calc_filter_kernel(self):        
        """ 
        return butterworth filter function 
        data lenth will be padded to an even number
        using global/default parameter: filter_type,fcut1,fcut2,sampling_frequency                
        
        Parameter
        ---------
         None
         
        Result
        -------
        filter kernel data
        """
         
        M     = self.data_length
        order = self.filter_order
        pad   = 0.0

       #---
        if self.filter_type.lower() in ['lp','hp','bp']:
           if self.fcut1 is None :
              self.fcut1 = self.fcut2
              logger.warning( "JuMEG_Filter_Bw.calc_filter_kernel value for fcut1 is not defined using fcut2 => %f" %(self.fcut2) )
           fcut1 = self.fcut1
           fcut2 = self.fcut2
           pad   = np.ceil( self.sampling_frequency / 2.0 / self.fcut1 )

       #---
        len = self.calc_zero_padding( M + pad  + 2.0 * self.settling_time_factor_timeslices)
       #---      
        nyq    = len / 2.0 + 1.0
        f_n    = self.sampling_frequency / 2.0
        f_res  = f_n  / nyq
       #---
        nyq_idx = np.int64(nyq)
        darray                  = np.arange( nyq_idx -1   ,dtype=np.float64 ) + 1.0
        self.filter_kernel_data = np.zeros( 2 * (nyq_idx -1),dtype=np.float64)
        fkd_part1               = self.filter_kernel_data[1:nyq_idx]
       #---  
 
        if self.filter_type == 'lp' :
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
           logger.info( "  -> FKD shape: {}".format( self.filter_kernel_data.shape )+ "===>DONE calc butterworth filter function")
        
        return self.filter_kernel_data

    def calc_filter_kernel_cplx_sqrt(self):
        """
         cal sqrt of complex filter function/kernel!!!
        """ 
        self.filter_kernel_data_cplx_sqrt = np.sqrt( self.filter_kernel_data.astype( np.complex64 ) )
         
        return self.filter_kernel_data_cplx_sqrt

    def init_filter_kernel(self):  
        """
         calculating complex butterworth filter function to filter data
         data_size => will be padded 
        """ 
        self.data_plane_isinit    = False     
        self.filter_kernel_isinit = False 
         
        self.calc_filter_kernel()         
        self.calc_filter_kernel_cplx_sqrt()
         
        self.filter_kernel_isinit = True
      
        return self.filter_kernel_isinit

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

    def do_apply_filter(self,data):  
        """
        do apply filter
        !!! filter works inplace!!!
        input data will be overwriten
        
        Parameters
        ----------
         data as np.array [ch,timepoints]
       
        Results
        -------
         filtered data 
        """  
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
        if self.remove_dcoffset : return data 
        
        data += dmean
        return data
