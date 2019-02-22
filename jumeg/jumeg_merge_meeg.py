#!/usr/bin/env python


'''Class JuMEG_MergeMEEG

Class to merge brainvision eeg data into MEG-fif file

Authors:
         Prank Boers     <f.boers@fz-juelich.de>
         Praveen Sripad  <praveen.sripad@rwth-aachen.de>
License: BSD 3 clause
----------------------------------------------------------------
merge ECG,EGO signals form BrainVision system into MEG-fif file
-> read meg raw fif
-> read bv eeg
-> find start events in raw and bv-eeg
-> lp-filter eeg and meg data
-> downsample eeg merge into meg-fif
-> adjust meg data size
-> save to disk [...eeg-raw.fif]

example via obj:
from jumeg.jumeg_merge_meeg import JuMEG_MergeMEEG
JMEEG = JuMEG_MergeMEEG()
JMEEG.meg_fname= my_meg_fname.fif
JMEEG.eeg_fname= my_eeg_fname.vhdr
JMEEG.run()

example  via function call
import jumeg.jumeg_merge_meeg
jumeg_merge_meeg(meg_fname= my_meg_fname.fif, eeg_fname = my_eeg_fname.vhdr)

---> update 09.01.2017 FB
     add cls for meg/eeg data
     error checking fix support for meg.raw
---> update 21.08.2017 FB
     update drift check function
---> update 19.10.2017 FB
     update drift check function
    
'''

import warnings
import os,sys,argparse
import numpy as np
import mne

from jumeg.jumeg_base                   import JuMEG_Base_IO
from jumeg.filter.jumeg_filter          import jumeg_filter


__version__= '2017.10.19.001'

class JuMEG_MergeMEEG_HiLoRate(object):
      def __init__(self, system='MEG'):
          super(JuMEG_MergeMEEG_HiLoRate,self).__init__()
          self.path     = None
          self.name     = None
          self.filename = None
          self.raw      = None
          self.system   = system
          self.startcode= 128
          
          self.__verbose = False
          self.__debug   = False
          
          self.data          = np.array([])
          self.times         = np.array([])
          self.ev_onsets     = np.array([])
          self.tsl_resamples = np.array([],dtype=np.int64)
          self.tsl_onset     = None
          self.time_delta    = None
          self.picks         = None
          self.is_data_size_adjusted = False
        
        #---meg defaults
          self.event_parameter = {'and_mask':255,'stim_type':'STIMULUS','response_shift':None}
          self.events          = {'consecutive': True, 'output': 'step', 'stim_channel': 'STI 014','min_duration': 0.00001, 'shortest_event': 1, 'mask': None}
               
      def reset(self):
          self.data          = np.array([])
          self.times         = np.array([])
          self.ev_onsets     = np.array([])
          self.tsl_resamples = np.array([],dtype=np.int64)
          self.tsl_onset     = None
          self.time_delta    = None
          self.picks         = None
          self.raw           = None
          
      def __get_stim_channel(self, v):
          return self.events['stim_channel']
      def __set_stim_channel(self, v):
          self.events['stim_channel'] = str(v)
      stim_channel = property(__get_stim_channel, __set_stim_channel)
    
      def __get_and_mask(self):
          return self.event_parameter['and_mask']
      def __set_and_mask(self, v):
          self.event_parameter['and_mask'] = v
      and_mask=property(__get_and_mask,__set_and_mask)
    
      def __get_stim_type(self):
          return self.event_parameter['stim_type']
      def __set_stim_type(self, v):
          self.event_parameter['stim_type'] = v
      stim_type=property(__get_stim_type,__set_stim_type)
    
      def __get_response_shift(self):
          return self.event_parameter['response_shift']
      def __set_response_shift(self, v):
          self.event_parameter['response_shift'] = v
      response_shift=property(__get_response_shift,__set_response_shift)
   
      def __get_min_duration(self):
          return self.event_parameter['min_duration']
      def __set_min_duration(self, v):
          self.event_parameter['min_duration'] = v
      min_duration=property(__get_min_duration,__set_min_duration)

      def __get_shortest_event(self):
          return self.event_parameter['shortest_event']
      def __set_shortest_event(self, v):
          self.event_parameter['shortest_event'] = v
      shortest_event=property(__get_shortest_event,__set_shortest_event)
      
     #--- verbose
      def __set_verbose(self,value):
          self.__verbose = value
      def __get_verbose(self):
          return self.__verbose
      verbose = property(__get_verbose, __set_verbose)
    
    #--- debug
      def __set_debug(self,value):
          self.__debug = value
          if self.__debug:
             self.verbose =True
           
      def __get_debug(self):
          return self.__debug
      debug = property(__get_debug, __set_debug)

  #---   
      def check_path(self):
          print(" --> check path & file: " + self.system + " file name: " + str(self.filename))
          assert( os.path.isfile( self.filename ) ),"---> ERROR " + self.system +" file: no such file: " + str(self.filename)
          print("  -> OK")
          return True

      def __get_time_onset(self):
          return self.raw.times[self.tsl_onset]
      time_onset = property(__get_time_onset)

      def __get_time_end(self):
          return self.raw.times[-1]
      time_end = property(__get_time_end)

      def __get_time_duration(self):
          return self.raw.times[self.raw.last_samp - self.tsl_onset]
      time_duration = property(__get_time_duration)

      def __get_tsl_end(self):
          return self.tsl_onset + self.times.shape[0]
      tsl_end = property(__get_tsl_end)

      def init_times(self,time_duration=None):

          if time_duration:
             dtsl = np.int64( self.raw.time_as_index( time_duration ) )
          else:
             dtsl = np.int64( self.raw.time_as_index( self.time_duration ) )

          self.times = np.zeros(dtsl)
          dtsl      += self.tsl_onset
          if (dtsl > self.raw.last_samp):
              dtsl = None # include last idx
          self.times[:]= self.raw.times[self.tsl_onset:dtsl]
          self.times  -= self.times[0]  # start time _onsetseq at zero

          # self.meg.tsl_end = self.meg.tsl_onset + self.eeg.tsl_resamples.shape[0]
      def time_info(self,channel_info=False):
          print("\n " + "-" * 50)
          print(" --> " + self.system)
          print("  -> Time start: %12.3f end: %12.3f delta: %12.3f" % (self.time_onset,self.time_end,self.time_duration))
          print("  -> TSL  start: %12.3f end: %12.3f" % (self.tsl_onset,self.raw.last_samp))
          print(" " + "-" * 50)

          if ( channel_info ):
             print("  -> channel info: ")
             print("  ->  names      : " + ",".join([self.raw.ch_names[i] for i in self.picks]))
             print("  ->  index      : " + str(self.picks))
             print(" -->  resamples  : %d" % (self.tsl_resamples.shape[0]))
             print(" " + "-" * 50)

      def adjust_data_size(self):
          self.raw._data = self.raw._data[:, self.tsl_onset:self.tsl_end]  #+1
          self.is_data_size_adjusted = True

     #---------------------------------------------------------------------------
     #--- get_onset_and_events
     #----------------------------------------------------------------------------
      def get_onset_and_events(self):
          print(" --> call mne find events")
          ev_start_code_onset_idx = None
          
          # print self.events
          ev = mne.find_events(self.raw, **self.events)
          #if self.verbose:
          #   print "\n -->EVENTS:"
          #   print ev
          #   print"\n"
          
          if self.and_mask:
             ev[:, 1] = np.bitwise_and(ev[:, 1], self.and_mask)
             ev[:, 2] = np.bitwise_and(ev[:, 2], self.and_mask)
  
        #--- brainvision response_shift correction if response channel
          if ( ( self.system =='EEG') and ( self.stim_type == 'RESPONSE') ):
             ev[:, 1] -= self.response_shift  
             ev[:, 2] -= self.response_shift  
      
          ev_onsets  = np.squeeze( ev[np.where( ev[:,2] ),:])  # > 0

        #--- check  if no startcode  -> startcode not in/any  np.unique(ev[:, 2])
          if ( self.startcode in np.unique(ev[:, 2]) ):
            
             if ev_onsets.ndim > 1 :
                ev_id_idx = np.squeeze( np.where( np.in1d( ev_onsets[:,2],self.startcode )))
                ev_start_code_onset_idx =  np.int64( ev_onsets[ ev_id_idx,:] )[0].flatten()[0]
             else: # only one event code
                ev_start_code_onset_idx =  ev_onsets[ 0 ]
                ev_onsets = np.array([ev_onsets])
                ev_id_idx = 0 
          else:
             print("---> ERROR no startcode found %d " % (self.startcode))
             print("---> Events: ")
             print(np.unique(ev[:, 2]))
             print("\n")
             assert"ERROR no startcode found in events"              
          
          if self.verbose:
             print("-"*50)
             print(" --> Onset & Events Info : " + self.system) 
             print("  -> Onset index         : %d"%( ev_start_code_onset_idx))
             if self.debug:
                print("  -> Onsets              :")
                print(ev_onsets)
             print("-"*50)
            
          return ev_start_code_onset_idx,ev_onsets

class JuMEG_MergeMEEG(JuMEG_Base_IO):
    def __init__(self,adjust_size=True,save=True,startcode=128,copy_eeg_events_to_trigger=False,
                 meg={'stim_channel':'STI 014','min_duration':0.002,'shortest_event':3 },eeg={'stim_channel':'STI 014','response_shift':1000,'stim_type':'STIMULUS','and_mask':None} ):
        # meg={'stim_channel':'STI 013'},eeg={'stim_channel':'STI 014','response_shift':1000,'stim_type':'RESPONSE','and_mask':None} 
        '''
        Class JuMEG_MergeMEEG
           -> merge BrainVision ECG/EOG signals into MEG Fif file
           -> finding common onset via startcode in TRIGGER channel e.g <STI 014> <STI 013>
           -> filter data FIR lp butter 200Hz / 400Hz
           -> downsampling eeg data, srate sould be igher than meg e.g. eeg:5kHz meg:1017.25Hz
           -> rename groups and channel names dynamic & automatic !!!
           meg EEG 0001 -> ECG group ecg
           meg EEG 0002 -> EOG_xxx group eog
           meg EEG 0003 -> EOG_xxx group ecg

           input
            adjust= True : ajust raw-obj data size to startcode-onset and last common sample beween MEG and EEG data
            save  = True : save merged raw-obj

            parameter set via obj:

             meg_fname  = None <full file name>
             eeg_fname  = None <eeg brainvision vhdr full file name> with extention <.vhdr>  
             meeg_extention    = ",meeg-raw.fif" output extention

             stim_channel               = 'STI 014'
             startcode                  = 128  start code for common onset in <stim_channel>
             
             brainvision_channel_type   = 'STIMULUS'
             brainvision_stim_channel   = 'STI 014'
             brainvision_response_shift = 1000 used for mne.io.read_raw_brainvision will be added to the response value

             flags:
              verbose        = False
              do_adjust_size = True
              do_save        = True
              do_filter_meg  = True
              do_filter_eeg  = True

             filter option: can be canged via <obj.filter>
              filter.filter_method = "bw"
              filter.filter_type   ='lp'
              filter.fcut1         = None -> automatic selected  200Hz or 400Hz
              
              
            return:
             raw obj; new meeg file name

        '''

        super(JuMEG_MergeMEEG,self).__init__()

        self.meg = JuMEG_MergeMEEG_HiLoRate(system='MEG')
        self.meg.stim_channel   = meg['stim_channel']
        self.meg.shortest_event = meg['shortest_event'] 
        self.meg.min_duration   = meg['min_duration']
             
        self.eeg = JuMEG_MergeMEEG_HiLoRate(system='EEG')
        self.eeg.stim_channel   = eeg["stim_channel"]
        self.eeg.response_shift = eeg["response_shift"] # brainvision_response_shift to mark response bits to higher 8bit in STI channel
        self.eeg.stim_type      = eeg["stim_type"]
        self.eeg.and_mask       = eeg['and_mask']
        #self.eeg.startcode      = startcode
        self.brainvision_response_shift = self.eeg.response_shift
        
        self.copy_eeg_events_to_trigger = copy_eeg_events_to_trigger
        
       #--- output
        self.meeg_fname     = None
        self.meeg_extention = ",meeg-raw.fif"

       #--- change channel names and group
       # self.channel_types = {'EEG 001': u'ecg', 'EEG 002': u'eog', 'EEG 003': u'eog'}
       # self.channel_names = {'EEG 001': u'ECG 001', 'EEG 002': u'EOG hor', 'EEG 003': u'EOG ver'}

      #--- filter obj
        self.filter     = jumeg_filter( filter_method="bw",filter_type='lp',fcut1=None,fcut2=None,remove_dcoffset=False,notch=[] )
        self.__event_id = 128
     
        self.verbose      = False
        self.debug        = False
        self.do_adjust_data_size = adjust_size
        self.do_save        = save
        self.do_filter_meg  = True
        self.do_filter_eeg  = True
        #self.do_check_data_drift = check_data_drift
        self.startcode      = startcode
      
        self.bads_list      = []
        
    def ___get_ev_event_id(self):
        return self.__event_id
    def ___set_ev_event_id(self, v):
        self.__event_id = v 
        self.meg.startcode = v
        self.eeg.startcode = v
        
    event_id  = property(___get_ev_event_id, ___set_ev_event_id)
    startcode = property(___get_ev_event_id, ___set_ev_event_id)
       
# ---------------------------------------------------------------------------
# --- check_data
# ----------------------------------------------------------------------------
    def check_ids(self):
        im= self.get_id(f=self.meg.filename)
        ie= self.get_id(f=self.eeg.filename)
        assert(set( im.split() ) == set( ie.split() ) ), "ERROR -> check IDs: MEG: " + str(im) +" EEG: " + str(ie)
        return True

# ---------------------------------------------------------------------------
# --- check_filter_parameterself.get_id(f=self.meg.filename)
# ----------------------------------------------------------------------------
    def check_filter_parameter(self):

        if self.meg.raw.info['sfreq'] > self.eeg.raw.info['sfreq']:
           assert "Warning EEG data sampling fequency %4.3f is lower than MEG data %4.3f " % (self.eeg.raw.info['sfreq'], self.meg.raw.info['sfreq'])
           # return False
        return self.filter.calc_lowpass_value( self.meg.raw.info['sfreq'] )

#---------------------------------------------------------------------------
# --- apply_filter_meg_eeg
# ----------------------------------------------------------------------------
    def apply_filter_meg_eeg(self):
        """inplace filter all meg and eeg data except trigger channel with  lp
        using meg butter filter (FB)
        """
      # --- meg
        if self.do_filter_meg:
           print("---> Start filter meg data")
           self.filter.sampling_frequency = self.meg.raw.info['sfreq']
           self.filter.fcut1 = self.check_filter_parameter()
           print(" --> filter info: " + self.filter.filter_info)
           self.filter.apply_filter(self.meg.raw._data, picks=self.meg.picks)
           print(" --> Done filter meg")
      # ---  bv eeg
        if self.do_filter_eeg:
           print("---> Start filter bv eeg data")
           self.filter.sampling_frequency = self.eeg.raw.info['sfreq']
           print(" --> filter info: " + self.filter.filter_info)
           self.filter.apply_filter(self.eeg.raw._data, picks=self.eeg.picks)
           print(" --> Done filter eeg")


#---------------------------------------------------------------------------
#--- get_resample_index
#----------------------------------------------------------------------------
    def get_resample_index(self,timepoints_high_srate=None,timepoints_low_srate=None,sfreq_high=None):
        """Downsampling function to resample signal of samp_length from
        higher sampling frequency to lower sampling frequency.

        Parameters
        ----------
        input:
          timepoints_low_srate : np.array of time points with lower  sampling rate less size than the other
          timepoints_high_self.meg.raw._data[self.meg.raw.ch_names.index(chname), self.meg.tsl_onset:self.meg.tsl_end] = srate: np.array of time points with higher sampling rate
          sfreq_high           : higher sampling frequency.

        Returns
        -------
        resamp_idx: np.array of index of <timepoints_high_srate> to downsampled high sampled signal.
        """
        
        import numpy as np
       #---ToDo implementation in C for speed
        eps_limit  = round((0.90 / sfreq_high), 6) # distance  beween timepoints high/low  us
        resamp_idx = np.zeros( timepoints_low_srate.shape[0],dtype=np.int64 )
        j = 0
        idx=0
        for idx in np.arange(timepoints_low_srate.shape[0],dtype=np.int64):
            while (timepoints_low_srate[idx] - timepoints_high_srate[j] ) > eps_limit:
                   j += 1
            resamp_idx[idx] = j
            j += 1
        return resamp_idx

# ----------------------------------------------------------------------------
# --- check_data_drift
# ----------------------------------------------------------------------------
    def check_data_drift(self):
        """check data drift: if last common meg/eeg event code onset is in time resolution uncertainty
        dt < 1 x meg sampling periode
        Parameters
        ----------
         None
        
        Returns
        ---------
        time differences from startcode onsets between meg and eeg as numpy array in samples
        """
        # self.meg.tsl_onset:self.meg.tsl_end] = self.eeg.raw._data[eeg_idx, self.eeg.tsl_resamples
                                             
        meg_samp_periode = 1000.0 / self.meg.raw.info['sfreq']
                        
        eeg_code_idx = np.where( ( self.eeg.ev_onsets[:,-1] < self.eeg.response_shift ) & ( self.eeg.ev_onsets[:,0] >= self.eeg.tsl_onset ) )[0]
        eeg_code     = self.eeg.ev_onsets[eeg_code_idx,-1].flatten()
        eeg_counts   = eeg_code.shape[0]
        
        meg_code_idx      = np.where( self.meg.ev_onsets[:,0] >= self.meg.tsl_onset )[0]
        meg_code          = self.meg.ev_onsets[meg_code_idx,-1].flatten()
        meg_counts        = meg_code.shape[0]
         
        meeg_counts = meg_counts
        if eeg_counts < meeg_counts:
           meeg_counts = eeg_counts  
        meeg_idx = meeg_counts -1    
        
        meg_dt_last_idx = self.meg.ev_onsets[ meg_code_idx[meeg_idx],0 ] - self.meg.tsl_onset
        meg_dt_last     = self.meg.raw.times[ meg_dt_last_idx ]
        eeg_dt_last_idx = self.eeg.ev_onsets[ eeg_code_idx[meeg_idx],0 ] - self.eeg.tsl_onset
        eeg_dt_last     = self.eeg.raw.times[ eeg_dt_last_idx ]
        dif_dt_last     = np.abs(meg_dt_last - eeg_dt_last)
        dmeeg    =  np.zeros(meg_counts)      
        
        if self.verbose:
           
           print("\n --> check data drift:")  
          
           if self.debug:
              print(" " + "-" * 50)
              print("\n  -> MEG codes:")
              print(meg_code)
              #print self.meg.ev_onsets[:,-1]
              print("\n  -> EEG codes [ exclude RESPONSE codes ]:")
              print(eeg_code)
              
              print("  -> Info MEG/EEG onsets:  code  t[s]")   
              print(" " + "-" * 50) 
              i=0
              dmeeg    =  np.zeros(meg_counts) 
            
              while i < meeg_counts:
                  megcode  = meg_code[i]
                  megonset = self.meg.raw.times[ self.meg.ev_onsets[meg_code_idx[i],0] - self.meg.tsl_onset]
                  eegcode  = eeg_code[i]
                  eegonset = self.eeg.raw.times[ self.eeg.ev_onsets[eeg_code_idx[i],0] - self.eeg.tsl_onset]
                  dmeeg[i] = megonset - eegonset
                  print("->% 5d MEG code:% 7.1f t:% 10.4f  EEG code:% 7.1f t:% 10.4f div: % 7.5f" % (i,megcode,megonset,eegcode,eegonset,dmeeg[i]))
                  
                  i+=1
    
           dmeeg_abs= abs(dmeeg)
        
           print(" " + "-" * 50) 
           print("     MEG event counts: %8d" % (meg_counts))
           print("     EEG event counts: %8d" % (eeg_counts))
           print(" " + "-" * 50) 
           print("  -> Last Common Event Code:")
           print("     MEG last event onset        [s] : %0.6f" % (meg_dt_last))
           print("     EEG last event onset        [s] : %0.6f" % (eeg_dt_last))
           print("     dt  last event              [ms]: %0.6f" % (dif_dt_last))
           print(" " + "-" * 50) 
           print("     MEG onset tsl            : %8d" % (self.meg.tsl_onset))
           print("     EEG onset tsl            : %8d" % (self.eeg.tsl_onset))
           print("     startcode                : %6d" % (self.startcode))
           print("     meg sampling periode [ms]: %0.6f" % (meg_samp_periode))
           print(" " + "-" * 50)
           if self.debug:
              print("     Diverence  abs(MEEG)")
              print("      AVG [ms]: % 3.5f" % ( dmeeg_abs.mean() ))
              print("      STD [ms]: % 3.5f" % ( dmeeg_abs.std()  ))
              print("      MIN [ms]: % 3.5f" % ( dmeeg_abs.min()  ))
              print("      MAX [ms]: % 3.5f" % ( dmeeg_abs.max()  ))
              print(" " + "-" * 50)

        if ( dif_dt_last > meg_samp_periode ):
           ws="\n -->ERROR Data Drift ->last common meg/eeg event-code-onset is not in time resolution uncertainty\n -->MEG: %s\n -->EEG: %s" %( self.meg.filename,self.eeg.filename)
           print(ws +"\n\n")
             
        return dif_dt_last
  
    def fill_zeros_with_last(self,v):
        ''' 
           fill_zeros_with_last -> solves EEG Marker sampling problem 
           only BV markers onset is recorded as one timepoint/tslsampled with 5kHz
           downsampling will may lose some onset markers
           solution: extend marker onset till the next maker value occures
           https://stackoverflow.com/questions/30488961/fill-zero-values-of-1d-numpy-array-with-last-non-zero-values   
        '''
        arr    = np.zeros(len(v)+1)
        arr[1:]= v
        prev   = np.arange(len(arr))
        prev[arr == 0] = 0
        prev = np.maximum.accumulate(prev)
        return arr[prev][1:]
        
#----------------------------------------------------------------------------
#--- run
#----------------------------------------------------------------------------
    def run(self):
        """ run merge meg and eeg data
        
        Parameters
        ----------
        start jumeg_merger_meeg.py with -h
        
        Returns
        ---------
        fname: fif-file name
        raw  : raw obj

        """
        self.meg.verbose = self.verbose
        self.eeg.verbose = self.verbose
        self.meg.debug   = self.debug
        self.eeg.debug   = self.debug
      
        
        print("\n---> Start JuMEG MEEG Merger")  
        print("-" * 50)
             
        self.meg.check_path()
        self.eeg.check_path()
        self.check_ids()

       
       #--- load BV eeg
        print("-" * 50)
        print(" --> load EEG BrainVision data: " + self.eeg.filename)
        self.eeg.reset()
        self.eeg.raw,_ = self.get_raw_obj(self.eeg.filename,raw=self.eeg.raw)
       #--- get start code onset
        self.eeg.tsl_onset,self.eeg.ev_onsets = self.eeg.get_onset_and_events()
       # --- get ECG & EOG picks
        self.eeg.picks = self.picks.eeg(self.eeg.raw)

       # --- load meg
        print("-" * 50)
        print(" --> load MEG data:             " + self.meg.filename)
        self.meg.reset()
        self.meg.raw, _   = self.get_raw_obj(self.meg.filename,raw=self.meg.raw)
       # --- start code onset
        self.meg.tsl_onset,self.meg.ev_onsets = self.meg.get_onset_and_events()
       # --- get MEG & Ref picks
        self.meg.picks = self.picks.exclude_trigger(self.meg.raw)

       #--- adjust data length
       # meg longer or shorter than eeg
        if (self.eeg.time_duration < self.meg.time_duration):
            self.meg.init_times(time_duration=self.eeg.time_duration)
            self.eeg.init_times(time_duration=self.eeg.time_duration)
        else:
            self.meg.init_times(time_duration=self.meg.time_duration)
            self.eeg.init_times(time_duration=self.meg.time_duration)

      #--- cp index tp form bv to meg-cannels
        self.eeg.tsl_resamples  = self.get_resample_index(timepoints_high_srate=self.eeg.times,timepoints_low_srate=self.meg.times,sfreq_high=self.eeg.raw.info['sfreq'])
        self.eeg.tsl_resamples += self.eeg.tsl_onset # shift index-values in array to onset

        self.meg.time_info(channel_info=False)
        self.eeg.time_info(channel_info=True)
        self.check_data_drift()

      # --- filter meg / eeg data; no trigger
        self.apply_filter_meg_eeg()

       #--- downsampe & merge
       #--- ToDo check if EEG channel in raw exist or add channel
        meg_ch_idx = 1
        eeg_idx    = 0

        for ch in ( self.eeg.raw.ch_names ):  # ECG EOG_hor EOG_ver
            if ch.startswith('E'):
               eeg_idx = self.eeg.raw.ch_names.index(ch)
               chname  = "EEG %03d" %(meg_ch_idx)
              #--- copy eeg downsapled data into raw
               self.meg.raw._data[self.meg.raw.ch_names.index(chname), self.meg.tsl_onset:self.meg.tsl_end] = self.eeg.raw._data[eeg_idx, self.eeg.tsl_resamples]
              #--- rename groups and ch names
               self.meg.raw.set_channel_types( { chname:ch.split('_')[0].lower() } ) # e.g. ecg eog
               self.meg.raw.rename_channels(   { chname:ch.replace('_',' ')      } )
               meg_ch_idx += 1
            eeg_idx+=1
       
        if self.copy_eeg_events_to_trigger:
           print("\n --> Do copy EEG events to Trigger channel: STI 014") 
           eeg_idx = self.eeg.raw.ch_names.index('STI 014')
           # meg_idx = self.meg.raw.ch_names.index('STI 014')
           chname  = 'STI 014'
           deeg    = self.eeg.raw._data[eeg_idx]
           deeg[:] = self.fill_zeros_with_last(deeg)
           
           d    = self.meg.raw._data[self.meg.raw.ch_names.index(chname), self.meg.tsl_onset:self.meg.tsl_end]
           d[:] = deeg[self.eeg.tsl_resamples] - self.eeg.response_shift
           d[ np.where( d < 0 ) ] = 0        
           
      # --- adjust data size to merged data block -> cut meg-data to meg-onset <-> meg-end tsls
        if self.do_adjust_data_size:
           self.meg.adjust_data_size()
           chname  = 'STI 014'
           print("Eventcode adjusted: " +chname)
           print(self.meg.raw._data[self.meg.raw.ch_names.index(chname)])
      
      #--- upate bads 
        if self.bads_list:
           self.update_bad_channels(None,raw=self.meg.raw,bads=self.bads_list)
           
       #--- save meeg
        print("\n*" + "-" * 50)
        self.meeg_fname = self.get_fif_name(raw=self.meg.raw,extention=self.meeg_extention,update_raw_fname=True)
        print("---> start saving MEEG data: "+ self.meeg_fname)
        if self.do_save:
           self.apply_save_mne_data(self.meg.raw,fname=self.meeg_fname,overwrite=True)
        else:
           print(" -->set Bads; bad-channels in meeg-raw file:")
           print(self.picks.bads(self.meg.raw))
           print()
        
        print("---> DONE merge BrainVision EEG data to MEG")
        print("*" + "-" * 50)
        print("\n\n")

        return self.meeg_fname,self.meg.raw

#=========================================================================================
# script part
#=========================================================================================

#-----------------------------------------------------------------------------------------
#--- get_args
#-----------------------------------------------------------------------------------------
def __get_args():

    info_global = """
         JuMEG Merge BrainVision EEG data to MEG Data

         ---> merge eeg data with meg fif file 
          jumeg_merge_meeg -fmeg  <xyz.fif> -feeg <xyz.vdr> -r
         
         ---> merge eeg data with meg fif file from text file 
          jumeg_merge_meeg.py -seeg /data/meg_store1/exp/INTEXT/eeg/INTEXT01/ -smeg /data/meg_store1/exp/INTEXT/mne/ -plist /data/meg_store1/exp/INTEXT/doc/ -flist intext_merge_meeg.txt -sc 5 -b -v -r
          
        """
    info_meg_stage="""
               meg stage: start path for meg files from list
               -> start path to fif file directory structure
                  e.g. /data/megstore1/exp/INTEXT/mne/
               """
    info_eeg_stage="""
               eeg stage: start path for eeg files from list
               -> start path to eeg file directory structure
                  e.g. /data/megstore1/exp/INTEXT/eeg/INTEXT01/
               """
    info_flist = """
          path to text file with fif/eeg file list:
          intext_merge_list.txt 
          
          207006_INTEXT01_170629_0852_2_c,rfDC-raw.fif --feeg=207006_2.vhdr
          
      """       
       
   # --- parser

    parser = argparse.ArgumentParser(info_global)

   # ---input files
    parser.add_argument("-fmeg", "--meg_fname",help="meg fif file + full path")
    parser.add_argument("-feeg", "--eeg_fname",help="<eeg vhdr> file + full path")
  
   #---files  
    parser.add_argument("-smeg","--stage_meg", help=info_meg_stage)
    parser.add_argument("-seeg","--stage_eeg", help=info_eeg_stage)
   
    parser.add_argument("-plist","--path_list", help=info_flist,default=os.getcwd() )
    parser.add_argument("-flist","--fname_list",help="text file with fif files")
    parser.add_argument("-bads", "--bads_list", help="apply bad channels to mark as bad works only with <--set_bads> flag",default=("MEG 007,MEG 142,MEG 156,RFM 011") ) 
   
   #--- parameter
    parser.add_argument("-sc",   "--startcode",         help="start code marker to sync meeg and eeg data",default=128)
    parser.add_argument("-megst","--meg_stim_channel",  help="meg stimulus marker channel",default='STI 014')
    parser.add_argument("-eegst","--eeg_stim_channel",  help="eeg stimulus marker chennel",default='STI 014')
    parser.add_argument("-eegrt","--eeg_stim_type",     help="eeg stim type",              default='STIMULUS')
    parser.add_argument("-eegrs","--eeg_response_shift",help="eeg respose shift",          default=1000)
    parser.add_argument("-eegam","--eeg_and_mask",      help="eeg and mask",               default=None)
       
   # ---flags:
    parser.add_argument("-b", "--set_bads",action="store_true", help="apply default bad channels to mark as bad") 
    parser.add_argument("-v", "--verbose", action="store_true", help="verbose mode")
    parser.add_argument("-d", "--debug",   action="store_true", help="debug mode")
    parser.add_argument("-r", "--run",     action="store_true", help="!!! EXECUTE & RUN this program !!!")

    return parser.parse_args(), parser

#=========================================================================================
#==== MAIN
#=========================================================================================
def main(argv):

    opt, parser = __get_args()
    df=40
    
    if opt.verbose :
       print("\n---> ARGV parameter:")
       print(" --> fmeg        : " + str(opt.meg_fname))
       print(" --> feeg        : " + str(opt.eeg_fname))
       print("-"*df)
       print(" --> stage meg   : " + str(opt.stage_meg))
       print(" --> stage eeg   : " + str(opt.stage_eeg))
       print("-"*df)
       print(" --> path  list  : " + str(opt.path_list))
       print(" --> fname list  : " + str(opt.fname_list))
       print("-"*df)
       print(" --> start code         : " + str(opt.startcode))
       print(" --> meg_stim_channel   : " + str(opt.meg_stim_channel))
       print(" --> eeg_stim_channel   : " + str(opt.eeg_stim_channel))
       print(" --> eeg_stim_type      : " + str(opt.eeg_stim_type))
       print(" --> eeg_response_shift : " + str(opt.eeg_response_shift))
       print(" --> eeg_and_mask       : " + str(opt.eeg_and_mask)) 
       print("-"*df)
       print(" --> bads list   : " + str(opt.bads_list))
       print(" --> set bads [uses channels from bads list] : " + str(opt.set_bads))
       print("-"*df)    
       print(" --> verbose     : " + str(opt.verbose))
       print(" --> debug       : " + str(opt.debug))
       print(" --> run         : " + str(opt.run))
       print("-"*df)
       print("\n\n")  
       

    if opt.run:
       JuMEEG = JuMEG_MergeMEEG()
     #---         
       JuMEEG.verbose = opt.verbose        
       JuMEEG.debug   = opt.debug        
     #---  
       JuMEEG.meg.stim_channel   = opt.meg_stim_channel
       JuMEEG.eeg.stim_channel   = opt.eeg_stim_channel
       JuMEEG.eeg.stim_type      = opt.eeg_stim_type
       JuMEEG.eeg.response_shift = int(opt.eeg_response_shift)
       if opt.eeg_and_mask:
          JuMEEG.eeg.and_mask = int(opt.eeg_and_mask)
       else:
          JuMEEG.eeg.and_mask = opt.eeg_and_mask 
       
       fn_raw_list = []
       opt_dict    = {}      
           
     #--- check for single file pair or list
       if opt.fname_list:
          fn_raw_list,opt_dict = JuMEEG.get_filename_list_from_file(opt.path_list + "/" + opt.fname_list,start_path=opt.stage_meg)
       else:
          fn_raw_list.append(opt.meg_fname)
          opt_dict[opt.meg_fname] = {'bads': None, 'feeg': opt.eeg_fname}
          
     #--- loop preproc for each fif file
       for fif_file in (fn_raw_list) :
              
           feeg=None
           fmeg=fif_file
           
           if opt_dict[fif_file]['feeg']:
              feeg = str(opt_dict[fif_file]['feeg'])
              if opt.stage_eeg:
                 if os.path.isfile(opt.stage_eeg + '/' + feeg):
                       feeg= opt.stage_eeg + '/' + feeg
           if opt.stage_meg:
              if os.path.isfile(opt.stage_meg + '/' + fmeg):
                 fmeg= opt.stage_meg + '/' + fmeg
      
           JuMEEG.meg.filename = fmeg    
           JuMEEG.eeg.filename = feeg
              
         #--- ck bags add global bds                          
           JuMEEG.bads_list = []
           if opt.set_bads:
              JuMEEG.bads_list = opt.bads_list.split(',')
         #--- ck add individual bads  
           if "bads" in opt_dict[fif_file]:
              JuMEEG.bads_list.append( opt_dict[fif_file]['bads'].split(',') )
         #--- individual startcode     
           if "startcode" in opt_dict[fif_file]:
              JuMEEG.startcode = int( opt_dict[fif_file]["startcode"] )
           else:   
              JuMEEG.startcode = int(opt.startcode)
          
           JuMEEG.run()


if __name__ == "__main__":
    main(sys.argv)


'''
EXAMPLE file

#--- 207006
#207006/INTEXT01/170629_0852/2/207006_INTEXT01_170629_0852_2_c,rfDC-raw.fif --feeg=207006_INTEXT01_002.vhdr
#207006/INTEXT01/170629_0852/3/207006_INTEXT01_170629_0852_3_c,rfDC-raw.fif --feeg=207006_INTEXT01_003.vhdr
#207006/INTEXT01/170629_0852/4/207006_INTEXT01_170629_0852_4_c,rfDC-raw.fif --feeg=207006_INTEXT01_004.vhdr
#207006/INTEXT01/170629_0852/5/207006_INTEXT01_170629_0852_5_c,rfDC-raw.fif --feeg=207006_INTEXT01_005.vhdr
#207006/INTEXT01/170629_0852/6/207006_INTEXT01_170629_0852_6_c,rfDC-raw.fif --feeg=207006_INTEXT01_006.vhdr
#207006/INTEXT01/170629_0852/7/207006_INTEXT01_170629_0852_7_c,rfDC-raw.fif --feeg=207006_INTEXT01_007.vhdr
#--- resting state
207006/INTEXT01/170629_1027/1/207006_INTEXT01_170629_1027_1_c,rfDC-raw.fif --feeg=207006_INTEXT01_restclosed.vhdr --startcode=128 --bads=MEG 007,MEG 142
#207006/INTEXT01/170629_1027/2/207006_INTEXT01_170629_1027_2_c,rfDC-raw.fif --feeg=207006_INTEXT01_restopen.vhdr --startcode=128
#
#--- 208548
#208548/INTEXT01/170706_0915/3/208548_INTEXT01_170706_0915_3_c,rfDC-raw.fif --feeg=208548_INTEXT01_001.vhdr 
#208548/INTEXT01/170706_0915/4/208548_INTEXT01_170706_0915_4_c,rfDC-raw.fif --feeg=208548_INTEXT01_002.vhdr 
#208548/INTEXT01/170706_0915/5/208548_INTEXT01_170706_0915_5_c,rfDC-raw.fif --feeg=208548_INTEXT01_003.vhdr 
#208548/INTEXT01/170706_0915/6/208548_INTEXT01_170706_0915_6_c,rfDC-raw.fif --feeg=208548_INTEXT01_004.vhdr 
#208548/INTEXT01/170706_0915/7/208548_INTEXT01_170706_0915_7_c,rfDC-raw.fif --feeg=208548_INTEXT01_005.vhdr 
#208548/INTEXT01/170706_0915/7/208548_INTEXT01_170706_0915_8_c,rfDC-raw.fif --feeg=208548_INTEXT01_006.vhdr 
#--- resting state
#208548/INTEXT01/170706_1030/1/208548_INTEXT01_170706_1030_1_c,rfDC-raw.fif --feeg=208548_INTEXT01_restclosed.vhdr
#208548/INTEXT01/170706_1030/2/208548_INTEXT01_170706_1030_2_c,rfDC-raw.fif --feeg=208548_INTEXT01_restopen.vhdr
#

#---  example cmd calls
#jumeg_merge_meeg.py -seeg /data/meg_store1/exp/INTEXT/eeg/INTEXT01/ -smeg /data/meg_store1/exp/INTEXT/mne/ -plist /data_meg_store1/exp/INTEXT/doc/ -flist intext_merge_meeg.txt -sc 5 -b -v -r

#jumeg_merge_meeg.py -seeg ~/MEGBoers/data/exp/INTEXT/eeg/INTEXT01/ -smeg ~/MEGBoers/data/exp/INTEXT/mne/ -plist ~/MEGBoers/data/exp/INTEXT/doc/ -flist intext_merge_meeg.txt -sc 5 -b -v -r -d

'''
