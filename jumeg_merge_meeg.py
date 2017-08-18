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

'''


import os,sys,argparse
import numpy as np
import mne

from jumeg.jumeg_base                   import JuMEG_Base_IO
from jumeg.filter.jumeg_filter          import jumeg_filter

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
          
          self.data          = np.array([])
          self.times         = np.array([])
          self.ev_onsets     = np.array([])
          self.tsl_resamples = np.array([],dtype=np.int64)
          self.tsl_onset     = None
          self.time_delta    = None
          self.picks         = None
          self.is_data_size_adjusted = False
        
        #---meg defaults
          self.event_parameter = {'and_mask':255,'stim_type':'RESPONSE','response_shift':None}
          self.events          = {'consecutive': True, 'output': 'step', 'stim_channel': 'STI 014','min_duration': 0.00001, 'shortest_event': 1, 'mask': 0}
      
      def ___get_stim_channel(self, v):
          return self.events['stim_channel']
      def ___set_stim_channel(self, v):
          self.events['stim_channel'] = str(v)
      stim_channel = property(___get_stim_channel, ___set_stim_channel)
    
      def ___get_and_mask(self):
          return self.event_parameter['and_mask']
      def ___set_and_mask(self, v):
          self.event_parameter['and_mask'] = v
      and_mask=property(___get_and_mask,___set_and_mask)
    
      def ___get_stim_type(self):
          return self.event_parameter['stim_type']
      def ___set_stim_type(self, v):
          self.event_parameter['stim_type'] = v
      stim_type=property(___get_stim_type,___set_stim_type)
    
      def ___get_response_shift(self):
          return self.event_parameter['response_shift']
      
      def ___set_response_shift(self, v):
          self.event_parameter['response_shift'] = v
      response_shift=property(___get_response_shift,___set_response_shift)
    
     #--- verbose
      def __set_verbose(self,value):
          self.__verbose = value
      def __get_verbose(self):
          return self.__verbose
      verbose = property(__get_verbose, __set_verbose)


  #---   
      def check_path(self):
          print " --> " + self.system + " file name: " + str(self.filename)
          assert( os.path.isfile( self.filename ) ),"---> ERROR " + self.system +" file: no such file: " + str(self.filename)
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
          print"\n " + "-" * 50
          print" --> " + self.system
          print"  -> Time start: %12.3f end: %12.3f delta: %12.3f" % (self.time_onset,self.time_end,self.time_duration)
          print"  -> TSL  start: %12.3f end: %12.3f" % (self.tsl_onset,self.raw.last_samp)
          print" " + "-" * 50

          if ( channel_info ):
             print"  -> channel info: "
             print"  ->  names      : " + ",".join([self.raw.ch_names[i] for i in self.picks])
             print"  ->  index      : " + str(self.picks)
             print" -->  resamples  : %d" % (self.tsl_resamples.shape[0])
             print" " + "-" * 50

      def adjust_data_size(self):
          self.raw._data = self.raw._data[:, self.tsl_onset:self.tsl_end]  #+1
          self.is_data_size_adjusted = True

     # ---------------------------------------------------------------------------
     # --- get_onset_and_events
     # ----------------------------------------------------------------------------
      def get_onset_and_events(self):
          print " --> call mne find events"
          ev_start_code_onset_idx = None

          ev = mne.find_events(self.raw, **self.events)
          print "\n -->EVENTS:"
          print ev
          print"\n"
          
          if self.and_mask:
             ev[:, 1] = np.bitwise_and(ev[:, 1], self.and_mask)
             ev[:, 2] = np.bitwise_and(ev[:, 2], self.and_mask)
  
        #--- brainvision response_shift correction if response channel
          if ( ( self.system =='EEG') and ( self.stim_type == 'RESPONSE') ):
             ev[:, 1] -= self.response_shift  
             ev[:, 2] -= self.response_shift  
      
          ev_onsets  = np.squeeze( ev[np.where( ev[:,2] ),:])  # > 0

         # ev_offset = np.squeeze( ev[np.where( ev[:,1] ),:])
       #--- check  if no startcode  -> startcode not in/any  np.unique(ev[:, 2])
          if ( self.startcode in np.unique(ev[:, 2]) ):
            
             if ev_onsets.ndim > 1 :
                ev_id_idx = np.squeeze( np.where( np.in1d( ev_onsets[:,2],self.startcode )))
                ev_start_code_onset_idx =  np.int64( ev_onsets[ ev_id_idx,:] )[0].flatten()[0]
             else: # only one event code
                ev_start_code_onset_idx =  ev_onsets[ 0 ]
                ev_onsets = np.array([ev_onsets])
                 
          else:
             print"---> ERROR no startcode found %d " % (self.startcode)
             print"---> Events: "
             print np.unique(ev[:, 2])
             print"\n"
             assert"ERROR no startcode found in events"
          
          if self.verbose:
             print "-"*50
             print" --> Onset & Events Info : " + self.system 
             print"  -> Onset index         : %d"%( ev_start_code_onset_idx)
             print"  -> Onsets              :"
             print ev_onsets
             print "-"*50
            
          return ev_start_code_onset_idx,ev_onsets

class JuMEG_MergeMEEG(JuMEG_Base_IO):
    def __init__(self,adjust_size=True,save=True,startcode=128,meg={'stim_channel':'STI 013'},eeg={'stim_channel':'STI 014','response_shift':1000,'stim_type':'RESPONSE','and_mask':None} ):
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
             meeg_extention    = ",eeg-raw.fif" output extention

             stim_channel               = 'STI 013'
             startcode                  = 128  start code for common onset in <stim_channel>
             
             brainvision_channel_type   = 'RESPONSE'
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
              filter.fcut1         = None automatic selected  200Hz or 400Hz
              
              
            return:
             raw obj; new meeg file name

        '''

        super(JuMEG_MergeMEEG,self).__init__()

        self.__version__= '2016.12.24.001'

        self.meg = JuMEG_MergeMEEG_HiLoRate(system='MEG')
        self.meg.stim_channel = meg['stim_channel']
        self.meg.startcode    = startcode
        
        self.eeg = JuMEG_MergeMEEG_HiLoRate(system='EEG')
        self.eeg.stim_channel   = eeg["stim_channel"]
        self.eeg.response_shift = eeg["response_shift"] # brainvision_response_shift to mark response bits to higher 8bit in STI channel
        self.eeg.stim_type      = eeg["stim_type"]
        self.eeg.and_mask       = eeg['and_mask']
        self.eeg.startcode      = startcode
        self.brainvision_response_shift = self.eeg.response_shift
        
        self.DO_MERGE_EEG_TRIGGER = False
        
       #--- output
        self.meeg_fname     = None
        self.meeg_extention = ",meeg-raw.fif"

       #--- change channel names and group
       # self.channel_types = {'EEG 001': u'ecg', 'EEG 002': u'eog', 'EEG 003': u'eog'}
       # self.channel_names = {'EEG 001': u'ECG 001', 'EEG 002': u'EOG hor', 'EEG 003': u'EOG ver'}

      #--- filter obj
        self.filter     = jumeg_filter( filter_method="bw",filter_type='lp',fcut1=None,fcut2=None,remove_dcoffset=False,notch=[] )
        self.__event_id = 128
     
        self.verbose        = False
        self.do_adjust_data_size = adjust_size
        self.do_save        = save
        self.do_filter_meg  = True
        self.do_filter_eeg  = True
        
        self.startcode      = startcode
      
    def ___get_ev_event_id(self):
        return self.__event_id
    def ___set_ev_event_id(self, v):
        self.__event_id = v 
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
           return False
        return self.filter.calc_lowpass_value( self.meg.raw.info['sfreq'] )

#---------------------------------------------------------------------------
# --- apply_filter_meg_eeg
# ----------------------------------------------------------------------------
    def apply_filter_meg_eeg(self):
       '''
          inplace filter all meg and eeg data except trigger channel with  lp
        '''
      # --- meg
       if self.do_filter_meg:
          print "---> Start filter meg data"
          self.filter.sampling_frequency = self.meg.raw.info['sfreq']
          self.filter.fcut1 = self.check_filter_parameter()
          print " --> filter info: " + self.filter.filter_info
          self.filter.apply_filter(self.meg.raw._data, picks=self.meg.picks)
          print " --> Done filter meg"
      # ---  bv eeg
       if self.do_filter_eeg:
          print "---> Start filter bv eeg data"
          self.filter.sampling_frequency = self.eeg.raw.info['sfreq']
          print " --> filter info: " + self.filter.filter_info
          self.filter.apply_filter(self.eeg.raw._data, picks=self.eeg.picks)
          print " --> Done filter eeg"


#---------------------------------------------------------------------------
#--- get_resample_index
#----------------------------------------------------------------------------
    def get_resample_index(self,timepoints_high_srate=None,timepoints_low_srate=None,sfreq_high=None):
        '''
        Downsampling function to resample signal of samp_length from
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
        '''
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
        '''  check data drift: if last common meg/eeg event code onset is in time resolution uncertainty
             dt < 2 x meg sampling periode
        '''
        meg_dtmax  = self.meg.raw.times[ self.meg.ev_onsets[-1,0] ] - self.meg.raw.times[ self.meg.ev_onsets[0,0] ]
        eeg_dtmax  = self.eeg.raw.times[ self.eeg.ev_onsets[-1, 0]] - self.eeg.raw.times[ self.eeg.ev_onsets[0,0] ]
        dif = np.abs(meg_dtmax - eeg_dtmax)

        if self.verbose:
           print" --> check data drift:"
           print"     MEG last event onset: %0.6f" % (meg_dtmax)
           print"     EEG last event onset: %0.6f" % (eeg_dtmax)
           print"     dt [ms]             : %0.6f" % (dif)
           print"     meg sampling periode [ms]: %0.6f" % (1000.0 / self.meg.raw.info['sfreq'])
           print" " + "-" * 50

        if ( dif > 2000.0 / self.meg.raw.info['sfreq'] ):
           assert('ERROR Data Drift ->last common meg/eeg event-code-onset is not in time resolution uncertainty')

        return dif

       #-- example stuff
       #  dt_meg    = tmeg - np.roll(tmeg, 1)
       #  dt_meg[0] = 0.0
       #  dt_eeg    = teeg - np.roll(teeg, 1)
       #  dt_eeg[0] = 0.0
       #  plt.plot( dt/dt_eeg )
       # mne.viz.plot_events(self.meg.ev_onsets, self.meg.raw.info['sfreq'],
       #                      self.meg.tsl_onset)  # color=color,event_id=event_id)

#----------------------------------------------------------------------------
#--- run
#----------------------------------------------------------------------------
    def run(self):
        """
            merge brainvision eeg data into MEG-fif file

            RETURN:
                   fname          : fif-file name,
                   raw            : raw obj

        """
        self.meg.check_path()
        self.eeg.check_path()
        self.check_ids()

        self.meg.verbose = self.verbose
        self.eeg.verbose = self.verbose
      
       #--- load BV eeg
        print"\n " + "-" * 50
        print " --> load EEG BrainVision data: " + self.eeg.filename
        self.eeg.raw,_ = self.get_raw_obj(self.eeg.filename,raw=self.eeg.raw)
       #--- get start code onset
        self.eeg.tsl_onset,self.eeg.ev_onsets = self.eeg.get_onset_and_events()
       # --- get ECG & EOG picks
        self.eeg.picks = self.picks.eeg(self.eeg.raw)

       # --- load meg
        print"\n " + "-" * 50
        print " --> load MEG data:             " + self.meg.filename
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

        if self.DO_MERGE_EEG_TRIGGER:
           eeg_idx = self.eeg.raw.ch_names.index('STI 014')
           # meg_idx = self.meg.raw.ch_names.index('STI 014')
           chname  = 'STI 014'
           d    = self.meg.raw._data[self.meg.raw.ch_names.index(chname), self.meg.tsl_onset:self.meg.tsl_end]
           d[:] = self.eeg.raw._data[eeg_idx, self.eeg.tsl_resamples] - self.eeg.response_shift
           d[ np.where( d < 0 ) ] = 0
           
           #self.meg.raw._data[self.meg.raw.ch_names.index(chname), self.meg.tsl_onset:self.meg.tsl_end] = self.eeg.raw._data[eeg_idx, self.eeg.tsl_resamples]
           
           #self.meg.raw._data[self.meg.raw.ch_names.index(chname), self.meg.tsl_onset:self.meg.tsl_end] -= self.eeg.response_shift
           
           
           
      # --- adjust data size to merged data block -> cut meg-data to meg-onset <-> meg-end tsls
        if self.do_adjust_data_size:
           self.meg.adjust_data_size()

       #--- save meeg
        print"\n*" + "-" * 50
        self.meeg_fname = self.get_fif_name(raw=self.meg.raw,extention=self.meeg_extention,update_raw_fname=True)
        print "---> start saving MEEG data: "+ self.meeg_fname
        if self.do_save:
           self.apply_save_mne_data(self.meg.raw,fname=self.meeg_fname,overwrite=True)
        print "---> DONE merge BrainVision EEG data to MEG"
        print"*" + "-" * 50
        print"\n"

        return self.meeg_fname,self.meg.raw


def __get_args():

    info_global = """
         JuMEG Merge BrainVision EEG data to MEG Data

         ---> porcess fif file for experiment MEG94T
          jumeg_merge_meeg -fmeg  <xyz.fif> -feeg <xyz.vdr>
        """

        # --- parser

    parser = argparse.ArgumentParser(info_global)

   # ---input files
    parser.add_argument("-fmeg", "--meg_fname", help="meg fif  file name with full path", default='None')
    parser.add_argument("-feeg", "--eeg_fname", help="eeg vhdr file name with full path", default='None')
   # ---flags:
    parser.add_argument("-v", "--verbose", action="store_true", help="verbose mode")
    parser.add_argument("-r", "--run", action="store_true", help="!!! EXECUTE & RUN this program !!!")

    return parser.parse_args(), parser

#=========================================================================================
#==== MAIN
#=========================================================================================
def main(argv):

    opt, parser = __get_args()

    JuMEEG = JuMEG_MergeMEEG()
    JuMEEG.meg.filename = opt.meg_fname
    JuMEEG.eeg.filename = opt.eeg_fname
    JuMEEG.verbose      = opt.verbose

    if opt.run:
       JuMEEG.run()

if __name__ == "__main__":
    main(sys.argv)


'''
jumeg_merge_meeg.py -r -fmeg ~/MEGBoers/data/exp/LDAEP/mne/200098/LDAEP02/130415_1526/1/200098_LDAEP02_130415_1526_1_c,rfDC-raw.fif -feeg ~/MEGBoers/data/exp/LDAEP/eeg/LDAEP02/200098_LDAEP02_2.vhdr

'''
