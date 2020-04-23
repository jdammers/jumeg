#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 15:50:22 2020

@author: fboers
"""
#!/usr/bin/env python3
# -+-coding: utf-8 -+-

#--------------------------------------------
# Authors:
# Frank Boers      <f.boers@fz-juelich.de>
# Jürgen Dammers   <j.dammers@fz-juelich.de>
#--------------------------------------------
# Date: 03.04.2020
#--------------------------------------------
# License: BSD (3-clause)
#--------------------------------------------
# Updates
#--------------------------------------------
import sys,os
import numpy as np
import mne

from numba import jit
#---
from jumeg.base                    import jumeg_logger
from jumeg.base.jumeg_base         import jumeg_base as jb
from jumeg.base.jumeg_base         import JUMEG_SLOTS

np.set_printoptions(formatter={'float': '{: 0.3f}'.format},threshold=sys.maxsize)
     
logger = jumeg_logger.get_logger()

__version__= "2020.04.21.001"

#@jit (nopython=True)
def get_chop_times_indices(times, chop_length=60., chop_nsamp=None, strict=False):
    """
    calculate chop times for every X s
    where X=interval.
    
    Author: J.Dammers

    Parameters
    ----------
    times: the time array
    chop_length: float  (in seconds)
    chop_nsamp: int (number of samples per chop)
                if set, chop_length is ignored

    strict: boolean (only when chop_samp=None)
            True: use be strict with the length of the chop
                  If the length of the last chop is less than X
                  the last chop is combined with the penultimate chop.
            False: (default) the full time is equally distributed across chops
                   The last chop will only have a few samples more

    Returns
    -------
    chop_times : list of float
                 Time range for each chop
    chop_time_indices : list of indices defining the time range for each chop

    """

    n_times = len(times)
   
    try:
        data_type = times.dtype()
    except:
        data_type = np.float64 

    if chop_nsamp:  # compute chop based on number of samples
        n_chops = int(n_times // chop_nsamp)
        if n_chops == 0:
            n_chops = 1
        n_times_chop = chop_nsamp
    else:  # compute chop based on duration given
        dt = times[1] - times[0]  # time period between two time samples
        n_chops, t_rest = np.divmod(times[-1], chop_length)
        n_chops = int(n_chops)
        # chop duration in s
        if strict:
            chop_len = chop_length
        else:
            chop_len = chop_length + t_rest // n_chops  # add rest to chop_length
        n_times_chop = int(chop_len / dt)
        # check if chop length is larger than max time (e.g. if strict=True)
        if n_times_chop > n_times:
            n_times_chop = n_times

    # compute indices for each chop
    ix_start = np.arange(n_chops) * n_times_chop  # first indices of each chop
    ix_end = np.append((ix_start - 1)[1:], n_times - 1)  # add last entry with last index

    # chop indices
    chop_indices = np.zeros([n_chops, 2], dtype=np.int)
    chop_indices[:, 0] = ix_start
    chop_indices[:, 1] = ix_end

    # times in s
    chop_times = np.zeros([n_chops, 2], dtype=data_type)
    chop_times[:, 0] = times[ix_start]
    chop_times[:, 1] = times[ix_end]

    return chop_times, chop_indices


@jit (nopython=True)
def find_events_in_window(events,window_tsl,ids=None):
    '''
    Parameters
    ----------
     events     : np.array([n,3]) mne events
     window_tsl : np.array [ts0,tsl1]
     ids        : ToDo search for id in id list
                  [event code list, optional,[1,2,3] for 'ECG','EOG hor','EOG ver'.]

    Returns
    -------
     np.array , mne events in window
    '''
     
   #--- find ECG,EOG hor, EOG ver 1,2,3 in window 
    evt = None
    if events.shape[0]:
       evt_idx = np.where( (events[:,0] > window_tsl[0]) &
                           (events[:,0] < window_tsl[1]) &
                           (events[:,2] > 0) )[0]
       if evt_idx.shape[0]:
          evt = events[evt_idx,:]
       
    return evt

#@jit (nopython=True)
def get_events_in_window(window_tsl=None,data=None,picks=None,events=None):
    """
   
    Parameters
    ----------
    window_tsl : TYPE, optional
        DESCRIPTION. The default is None.
    raw : TYPE, optional
        DESCRIPTION. The default is None.
    picks : TYPE, optional
        DESCRIPTION. The default is None.
    events : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    evt : TYPE
        DESCRIPTION.

    """
    evt = None   
    
       #--- stim data  
    if events is not None:
       d = np.zeros([len(picks) +1, window_tsl[1] - window_tsl[0] ],dtype=np.float)
       d[0:-1,:] += data[picks,window_tsl[0]:window_tsl[1] ]
    
      #--- ToDo vectorize NUMA 
       onset  = events[:,0] - window_tsl[0] -2
       idx = np.where(onset<0)[0]
       if idx.shape[0]:
           onset[idx] = 0
       
       offset = onset + 4
       idx    = np.where(offset > data.shape[-1])[0]
       offset[idx] = d.shape[-1] 
       
       for idx in range( onset.shape[0] ):
           d[ -1,onset[idx]:offset[idx] ]+=1
           
       d = d.sum(axis=0).astype(np.int)
    else:
       d = data[picks,window_tsl[0]:window_tsl[1] ].sum(axis=0).astype(np.int)
  
    didx = np.where( d[0:-1] - d[1:] )[0]+1

    #--- if codes in window -> find events in window
    if didx.shape[0]: 
       #--- mne event array
       evt        = np.zeros( [ didx.shape[0],3 ],dtype=np.int) 
       evt[:,0]  += didx + window_tsl[0]
       evt[:,2]  += d[didx]
       evt[1:,1] += d[didx][0:-1]
       
       #logger.debug(" window tsl: {}\n events:\n{}\n".format(window_tsl,evt) )
 
    return evt

    
def copy_crop_and_chop(raw,chop,verbose=False):
    '''
    chop MNE RAW obj [copy & crop]
    
    Parameters
    ----------
    raw     : mne RAW obj
    chop    : np.array or list [chop start, chop end] in sec
    verbose : bool, <False>
    Returns
    -------
    raw_crop : croped RAW obj
    '''
    raw_crop = raw.copy().crop(tmin=chop[0],tmax=chop[1])
    if verbose:
       logger.info("RAW Crop Annotation : {}\n  -> tmin: {} tmax: {}\n {}\n".format(jb.get_raw_filename(raw),chop[0],chop[1],raw_crop.annotations))
    return raw_crop
        
     
    
class JuMEG_PIPELINES_CHOPPER(JUMEG_SLOTS):
    '''
    estimate ica chops based on chop-length,
    adjust chops for artifact,stimulus and response events
    e.g. try to avoid chop onset/offset close to this events
    -> search for longest time period with no events arround estimated
       chop time [offset] in a defined time window 
    -> store chop offsets in annotations with <description>
    -> if <show>: show chops offset with raw.plot()
    
    Parameters
    ----------
    raw     : mne RAW obj, <None>
    length  : chop length in sec
              default: 120.0
    
    time_window_sec: np.array,list,tuple
              search for <no event period> in 
              pre/post time 
              default: [5.0,5.0]
       
    description: string, label of annotation description 
                 to store chop offsets in annotations 
                 default: ica-chops
        
    show    : bool, <False>  show end of chop via raw.plot()
    verbose : bool, <False>
    debug   : bool, <False>
    
    Returns
    -------
    
    
    Example
    -------
    from jumeg.base                                    import jumeg_logger
    from jumeg.base.jumeg_base                         import jumeg_base as jb
    from jumeg.base.pipelines.jumeg_pipelines_chopper  import JuMEG_PIPELINES_CHOPPER

    logger = jumeg_logger.get_logger()

    fin  = test.fif
    raw,fname = jb.get_raw_obj(fname=fin)
   
    jCP = JuMEG_PIPELINES_CHOPPER()
    jCP.update(raw=raw,verbose=True,debug=True,show=True)
    
    '''
        
    __slots__=["length","verbose","debug","show","_time_window_sec","_raw","_description",
               "_estimated_chops","_estimated_indices","_chops","_indices"]
    
    
    def __init__(self,**kwargs):
        self._init(**kwargs)
        
        self.length          = 120.0
        self.description     = "ica-chops"
        self.time_window_sec = np.array([5.0,5.0])
        
    def _update_from_kwargs(self,**kwargs):
        super()._update_from_kwargs(**kwargs)
        
        if "raw" in kwargs:
           self.raw = kwargs.get("raw")
        if "time_window_sec" in kwargs:
           self.time_window_sec = kwargs.get("time_window_sec")
        if "description" in kwargs:   
           self.description = kwargs.get("description")    
        
    @property
    def description(self): return self._description
    @description.setter
    def description(self,v):
        self._description=v
        
    @property
    def time_window_sec(self): return self._time_window_sec
    @time_window_sec.setter
    def time_window_sec(self,v):
        self._time_window_sec = np.zeros(2,dtype=np.float)
        if isinstance(v,(list,tuple,np.ndarray)):
           self._time_window_sec[0] = v[0]
           self._time_window_sec[1] = v[-1]
           if not self._time_window_sec[1]:
              self._time_window_sec[1] = v[0]
        else:
           self._time_window_sec += v
           
    @property  
    def chops(self):   return self._chops
    @property  
    def n_chops(self):
        try:
           return self._chops.shape[0]
        except:
          return None
      
    @property  
    def indices(self): return self._indices
     
    @property  
    def estimated_chops(self):   return self._estimated_chops
    @property  
    def estimated_indices(self): return self._estimated_indices
   #--- 
    @property
    def raw(self): return self._raw
    @raw.setter
    def raw(self,v): 
        self._raw = v
    @property  
    def times(self): return self._raw.times 
    @property
    def sfreq(self): return self._raw.info['sfreq']  
 
    def chops_as_string(self):
        '''
        
        Returns
        -------
        None.

        '''
        s = ""
        for cp in self.chops:
            s+="{:0.3f}-{:0.3f} ".format(cp[0],cp[1])
        return s
      
    def indices_as_string(self):
        '''
        
        Returns
        -------
        None.

        '''
        s = ""
        for cp in self.indices:
            s+="{}-{} ".format(cp[0],cp[1])
        return s
        
    def _calc_estimated_chops_from_timepoints(self):
        """
        calc chop times & indices

        Returns 
         chops [sec], indices [samples] 
        -------
        TYPE
            np.array [nchops [ onset,offset]] 
      
        DESCRIPTION.
         estimated 

        """
        if self.verbose:
           logger.info("Start calc Chop Times: length: {} raw time: {}".format(self.length,self.times[-1]))
        
        self._chops             = None
        self._indices           = None
        self._estimated_chops   = None
        self._estimated_indices = None
        
        #--- warn if times less than chop length
        if self.times[-1] <= self.length:
           logger.warning("<Raw Times> : {} smaler than <Chop Times> : {}\n\n".format(self.times[-1],self.length))
                       
        self._estimated_chops,self._estimated_indices = get_chop_times_indices(self.times,chop_length=self.length) 
        
        if self.verbose:
           self.GetInfo()
           
        return self._estimated_chops,self._estimated_indices
   
  
    def _calc_chop_from_events(self,evt):
        '''
        calc chop from events
        find the maximum distance in samples between events 
        onset/offset

        Parameters
        ----------
        evt : np.array([n,3])
              mne event struct

        Returns
        -------
         int, sample
     
        '''
       #--- find zero code windows
        didx = np.where( evt[:,2] == 0 )[0] # get offsets
       #--- ck for no offset / onset => no zero event code
        if not didx.shape[0]:   # no zero code ?!!!
           return None
           
       #--- ck if last one is zero code
        if didx[-1] == evt.shape[0]-1:
           didx = didx[0:-1]    
               
        zeros_onset_tsl  = evt[didx,0]
        zeros_offset_tsl = evt[didx+1,0]
        diff_tsl         = zeros_offset_tsl - zeros_onset_tsl
            
        idx_tsl = np.argmax( np.abs(diff_tsl) )
        evt_idx = didx[idx_tsl]
        return int( evt[evt_idx,0] + diff_tsl[idx_tsl] / 2.0 )
    
    def _adjust_chops_for_stimulus_response(self,chops=None):
        """
        adjust chop onset not in stimulus / response window 
        chops: TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
         chops,indices
       
        """
     
       #--- define search window for events 
        time_window_tsl = np.array(self.time_window_sec * self.sfreq).astype(np.int)
       
        picks = jb.picks.stim_response(self.raw)
        anno_evt,anno_labels = mne.events_from_annotations(self.raw)
       
        self._chops   = None
        self._indices = None
        
        if not chops:
           chops   = self._estimated_chops.copy()
           indices = self._estimated_indices.copy()
        else:
           indices = (chops * self.sfreq).astype(int)
        n_chops = len(chops)
        
        if self.debug:
           logger.debug("Input: {}\n  -> Chops :{}\n  -> Indices: {}".format(n_chops,chops,indices))
        
        idx = 0
        stim_indices = np.zeros(n_chops,dtype=np.int)
          
         
        for tsls in indices:
            stim_indices[idx] = tsls[1]
            if self.debug:
               logger.debug("STIM last sample: {}\n  -> TSLS: {}".format(self.raw.last_samp,tsls))
            
            if tsls[1] >= self.raw.last_samp:
               idx += 1
               continue 
            if self.debug:
               logger.debug("STIM: {}\n TSLS: {}\n stim indices: {}\n".format(idx,tsls,stim_indices))
        
            window_tsl =[ tsls[1] - time_window_tsl[0],tsls[1] + time_window_tsl[1]]
           
           #--- find ECG,EOG hor, EOG ver 1,2,3 in window 
            evt_artifacts = find_events_in_window(anno_evt,window_tsl)
           #--- get onset & offset from stim,response and artifacts events
            evt = get_events_in_window(data=self.raw._data,window_tsl=window_tsl,picks=picks,events=evt_artifacts)
           
            if self.debug:
               logger.debug("idx: {}\n Chops:\n{}\n indices:\n{}\n events:\n{}\n".format(idx,chops,indices,evt) )
            '''
            Chop:    [ 0.000  152.999]
            indices: [     0   155638]
            stim events:[
                 [150827      0      0]
                 [150980      0     40]
                 [155935     40      0]
                 [156087      0     30]
                 [157988     30      0]
                 [158140      0     55]
                 [159431     55      0]
                 [159584      0     80]]
            '''
     
            if evt is None:
               idx += 1
               continue
            
            stim_indices[idx] = self._calc_chop_from_events(evt)
            if not stim_indices[idx]: 
               msg=["can not find chop window with zero event code =>\n",
                    " using TSL : {}\n".format(tsls[1]),
                    "  -> idx   : {}]\n".format(idx),
                    "  -> chop  : {}\n".format(chops[idx]), 
                    "  -> events:\n{}\n".format(evt)]
                     
               logger.warning( "".join(msg) )
               idx += 1
               continue
           
            idx += 1
           
       #--- set annotations 
        indices[:,1]  = stim_indices #[idx]
        indices[1:,0] = indices[0:-1,1]
        
        for idx in range(n_chops):
            chops[idx][0]= self.raw.times[ indices[idx][0] ]
            chops[idx][1]= self.raw.times[ indices[idx][1] ]
        
        logger.info("adjust chops for stimulus,response and artifact on/offsets:\n{}\n Chop:\n{}\n => indices:\n{}\n stim\n{}\n".format(idx,chops,indices,stim_indices) )
        
        self._chops   = chops
        self._indices = indices
        
        return chops,indices       
        
    def _update_annotations(self):
       
        duration   = np.ones( self.chops.shape[0] ) / self.sfreq  # one line in raw.plot
        chop_annot = mne.Annotations(onset      = self.chops[:,1].tolist(),
                                     duration   = duration.tolist(),
                                     description=self.description,
                                     orig_time  = None)
        
        msg = ["Update Annotations with description: {}".format(self.description)]
      
        try:
            raw_annot  = self.raw.annotations #.copy()
            raw_annot.orig_time = None
        except:
            raw_annot = None
       
        if raw_annot:
           #--- clear old annotations
            kidx = np.where( raw_annot.description == self.description)[0] # get index
            if kidx.any():
                msg.append("  -> delete existing annotation {} counts: {}".format(self.description,kidx.shape[0]) )
                raw_annot.delete(kidx)
       
            self.raw.set_annotations( raw_annot + chop_annot)
        else:
            self.raw.set_annotations(chop_annot)
       
        if self.verbose:
           idx = np.where(self.raw.annotations.description == self.description)[0]
           msg.extend(["  -> onsets:\n{}".format(self.raw.annotations.onset[idx]),
                       "-"*40,
                       " --> mne.annotations in RAW:\n  -> {}".format(self.raw.annotations)])
           
           logger.info("\n".join(msg))
        
  
    def update(self,**kwargs):
        """
        calc chops from timepoints 
        adjust for stimulus/response onset/offset
        adjust for EOG,ECG onsets
        """
        self._update_from_kwargs(**kwargs)
       #--- calc estimated chops from chop length
        self._calc_estimated_chops_from_timepoints()
       #--- adjust chops for stimulusresponse
        self._adjust_chops_for_stimulus_response()
       #--- update annotations
        self._update_annotations()
       #--- get info
        if self.verbose:
           self.GetInfo()
       #--- show plot
        if self.show:
           self.show_chops() 
  
    def copy_crop_and_chop(self,raw=None,chop=None):
        '''
        chop raw using copy & crop
        
        Parameters
        ----------
        raw : mne.RAW obj, optional
              The default is None.
              if None: use raw obj from class [self.raw]
            
        chop : np.array,list,tuple [start time,end time]
               if None return None
            
        Returns
        -------
        raw_crop : copy of part of mne.RAW obj in range of chop time
        or None if chop is None
    
        '''
        
        if not chop : return None
      
        if not raw:
           raw= self.raw
            
        raw_crop = raw.copy().crop(tmin=chop[0],tmax=chop[1])
        if self.verbose:
           logger.info("RAW Crop Annotation : {}\n  -> tmin: {} tmax: {}\n {}\n"
                       .format(jb.get_raw_filename(raw),chop[0],chop[1],raw_crop.annotations))
        return raw_crop
        
      
        
    def show_chops(self,raw=None,time_window_sec=None,description=None,picks=None):
        """
        show chops with raw.plot() 
        
        Parameters
        ----------
        raw : TYPE, optional
            DESCRIPTION. The default is None.
        time_window_sec : TYPE, optional
            DESCRIPTION. The default is None.
        description : TYPE, optional
            DESCRIPTION. The default is None.
        picks: channel indices to plot e.g. picks from mne raw  
            DESCRIPTION. The default is None, use stim and resp picks.
        Returns
        -------
        None

        """
        
        if not time_window_sec:
           time_window_sec = self.time_window_sec
        if not raw:
           raw = self.raw
        if not description:
           description = self.description
        if not picks:   
           picks  = jb.picks.stim_response(raw)
        labels = jb.picks.picks2labels(raw, picks)
        labels.append('STI SUM') #  sum off all stim channels
        
        stim_data = np.zeros([ len(labels), len(raw.times)])
        stim_data[0:-1,:] += raw._data[picks,:]
        stim_data[-1,:]   += raw._data[picks,:].sum(axis=0)
        scalings={'stim':16}
    
        #-- create raw to plot 
        #-- make combined sti-resp raw obj
        # https://github.com/mne-tools/mne-python/issues/4208
        #info      = mne.create_info(['STI 013','STI 014','STI SUM'], self.sfreq, ['stim','stim','stim'])
        #stim_data = np.zeros([3, len(self.raw.times)])
      
        info     = mne.create_info(labels,raw.info['sfreq'],['stim' for l in range( len(labels))] )
        stim_raw = mne.io.RawArray(stim_data, info)
      
       #-- cp annotations
        raw_annot = self.raw.annotations
        raw_annot.orig_time = None
        stim_raw.set_annotations(raw_annot)
      
        idx = np.where(raw_annot.description == description)[0]
        chop_onsets = raw_annot.onset[idx]
     
        logger.info("STIM Annotations chop time onsets: {}".format(chop_onsets))
        for onset in chop_onsets:
            if onset < stim_raw.times[-1]:
               stim_raw.plot(show=True,block=True,scalings=scalings,duration=time_window_sec[0] + time_window_sec[1],
                             start=onset - time_window_sec[0])
           
        
    def GetInfo(self):
        msg = ["Chops Info:"]      
        msg.extend([" --> estimated chops",
                    "  -> length [sec]: {}".format(self.length), 
                    "  -> times  :\n{}".format(self.estimated_chops), 
                    "  -> indices:\n{}".format(self.estimated_indices)])
                   
        
        
        if self.chops is not None:
           lt = self.chops[:,1]   - self.chops[:,0]
           li = self.indices[:,1] - self.indices[:,0]
           msg.extend(["-"*40,
                       " --> calculated chops",
                       "  -> length [sec]    : {}".format(lt),
                       "  -> length [samples]: {}".format(li),
                       "  -> times  :\n{}".format(self.chops_as_string()),
                       "  -> indices:\n{}".format(self.indices_as_string)])
        msg.append("-"*40)          
        logger.info( "\n".join(msg) )   
 

def test():
    
    stage= "$JUMEG_TEST_DATA/mne/201772/INTEXT01/190212_1334/2"
    fn   = "201772_INTEXT01_190212_1334_2_c,rfDC,meeg,nr,bcc,int-raw.fif"
    fin  = os.path.join(stage,fn)

    raw,fname = jb.get_raw_obj(fname=fin)

    logger.info("RAW annotations: {}".format(raw.annotations))
   
    jCP = JuMEG_PIPELINES_CHOPPER()
    jCP.update(raw=raw,verbose=True,debug=True,show=True)


if __name__ == "__main__":
    
   logger = jumeg_logger.setup_script_logging(logger=logger)
   test() 
    
    