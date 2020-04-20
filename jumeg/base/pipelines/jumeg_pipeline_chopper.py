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

import os,os.path as op
import numpy as np
import pandas as pd

import time,datetime,logging
import matplotlib.pyplot as pl

from distutils.dir_util import mkpath

import mne
from mne.preprocessing import find_ecg_events, find_eog_events

#from jumeg.decompose.ica_replace_mean_std import ICA, read_ica, apply_ica_replace_mean_std
#from jumeg.jumeg_preprocessing            import get_ics_cardiac, get_ics_ocular
#---
from jumeg.base                    import jumeg_logger
from jumeg.base.jumeg_base         import jumeg_base as jb
from jumeg.base.jumeg_base         import JUMEG_SLOTS
#from jumeg.base.jumeg_base_config  import JuMEG_CONFIG as jCFG
#---
from jumeg.base.pipelines.jumeg_pipelines_ica_perfromance  import JuMEG_ICA_PERFORMANCE
#from jumeg.base.pipelines.jumeg_pipelines_report           import JuMEG_REPORT
#---
#from jumeg.filter.jumeg_mne_filter import JuMEG_MNE_FILTER

logger = logging.getLogger("jumeg")

__version__= "2020.04.03.001"


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


    
class JuMEG_CHOPS(JUMEG_SLOTS):
    __slots__=["verbose","length","groups","show","_time_window_sec","_raw","_description",
               "_estimated_chops","_estimated_indices","_chops","_indices"]
    
    def __init__(self,**kwargs):
        self.init(**kwargs)
        if not self._time_window_sec:
           self._time_window_sec = np.array([5.0,5.0],dtype=np.float) 
        if not self._description:
           self._description="ICA chops" 
       
    @property
    def description(self): return self._description
    @description.setter
    def description(self,v):
        self._description=v
        
    @property
    def time_window_sec(self): return self._time_window_sec
    @time_window_sec.setter
    def time_window_sec(self,v):
        if isinstance(v,(list,tuple,np.ndarray)):
           self._time_window_sec = v
           if not v[-1]:
              v[-1] = v[0] 
        else:
           self._time_window_sec = [v,v]
           
    @property  
    def chops(self):   return self._chops
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
    
    
    def _calc_estimated_chops_from_timepoints(self):
        """
        calc chop times & indices

        Returns
        self._chop_times,self._chop_indices
        -------
        TYPE
            DESCRIPTION.

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
         
   
    def _get_events_in_window(self,window_tsl=None,raw=None,picks=None,events=None):
               
        if not raw:
           raw = self._raw 
        
            
        
        evt = None   
       #--- stim data  
     
        if events is not None:
           data = np.zeros([len(picks) +1, window_tsl[1] - window_tsl[0] ],dtype=np.float)
           data[0:-1,:] += raw._data[picks,window_tsl[0]:window_tsl[1] ]
          #--- ToDo vectorize NUMA 
           for e in events:
               onset = e[0] - window_tsl[0] -2
               if onset < 0: 
                  onset = 0
               
               offset = e[0] - window_tsl[0] +2
               if offset <= onset:
                  offset = onset + 2    
              
               if offset > data.shape[-1]:
                  offset = data.shape[-1] 
               data[-1,onset:offset]+=1
               
           data = data.sum(axis=0).astype(np.int)
        else:
           data = raw._data[picks,window_tsl[0]:window_tsl[1] ].sum(axis=0).astype(np.int)
      
        didx = np.where( data[0:-1] - data[1:] )[0]+1

        #--- if codes in window -> find events in window
        if didx.shape[0]: 
           #--- mne event array
           evt        = np.zeros( [ didx.shape[0],3 ],dtype=np.int) 
           evt[:,0]  += didx + window_tsl[0]
           evt[:,2]  += data[didx]
           evt[1:,1] += data[didx][0:-1]
           
           logger.debug(" window tsl: {}\n events:\n{}\n".format(window_tsl,evt) )
     
        return evt
              
    def _adjust_chops_for_stimulus_response(self,chops=None):
        """
        adjust chop onset not in stimulus / response window 

        Returns
        -------
         chops,indices
       
        """
        import sys
        np.set_printoptions(formatter={'float': '{: 0.3f}'.format},threshold=sys.maxsize)
       
        '''
        define search window for events 
        
        '''
        #-- make combined sti-resp raw obj
        # https://github.com/mne-tools/mne-python/issues/4208
        #info      = mne.create_info(['STI 013','STI 014','STI SUM'], self.sfreq, ['stim','stim','stim'])
        #stim_data = np.zeros([3, len(self.raw.times)])
       
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
        
        logger.info("Input: {}\n  -> Chops :{}\n  -> Indices: {}".format(n_chops,chops,indices))
        
        idx = -1
        stim_indices = np.zeros(n_chops,dtype=np.int)
          
         
        for tsls in indices:
            idx += 1
            stim_indices[idx] = tsls[1]
            logger.info("STIM last sample: {}\n  -> TSLS: {}".format(self.raw.last_samp,tsls))
            
            if tsls[1] >= self.raw.last_samp:
               continue 
            logger.info("STIM: {}\n TSLS: {}\n stim indices: {}\n".format(idx,tsls,stim_indices))
            
            #window_tsl0 = tsls[1] - window_tsl[0]
            #window_tsl1 = tsls[1] + window_tsl[1]
        
            window_tsl =[ tsls[1] - time_window_tsl[0],tsls[1] + time_window_tsl[1]]
           
           #--- find ECG,EOG hor, EOG ver 1,2,3 in window 
            evt_artifacts = None
           
            if anno_evt.shape[0]:
               logger.info("idx: {}\n Chops:\n{}\n window tsl:\n{}\n annos:\n{}\n".format(idx,chops,window_tsl,anno_evt) )
               logger.info("idx: {}\n Chops:\n{}\n window tsl:\n{}\n".format(idx,chops,window_tsl) )
               #onsets = evt_from_annotations[0][]:,0]
               evt_eog_idx = np.where( (anno_evt[:,0] > window_tsl[0]) &
                                       (anno_evt[:,0] < window_tsl[1]) &
                                       (anno_evt[:,2] > 0) )[0]
            
               if evt_eog_idx.shape[0]:
                  evt_artifacts = anno_evt[evt_eog_idx,:]
                
            
            evt = self._get_events_in_window(window_tsl=window_tsl,picks=picks,events=evt_artifacts)
         
            logger.info("idx: {}\n Chops:\n{}\n indices:\n{}\n events:\n{}\n".format(idx,chops,indices,evt) )
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
               continue
          
        #--- find zero code windows
            didx = np.where( evt[:,2] == 0 )[0] # get offsets
         
           #--- ck for no offset / onset => no zero event code
            if not didx.shape[0]:   # no zero code ?!!!
               msg=["can not find chop window with zero event code =>\n",
                    " using TSL : {}\n".format(tsls[1]),
                    "  -> idx   : {}]\n".format(idx),
                    "  -> chop  : {}\n".format(chops[idx]), 
                    "  -> events:\n{}\n".format(evt)]
                     
               logger.warning( "".join(msg) )
               continue
           
           #--- ck if last one is zero code
            if didx[-1] == evt.shape[0]-1:
               didx = didx[0:-1]    
               
            zeros_onset_tsl  = evt[didx,0]
            zeros_offset_tsl = evt[didx+1,0]
            diff_tsl         = zeros_offset_tsl - zeros_onset_tsl
            
            idx_tsl    = np.argmax( np.abs(diff_tsl) )
           
            onset = np.max(zeros_offset_tsl - zeros_onset_tsl)
            
            tsl = evt[1:,0]-evt[0:-1,0]
            tp  = tsl / self.sfreq
            logger.info("didx: {}\n onset: {}\n time: {}".format(didx,tsl,tp) )
        
            logger.info("evt: {}\n".format(evt[:,0]/self.sfreq) )
        
        
            logger.info("diff: {}\n{}\n{}\n{}".format(diff_tsl,idx_tsl,zeros_offset_tsl[idx_tsl],diff_tsl[idx_tsl] ) )
        
        
            evt_idx = didx[idx_tsl]
            stim_indices[idx] = int(  evt[evt_idx,0] + diff_tsl[idx_tsl] /2.0 )
       
            
       #--- set annotations 
       
        indices[:,1]  = stim_indices #[idx]
        indices[1:,0] = indices[0:-1,1]
        
        for idx in range(n_chops):
            chops[idx][0]= self.raw.times[ indices[idx][0] ]
            chops[idx][1]= self.raw.times[ indices[idx][1] ]
        
        logger.info("STIM chops:\n{}\n Chop:\n{}\n => indices:\n{}\n stim\n{}\n".format(idx,chops,indices,stim_indices) )
        
        self._chops   = chops
        self._indices = indices
        
        return chops,indices       
       
    
    def update(self,**kwargs):
        """
        calc chops from timepoints 
        adjust for stimulus/response onset/offset
        adjust for EOG,ECG onsets
        """
        self._update_from_kwargs(**kwargs)
        #--- calc estimated chops from chop length
        self._calc_estimated_chops_from_timepoints()
        #---
        self._adjust_chops_for_stimulus_response()
        #---
        self._update_annotations()
        #---
        if self.show:
           self.show_chops() 
        
    def _update_annotations(self):
        
        try:
            raw_annot  = self.raw.annotations #.copy()
            raw_annot.orig_time = None
        except:
            raw_annot = None
       
        duration   = np.ones( self.chops.shape[0] ) / self.sfreq  # one line in raw.plot
        chop_annot = mne.Annotations(onset=self.chops[:,1].tolist(),
                                     duration=duration.tolist(),
                                     description=self.description,
                                     orig_time=None)
        
        msg = ["update raw.annotations: {}".format(self.description)]
      
        if raw_annot:
            msg.append(" --> found mne.annotations in RAW:\n  -> {}".format(raw_annot))
           #--- clear old annotations
            kidx = np.where( raw_annot.description == self.description)[0] # get index
            if kidx.any():
                msg.append("  -> delete existing annotation {} counts: {}".format(self.description,kidx.shape[0]) )
                raw_annot.delete(kidx)
       
            self.raw.set_annotations( raw_annot + chop_annot)
        else:
            self.raw.set_annotations(chop_annot)
 
        idx = np.where(self.raw.annotations.description == self.description)[0]
        logger.info("Update Annotations for chops: {}\n  -> {}\n   -> chop onset\n{}".format(self.description,self.raw.annotations,
                                                                                            self.raw.annotations.onset[idx] ))
        
        
    def show_chops(self,time_window_sec=None):
        """
        show chops with raw.plot() 
        
        Parameters
        ----------
        time_window_sec: None
              
        Returns
        -------
        None

        """
        
        if not time_window_sec:
           time_window_sec  = self.time_window_sec
             
        picks  = jb.picks.stim_response(self.raw)
        labels = jb.picks.picks2labels(self.raw, picks)
        labels.append('STI SUM') #  sum off all stim channels
        
        stim_data = np.zeros([ len(labels), len(self.raw.times)])
        stim_data[0:-1,:] += self.raw._data[picks,:]
        stim_data[-1,:]   += self.raw._data[picks,:].sum(axis=0)
        scalings={'stim':16}
    
       #--- create raw to plot 
        info     = mne.create_info(labels,self.sfreq,['stim' for l in range( len(labels))] )
        stim_raw = mne.io.RawArray(stim_data, info)
      
       #--- cp annotations
        raw_annot = self.raw.annotations
        raw_annot.orig_time = None
        stim_raw.set_annotations(raw_annot)
      
        idx = np.where(raw_annot.description == self.description)[0]
        chop_onsets = raw_annot.onset[idx]
     
        logger.info("STIM Annotations chop time onsets: {}".format(chop_onsets))
        for onset in chop_onsets:
            if onset < stim_raw.times[-1]:
               stim_raw.plot(show=True,block=True,scalings=scalings,duration=time_window_sec[0] + time_window_sec[1],
                             start=onset - time_window_sec[0])
           
        
    def GetInfo(self):
        msg = ["Chops Info:"]      
        msg.append("estimeted chops\n --> length: {}\n --> times:\n{}\n --> indices:\n{}\n".
                   format(self.length,self._estimated_chops,self._estimated_indices))
        logger.info( "\n".join(msg) )   
         

class JuMEG_PIPELINES_CHOPPER(object):
    """
    
    chop RAW  time intervall of chop_length_sec
    check for overlap with  ECG/EOG events in annotations
    check for overlap with in events stim/resp
        
    """
    
    def __init__(self,**kwargs):
        super().__init__()
        
        
       # self._CFG           = jCFG(**kwargs)
        self._CHOPS        = JuMEG_CHOPS()
        self.chop_length   = 120.0
        self.verbose       = False
        self.debug         = False
        
    
              
    @property
    def verbose(self): return self.Chops.verbose
    @verbose.setter
    def verbose(self,v):
        self.Chops.verbose=v
  
    @property
    def raw(self):   return self._CHOPS.raw
    
    @property
    def Chops(self): return self._CHOPS

      
    @property
    def CFG(self): return self._CFG
    @property
    def cfg(self): return self._CFG._data

        
    def _update_from_kwargs(self,**kwargs):
        if "raw" in kwargs:
           self.Chops.raw = kwargs.get("raw")
        
        self.verbose = kwargs.get("verbose",self.verbose)
        self.debug   = kwargs.get("debug",self.debug)
        
        #self.path      = kwargs.get("path",self._path)
        # self._stage    = kwargs.get("stage",self.stage)
        #self.raw_fname = kwargs.get("raw_fname",self._raw_fname)

    
        
    def _copy_crop_and_chop(self,raw,chop):
        """
        copy raw
        crop
        :param raw:
        :param chop:
        :return:
        """
        if self.Chops.times.shape[0] > 1:
           raw_crop = raw.copy().crop(tmin=chop[0],tmax=chop[1])
           if self.debug:
              logger.info("RAW Crop Annotation : {}\n  -> tmin: {} tmax: {}\n {}\n".format(jb.get_raw_filename(raw),chop[0],chop[1],raw_crop.annotations))
           return raw_crop
        return raw
        
  
    def apply(self,**kwargs):
        self._update_from_kwargs(**kwargs)
        #logger.info("-"*40) 
        #logger.info(kwargs)
        #logger.info(self.raw.times)
                
        self.Chops.update(timepoints=self.raw.times,length=self.chop_length,show=True)
        
        #self.Chops.GetInfo()


    def calc_chops(self):
        sampling_freq = raw_A.info['sfreq']
        cptime=chops[0][1]
        dt=2.0
        
        chop_time = np.array( [ cptime-dt,cptime+dt ] )
        tsl0,tsl1 = (chop_time * sampling_freq).astype(int)
        
        x=raw_A.times[tsl0:tsl1]
        
        raw_stim = raw_A[picks_stim,tsl0:tsl1]
        raw_eeg  = raw_A[picks_eeg, tsl0:tsl1]
        print(raw_stim)
        
        raw_A.annotations
        idx=indices[0][1]
        dt_idx= int(dt*raw.info["sfreq"])  
        print(idx)
        print(dt_idx)
        
        
        evt_ana=mne.events_from_annotations(raw)
        print(evt_ana)
        print(evt_ana[0].shape)
        
        events_stim = {'consecutive': True, 'output': 'step', 'stim_channel': 'STI 014','min_duration': 0.00001, 'shortest_event': 1, 'mask': None}
        evt_stim=mne.find_events(raw,**events_stim)
        
        
        events_resp = {'consecutive': True, 'output': 'step', 'stim_channel': 'STI 013','min_duration': 0.00001, 'shortest_event': 1, 'mask': None}
        evt_resp=mne.find_events(raw,**events_resp)
        
        
        df=pd.DataFrame( evt_ana[0],columns=["onset","duration","id"])
        
        
        df[ (tsl0 < df["onset"]) & (df["onset"] <tsl1)]
        
        #-- ECG id == 1
        ecg_onsets=df[ (tsl0 < df["onset"]) & (df["onset"] <tsl1) & (df["id"]==1)]
        print(ecg_onsets)
        
        
        #-- ECG id == 1
        dton = ecg_onsets["onset"] / sampling_freq - cptime
        print(dton)
        #-- EOG id == 1
        eog_onsets=df[ (tsl0 < df["onset"]) & (df["onset"] <tsl1) & (df["id"]==3)]
        print(eog_onsets)
        
        
        
        df_stim= pd.DataFrame(evt_stim)
        hits=np.where( (tsl0 < evt_stim[:,0]) & (evt_stim[:,0] < tsl1) )[0]
        print(hits)
        print(evt_stim[hits])
        print(evt_stim[hits,0]/sampling_freq )
        
        evt_stim[hits,0]/sampling_freq - cptime
        
        
                


def run():
    
    stage= "/home/fboers/MEGBoers/data/exp/JUMEGTest/mne/201772/INTEXT01/190212_1334/2"
    fn   = "201772_INTEXT01_190212_1334_2_c,rfDC,meeg,nr,bcc,int-raw.fif"
    fin  = os.path.join(stage,fn)

    raw,fname = jb.get_raw_obj(fname=fin)

    #ICAPerformance = JuMEG_ICA_PERFORMANCE()    
    #ICAPerformance.ECG.find_events(raw=raw,set_annotations=True) #,**ecg)
    #ICAPerformance.EOG.find_events(raw=raw,set_annotations=True) #,**ecg)
    #raw.save(fname=fname,overwrite=True)
    
    logger.info("RAW annotations: {}".format(raw.annotations))
    
   
    jCP = JuMEG_PIPELINES_CHOPPER()
    jCP.apply(raw=raw,verbose=True,debug=True)




if __name__ == "__main__":
    
   logger = jumeg_logger.setup_script_logging(logger=logger)
   run() 
    
    