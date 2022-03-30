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

# np.set_printoptions(formatter={'float': '{: 0.3f}'.format},threshold=sys.maxsize)
     
logger = jumeg_logger.get_logger()

__version__= "2020.05.06.001"


def get_chop_times_indices(times, chop_length=180., chop_nsamp=None, strict=False, exit_on_error=False):
    """
    calculate chop times for every X s
    where X=interval.

    Author: J.Dammers
    update: F.Boers

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

    exit_on_error: boolean <False>
                   error occures if chop_length < times
                    -> if True : exit on ERROR
                    -> if False: try to adjust chop_time
                       e.g. chop_times: is one chop with [ times[0],times[-1] ]

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
            # ToDo check for times[-1] < chop_length
            chop_len = chop_length
        else:
            chop_len = chop_length + t_rest // n_chops  # add rest to chop_length

        msg1 = [
            "  -> number of chops      : {}".format(n_chops),
            "  -> calculated chop legth: {}".format(chop_len),
            "  -> rest [s]             : {}".format(t_rest),
            "-" * 40,
            "  -> chop length          : {}".format(chop_length),
            "  -> numer of timepoints  : {}".format(n_times),
            "  -> strict               : {}".format(strict),
            "-" * 40,
            "  -> exit on error        : {}\n".format(exit_on_error)
        ]

        try:
            n_times_chop = int(chop_len / dt)
        except:
            if exit_on_error:
                msg = ["EXIT on ERROR"]
                msg.extend(msg1)
                logger.exception("\n".join(msg))
                assert (chop_len > 0), "Exit => chop_len: {}\n".format(chop_len)
            else:  # data size < chop_length
                msg = ["setting <chop_len> to number of time points!!!"]
                msg.extend(msg1)
                logger.error("\n".join(msg))

                n_times_chop = n_times
                n_chops = 1
                msg = ["data length smaller then chop length !!!",
                       " --> Adjusting:",
                       "  -> number of chops: {}".format(n_chops),
                       "  -> chop time      : {}".format(n_times_chop)
                       ]
                logger.warning("\n".join(msg))

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
def get_events_in_window(window_tsl=None,data=None,picks=None,events=None,and_mask=255):
    """
   
    Parameters
    ----------
    window_tsl : TYPE, optional
        DESCRIPTION. The default is None.
    data : np.array
        DESCRIPTION. The default is None.
    picks : list,np.array, optional
        DESCRIPTION.  channels to pick The default is None.
    events : TYPE, optional
        DESCRIPTION. The default is None.
    and_mask: int, The default is 255
        
    Returns
    -------
    evt : TYPE
        DESCRIPTION.

    """
    dtsl = 2
    evt  = None   
   
   #--ToDo warning
    if data.ndim > 2: return
   
   #--- make np.array
    if data.ndim > 1:  # [picks,:]
       if picks is not None:
          d = np.zeros([len(picks),window_tsl[1] - window_tsl[0]])
          d[:,:] += data[picks,window_tsl[0]:window_tsl[1] ]   
       else:
          d = np.zeros([data.shape[0],window_tsl[1] - window_tsl[0]])
          d[:,:] += data[:,window_tsl[0]:window_tsl[1] ]   
    else:
          d = np.zeros(window_tsl[1] - window_tsl[0])
          d[:] += data[window_tsl[0]:window_tsl[1] ]   
    
    if and_mask:
       d = np.bitwise_and(d.astype(np.int),and_mask)
   
   #--- add events      
    if events is not None:
       onset = events[:,0] - window_tsl[0] - dtsl       
      #--- evt onset < window set onset= 0
       idx = np.where(onset<0)[0]
       if idx.shape[0]:
          onset[idx] = 0
       
       offset = onset + dtsl *2  # 4
       idx    = np.where(offset > d.shape[-1])[0]
      #--- evt onset > window set onset= max data.shape  
       offset[idx] = d.shape[-1] 
       
       for idx in range( onset.shape[0] ):
           d[ -1,onset[idx]:offset[idx] ]+=1
       
    d    = d.sum(axis=0).astype(np.int)
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
        

def concat_and_save(raws,fname=None,save=False,annotations=None,clear=True):
    '''
    concatenate a list of raw obj's to a new raw obj
    calling mne.concatenate_raw
    clear raws in list
    
    with new raw obj:
     set fname, annotations
     save raw
    
    Parameters
    ----------
    raws  : list of raw objs
    fname : full filename <None>
    save  : <False>
    clear : close all raw obj in raws <True>
    annotations: set annotations in raw obj <None>.
  
    Returns
    -------
    raw obj concatenated
    '''
    raw_concat = None
    
    if raws:
       raw_concat = mne.concatenate_raws(raws)
       if clear:
          while raws:
                raws.pop().close()
          del raws
       if annotations:
          raw_concat.set_annotations(annotations)
       
       if fname:
          jb.set_raw_filename(raw_concat,fname)
       if save:
          jb.apply_save_mne_data(raw_concat,fname=fname)
              
    return raw_concat


def compare_data(d1,d2,picks=None,verbose=False):
    '''
    compare two data sets
    
    Parameters
    ----------
    d1 : TYPE
        DESCRIPTION.
    d2 : TYPE
        DESCRIPTION.
    picks : TYPE, optional
        DESCRIPTION. The default is None.
    verbose : TYPE, optional
        DESCRIPTION. The default is False.

    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    # np.not_equal([1.,2.], [1., 3.],where=True)
    if picks:
       dif_idx = np.where( d1 [picks,:] - d2[picks,:] )[0]
    else:
       dif_idx = np.where( d1 - d2 )[0]
       
        
    msg = ["Compare",
              "  -> data 1 : {}".format(d1.shape),
              "  -> data 2 : {}".format(d2.shape),
              "-"*40,
              "  -> non equal data : {}".format(dif_idx.shape[0])
          ]
    
    if dif_idx.shape[0]:
       raise ValueError(" ERROR chop crop data\n".join(msg))
    elif verbose:
       logger.info( "\n".join(msg))
  
    if dif_idx.shape[0]:
       return False
    return True

def compare_data_shapes(shapes,labels,verbose=False):
    '''
    wrapper <compare_data_shapes>

    Parameters
    ----------
    shapes : list of np.arrays e.g. data.shape 
    labels : list of strings, shape label
      
    Returns
    -------
    True/False
    '''      
    ck_shape = True
    msg = ["Check data shapes"]
  
    for label,shape in zip(labels,shapes): 
        if ( shapes[0] != shape ): ck_shape = False
        msg.append(" --> {} compare: {} shapes: {} ".format(label,ck_shape,shape))
    
    if not ck_shape:
       raise ValueError(" ERROR in <compare data shapes>\n".join(msg))
    elif verbose:
       logger.info( "\n".join(msg) )
 
    return ck_shape
           
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
        
    __slots__=["length","verbose","debug","show","and_mask","exit_on_error",
               "_time_window_sec","_raw","_description",
               "_estimated_chops","_estimated_indices","_chops","_indices"]
    
    
    def __init__(self,**kwargs):
        self._init(**kwargs)
        
        self.and_mask        = 255
        self.length          = 120.0
        self.exit_on_error   = True
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
        if v is None:
           self._time_window_sec = None
           return   
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
 
    @property   
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
    
    @property  
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
                       
        self._estimated_chops,self._estimated_indices = get_chop_times_indices(self.times,
                                                                               chop_length=self.length,
                                                                               exit_on_error=self.exit_on_error) 
        
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
        only if <time_window_sec> is not None
        
        chops: TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
         chops,indices
       
        """
       #-- ck no stimulus adjustment 
        if self.time_window_sec is None:
           self._chops   = self._estimated_chops.copy()
           self._indices = self._estimated_indices.copy() 
           msg=["Warning: chops not adjusted for stimulus,response and artefacts",
                "  -> using time estimated chops & indices: <time_window_sec> is <None>"
                "  -> chops  :\n{}".format(self.chops),
                "  -> indices:\n{}".format(self.indices),
               ]           
           logger.warning("\n".join(msg))  
           return self._chops,self._indices
       
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
            evt = get_events_in_window(data=self.raw._data,window_tsl=window_tsl,picks=picks,
                                       events=evt_artifacts,and_mask=self.and_mask)
           
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
        indices[1:,0] = indices[0:-1,1]+1
        
        for idx in range(n_chops):
            chops[idx][0]= self.raw.times[ indices[idx][0] ]
            chops[idx][1]= self.raw.times[ indices[idx][1] ]
       
       #--- set last chop/indice offset to last timepoint/sample 
        chops[-1][1]   = self.raw.times[-1 ]
        indices[-1:,1] = self.raw.last_samp
        
        if self.debug:
           msg=[
                "adjust chops for stimulus,response and artifact on/offsets: {}".format(chops.shape[0]),
                "  -> chops  :\n{}".format(chops),
                "  -> indices:\n{}".format(indices),
                "  -> stim   :\n{}".format(stim_indices)
               ] 
           logger.debug("\n".join(msg))
        self._chops   = chops
        self._indices = indices
        
        return chops,indices       
        
    def _update_annotations(self):
        '''
        store chop offset as timepoints in raw.annotations
        e.g.:
            description: ica-chops
            onsets: [ 152.129  310.137  461.694]
        
        Returns
        -------
        None.

        '''
       
        jb.update_annotations(self.raw,description=self.description,onsets=self.chops[:,1],verbose=self.verbose)
            
      
        
  
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
        wrapper call <copy_crop_and_chop>

        Parameters
        ----------
        raw  : rawobj, optional
        chop : np.array[2], optional
       
        Returns
        -------
         raw chop
        '''
        if not raw:
           raw = self.raw 
        if chop is None:
           logger.exception("ERROR chop is None")   
           return None
       
        return copy_crop_and_chop(raw=raw,chop=chop,verbose=self.verbose) 
    
    def concat_and_save(self,raws,fname=None,annotations=None,save=False,clear=True):
        '''
        wrapper <concat_and_save>
        Parameters
        ----------
        raws : TYPE
            DESCRIPTION.
        fname : TYPE, optional
            DESCRIPTION. The default is None.
        annotations : TYPE, optional
            DESCRIPTION. The default is None.
        save : TYPE, optional
            DESCRIPTION. The default is False.
        clear : TYPE, optional
            DESCRIPTION. The default is True.

        Returns
        -------
        TYPE
            DESCRIPTION.

        '''
        
        return concat_and_save(raws,fname=fname,save=save,
                               annotations=annotations,clear=clear)
        
    def compare_data_shapes(self,shapes,labels):
        '''
        wrapper <compare_data_shapes>

        Parameters
        ----------
        shapes : list of np.arrays e.g. data.shape 
        labels : list of strings, shape label
      
        Returns
        -------
         True/False
        '''
        
        return compare_data_shapes(shapes,labels,verbose=self.verbose)
        
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
      
        info = mne.create_info(labels,raw.info['sfreq'],['stim' for l in range( len(labels))] )
        info['meas_date' ] = self.raw.info['meas_date'] # for annotations.orig_time
        
        stim_raw = mne.io.RawArray(stim_data, info)
      
       #-- cp annotations
        raw_annot = self.raw.annotations.copy() 
        stim_raw.set_annotations(raw_annot)
      
        idx = np.where(raw_annot.description == description)[0]
        chop_onsets = raw_annot.onset[idx]
       
        if self.verbose:
           msg=[
                "Plot STIM Annotations: {}".format(raw_annot),
                "  -> chop time onsets: {}".format(chop_onsets)]
           logger.info("\n".join(msg))
        
        for onset in chop_onsets:
            logger.info("plot raw chop offset [sec]: {:0.3f}".format(onset))
            stim_raw.plot(show=True,block=True,scalings=scalings,duration=time_window_sec[0] + time_window_sec[1],
                          start=onset - time_window_sec[0])
        
        
    def GetInfo(self):
        msg = ["Info estimated Chops:"]      
        msg.extend(["  -> length [sec]: {}".format(self.length), 
                    "  -> times  :\n{}".format(self.estimated_chops), 
                    "  -> indices:\n{}".format(self.estimated_indices)])
                   
        if self.chops is not None:
           lt = self.chops[:,1]   - self.chops[:,0]
           li = self.indices[:,1] - self.indices[:,0]
           msg.extend(["",
                       "---> Info adjusted Chops:",
                       "  -> length [sec]    : {}".format(lt),
                       "  -> length [samples]: {}".format(li),
                       "  -> times  : {}".format(self.chops_as_string),
                       "  -> indices: {}".format(self.indices_as_string),
                       "  -> samples in raw : {}".format(self._raw._data.shape[1]),
                       "  -> last sample    : {}".format(self._raw.last_samp)
                      ])
        msg.append("-"*40)          
        logger.info( "\n".join(msg) )   
 

def test():
    '''
    from jumeg.base.jumeg_base                                 import jumeg_base as jb
    from jumeg.base.pipelines.jumeg_pipelines_ica_perfromance  import ICAPerformance
    from jumeg.base.pipelines.jumeg_base_pipelines_chopper import JuMEG_PIPELINES_CHOPPER,copy_crop_and_chop,concat_and_save
    
    '''

    stage= "$JUMEG_TEST_DATA/mne/201772/INTEXT01/190212_1334/2"
    fn   = "201772_INTEXT01_190212_1334_2_c,rfDC,meeg,nr,bcc,int-raw.fif"
    
    #stage="/media/fboers/USB_2TB/exp/INTEXT/mne/208548/INTEXT01/181023_1355/1"
    #fn="208548_INTEXT01_181023_1355_1_c,rfDC,meeg,nr,bcc,int-raw.fif"
    fin  = os.path.join(stage,fn)


    raw,fname = jb.get_raw_obj(fname=fin)
 
    #--- ck for annotations in raw 
    try:
      annota = raw.annotations
    except:
      from jumeg.base.pipelines.jumeg_pipelines_ica_perfromance import ICAPerformance
      IP = ICAPerformance()
     #--- find ECG
      IP.ECG.find_events(raw=raw)
      IP.ECG.GetInfo(debug=True)
     #--- find EOG
      IP.EOG.find_events(raw=raw)
      IP.EOG.GetInfo(debug=True)
    
     
    jCP = JuMEG_PIPELINES_CHOPPER()
    jCP.update(raw=raw,verbose=True,debug=True,show=True)
   
   #---  test chop crop
    raw_chops=[]
    
   # raw = jCP.stim_raw
    
    for chop in jCP.chops:
        raw_chop = copy_crop_and_chop(raw=raw,chop=chop)
        raw_chops.append(raw_chop)
   #--- concat chpos
    raw_concat = concat_and_save(raw_chops,annotations=raw.annotations)

    if not compare_data(raw._data[0],raw_concat._data[0],verbose=False):
       logger.exception("Error raw and raw_concat not equal") 
       sys.exit()
   
    
if __name__ == "__main__":
    
   logger = jumeg_logger.setup_script_logging(logger=logger)
   test() 
    
    