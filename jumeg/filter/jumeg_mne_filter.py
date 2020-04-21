#!/usr/bin/env python3
# -+-coding: utf-8 -+-

"""
"""

#--------------------------------------------
# Authors: Frank Boers <f.boers@fz-juelich.de> 
#
#-------------------------------------------- 
# Date: 12.12.19
#-------------------------------------------- 
# License: BSD (3-clause)
#--------------------------------------------
# Updates
#--------------------------------------------

import mne
import numpy as np
from jumeg.base.jumeg_base import jumeg_base as jb
from jumeg.base            import jumeg_logger

logger = jumeg_logger.get_logger()

__version__= "2020.03.19.001"



class JuMEG_MNE_FILTER(object):
    """
    wrapper cls to wrap mne.filter MNE version 19.2 in juMEG
     call MNE filter e.g.:
              raw.filter(l_freq=flow,h_freq=fhigh,picks=picks)
        
    save and rename filterd raw file
    
    
    raw      : <None> raw obj
    flow     : <None> mne <l_freq>
    fhigh    : <None> mne <h_freq>
    picks    : <None> => if None then exclude channels from <stim> group
    save     : <False> / True
    overwrite: <False> if save overwrite existing filtered file
    verbose  : <False> tell me more
    debug    : <False>
    postfix  : <None>  postfix for filename
    
    Returns:
    --------
    filename of filtered raw
    !!! raw is filtered in place !!!
    
    Example:
    --------
    from jumeg.base.jumeg_base import jumeg_base as jb
    from jumeg.filter.jumeg_mne_filter import JuMEG_MNE_FILTER
   #--- load raw
    raw = raw_fname = jb.get_raw_obj(fname,raw=None)
  
   #--- ini MNE_Filter class
    jfi= JuMEG_MNE_FILTER()
  
   #--- filter inplace
    fname_fitered_raw = jfi.apply(raw=raw,flow=0.1,fhigh=45.0,picks=None,save=True,verbose=True,overwrite=True)
    
    """
    __slots__ = ["raw","flow","fhigh","picks","save","overwrite","verbose","debug","_is_filtered","_is_reloaded","_fname_orig","annotations"]
    
    def __init__(self,**kwargs):
        #super().__init__()
        
        self.clear()
        self._update_from_kwargs(**kwargs)
        
    @property
    def fname_orig(self): return self._fname_orig
    @property
    def fname(self):
        return jb.get_raw_filename(self.raw,index=0)
    
    @property
    def isFiltered(self):
        return self._is_filtered
    
    @property
    def isReloaded(self):
        return self._is_reloaded

    @property
    def postfix(self):
        return self._update_postfix()

    def clear(self):
        for k in self.__slots__:
            self.__setattr__(k,None)
    
    def _update_from_kwargs(self,**kwargs):
        for k in self.__slots__:
            self.__setattr__(k,kwargs.get(k,self.__getattribute__(k)))
    
    def _update_postfix(self,**kwargs):
        """return filter extention """
        self._update_from_kwargs(**kwargs)
        fi_fix = None
        
        if self.flow and self.fhigh:
            fi_fix = "fibp"
            fi_fix += "%0.2f-%0.1f" % (self.flow,self.fhigh)
        elif self.flow:
            fi_fix = "fihp"
            fi_fix += "%0.2f" % self.flow
        elif self.fhigh:
            fi_fix = "filp"
            fi_fix += "%0.2f" % (self.fhigh)
        
        # self.postfix = fi_fix
        return fi_fix

   
    def apply(self,**kwargs):
        """
        wrapper function for MNE filter cls
        raw is filtered with MNE filter function inplace
        data in raw-obj will be overwritten
        filename is updated in raw-obj
        
        call MNE filter e.g.:
            raw.filter(l_freq=flow,h_freq=fhigh,picks=picks)
        
        :param kwargs:
         flow,fhigh,raw,picks
        
        Example
        --------
        -> filter all chanels 0.1 -45.0 Hz except STIM
        
        from jumeg.base.jumeg_base import jumeg_base as jb
        from jumeg.filter.jumeg_mne_filter import JUMEG_FILTER
        
        jFI = JUMEG_FILTER()
        fname = jFI.apply(
                  flow = 0.1,
                  fhigh = 45.0,
                  save  = True,
                  raw   = raw,
                  picks = jb.picks.exclude_trigger(raw) )
 
        :return:
         fname
        

        """
        
        self._update_from_kwargs(**kwargs)
        self._is_filtered = False
        self._is_reloaded = False
        
        v = jb.verbose
        jb.verbose = self.verbose
        
        logger.info("Filter start: {}".format(self.fname))
        
        self._update_postfix()
        fname,ext = self.fname.rsplit('-',1)  #raw.fif'
        fname += "," + self.postfix + "-" + ext
        
        #--- ck if load from disk
        if not self.overwrite:
            if jb.isFile(fname):
                logger.debug("Filtered RAW reloading from disk ...")
                self.raw,fname    = jb.get_raw_obj(fname,None)
                self._fname_orig  = fname
                
                if self.annotations:
                   self.raw.set_annotations(self.annotations)
           
                self._is_filtered = True
                self._is_reloaded = True
        
        if not self._is_filtered:
            logger.info("Filter start MNE filter ...")
            if isinstance(self.picks,(list,np.ndarray)):
               picks = self.picks
            else:
               logger.warning("WARNING: picks not defined : excluding channel group <stim> and <resp>")
               picks = jb.picks.exclude_trigger(self.raw)
               
            self.raw.filter(l_freq=self.flow,h_freq=self.fhigh,picks=picks)
            self._fname_orig = jb.get_raw_filename(self.raw)
            self._is_filtered = True
          
            if self.annotations:
               self.raw.set_annotations( self.annotations.copy() )
            
            fname,_ = jb.update_and_save_raw(self.raw,fout=fname,save=self.save,overwrite=True,update_raw_filename=True)
          
        if self.verbose:
           self.GetInfo()
        
        return fname
    
    def GetInfo(self,msg=None):
        """

        :param msg:
        :return:
        """
        _msg = ["Filter      : {}".format(self.isFiltered),
                " --> raw filtered: {}".format(self.fname),
                "  -> postfix: {}".format(self.postfix),
                "  -> flow   : {}".format(self.flow),
                "  -> fhigh  : {}".format(self.fhigh),
                "  -> save   : {}".format(self.save)
                ]
        try:
            annota =self.raw.annotations
        except:
           annota = None
        _msg.append("  -> mne.annotations in RAW:\n  -> {}".format(annota))

        if self.debug:
           _msg.extend(["-"*20,
                        "->  MNE version: {}".format(mne.__version__),
                        "->      version: {}".format(__version__) ])
        if msg:
            msg.extend(_msg)
            return msg
        else:
            logger.info(_msg)


class JuMEG_MNE_NOTCH_FILTER(JuMEG_MNE_FILTER):
    """
        wrapper cls to wrap mne.notch_filter  MNE version 19.2 in juMEG
         call MNE notch_filter e.g.:
                  raw.notch_filter(l_freq=flow,h_freq=fhigh,picks=picks)

        save and rename filterd raw file
 
        call MNE <raw.notch_filter>
        notch_filter(self,freqs,picks=None,filter_length='auto',notch_widths=None,trans_bandwidth=1.0,n_jobs=1,method='fir',
                     iir_params=None,mt_bandwidth=None,p_value=0.05,phase='zero',fir_window='hamming',fir_design='firwin',
                     pad='reflect_limited',verbose=None)[source]
        
         Example
        --------
        -> notch all chanels 50.0,100.0,150.0 Hz except STIM
        
        from jumeg.base.jumeg_base import jumeg_base as jb
        from jumeg.filter.jumeg_mne_filter import JUMEG_NOTCH_FILTER
        
        jNFI = JUMEG_NOTCH_FILTER()
        
        fname = jNFI.apply(
                           freqs = [50.0,100.0,150.0]
                           picks = jb.picks.exclude_trigger(raw)
                           )
        
    """
    __slots__ = ["raw","freqs","picks","filter_length","notch_widths","trans_bandwidth","n_jobs","method",
                 "iir_params","mt_bandwidth","p_value","phase","fir_window","fir_design","pad","verbose",
                 "save","overwrite","verbose","debug","_is_filtered","_is_reloaded","_fname_orig"]
    
    def __init__(self,**kwargs):
        #super().__init__()
        
        self.clear()
        
        self._update_from_kwargs(**kwargs)

    def clear(self):
        for k in self.__slots__:
            self.__setattr__(k,None)
            
        self.filter_length   = 'auto'
        self.trans_bandwidth = 1.0
        self.n_jobs          = 1
        self.method          = 'fir'
        self.p_value         = 0.05
        self.phase           = 'zero'
        self.fir_window      ='hamming'
        self.fir_design      ='firwin'
        self.pad             ='reflect_limited'
        
    def _update_from_kwargs(self,**kwargs):
        
        for k in self.__slots__:
            self.__setattr__(k,kwargs.get(k,self.__getattribute__(k)))

    def _update_postfix(self,**kwargs):
        """return filter extention """
        self._update_from_kwargs(**kwargs)

        fi_fix = "fin"
        if isinstance(self.freqs,(list,np.ndarray)):
           fi_fix += "{%0.2f}x{}".format(self.freqs[0],len(self.freqs))
        else:
           fi_fix += "{%0.2f}x1".format(self.freqs)
        return fi_fix

    def apply(self,**kwargs):
        """
        wrapper function for MNE version 19.2 notch filter cls
        data in raw-obj will be overwritten
        filename is updated in raw-obj

        call MNE <raw.notch_filter>
        notch_filter(self,freqs,picks=None,filter_length='auto',notch_widths=None,trans_bandwidth=1.0,n_jobs=1,method='fir',
                     iir_params=None,mt_bandwidth=None,p_value=0.05,phase='zero',fir_window='hamming',fir_design='firwin',
                     pad='reflect_limited',verbose=None)[source]
      

        :param kwargs:
        
        Example
        --------
        -> notch all chanels 50.0,100.0,150.0 Hz except STIM
        
        from jumeg.base.jumeg_base import jumeg_base as jb
        from jumeg.filter.jumeg_mne_filter import JUMEG_NOTCH_FILTER
        
        jNFI = JUMEG_NOTCH_FILTER()
        
        fname = jNFI.apply(
                           freqs = [50.0,100.0,150.0]
                           picks = jb.picks.exclude_trigger(raw)
                           )
        Example
        --------
        -> filter all chanels 0.1 -45.0 Hz except STIM

        from jumeg.filter.jumeg_mne_filter import JUMEG_FILTER
        jFI = JUMEG_FILTER()
        fname = jFI.apply(
                  flow = 0.1,
                  fhigh = 45.0,
                  save  = True,
                  raw   = raw,
                  picks = jb.picks.exclude_trigger(raw) )

        
        :return:
         fname
        """
    
        self._update_from_kwargs(**kwargs)
        self._is_filtered = False
        self._is_reloaded = False
    
        v = jb.verbose
        jb.verbose = self.verbose
    
        logger.info("---> Filter start: {}".format(self.fname))
    
        self._update_postfix()
        fname,ext = self.fname.rsplit('-',1)  #raw.fif'
        fname += "," + self.postfix + "-" + ext
    
        #--- ck if load from disk
        if not self.overwrite:
            if jb.isFile(fname):
                logger.debug("Notch Filtered RAW reloading from disk ...")
                self.raw,fname = jb.get_raw_obj(fname,None)
                self._is_filtered = True
                self._is_reloaded = True
    
        if not self._is_filtered:
            logger.info("Notch Filter start MNE filter ...")
            if isinstance(self.picks,(list,np.ndarray)):
                picks = self.picks
            else:
                logger.warning("picks not defined : excluding channel group <stim> and <resp>")
                picks = jb.picks.exclude_trigger(self.raw)
        
            self.raw.notch_filter(self.freqs,picks=picks,filter_length=self.filter_length,notch_widths=self.notch_widths,
                                  trans_bandwidth=self.trans_bandwidth,n_jobs=self.n_jobs,method=self.method,
                                  iir_params=self.iir_params,mt_bandwidth=self.mt_bandwidth,
                                  p_value=self.p_value,phase=self.phase,fir_window=self.fir_window,
                                  fir_design=self.fir_design,pad=self.pad,verbose=self.verbose)
       
            self._fname_orig = jb.get_raw_filename(self.raw)
            self._is_filtered = True
        
            if self.save:
                logger.info("Notch Filter saving data")
                fname = jb.apply_save_mne_data(self.raw,fname=fname,overwrite=True)
            else:
                jb.set_raw_filename(self.raw,fname)
    
        logger.info("Notch Filter done: {}\n".format(self.fname) +
                    "  -> reloaded from disk: {}".format(self._is_reloaded)
                    )
    
        jb.verbose = v
        return fname

    def GetInfo(self,msg=None):
        """

        :param msg:
        :return:
        """
        _msg = ["Notch Filter: {}".format(self.isFiltered),
                " --> raw filtered: {}".format(self.fname),
                "  -> postfix: {}".format(self.postfix),
                "  -> save   : {}".format(self.save),
                "---> Parameter:",
                "  -> freqs               : {}".format(self.freqs),
                "  -> notch_widths        : {}".format(self.notch_widths),
                "  -> trans_bandwidthfreqs: {}".format(self.trans_bandwidth),
                "  -> method              : {}".format(self.method),
                "  -> irr_params          : {}".format(self.irr_params),
                "  -> mt_bandwidth        : {}".format(self.mt_bandwidth),
                "  -> phase               : {}".format(self.phase),
                "  -> fir_window          : {}".format(self.fir_window),
                "  -> fir_design          : {}".format(self.fir_design)
                ]
        
        if self.debug:
           _msg.extend(["-"*20,
                        "->  MNE version: {}".format(mne.__version__),
                        "->      version: {}".format(__version__) ])
        if msg:
            msg.extend(_msg)
            return msg
        else:
            logger.info(_msg)




#---
def jumeg_mne_filter(raw=None,fname=None,**kwargs):
    jfi   = JuMEG_MNE_FILTER(raw=raw,fname=fname)
    fname = jfi.apply(**kwargs)
    return raw,fname
    
#if cfg.post_filter.run:
#   self.PostFilter.apply(
#                         flow  = cfg.post_filter.flow,
#                         fhigh = cfg.post_filter.fhigh,
#                         save  = cfg.post_filter.save,
#                         raw   = raw_unfiltered_clean, # ????
#                         picks = jb.picks.exclude_trigger(raw_filtered_clean)
#                       )
# return self.PostFilter.raw

    
    