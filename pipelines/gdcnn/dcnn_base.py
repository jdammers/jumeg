#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Authors: 
 - Frank Boers: f.boers@fz-juelich.de
 - Jürgen Dammers: j.dammers@fz-juelich.de

"""

import yaml
from os import path as op
from copy import deepcopy
import numpy as np
import collections, pprint

import mne
from mne.viz import topomap

from dcnn_utils import logger, isFile, isPath, rescale, read_raw, get_raw_filename, expandvars
from dcnn_utils import apply_noise_reduction_4d, fig2rgb_array

__version__ = "2020.08.10.001"

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
# SLOTS class
# ck for meta cls __all_slots__
# https://codereview.stackexchange.com/questions/85509/meta-class-to-allow-inspection-of-all-slots-of-a-class
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class _SLOTS(object):
    __slots__ = ('_verbose', '_cls_name')
   
    def __init__(self,**kwargs):
        super().__init__()
        self.verbose = False
        self._init()
        self._update_from_kwargs(**kwargs)
        
    @property
    def verbose(self): return self._verbose

    @verbose.setter
    def verbose(self,v):
        self._verbose= v
     
    def _init(self):
        #--- init slots
        for k in self.__slots__:
            self.__setattr__(k,None)
        # self._update_from_kwargs(**kwargs)
 
    def init(self, **kwargs):
        self._update_from_kwargs(**kwargs)
    
    def __get_slots_attr(self):
        data = dict()
        for k in self.__slots__:
            if k.startswith("_"): continue
            data[k]= getattr(self,k)
        return data

    def _get_slots_attr_hidden(self,lstrip=True):
        data = dict()
        for k in self.__slots__:
            if k.startswith("_"):
                if lstrip:
                   data[ k.lstrip("_") ]= getattr(self,k)
                else:
                   data[k] = getattr(self,k)
        return data

    def _set_slots_attr_hidden(self,**kwargs):
        for k in kwargs:
            slot = "_"+ k
            if slot in self.__slots__:
               self.__setattr__(slot,kwargs.get(k) )

    def _update_from_kwargs(self, **kwargs):
        if not kwargs: return
        for k in kwargs:
            try:
                if k in self.__slots__:
                    self.__setattr__(k,kwargs.get(k))
            except:
                pass
    
    def clear(self,**kwargs):
        """
        set all values to None
        :param kwargs:
        :return:
        """
        #--- clear slots
        for k in self.__slots__:
            if k.startswith("_"): continue
            self.__setattr__(k,None)
    
    def update(self,**kwargs):
        self._update_from_kwargs(**kwargs)

    def get_info(self,hidden=False,msg=None):
        """
        log class parameter in slots
        Parameters
        ----------
        hidden: if True log hidden slots e.g. _data
        msg: string or list, message to append
        Returns
        -------

        """
        _msg=["Info => {}".format(self._cls_name)]
        for k in self.__slots__:
            if k.startswith("_"): continue
            _msg.append( "  -> {} : {}".format(k,getattr(self,k)) )
        if hidden:
           _msg.append("-"*20)
           for k,v in self._get_slots_attr_hidden().items():
               _msg.append( "  -> {} : {}".format(k,v) )
        if isinstance(msg,(list)):
           _msg.extend(msg)
        elif msg:
           _msg.append(msg)
        try:
            logger.info("\n".join(_msg))
        except:
            print("--->"+"\n".join(_msg))
   
    def dump(self):
        """
        return 
        dump of slots, who starts with a letter [aA..zZ]  
       
        Returns
        -------
        dict() key/values in __slots__
       
        """
        return self.__get_slots_attr() 
       
    '''
    future plans
    def __reduce_ex__(self, protocol):
        
        https://docs.python.org/3/library/pickle.html#performance
        https://diveintopython3.net/special-method-names.html

        Parameters
        ----------
        protocol : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.
        TYPE
            DESCRIPTION.
        TYPE
            DESCRIPTION.

        
        if protocol >= 5:
            return type(self)._reconstruct, (PickleBuffer(self),), None
        else:
            # PickleBuffer is forbidden with pickle protocols <= 4.
            return type(self)._reconstruct, (bytearray(self),)
    '''    


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# CONFIG class
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class DCNN_CONFIG(object):
   '''
   loading config file [yaml]
   
   Parameter:
   ----------
     defaults: 
     keys_to_check
     path_list" in kwargs:   
     fname
     config
   
   Example:
   ---------    
   
   '''

   def __init__(self,**kwargs):
       super().__init__()
       self._cfg          = None
       self._missing_keys = None
       self._fname        = None

       self.defaults = { 
                         'version': 'v0.2',

                         'n_jobs': 1,
                         'path': {
                             'model': 'Model'
                         },

                         'meg': {
                             'apply_filter': False,
                             'flow_raw': 0.3,
                             'fhigh_raw': 45.0,
                             'reject': None,
                             'apply_noise_reduction':True,
                             'nr_reflp': 5.0,
                             'nr_refhp': 0.1,
                             'apply_notch': False,
                             'line_freqs': [50., 100., 150., 200., 250., 300., 350.]
                         },

                         'ica': {
                             'n_components': 40,
                             'chop_len_init': 180.0,
                             'ica_method': 'fastica',
                             'fit_params': None,
                             'random_state': 2020,
                             'flow_ica': 2,
                             'fhigh_ica': 40,
                             'flow_ecg': 8,
                             'fhigh_ecg': 25,
                             'flow_eog': None,
                             'fhigh_eog': 20,
                             'ecg_thresh_ctps': 0.20,
                             'ecg_thresh_corr': 0.20,
                             'eog_thresh_ver': 0.20,
                             'eog_thresh_hor': 0.20
                         },

                         'res': {'res_time':60000,'res_space':64}
                         }
       
       self.keys_to_check={
                   'path': ['basedir', 'data_meg', 'data_train', 'report'],
                   'meg' : ['vendor', 'exp_info', 'ecg_ch', 'eog_ch1',
                            'apply_notch', 'line_freqs'],
                   'ica' : ['n_components', 'ica_method', 'fit_params',
                            'flow_ica', 'fhigh_ica', 'flow_ecg', 'fhigh_ecg',
                            'flow_eog', 'fhigh_eog',
                            'ecg_thresh_ctps', 'ecg_thresh_corr',
                            'eog_thresh_ver', 'eog_thresh_hor']       
                   }

       self._update_from_kwargs(**kwargs)

   @property
   def config(self): return self._cfg

   @property
   def fname(self): return self._fname
   @fname.setter
   def fname( self,v ):
       self._fname= expandvars(v)

   @config.setter
   def config(self,v):
       self._missing_keys = None
       self._cfg          = None #deepcopy( self.defaults )
       if not v: return
       self._cfg,self._missing_keys = self._merge_and_check(v)
       # self._add_base_dir()
       
   @property
   def missing_keys(self): return self._missing_keys
   
   def _update_from_kwargs(self,**kwargs):
       if "defaults" in kwargs:
          self.defaults = kwargs.get("defaults")
       if "keys_to_check" in kwargs:
          self.keys_to_chek = kwargs.get("keys_to_check")
       if "fname" in kwargs:
          self.fname = kwargs.get("fname")
       if "config" in kwargs:
          self.config = kwargs.get("config")              
  
   def _merge_and_check(self,cfg):
       '''
       merge <default dict> with > config dict>

       Parameters
       ----------
       cfg : TYPE
           DESCRIPTION.

       Returns
       -------
       config : dict, config dict 
       missing_keys : dict, <missing keys> in config dict
       '''
       #--  merge <default dict> with > config dict>
       cfg = self._update_and_merge(self.defaults, cfg, depth=-1, do_copy=True)
      
       #--- ck for missing keys
       return self._check_keys_in_config(config=cfg)
    
   def _update_and_merge(self, din, u, depth=-1, do_copy=True):
       """ update and merge dict parameter overwrite defaults
        
       Parameters
       ----------
       dict with defaults parameter
       dict with parameter to merge or update
       depth: recusive level <-1>
                
       Result
       -------
       dict with merged and updated parameters
        
       Example
       -------
       copy from:
       http://stackoverflow.com/questions/3232943/update-value-of-a-nested-dictionary-of-varying-depth
        
       Recursively merge or update dict-like objects. 
       >>> update({'k1': {'k2': 2}}, {'k1': {'k2': {'k3': 3}}, 'k4': 4})
       {'k1': {'k2': {'k3': 3}}, 'k4': 4}
       return dict
       """
        
       if do_copy:
          d = deepcopy(din)
       else:
          d = din 
       for k, v in u.items():
           if isinstance(v, collections.Mapping) and not depth == 0:
              r = self._update_and_merge(d.get(k, {}), v, depth=max(depth - 1, -1))
              d[k] = r
           elif isinstance(d, collections.Mapping):
              d[k] = u[k]
           else:
              d = {k: u[k]}
        
       return d    
       
   def _check_keys_in_config(self,config=None,keys_to_check=None):
       '''
       

       Parameters
       ----------
       config : TYPE, optional
           DESCRIPTION. The default is None.
       keys_to_check : TYPE, optional
           DESCRIPTION. The default is None.

       Returns
       -------
       config : dict, config dict 
       missing_keys : dict, <missing keys> in config dict

       '''
       if not config:
          config = self.config 
       k = config.keys()
       
       if not keys_to_check:
          keys_to_check = self.keys_to_check
       
       missing_keys = dict() 
       #-- ToDo make recursive
       for k in keys_to_check.keys():
           if k in config:
               
              kdefaults = keys_to_check[k]
              if isinstance(config[k],dict):
                 kcfg = config[k].keys()
              else:
                 kcfg = config[k] 
              kdiff     = ( set( kdefaults) - set( kcfg ) )
              if kdiff:
                 missing_keys[k] = kdiff  
           else:
              missing_keys[k] = []  
              
       if missing_keys:  
          msg = ["ERROR  missing keys in config"]
          for k in missing_keys:
              msg.append(" --> {}:\n  -> {}".format(k,missing_keys))
          logger.error("\n".join(msg))
       
       return config,missing_keys     
    
   def _dict2str(self,d):
       '''
       helper for pretty printing
 
       Parameters
       ----------
       d : TYPE
           DESCRIPTION.
 
       Returns
       -------
       TYPE
           DESCRIPTION.
 
       '''
       pp = pprint.PrettyPrinter(indent=2)
       return ''.join(map(str,pp.pformat(d)))     

   def load(self,**kwargs):
       '''
        load config [yaml]
        in config['path'] add basedir to path in path_list
       
        Parameters
        ----------
        fname : string, optional config filename
                if None use class.filename else set filename in class and load config
                The default is 'config.yaml'.
        path_list : TYPE, optional
                 if None use class.path_list to add config[basedir] as prefix 
                 The default is ['data_meg','data_train','report'].
    
        Returns
        -------
        config : TYPE
            DESCRIPTION.
    
       '''
       self._update_from_kwargs(**kwargs)
      
       self._cfg = None 
   
       if isFile(self.fname):
          self.config = yaml.safe_load(open(self.fname))
       else:
          logger.warning("config file is not a file: {}".format(self.fname))
       return None
    
   def info(self):
       '''
       shows/logs data in config class

       Returns
       -------
       None.

       '''
       msg = ["Config fname  : {}".format(self.fname)]
       msg.extend(["  -> parameter:\n{}".format(self._dict2str(self.config)),
                   "-"*40,
                   "  -> defaults :\n{}".format(self._dict2str(self.defaults)),
                   "-"*40,
                   "  -> keys to check for:\n{}".format(self._dict2str(self.keys_to_check)),
                   "-"*40,
                   "  -> missing keys:\n{}".format(self._dict2str(self.missing_keys))
                   ])
       logger.info("\n".join(msg))


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# PATH class
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class DCNN_PATH(_SLOTS):

    __slots__= ("expandvars", "exit_on_error", "logmsg", "mkdir", "overwrite", "add_basedir",
                "_basedir", "_data_meg", "_data_train", "_model", "_report")

    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self._cls_name  = "DCNN_PATH"
        self.init(**kwargs)

    def init(self,**kwargs): # set overwrite=True
        self.clear()
        self._update_from_kwargs(**kwargs)
        #-- set overwrite to True for real init if not set
        self.overwrite     = kwargs.get("overwrite",True)
        self.add_basedir   = kwargs.get("add_basedir",True)
        self.expandvars    = kwargs.get("expandvars",True)
        self.exit_on_error = kwargs.get("exit_on_error",True)
        self.logmsg        = kwargs.get("logmsg",True)
        self.mkdir         = kwargs.get("mkdir",True)
        # --
        self.basedir    = kwargs.get("basedir")
        self.data_meg   = kwargs.get("data_meg")
        self.data_train = kwargs.get("data_train")
        self.report     = kwargs.get("report")
        self.model      = kwargs.get("model")
        #-- set overwrite to False for update if not set
        self.overwrite  = kwargs.get("overwrite",False)

        #self.__set_slots_attr_hidden(**kwargs)  # sets hidden slots e.g. _data-xyz

    @property
    def basedir(self): return self._basedir
        # self._get_path(self._basedir)
    @basedir.setter
    def basedir(self,v):
        self._set_path(v.rstrip("/"),"basedir")

    @property
    def data_meg(self):
        return self._get_path(self._data_meg)
    @data_meg.setter
    def data_meg(self,v):
        self._set_path(v,"data_meg")

    @property
    def data_train(self):
        return self._get_path(self._data_train)
    @data_train.setter
    def data_train(self,v):
        self._set_path(v,"data_train")

    @property
    def model(self):
        return self._get_path(self._model)
    @model.setter
    def model(self,v):
        self._set_path(v,"model")

    @property
    def report(self):
        return self._get_path(self._report)
    @report.setter
    def report(self,v):
        self._set_path(v,"report")

    def _get_path(self,v):
        if self.expandvars:
            return expandvars(v)
        return v

    def _set_path(self,path,label):
        if not path:
           logger.warning("Warning {} can not set path, not defined for: {}".format(self._cls_name,label))
           return None

        if self.add_basedir and self.basedir:
           if not path.startswith(self.basedir):
              if path.startswith("/"):
                 path = self.basedir
              elif path != self.basedir:
                 path = op.join(self.basedir,path)
           # logger.warning(" add basedir: {} => {}".format(self.basedir,path))

        is_path = isPath(path,head="{} check path: {}".format(self._cls_name,label),
                         exit_on_error=self.exit_on_error,logmsg=self.logmsg,mkdir=self.mkdir)
        if not getattr(self,"_" +label) or self.overwrite:
           self.__setattr__("_" +label,path)
        else:
           pass

        if self.verbose:
           msg = ["Warning in {} not allowed to overwrite path settings for: <{}>".format(self._cls_name,label),
                  "  -> path orig  : {}".format(getattr(self,"_"+label)),
                  "  -> path       : {}".format(path),
                  "  -> overwrite  : {}".format(self.overwrite),
                  "  -> add basedir: {}".format(self.add_basedir),
                  "  -> basedir    : {}".format(self.basedir)]
           logger.warning("\n".join(msg))


    def dump(self):
        d1 = super().dump()
        d2 = self._get_slots_attr_hidden()
        return {**d1,**d2}

    def get_info(self):
        super().get_info(hidden=True,msg="basedir expand: {}\n".format( expandvars(self.basedir) ))


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# PICKS class
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class PICKS(_SLOTS):
    __slots__= ('_raw','_aux','_meg','_chan','_n_chan','_aux_labels','_aux_types')

    def __init__(self, raw=None):
        super().__init__()
        self._cls_name = "PICKS"
        self.raw = raw

    @property
    def aux_labels(self):
        return self._aux_labels

    @aux_labels.setter
    def aux_labels(self,v):
        self._aux_labels = v
      
    @property
    def aux_types(self): return self._aux_types
   
    @property
    def info(self):
        return self.raw.info
    
    @property 
    def raw(self): return self._raw

    @raw.setter
    def raw(self,v):
        if v:
           self._raw    = v
           self._meg    = self._get_meg()
           self._aux    = self._get_aux()
           self._chan   = self._get_chan()
        else:
           self.clear() 
   #---     

    @property
    def meg(self): return (self._meg)

    @property
    def n_meg(self): return len(self._meg)

    @property
    def aux(self): return (self._aux)

    @property
    def n_aux(self): return len(self._aux)

    @property
    def chan(self): return (self._chan)

    @property
    def n_chan(self): return len(self._chan)

    def _get_meg(self):
        '''
        get meg picks
    
        Returns
        -------
        list of indices, meg picks        
        '''
        return  mne.pick_types(self.info, meg=True, eeg=False, eog=False, ecg=False,
                               ref_meg=False, exclude=[])

    def _get_aux(self):
        '''
        get aux by aux label list ecg1,eog1,eog2

        # Auxilliary channels must be of good quality!
        ecg_ch: 'ECG 001'       # ECG channel
        eog_ch1: 'EOG 001'      # vertical EOG channel
        eog_ch2: 'EOG 002'

        :return:
        np.array , picks [ECG,EOG]
        '''

        error_msg = "\nERROR => Wrong or missing aux channel labels in config!\n" \
                    "  -> aux labels in config: {}\n".format(self.aux_labels)+\
                    "  -> aux labels in raw   : {}".format(self.get_aux_labels_in_raw())
        try:
            picks = self.labels2picks(self.aux_labels)

            assert picks is not None, error_msg
            assert (len(picks) == len(self._aux_labels)), error_msg + "\n  -> picks found: {}\n".format(picks)

            self._aux_types = self.raw.get_channel_types(picks, unique=False, only_data_chs=False)
        except ValueError:
            # logger.exception( error_msg +"\n  -> picks: {}".format(picks) )
            raise SystemExit(error_msg + "\n  -> picks: {}".format(picks))

        return picks

    def get_aux_picks_in_raw(self):
        '''
        get all aux picks

        Returns
        -------
        list of indices, aux picks
        '''
        return mne.pick_types(self.info, ecg=True, eog=True, meg=False, eeg=False,
                              ref_meg=False,exclude=[])

    def get_aux_labels_in_raw( self ):
        picks = self.get_aux_picks_in_raw()
        return self.picks2labels(picks)

    def _get_chan(self):
        '''
        get meg,ecg,eog picks
       
        Returns
        -------
        list of indices , meg,ecg,eog picks
        '''
        return mne.pick_types(self.info, meg=True, eog=True, ecg=True, eeg=False,
                              ref_meg=False,exclude=[])

    def picks2labels(self, picks):
        '''
        get channel labels from picks in  raw-obj
        (raw in this CLS)

        Parameter
        ---------

         picks <numpy array int64>

        Result
        -------
         return label list
        '''
        if isinstance(picks, (int, np.int64)):
            return self.raw.ch_names[picks]
        return ([self.raw.ch_names[i] for i in picks])

    def labels2picks(self, labels):
        """
        get picks from channel labels in raw-obj
        (raw in this CLS)
        call to < mne.pick_channels >
        picks = mne.pick_channels(raw.info['ch_names'], include=[labels])

        Parameter
        ---------
         channel label or list of labels

        Result
        -------
         picks as numpy array int64
        """
        if isinstance(labels, (list)):
            return mne.pick_channels(self.raw.info['ch_names'], include=labels)
        else:
            return mne.pick_channels(self.raw.info['ch_names'], include=[labels])

    def set_channel_types(self, names=None, types=None):
        '''
        wrapper to set channel types for channel names in raw
        init a dict from <names> ans >types> 
        and call raw.set_channel_types

        Parameters
        ----------
        names : string, list of strings
                The default is None.
        types : string, list of strings
                list must have the same length as names
                The default is None.

        Returns
        -------
        True/False

        '''
        if not isinstance(names,(list)):
           names = list(names)
       
       #- ck if labels in raw
        _names = list(set(names).intersection( set( self.raw.info["ch_names"] )) )
        if _names != names:
           msg=["ERROR some <channel names> not in RAW",
                "  -> found in RAW : {}".format(_names),
                "  -> original     : {}".format(names)] 
           
           logger.exception("\n".join(msg))
           return None
       
        chtps = dict()
        if isinstance(types,(list)):
           for l,t in zip(names,types):
              chtps[l]=t 
        else:    
           for l in names:
               chtps[l] = types
            
        self.raw.set_channel_types(chtps)

    def init(self, **kwargs):
        self.clear()
        self._update_from_kwargs(**kwargs)
        self._chan = kwargs.get("chan")
        self._meg  = kwargs.get("meg")
        self._aux  = kwargs.get("aux")
        self._aux_labels = kwargs.get("aux_labels",[])
        self._aux_types  = kwargs.get("aux_types",[])
  
    def dump(self):
        d = super().dump()
        d['meg']    = self._meg
        d['n_meg']  = len(self._meg)
        d['aux']    = self._aux
        # -- FB used in plot performance
        d['aux_labels'] = self.aux_labels
        d['aux_types']  = self.aux_types

        d['n_aux']  = len(self._aux)
        d['chan']   = self._chan
        d['n_chan'] = len(self._chan)
        return d



# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# MEG data class
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class DCNN_MEG(_SLOTS):
    __slots__= ('exp_info','exp_name','exp_type','location','system','vendor',
                'ecg_ch','eog_ch1','eog_ch2','n_jobs',
                'apply_noise_reduction','apply_notch','apply_filter',
                'line_freqs','nr_reflp','nr_refhp','flow_raw','fhigh_raw','reject',
                'sfreq_orig', 'sfreq_ds', 'chop_length','verbose',
                '_PICKS','_info_orig', '_times_orig', '_fname',
                '_isGradientCompensated','_isNoiseReduced','_isFiltered','_isNotched',
                '_isInterpolatedBads')
 
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self._PICKS = PICKS( raw=kwargs.get("raw") )
        self.raw    = kwargs.get("raw")
        self._cls_name = "MEG"
    
    @property
    def picks(self): return self._PICKS

    @property
    def raw(self): return self._PICKS.raw

    @raw.setter
    def raw(self,v):
        self._clear4raw()

        self._PICKS.aux_labels = [self.ecg_ch,self.eog_ch1]
        if self.eog_ch2:
            self._PICKS.aux_labels.append(self.eog_ch2)

        self._PICKS.raw  = v
        if v:
           self._info_orig  = deepcopy(v.info)
           self._times_orig = v.times.copy()

    @property
    def fname(self): return self._fname

    @property
    def info_orig(self): return self._info_orig

    @property
    def times_orig(self): return self._times_orig
      
    @property
    def isFiltered(self): return self._isFiltered

    @property
    def isNotched(self): return self._isNotched

    @property
    def isNoiseReduced(self): return self._isNoiseReduced

    @property
    def isGradientCompensated(self): return self._isGradientCompensated

    @property
    def isInterpolatedBads(self): return self._isInterpolatedBads
    
    def init(self, **kwargs):
        self.clear()
        self._clear4raw()
        self._update_from_kwargs(**kwargs)
        if "picks" in kwargs:
           # self._PICKS.init(**kwargs.get("picks"))
           self.picks.init(**kwargs.get("picks"))
        self._info_orig  = kwargs.get("info_orig")
        self._times_orig = kwargs.get("times_orig")
        self._fname = kwargs.get("fname")

    def _clear4raw(self):
        '''
        clear settings for new raw obj

        Returns
        -------
        None.

        '''
        self._info_orig      = None
        self._times_orig     = None
        self._fname          = None
        #-- flags set in apply_noise_reduction
        self._isFiltered     = False
        self._isNotched      = False
        self._isNoiseReduced = False
        self._isGradientCompensated = False
        self._isInterpolatedBads = False
    
    def read_raw(self, raw=None, fname=None):
        '''
        load raw obj from file or
        get from obj
     
        Parameters
        ----------
        raw : raw obj, optional,
              if raw use this object, the default is None.
        fname : string, optional
              fielname of raw obj, if raw is None load this

        Returns
        -------
        raw,fname

        '''
        self.raw = None
        
        raw, fname = read_raw(fname,raw=raw,verbose=self.verbose)

        if not raw:
           assert raw ("---> ERROR RAW is None => check input file: {}".format(fname)) 
        
        self.raw = raw
        self._fname = get_raw_filename(self.raw)
        self.sfreq_orig = raw.info['sfreq']
        self._set_channel_types()
        self.picks._aux = self.picks._get_aux()

        return raw, fname
    
    def _set_channel_types(self,raw=None):
        '''
        in raw obj set channel type  for chanel labels
        ECG: ecg
        EOG ver: eog, EOG hor: eog
        
        ToDo: use list of labels/channels, avoid fixed ECG,EOG channels
         
        Parameters
        ----------
        raw : TYPE, optional,if None use raw from class
              The default is None.

        Returns
        -------
        raw
        '''
        if not raw:
           raw = self.raw
     
       # settings: set channel type for ECG and EOG channels
        raw.set_channel_types({
            self.ecg_ch: 'ecg',
            self.eog_ch1: 'eog'})
        # optional additional EOG channel
        if self.eog_ch2:
           raw.set_channel_types({self.eog_ch2: 'eog'})
    
        return raw
       
    def _noise_reduction(self):
        
        if self.apply_noise_reduction:
            #- CTF gradient compensation
            if self.vendor == 'CTF' and self.raw.compensation_grade != 3:
               if self.isGradientCompensated:
                  logger.warning("WARNING Gradient Compensation in RAW applied before")
               self.raw.apply_gradient_compensation(3)
               self._isGradientCompensated = True
               
            #- 4D noise reduction by Eberhard Eich 4D ,4DHPC
            elif self.vendor.startswith('4D'):
                 if self.isNoiseReduced:
                    logger.warning("WARNING Noise Reduction in RAW applied before")
                 
                 self.raw = apply_noise_reduction_4d(self.raw,refnotch=self.line_freqs,vendor=self.vendor,
                                                     reflp=self.nr_reflp,refhp=self.nr_refhp)
                 self._isNoiseReduced = True

            if self.apply_notch:
               if self.isNotched:
                  logger.warning("WARNING Notch Filter in RAW applied before")
               
               self.raw.notch_filter(self.line_freqs,filter_length='auto',phase='zero',n_jobs=self.n_jobs)
               self._isNotched = True
               
            if self.apply_filter:
               if self.isFiltered:
                  logger.warning("WARNING Filter in RAW applied before")
             
               self.raw.filter(self.flow_raw,self.fhigh_raw, picks=self.picks.meg,
                               filter_length='auto', l_trans_bandwidth='auto', h_trans_bandwidth='auto',
                               method='fir', phase='zero', fir_window='hamming', n_jobs=self.n_jobs)
               self._isFiltered = True
    
    def _interpolate_bads(self):
        if self.isInterpolatedBads:
           logger.warning("WARNING Interpolated Bads in RAW applied before")
            
        if self.raw.info['bads']:
           logger.info("Start Interpolating Bads`s: {}".format( self.raw.info['bads'] ))  
           self.raw.interpolate_bads(reset_bads=True)
           logger.info(" Done: Interpolating Bads`s: {}".format( self.raw.info['bads'] )) 
           self._isInterpolatedBads = True

    def update(self, fname=None, raw=None):
        '''
        read raw file (no epochs!) and
        1. get/set settings
        2. apply noise reduction (and or notches)
        3. interpolate bad channels
  
        Parameters
        ----------
        fname : TYPE, raw file name
       
        Returns
        -------
        raw obj
        
        '''
       #- 1 load or init raw obj
        if not self.read_raw(raw=raw, fname=fname):
           logger.exception(" ERROR could not read raw: {} !!!".format(fname))
           return 
       
       #- set ECG/EOG channel types
        # TODO: this should be done in dcnn_utils.read_raw()
        #       otherwise (for some CTF data) aux picks are not set correctly
        #       workaround: do this in dcnn_base.read_raw()
        # self._set_channel_types()
        
       #- 2. apply noise_reduction (optional)
        self._noise_reduction()
         
       #- 3. apply interpolation on channels in case bads where are defined
        self._interpolate_bads()
      
        return self.raw    

    def dump(self):
        d = super().dump()
        d['fname']       = self.fname
        d['info_orig']   = self.info_orig
        d['times_orig']  = self.times_orig
        d['sfreq_ds']    = self.sfreq_ds
        d['chop_length'] = self.chop_length
        d['picks'] = self.picks.dump()        # will not dump picks.raw
        return d



# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ICA EXCLUDE class
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class _ICA_EXCLUDE(_SLOTS):
    __slots__= ('exclude','ecg','ecg_ctps','ecg_ctps_scores',
                'ecg_corr','ecg_corr_scores',
                'eog','eog_ver','eog_ver_scores','eog_hor','eog_hor_scores')
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.clear()
        self._cls_name = "ICA_EXCLUDE"
        
    def init(self, **kwargs):
        self.clear()
        self._update_from_kwargs(**kwargs)
        
    def update(self, **kwargs):
        self._update_from_kwargs(**kwargs)
    
    def clear(self):
        for k in self.__slots__:
            self.__setattr__( k,list() )

    def dump(self):
        d = super().dump()
        return d


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ICA CHOP class
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class _ICA_CHOP(_SLOTS):
    __slots__= ('times','indices','n_samples','sfreq_ds','sfreq_last','picks','ica',
                '_EXCLUDE'
               )
 
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._cls_name = "ICA_CHOP"
        self._EXCLUDE = _ICA_EXCLUDE()        
        
    def update(self,**kwargs):
        self._update_from_kwargs(**kwargs)

    def init(self,**kwargs):
        self.clear()
        self._update_from_kwargs(**kwargs)
        if "exclude" in kwargs:
            self.exclude.init(**kwargs.get("exclude"))
        
    @property
    def exclude(self): return self._EXCLUDE
    
    @property
    def dt(self): return 1.0 / self.sfreq_ds
    
    @property
    def n_chop(self): return len(self.times)
    
    @property
    def n_chan(self): return len(self.picks)

    @property
    def n_picks(self): return len(self.picks)
 
    def clear(self):
        super().clear()
        self._EXCLUDE.clear()
         
    def dump(self):
        d = super().dump()
        d["exclude"] = self.exclude.dump()
        return d


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ICA TOPOGRAPHIES class
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class _ICA_TOPO(_SLOTS):
    __slots__ = ('_data', '_images', '_images_head')
    
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self._cls_name = "ICA_TOPO"
        
    @property
    def data(self): return self._data

    @property
    def images(self): return self._images

    @property
    def images_head(self): return self._images_head

    def init(self, **kwargs):
        self.clear()
        self._update_from_kwargs(**kwargs)
        self._data   = kwargs.get("data")
        self._images = kwargs.get("images")
        self._images_head = kwargs.get("images_head")


    # -----------------------------------------------------
    #  compute (interpolated) images from component maps
    #  Note: this only works with mne-python <=0.19
    #        and will be removed in the next version
    # -----------------------------------------------------
    def _calc_images_old(self, data, pos, outlines, res=64, norm=True):
    
        # Note: this only works for MNE version <=0.19
    
        from mne.viz.topomap import _GridData
        extrapolate = 'box'
    
        data = np.asarray(data)
    
        # find mask limits
        xlim = np.inf, -np.inf,
        ylim = np.inf, -np.inf,
        mask_ = np.c_[outlines['mask_pos']]
        xmin, xmax = (np.min(np.r_[xlim[0], mask_[:, 0]]),
                      np.max(np.r_[xlim[1], mask_[:, 0]]))
        ymin, ymax = (np.min(np.r_[ylim[0], mask_[:, 1]]),
                      np.max(np.r_[ylim[1], mask_[:, 1]]))
    
        # interpolate the data, we multiply clip radius by 1.06 so that pixelated
        # edges of the interpolated image would appear under the mask
        head_radius = (None if extrapolate == 'local' else
                       outlines['clip_radius'][0] * 1.06)
        xi = np.linspace(xmin, xmax, res)
        yi = np.linspace(ymin, ymax, res)
        Xi, Yi = np.meshgrid(xi, yi)
        interp = _GridData(pos, extrapolate, head_radius).set_values(data)
        _image = interp.set_locations(Xi, Yi)()
    
        # rescale image [-1.0, +1.0]
        if norm:
            _image = rescale(_image, -1, 1)
    
        return _image
    
    
    # -----------------------------------------------------
    #  Note: this only works with mne-python <=0.19
    #        and will be removed in the next version
    #        requires gDCNN._calc_topo_image()
    # make images from topographies
    # this bypasses the problem that different MEG systems
    # do have different number of MEG channels
    # the image has fixed size of 64 x 64 (=4096)
    # uses _calc_topo_image()
    # -----------------------------------------------------
    def _get_images_old(self, icas=None, n_chop=None, n_components=None, res=64, merge_grads=False):
    
       # Note: this only works for MNE version <=0.19
       # get details
      
        self._images = np.zeros([n_chop,n_components, res, res])
    
        # loop across all ICA chops
        # print('-> make images from topographies ....')
        for ichop in range(n_chop):
            ica   = icas[ichop]
            topos = self.data[ichop]
    
            # prepare topoplot for magnetometers
            sensor_layout = mne.find_layout(ica.info, 'meg')
            if merge_grads:
                ch_type = 'grad'
            else:
                ch_type = 'mag'
            # Note: this has changed in MNE-Python 0.20
            picks, pos, merge_grads, names, _ = topomap._prepare_topo_plot(ica, ch_type, sensor_layout)
            pos, outlines = topomap._check_outlines(pos, 'head', None)
            topos = topos[picks]  # shape is [n_chan_mag, n_comp_ica]
    
            # loop across components
            for icomp in range(n_components):
                topo = topos[:, icomp]
                # use modified routine to not plot the image
                img = self._calc_image_old(topo.flatten(), pos, outlines, res=res)
    
                # collect topos for all ICs
                self._images[ichop, icomp] = img
    
        return self._images


    # -----------------------------------------------------
    # get topo images with head
    #    => uses MNE plot routine
    # -----------------------------------------------------
    def _get_topo_head_images(self, icas=None, n_chop=None, n_components=None, res=300):

        from mne.utils.check import _check_sphere
        from mne.viz.topomap import _make_head_outlines, _setup_interp
        from mne.viz.topomap import _prepare_topomap_plot
        from mne.viz.topomap import plot_topomap
        import matplotlib
        import matplotlib.pylab as plt

        backend_orig = matplotlib.get_backend()  # store original backend
        matplotlib.use('Agg')                    # switch to agg backend to not show the plot

        # get details
        topo_data = self.data
        self._images_head = np.zeros([n_chop, n_components, 300, 300, 3])

        # loop across all ICA chops
        logger.info('make images from topographies using MNE plot_topomap() ....')

        for ichop in range(n_chop):
            ica = icas[ichop]
            data = topo_data[ichop, :]

            for icomp in range(n_components):
                data_picks, pos, merge_channels, names, _, sphere, clip_origin = \
                    _prepare_topomap_plot(ica.info, 'mag')
                outlines = _make_head_outlines(sphere, pos, 'head', clip_origin)

                pos = pos[:, :2]
                data_ = data[data_picks, icomp].ravel()

                # -----------------------------------
                #  just compute the interpolated topo
                #  image without using plot routines
                # -----------------------------------
                # here we just compute the image without using the plot routine
                sphere = _check_sphere(sphere)
                extent, Xi, Yi, interp = _setup_interp(pos, res, 'box', sphere, outlines, 0)
                interp.set_values(data_)

                fig, ax = plt.subplots(figsize=(2.5,2.5),dpi=150)
                im = plot_topomap(
                    data_, pos, vmin=-1.0, vmax=1.0, res=res, axes=ax,
                    cmap='RdBu_r', outlines=outlines, contours=0, sphere=sphere,
                    image_interp='bilinear', show=False, sensors=False)[0]
                # plt.tight_layout()
                img_head = rescale(fig2rgb_array(fig), 0, 1)
                # plt.figure(); plt.imshow(img_head[25:325,45:345 ]); plt.show()

                # collect topos for all ICs
                # now we create a larger figure and need to crop the image
                # by this we reduce the white border
                self._images_head[ichop, icomp] = img_head[25:325,45:345 ] # crop to fit size
                plt.close()

        matplotlib.use(backend_orig)

        return self._images_head


    # -----------------------------------------------------
    # Note, this version only works with mne-python >= 0.20
    #
    # grabs images from topographies (no clipping)
    # it bypasses the problem that different MEG systems
    # do have different number of MEG channels
    # the topo image has a fixed size of 64 x 64 (=4096 pixel)
    # -----------------------------------------------------
    def _get_images(self, icas=None, n_chop=None, n_components=None, res=64):
    
        from mne.utils.check import _check_sphere
        from mne.viz.topomap import _make_head_outlines, _setup_interp
        from mne.viz.topomap import _prepare_topomap_plot

        # get details
        topo_data = self.data
        self._images = np.zeros([n_chop,n_components, res, res])

        # loop across all ICA chops
        logger.info('make images from topographies ....')
        
        for ichop in range(n_chop):
            ica = icas[ichop]
            data = topo_data[ichop, :]
    
            for icomp in range(n_components):
    
                data_picks, pos, merge_channels, names, _, sphere, clip_origin = \
                    _prepare_topomap_plot(ica, 'mag')
                outlines = _make_head_outlines(sphere, pos, 'head', clip_origin)
    
                pos = pos[:, :2]
                data_ = data[data_picks, icomp].ravel()

                # -----------------------------------
                #  just compute the interpolated topo
                #  image without using plot routines
                # -----------------------------------
                # here we just compute the image without using the plot routine
                sphere = _check_sphere(sphere)
                extent, Xi, Yi, interp = _setup_interp(pos, res, 'box', sphere, outlines, 0)
                interp.set_values(data_)
                img_arr = interp.set_locations(Xi, Yi)()
                # collect topos for all ICs
                self._images[ichop, icomp] = rescale(img_arr[::-1], -1.0, 1.0)   # rescale between -1 and +1
    
        return self._images


    def update_images(self, icas=None, n_chop=None, n_components=None, res=64):
            '''
            Parameters
            ----------
            res : TYPE, optional
                DESCRIPTION. The default is None.
    
            Returns
            -------
            None.
    
            '''
            if (np.float(mne.__version__[0:3]) < 0.2):
               return self._get_images_old(icas=icas, res=res, n_chop=n_chop,
                                           n_components=n_components, merge_grads=False)
            else:
                return self._get_images(icas=icas, res=res, n_chop=n_chop,
                                       n_components=n_components)


    def update_images_head(self, icas=None, n_chop=None, n_components=None, res=300):
        '''
        Parameters
        ----------
        res : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.

        '''

        return self._get_topo_head_images(icas=icas, res=res, n_chop=n_chop, n_components=n_components)

    # -----------------------------------------------------
    # get ica weights to construct topographic component maps
    # -----------------------------------------------------
    def update_data(self, icas=None, n_chop=None, n_chan=None, n_components=None):

        self._data = np.zeros([n_chop, n_chan, n_components])
        
        for ichop in range(n_chop):
            ica = icas[ichop]
            # get topographies from all sources
            self._data[ichop] = ica.get_components()  # shape is [n_chan_meg_all, n_comp_ica]
    
        return self._data
                
    def dump(self):
        d = super().dump()
        d["data"]   = self.data
        d["images"] = self.images
        d["images_head"] = self.images_head
        return d


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ICA SCORE class
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class _ICA_SCORE(_SLOTS):
    __slots__ = ('_ecg_ctps', '_ecg_corr', '_eog_ver', '_eog_hor')

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.clear()
        self._cls_name = "ICA_SCORE"

    @property
    def ecg_ctps(self): return self._ecg_ctps

    @property
    def ecg_corr(self): return self._ecg_corr

    @property
    def eog_ver(self): return self._eog_ver

    @property
    def eog_hor(self): return self._eog_hor

    def init(self, **kwargs):
        self.clear()
        self._update_from_kwargs(**kwargs)
        self._ecg_ctps = kwargs.get("ecg_ctps")
        self._ecg_corr = kwargs.get("ecg_corr")
        self._eog_ver  = kwargs.get("eog_ver")
        self._eog_hor  = kwargs.get("eog_hor")

    def update(self, **kwargs):
        self._update_from_kwargs(**kwargs)

    def clear(self):
        for k in self.__slots__:
            self.__setattr__(k, list())

    def dump(self):
        d = super().dump()
        d['ecg_ctps']  = self.ecg_ctps
        d['ecg_corr']  = self.ecg_corr
        d['eog_ver'] = self.eog_ver
        d['eog_hor'] = self.eog_hor
        return d


#######################################################################
#
# DCNN ICA class
#
#######################################################################
class DCNN_ICA(_SLOTS):
    __slots__= ('chop_len_init','chop_n_times','chop_length','random_state','n_components', 'n_chop',
                'ica_method','ecg_thresh_corr','ecg_thresh_ctps','eog_thresh_hor','eog_thresh_ver',
                'fhigh_ecg','fhigh_eog','flow_ecg','flow_eog','flow_ica','fhigh_ica','fit_params',
                '_CHOP','_TOPO'
                )
 
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        
        self._cls_name = "DCNN_ICA"
        self._CHOP = _ICA_CHOP()
        self._TOPO = _ICA_TOPO()

    @property
    def chop(self): return self._CHOP

    @property
    def topo(self): return self._TOPO

    def init(self,**kwargs):
        self.clear()
        self._update_from_kwargs(**kwargs)
        if "topo" in kwargs:
            self.topo.init( **kwargs.get("topo") )

        if "chop" in kwargs:
            self.chop.init( **kwargs.get("chop") )

    def update(self,**kwargs):
        self._update_from_kwargs(**kwargs)

    def update_topo_data(self):
        return self.topo.update_data(icas=self.chop.ica,
                                     n_chop=self.chop.n_chop,
                                     n_chan=self.chop.n_chan,
                                     n_components=self.n_components)

    def update_topo_images(self, res=64):
        """

        Parameters
        ----------
        res : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.

        """
        
        return self.topo.update_images(icas=self.chop.ica,
                                       n_chop=self.chop.n_chop,
                                       n_components=self.n_components,
                                       res=res)

    def update_topo_images_head(self, res=300):
        """

        Parameters
        ----------
        res : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.

        """
        return self.topo.update_images_head(icas=self.chop.ica, n_chop=self.chop.n_chop,
                                            n_components=self.n_components, res=res)

    def clear(self):
        if self._CHOP:
            self._CHOP.clear()
        if self._TOPO:
            self._TOPO.clear()

        for k in self.__slots__:
            if k.startswith("_"): continue
            self.__setattr__(k, None)

    def dump(self):
        d = super().dump()
        self.topo.get_info()
        d["topo"] = self.topo.dump()
        d["chop"] = self.chop.dump()
        return d  
   
    def get_info(self):
        super().get_info()
        self.topo.get_info()
        self.chop.get_info()



#######################################################################
#
# DCNN ICA SOURCES class
#
#######################################################################
class DCNN_SOURCES(_SLOTS):
    __slots__ = ('_data_ica',  '_data_aux', '_labels', '_SCORE', '_TOPO',
                 '_events_ecg', '_events_eog_ver', '_events_eog_hor')

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._cls_name = "DCNN_SOURCES"
        self._labels   = []
        self._SCORE    = _ICA_SCORE()
        self._TOPO     = _ICA_TOPO()

    @property
    def topo(self): return self._TOPO

    @property
    def score(self): return self._SCORE

    @property
    def data_ica(self): return self._data_ica

    @property
    def data_aux(self): return self._data_aux

    @property
    def labels(self): return self._labels

    @property
    def events_ecg(self): return self._events_ecg

    @property
    def events_eog_ver(self): return self._events_eog_ver

    @property
    def events_eog_hor(self): return self._events_eog_hor

    def init(self, **kwargs):
        self.clear()
        self._update_from_kwargs(**kwargs)
        if "data_ica" in kwargs:
            self._data_ica = kwargs.get("data_ica")
        if "data_aux" in kwargs:
            self._data_aux = kwargs.get("data_aux")
        if "labels" in kwargs:
            self._labels   = kwargs.get("labels")
        if "score" in kwargs:
            self.score.init(**kwargs.get("score"))
        if "topo" in kwargs:
            self.topo.init( **kwargs.get("topo") )
        self._events_ecg     = kwargs.get("events_ecg")
        self._events_eog_ver = kwargs.get("events_eog_ver")
        self._events_eog_hor = kwargs.get("events_eog_hor")


    def update(self, **kwargs):
        self._update_from_kwargs(**kwargs)

    def clear(self):
        if self._TOPO:
           self._TOPO.clear()
        if self._SCORE:
            self._SCORE.clear()
        for k in self.__slots__:
            if k.startswith("_"): continue
            self.__setattr__(k, None)

    def dump(self):
        d = super().dump()
        d['data_ica'] = self._data_ica
        d['data_aux'] = self._data_aux
        d['labels']   = self._labels
        d["topo"]     = self.topo.dump()
        d["score"]    = self.score.dump()
        d['events_ecg']     = self._events_ecg
        d['events_eog_ver'] = self._events_eog_ver
        d['events_eog_hor'] = self._events_eog_hor
        return d

    def get_info(self):
        super().get_info()
        logger.info("Info {}\n{}".format(self._cls_name,self.dump()))
