#!/usr/bin/env python3
# -+-coding: utf-8 -+-

"""
"""

#--------------------------------------------
# Authors: Frank Boers <f.boers@fz-juelich.de> 
#
#-------------------------------------------- 
# Date: 08.10.19
#-------------------------------------------- 
# License: BSD (3-clause)
#--------------------------------------------
# Updates
#--------------------------------------------

"""
https://stackoverflow.com/questions/6866600/how-to-parse-read-a-yaml-file-into-a-python-object
"""
import os,os.path as op
import logging,yaml

from jumeg.base.jumeg_base import jumeg_base as jb
from jumeg.base            import jumeg_logger

logger = logging.getLogger("jumeg")

__version__= "2019.10.08.001"

class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)

class dict2obj(dict):
    def __init__(self, dict_):
        super(dict2obj, self).__init__(dict_)
        for key in self:
            item = self[key]
            if isinstance(item, list):
                for idx, it in enumerate(item):
                    if isinstance(it, dict):
                        item[idx] = dict2obj(it)
            elif isinstance(item, dict):
                self[key] = dict2obj(item)

    def __getattr__(self, key):
        # Enhanced to handle key not found.
        if self.has_key(key):
            return self[key]
        else:
            return None
        
class JuMEG_CONFIG_YAML_BASE(object):
    """
    load or get yaml config as dict
    convert to object
    
    Example:
    --------
     cfg["test"] => cfg.test
    """
    def __init__(self,**kwargs):
        self._fname  = None
        self._data   = None
        self.verbose = False
        self.debug   = False
        self._init(**kwargs)
    
    @property
    def filename(self): return self._fname
    
    @property
    def data(self): return self._data
    
    def _init(self,**kwargs):
        pass
    
    def load_cfg(self,fname=None,key=None):
        if fname:
            self._fname = fname
        with open(self._fname,'r') as f:
            data = yaml.full_load(f)
            if key:
                self._data = dict2obj( data.get(key))
            else:
                self._data = dict2obj(data)
        return self._data
    
    def update(self,cfg,key=None):
        if isinstance(cfg(dict)):
            if key:
                self._data = dict2obj( cfg.get(key) )
            else:
                self._data = dict2obj( cfg)
        else:
            self.load_cfg(fname=cfg)