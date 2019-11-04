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
import logging,yaml,pprint

from jumeg.base.jumeg_base import jumeg_base as jb
from jumeg.base            import jumeg_logger

logger = logging.getLogger("jumeg")

__version__= "2019.10.08.001"

class _Struct:
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

class Struct(object):
    """
    https://stackoverflow.com/questions/1305532/convert-nested-python-dict-to-object
    Nr: 30
    """
    def __init__(self, data):
        for name, value in data.items():
            setattr(self, name, self._wrap(value))

    def _wrap(self, value):
        if isinstance(value, (tuple, list, set, frozenset)):
            return type(value)([self._wrap(v) for v in value])
        else:
            return Struct(value) if isinstance(value, dict) else value
        
        
class JuMEG_CONFIG_YAML_BASE(object):
    """
    load or get yaml config as dict
    convert to object
    
    Example:
    --------
     cfg["test"] => cfg.test
     cfg["test"]["A"] => cfg.test.A
    """
    def __init__(self,**kwargs):
        self._fname = None
        self._data  = None
        self._cfg   = None
        self._init(**kwargs)
    
    @property
    def data(self): return self._data
   
    @property
    def filename(self): return self._fname
    
    def _init(self,**kwargs):
        pass
    
    def info(self):
        """print config dict"""
        logger.info("---> config file: {}\n{}/n".format(self.filename,pprint.pformat(self._cfg,indent=4)))
        

    #def GetData(self,key=None):
    #    if key:
    #        return self._cfg.et(key)
    #    return self._cfg
    
    def GetDataDict(self,key=None):
        if key:
           return self._cfg.get(key)
        return self._cfg
    
    def _init(self,**kwargs):
        pass
    
    def load_cfg(self,fname=None,key=None):
        if fname:
            self._fname = fname
        with open(self._fname,'r') as f:
            self._cfg = yaml.full_load(f)
            if key:
                self._cfg = self._cfg.get(key)
            self._data = Struct( self._cfg )
            
        return self._data
    
    def update(self,**kwargs):
        """
        update config obj
        :param config: config dict or filename
        :param key: if <key> use part of config e.g.: config.get(key)
        :return:
        """
        self._cfg = kwargs.get("config",None)
        key = kwargs.get("key",None)
        
        if isinstance(self._cfg,(dict)):
           if key:
              self._cfg  = self._cfg.get(key)
           self._data  = Struct(self._cfg)
           self._fname = None
        else:
           self.load_cfg(fname=self._cfg,key=key)