import os,sys
import numpy as np

import mne
from jumeg.base.jumeg_base import JuMEG_Base_IO

import logging
logger = logging.getLogger('jumeg')

__version__="2019-04-11-001"

#print "########## Refchan geo data:"
# This is just for info to locate special 4D-refs.
#for iref in refpick:
#    print raw.info['chs'][iref]['ch_name'],
#raw.info['chs'][iref]['loc'][0:3]

#----------------------------------------------------------------------------------------
class JuMEG_TSV_Utils_IO_Data(JuMEG_Base_IO):
    __slots__ =["_fname","_path","_raw","experiment","bads","dtype_original","dtype_plot","verbose","append_bads","_isLoaded"]
    def __init__(self,**kwargs):
        super().__init__()
        self._init(**kwargs)
        
    @property
    def isLoaded(self): return self._isLoaded
    @property
    def raw(self): return self._raw
    @property
    def fname(self): return self._fname
    @property
    def path(self): return self._path
    @property
    def filename(self): return  os.path.join(self._path,self._fname)
    
    def GetBads(self):
        self._raw.info.get('bads')

    #def SetBads(self,bads):
    #    self._raw.update_bad_channels(raw=self.raw,bads=bads)

    def get_parameter(self,key=None):
       """
       get  host parameter
       :param key:)
       :return: parameter dict or value of  parameter[key]
       """
       if key: return self.__getattribute__(key)
       return {slot: self.__getattribute__(slot) for slot in self.__slots__}
  
    def _init(self,**kwargs):
      #--- init slots
       for k in self.__slots__:
           self.__setattr__(k,None)
          
       self.append_bads    = True
       self.dtype_original = None
       self.dtype_plot     = np.float32
   
       self._update_from_kwargs(**kwargs)
  
    def _update_from_kwargs(self,**kwargs):
        if not kwargs: return
        for k in kwargs:
            try:
               self.__setattr__(k,kwargs.get(k))
             # self.__setattr__(k,kwargs.get(k)) #,self.__getattribute__(k)))
            except:
               pass#
    
    def update(self,**kwargs):
        """
        :param path:
        :param fname:
        :param raw:
        :param reload:
        :return:
        """
        self._update_from_kwargs(**kwargs)
        
        self.verbose=True
        
        if (kwargs.get("reload") and self.isLoaded):
           fname  = self.get_fif_name(raw=self.raw)
           self.load_data(fname=fname)
        else:
           self.load_data(fname=kwargs.get("fname"),path=kwargs.get("path"),raw=kwargs.get("raw"),ica=kwargs.get("ica"))
    
    def info(self):
        logger.info("---> JuMEG TSV IO data loaded\n" +
                    "  -> path : {}\n".format(self.path) +
                    "  -> file : {}\n".format(self.fname) +
                    "  -> ICA  : {}\n".format(ica))

    def load_data(self,raw=None,fname=None,path=None,ica=False):
        """
          
        :param self:
        :param raw:
        :param fname:
        :param ica : <False>
        :return:
        """
        
        self._isLoaded = False
        
        self._raw,self._fname = self.get_raw_obj(fname,raw=raw,path=path,preload=True)
       
        if not self._raw:
           return
        self._path,self._fname = os.path.split( self.fname )
        self.bads            = self._raw.info.get('bads')
        self.dtype_original  = self._raw._data.dtype
        self._raw._data      = self._raw._data.astype(self.dtype_plot)
        
        self._isLoaded   = True
        
        if self.verbose: self.info()
        
        return self.raw,self.bads
    
        
    def save_bads(self):
        self._raw._data      = self._raw._data.astype(self.dtype_original)
        return self.update_bad_channels(raw=self.raw,bads=self.bads,append=self.append_bads,save=True)


class JuMEG_TSV_Utils_IO_ICAChop(JuMEG_TSV_Utils_IO_Data):
    __slots__ = ["_fname","_path","_raw","experiment","bads","dtype_original","dtype_plot","verbose",
                 "append_bads","_isLoaded"]
    """
    ToDo
    chop_times -> chop_list

    """
    
    def __init__(self,**kwargs):
        super().__init__()
        self._init(**kwargs)


    def load_data(self,raw=None,fname=None,path=None,ica=False):
        """
    
        :param self:
        :param raw:
        :param fname:
        :param ica : <False>
        :return:
        """
        
        self._isLoaded = False
        
        self.get_ica_raw_obj(fname,ica_raw=raw)
        
        if not self._raw:
            return
        
        self._path,self._fname = os.path.split(self.fname)
        
        self.bads = self._raw.info.get('bads')
        self.dtype_original = self._raw._data.dtype
        self._raw._data = self._raw._data.astype(self.dtype_plot)
        
        self._isLoaded = True
        
        if self.verbose:
            self.info()
        
        return self.raw,self.bads


class JuMEG_TSV_Utils_IO_ICAData(JuMEG_TSV_Utils_IO_Data):
    __slots__ =["_RAW","_ICA","verbose"]
    """
    ToDo
    chop_times -> chop_list
    
    """
    def __init__(self,**kwargs):
        super().__init__()
        self._RAW = JuMEG_TSV_Utils_IO_Data(**kawargs)
        self._ICA = JuMEG_TSV_Utils_IO_ICAChop(**kawargs)

        self._init(**kwargs)

        
        
'''
ToDo
fraw
load ica/chops/  *-ica.fif to
chop list list put in combobox

load ica chops
 deselect/save ICs
 
apply  filter, ica_transform


'''
