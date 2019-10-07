# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 09:19:26 2015

@author: fboers
"""
import re,time
import numpy as np
from pubsub import pub
import mne # groups channel info
from jumeg.base.jumeg_base import jumeg_base as jb
# import wx.lib.colourdb as WX_CDB

import logging
logger = logging.getLogger("jumeg")

__version__="2019-09-23-001"

class ColourSettings( object ):
      """
      plot colour cls      
      
      
      Reminder:
      ---------
      HoToGet some wx.Colours
      
      import wx
      import wx.lib.colourdb as cdb
      cdb.getColourInfoList()  # ~ 670
      cl = cdb.getColourList()
      
      cl.index('RED') =>  182
      cil=cdb.getColourInfoList()
      cil[182] => ('RED', 255, 0, 0)
      
      cidx = [ cl.index(c) for c in label_list]
      cil  = cdb.getColourInfoList()
      colours = [ cil[i] for i in cidx]
      labels = [c[0] for c in  l]
      colours= [c[1:] for c in  l]
      """
      __slots__ = ["_labels","_colours","_default_colour"]
      
      def __init__(self):
          
          self._labels = ['RED','GREEN','LIME GREEN','DARKGREEN','AQUAMARINE','BLUE','MEDIUMBLUE','MIDNIGHTBLUE','ROYALBLUE','NAVYBLUE','CYAN','YELLOW','MAGENTA',
                          'VIOLET','PURPLE','GREY40','GREY50','GREY60','GREY70','GOLD','PERU','BROWN','ORANGE','DARKORANGE','PINK','HOTPINK','MAROON','ORCHID1','WHITE','BLACK']
          self._colours = np.array([(255, 0, 0),(0, 255, 0),(50, 205, 50),(0, 100, 0),(127, 255, 212),(0, 0, 255), (0, 0, 205), (25, 25, 112), (65, 105, 225), (0, 0, 128), (0, 255, 255), (255, 255, 0),
                                   (255, 0, 255),(238, 130, 238),(155, 48, 255),(102, 102, 102), (127, 127, 127),(153, 153, 153), (179, 179, 179), (255, 215, 0), (205, 133, 63), (165, 42, 42), (255, 165, 0),
                                   (255, 140, 0),(255, 192, 203),(255, 105, 180),(176, 48, 96),(255, 131, 250),(255,255,255),(0,0,0)], dtype=np.float32 )
          
          
          self._default_colour   = 'GREY50'
      
      @property
      def colours(self): return self._colours
      
      @property
      def n_colours(self): return len( self._colours )
      
      @property
      def labels(self): return self._labels
      
      @property
      def default_colour(self):
          return self._default_colour
      @default_colour.setter
      def default_colour(self,c):
          if self.iscolour(c):
             self._default_colour = c
      
      def label2index(self,label):
          return self._labels.index( label.upper() )
    
      def index2colour(self,index):
          return self._colours[index] / 255.0

      def index2RGB(self,index):
          c = self._colours[index] / 255.0
          return c[0],c[1],c[2]

      def label2colour(self,label):
          return self._colours[ self.label2index( label) ]

      def label2colourRBG(self,label):
          return self.index2RGB( self.label2index( label) )
      
      def colour2index(self,c):
          c_in_colours = np.logical_and(self._colours[:,0] == c[0],self._colours[:,1] == c[1])
          cidx = np.where(c_in_colours)[0]
          if cidx.shape[0] == 1:
             return cidx[0]
          cidx = np.where( self._colours[cidx,2] == c[2] )[0]
          if cidx.shape[0]:
             return cidx[0]
          
      def iscolour(self,label):
          return (label in self._labels)
 
#----------------------------------------
class UnitSettings(object):
      """
       SET:
        -> scale  pre-unit si-unit: 100mT => scale: 100 * 1e-3
        -> index scale: 13,  index pre-unit: 1,  si-unit: T  => 100mT => scale: 100 * 1e-3

       GET:
        -> Unit: mT
        -> Scale : 100 * 1e-3
       
       GET via Index
       
      """
      __slots__=["_prescale","_prefix","_unit","_prescales","_units"]
      def __init__(self,**kwargs):
      
         self._prescales = ["1","2","3","4","5","8","10","15","16","20","25","30","32","40","50","64","75","100","128","150","200","250","256","400","500","512","750","1000","1024"]
         self._units     = {"prefixes": ["","m","u","n","p","f","a"],
                            "factors" : [1,1e-3,1e-6,1e-9,1e-12,1e-15,1e-18],
                            "labels"  : ["T","V","bit","s","db","AU"]}
         
         self._unit   = "AU"
         self._prefix = ""
         self._prescale  = 1.0
         
         self._update_from_kwargs(**kwargs)
      
      @property
      def prefixes(self): return self._units["prefixes"]
      @property
      def prescales(self): return self._prescales
      
      def _update_from_kwargs(self,**kwargs):
          self._unit     = kwargs.get("unit",    self._unit)
          self._prefix   = kwargs.get("prefix",  self._prefix)
          self._prescale = kwargs.get("prescale",self._prescale)
      
      @property
      def prefix(self): return self._prefix
      @prefix.setter
      def prefix(self,v):
          if len(v)>1:
             v = v[0]
          if v in self._units["factors"] :
             self._prefix = v
          else:
             self._prefix = ""
          
      @property
      def unit(self): return self._prefix + self._unit
      @unit.setter
      def unit(self,v):
         #--- no digits, no space
          v = re.sub(r"\d|\s+", "", v)
          if not v:
             self._unit = "AU"
             return
          if v[0] in self._units["prefixes"]:
             self._prefix = v[0]
             v = v[1:]
          else:
             self._prefix = ""
          if v in self._units["labels"]:
             self._unit = v
              
      @property
      def prescale(self): return self._prescale
      @prescale.setter
      def prescale(self,v):
          self._prescale = v #float(v) # re.sub(r"\D", "", v) )
          
      def GetUnits(self):
          return [ u + self._unit for u in self.prefixes ]
          
      def GetScaleFromIndex(self,pre_scale_idx,factor_idx):
          """
          
          :param scale_idx:
          :param factor_idx:
          :return:
          """
          return int( self._prescales[pre_scale_idx] ) * self._units["factor"][factor_idx]
      
      def GetScale(self,prescale=None,unit=None):
          """
          
          :param pre_scale:
          :param unit:
          :return:
          """
          #factor_idx = 0
          #fac =0.0
          
          if prescale:
             self._prescale = prescale
          if unit:
             self.unit = unit
          try:
             factor_idx = self._units["prefixes"].index( self._prefix )
             fac        = float( self._units["factors"][factor_idx] )
          except:
              fac = 1.0
          return float(self._prescale) * fac
          
      def update(self,**kwargs):
          self._update_from_kwargs(**kwargs)

class STATUS(object):
    __slots__=["selected","colour","scale","dcoffset","update"]
    def __init__(self):
        self.selected = 1
        self.colour   = 2
        self.scale    = 4
        self.dcoffset = 8
        self.update   = 255
      
class SCALE_MODE(object):
    """
    helper cls for scale mod: scale/div, scale on MIMAX, window or global
    """
    def __init__(self):
        
        pass
    @property
    def division(self): return 0
    @property
    def minmax_on_global(self): return 1
    @property
    def minmax_on_window(self): return 2

class GroupSettingsBase(object):
      """
      ToDO support more MNE Channel Groups
      MNE Channel Groups
      https://www.nmr.mgh.harvard.edu/mne/stable/auto_tutorials/intro/plot_info.html
      eeg : For EEG channels with data stored in Volts (V)
      meg (mag) : For MEG magnetometers channels stored in Tesla (T)
      meg (grad) : For MEG gradiometers channels stored in Tesla/Meter (T/m)
      ecg : For ECG channels stored in Volts (V)
      seeg : For Stereotactic EEG channels in Volts (V).
      ecog : For Electrocorticography (ECoG) channels in Volts (V).
      fnirs (HBO) : Functional near-infrared spectroscopy oxyhemoglobin data.
      fnirs (HBR) : Functional near-infrared spectroscopy deoxyhemoglobin data.
      emg : For EMG channels stored in Volts (V)
      bio : For biological channels (AU).
      stim : For the stimulus (a.k.a. trigger) channels (AU)
      resp : For the response-trigger channel (AU)
      chpi : For HPI coil channels (T).
      exci : Flux excitation channel used to be a stimulus channel.
      ias : For Internal Active Shielding data (maybe on Triux only).
      syst : System status channel information (on Triux systems only).
      """
      def __init__(self,**kwargs):
  
          self.Status  = STATUS()
          self._labels = [] #['grad','mag','ref_meg','eog','emg','ecg','stim','resp','eeg'] #,'resp'] #,'exci','ias','syst','misc','seeg','chpi']
          
          #---ToDo get groups/channel_types from mne
          '''
          grps=mne.io.pick.get_channel_types()
          grps.keys()
         # dict_keys(['grad', 'mag', 'ref_meg', 'eeg', 'stim', 'eog', 'emg', 'ecg', 'resp', 'misc', 'exci', 'ias', 'syst', 'seeg', 'bio', 'chpi', 'dipole', 'gof', 'ecog', 'hbo', 'hbr'])
          
          '''
          self._grp = {}
         #---fixed pos to display
          self._default_labels = ['mag','grad','ref_meg', 'ecg','eog','emg','eeg','stim','resp','misc']
          
          self._default_grp={
                       'mag':    {"selected":True,"colour":"RED",         "prescale":500,"unit":"fT"},
                       'grad':   {"selected":True,"colour":"BLUE",        "prescale":500,"unit":"fT"},
                       'ref_meg':{"selected":True,"colour":"DARKGREEN",   "prescale":4,  "unit":"pT"},
                       'eeg':    {"selected":True,"colour":"MIDNIGHTBLUE","prescale":1,  "unit":"uV"},
                       'eog':    {"selected":True,"colour":"PURPLE",      "prescale":100,"unit":"uV"},
                       'emg':    {"selected":True,"colour":"DARKORANGE",  "prescale":100,"unit":"uV"},
                       'ecg':    {"selected":True,"colour":"MAROON",      "prescale":1,  "unit":"mV"},
                       'stim':   {"selected":True,"colour":"MAGENTA",     "prescale":10, "unit":"bit"},
                       'resp':   {"selected":True,"colour":"NAVYBLUE",    "prescale":10, "unit":"bit"},
                       'default':{"selected":True,"colour":"PURPLE",      "prescale":1,  "unit":"AU"}
              }
          for g in self._default_grp.keys():
              self._default_grp[g]["status"]       = 0
              self._default_grp[g]["scale_mode"]   = 2  # min max window
              self._default_grp[g]["DCOffsetMode"] = 0 # fit to window
              self._default_grp[g]["bads"]         = []
 
          for g in ['stim','resp']:
              self._default_grp[g]["scale_mode"]=0
       
        #--- set meg mags
          g  = "mag"
          self._default_grp[g]["scale_mode"] = 0
          self._default_grp[g]["prescale"]   = 1
          self._default_grp[g]["unit"]       = "pT"
          
          
      @property
      def labels(self): return self._labels
      @labels.setter
      def labels(self,v):
          self._labels=v
      
      def DeleteNonExistingGroup(self,grps):
          """
          
          :param grps:  label or list of labels
          :return:
          """
          if not isinstance(grps,(list)):
             grps=[grps]
          grps=list( set(grps) )
          grps2delete = [ item for item in grps if item not in self.labels ]
          
          for grp in grps2delete:
              if self._grp.pop(grp,None):
                 try:
                      idx = self.labels.index( grp )
                      self.labels.pop(idx)
                 except:
                      pass
                      
      def GetStatus(self):
          """
          check for each group if parameter changed
          :return: group list
          """
          gs = []
          for g in self._grp:
              if self._grp["status"]:
                 gs.append(g)
          if len(g):
             return g
          return 0
     
      def _get_grp_key(self,grp,key):
          if grp:
             if key:
                return self._grp[grp][key]
             else:
                return self._grp[grp]

      def _set_grp_key(self,grp,key,v):
          if grp:
             if key:
                self._grp[grp][key] = v
      
      def GetGroup(self,grp):
          if grp:
             return self._get_grp_key(grp,None)
          return self._grp
          
      def SetGroup(self,grp,v):
          if grp:
             self._grp[grp]=v
             self._grp[grp]["status"] = self.Status.allkeys
          if grp not in self._labels:
             self._labels.append(grp)

      def GetSelected(self,grp):
          return self._get_grp_key(grp,"selected")
      def SetSelected(self,grp,v):
          self._set_grp_key(grp,"selected",v)
          self._grp[grp]["status"] = self._grp[grp]["status"] | self.Status.selected
      def GetStatusSelected(self,grp):
          return self._grp[grp]["status"] & self.Status.selected
      
      def GetGroupNameFromIndex(self,idx):
          return self._labels[idx]
      
      def GetIndex(self,grp):
          # return self._get_grp_key(grp,"index")
          return self._labels.index(grp)
      #def SetIndex(self,grp,v):
      #    self._set_grp_key(grp,"index",v)
          
      def GetColour(self,grp):
          return self._get_grp_key(grp,"colour")
      def SetColour(self,grp,v):
          self._set_grp_key(grp,"colour",v)
          self._grp[grp]["status"] = self._grp[grp]["status"] | self.Status.colour
      def GetStatusColour(self,grp):
          return self._grp[grp]["status"] & self.Status.colour

      def GetPreScale(self,grp):
          return self._get_grp_key(grp,"prescale")
      def SetPreScale(self,grp,v):
          self._set_grp_key(grp,"prescale",v)
          self._grp[grp]["status"] = self._grp[grp]["status"] | self.Status.scale
          
      def GetUnit(self,grp):
          return self._get_grp_key(grp,"unit")
      def SetUnit(self,grp,v):
          self._set_grp_key(grp,"unit",v)
          self._grp[grp]["status"] = self._grp[grp]["status"] | self.Status.scale
 
      def GetDCOffsetMode(self,grp):
          return self._get_grp_key(grp,"DCOffsetMode")
      def SetDCOffsetMode(self,grp,v):
          self._set_grp_key(grp,"DCOffsetMode",v)
          self._grp[grp]["status"] = self._grp[grp]["status"] | self.Status.dcoffset
      def GetStatusDCOffsetMode(self,grp):
          return self._grp[grp]["status"] & self.Status.dcoffset
        
      def GetStatusScale(self,grp):
          return self._grp[grp]["status"] & self.Status.scale
      
      def GetStatus(self,grp):
          return self._grp[grp]["status"]
      def SetStatus(self,grp,v):
          self._grp[grp]["status"] = v

      def UpdateLabels(self,lin):
          
          lin = list(set(lin))
          self._labels = []
          for label in self._default_labels:
              if label in lin:
                 self._labels.append(label)
                 del lin[ lin.index(label) ]
          self._labels.extend(lin)
          

      def ResetStatus(self,grp,status=None):
          """
          
          :param status:
          :return:
          """
          if status:
             self._grp[grp]["status"] = self._grp[grp]["status"] ^ status
          else:
             self._grp[grp]["status"] = 0
          
class GroupSettings(GroupSettingsBase):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
       # self._defaults = GroupSettingsBase(**kwargs)
        self.Colour    = ColourSettings()
        self.Unit      = UnitSettings()
        
        self._scale_modes   = ["Divsion","Global","Window"]
        self._dcoffset_modes= ["None","Global","Window"]
        
    @property
    def ScaleModes(self): return self._scale_modes

    @property
    def DCOffsetModes(self):
        return self._dcoffset_modes

    def GetScaling(self,grp):
        return self.Unit.GetScale(prescale=self._grp[grp]["prescale"],unit=self._grp[grp]["unit"] )
        
    def SetColour(self,grp,c):
        if not isinstance(c,(str)):
           idx = self.Colour.colour2index(c)
           c   = self.Colour.labels[idx]
        super().SetColour( grp, c )

    def GetColourIndex(self,grp):
        clabel = self.GetColour(grp)
        return self.Colour.label2index(clabel)
    
    def reset(self):
        self._grp   = {}
        self._labels= []
    
    def SetChannelIndex(self,grp,idx):
        self._grp[grp]["channel_index"]=idx

    def GetChannelIndex(self,grp):
        return self._grp[grp]["channel_index"]

    def AddChannelName(self,grp,ch):
        self._grp[grp]["channels"].append(ch)
        
    def Add(self,grp,**kwargs):
        """
        ToDo
        add update keep settings if grp exist in new raw
        
          :param grp:
          :param selected: True
          :param colour  : default colour,
          :param prescale: 1
          :param unit    : "AU"
          
          :return:
        """
        
        if grp not in self._labels:
           self._labels.append(grp)
         
        if self._grp.get(grp,None):
           return
        elif self._default_grp.get(grp,None):
             self._grp[grp] = self._default_grp[grp].copy()
        else:
             self._grp[grp] = self._default_grp["default"].copy()
             
        self._grp[grp]["channels"]      = []
        self._grp[grp]["channel_index"] = np.array([],dtype=np.uint32)
        
        #self._grp[grp]["bads"]          = None

    def GetInfo(self):
        msg=["---> Group Info"]
        for grp in self.labels:
            msg.append("\n --> {}: ".format(grp))
            msg.append("  -> {}: ".format(self.GetGroup(grp)))
        
        logger.info("\n".join(msg))

    def GetScaleMode(self,grp):
        return self._grp[grp]["scale_mode"]

    def SetScaleMode(self,grp,v):
        self._grp[grp]["scale_mode"] = v
        self._grp[grp]["status"] = self._grp[grp]["status"] | self.Status.scale

class ChannelSettings(object):
    __slots__ = ["colour_index","colour","selected","scale","group_index","bads","labels","verbose","debug","scale_mode","bit_index",
                 "dcoffset","dcoffset_mode","mean","minmax_global","_colour_bads","_ScaleMode","_DC_OFFSET_WINDOW"]
    
    def __init__(self,**kwargs):
        self._init(**kwargs)
    
    def _init(self,**kwargs):
        raw = kwargs.get("raw",None)
        
        self._ScaleMode = SCALE_MODE()
        self._DC_OFFSET_WINDOW = 2
        
        if raw:
           self.labels    = raw.info['ch_names']
           n_chan         = self.n_channels
           self.bit_index = [] #np.array([],dtype=np.int32)
           self.colour    = np.zeros([n_chan,3],dtype=np.float32)
           self.calc_min_max_mean(raw)
        else:
           self.labels = kwargs.get("labels",None)
           self.colour = np.zeros([],dtype=np.float32)
           self.mean   = np.zeros([],dtype=np.float32)
           self.minmax_global = np.zeros([],dtype=np.float32)

        self.verbose = kwargs.get("verbose",False)
        self.debug   = kwargs.get("debug",False)
        self._colour_bads = kwargs.get("ColourBads",np.array([0.7,0.7,0.7],dtype=np.float32) )
        
        n_chan = self.n_channels
        self.selected     = np.zeros(n_chan,dtype=np.bool)
        self.bads         = np.zeros(n_chan,dtype=np.bool)
        self.scale        = np.zeros(n_chan,dtype=np.float32)
        self.colour_index = np.zeros(n_chan,dtype=np.uint32)
        self.group_index  = np.zeros(n_chan,dtype=np.uint32)
        self.scale_mode   = np.zeros(n_chan,dtype=np.uint32)
        self.dcoffset_mode= np.zeros(n_chan,dtype=np.uint32)   # 0=none,1=global,2=window
        self.dcoffset     = np.zeros(n_chan,dtype=np.float32)
    
    def calc_min_max_mean(self,raw):
        self.mean          = np.array( raw._data.mean(axis=1),dtype=np.float32)
        self.minmax_global = np.array([ raw._data.min(axis=1),raw._data.max(axis=1) ],dtype=np.float32)
 
    def GetDCoffset(self,raw=None,tsls=None):
        """
        calc DC Offset for each channel depending on DCOffsetMode:
        None => no DCOffset, GlobalDC=> global mean,  WindowDc => mean in TimeWindow range
        dcoffset will be subsracted in OGL Vertex Shader
        
        :param raw: raw obj
        :param tsls: start,end of DCoffset window
        :return: dcoffset np.array[n_channesl]
        """
       #--- get all dcoffsets:  None => 0, DCGlobal=> 1  => substract mean in GLShader
        # t0 = time.time()
        idx = np.where(self.dcoffset_mode < self._DC_OFFSET_WINDOW)[0]
        if idx.shape[0]:
           if idx.shape[0] < self.dcoffset.shape[0]: # all None or Global
              self.dcoffset[idx] = self.mean[idx] * self.dcoffset_mode[idx]
           else:
              self.dcoffset = self.mean * self.dcoffset_mode
              return self.dcoffset
           
       #-- DC Window => self._DCMode.DCWindow = 2
        if raw:
           if tsls:
              idx = np.where(self.dcoffset_mode == self._DC_OFFSET_WINDOW)[0]
              if idx.shape[0]:
                 self.dcoffset[idx] = raw._data[idx,tsls[0]:tsls[1]].mean(axis=1)
              
        # t1 = time.time()
        # logger.info("--> DC Offset Time: {}".format(t1-t0))
        
        return self.dcoffset

    def GetMinMaxScale(self,raw=None,tsls=None,picks=None,div=1.0):
        """
        check for scale => min-max global,min-max window
        :param tsls:
        :param divisions:
        :return: index for division,minmax_global,min,max-window
        """
        
        scales = np.zeros([self.n_channels,2],dtype=np.float32) #scale,offset
        found=0
        
       #--- get scale for minmax window
        picks_idx = np.asarray(self.scale_mode[picks] == self._ScaleMode.minmax_on_window).nonzero()[0]
        if picks_idx.shape[0]:
           idx = picks[picks_idx]
           data = raw._data[idx,tsls[0]:tsls[1]]
           min = data.min(axis=1)
           max = data.max(axis=1)
           dmm = np.abs( (max - min) / div )
           scales[idx,0] = data.min(axis=1) - dmm
           scales[idx,1] = data.max(axis=1) + dmm

           if picks_idx.shape[0] == self.n_channels:
              return scales
           found = picks_idx.shape[0]
           
       #--- get scale for minmax global
        picks_idx = np.asarray(self.scale_mode[picks] == self._ScaleMode.minmax_on_global).nonzero()[0]
        if picks_idx.shape[0]:
           idx = picks[picks_idx]
           scales[idx,0] = self.minmax_global[0,idx]
           scales[idx,1] = self.minmax_global[1,idx]
           dmm = np.abs((scales[idx,1] - scales[idx,0]) / div)
           scales[idx,0] -= dmm
           scales[idx,1] += dmm

           if picks_idx.shape[0] + found >= self.n_channels:
              return scales

       #--- get scale division
        picks_idx = np.asarray(self.scale_mode[picks] == self._ScaleMode.division).nonzero()[0]
        if picks_idx.shape[0]:
           idx = picks[picks_idx]
           scales[idx,1] =  self.scale[idx] * div /2
           scales[idx,0] = - scales[idx,1]
           if self.bit_index:  #.shape[0]:
              scales[self.bit_index,0] = - 1.0  # set min for ch in  stim,res group
        
        return scales
        
    
    @property
    def n_channels(self):
        if self.labels:
           return len(self.labels)
        return None
    
    @property
    def ColourBads(self):
        return self._colour_bads
    @ColourBads.setter
    def ColourBads(self,v):
        self._colour_bads = v
        
    def reset(self,**kwargs):
        """
        :param labels: pointer to  raw.info['ch_names']
        :param verbose
        :param debug
        :return:
        """
        self._init(**kwargs)
        
    #---
    def GetBadsPicks(self):
        return np.where(self.bads)[0]
    #---
    def GetBadsLabels(self):
        picks = self.GetBadsPicks()
        if picks.shape:
           bads = [self.labels[i] for i in picks]
           bads = list(set( bads ) )
           bads.sort()
           return bads
        return []

    def GetNoBads(self,idx):
        return np.where(self.bads == False)[0]
   
    def SetBadsColour(self,colour=None,index=None):
        """
        :param index : index in colour tab
        :param colour: np.arrax [3] dtype float32
        :return:
        """
        if colour:
           self._colour_bads = colour
        picks = np.where(self.bads)[0].flatten()
       
       # logger.info("set BADs colour:  picks: {}  colour: {}".format(picks,self._colour_bads))
        
        if not picks.shape[0]: return
        if index:
           self.colour_index[picks] = index
        self.colour[picks] = self._colour_bads
        
    def GetSelected(self):
        return np.where(self.selected)[0]
    
    def GetDeselected(self):
        return np.where(self.selected == False)[0]
    
    #---
    def GetScaleFromIndex(self,idx):
        """
        
        :param self:
        :param idx:
        :return:
        """
        return self.scale[idx]
    
    def SetScaleFromIndex(self,idx,v):
        """
        
        :param self:
        :param idx:
        :param v:
        :return:
        """
        self.scale[idx] = v
    
    #---
    def GetGroupIndex(self,idx):
        return self.group_index[idx]
    
    def SetGroupIndex(self,idx,v):
        self.group_index[idx] = v
  
   #--- selected channels
    def selected_index(self):
        return np.array( np.where(self.selected == True),dtype=np.uint32).flatten()
 
    def selected_label(self):
        return self.label[ self.selected_index() ]

    def SetSelected(self,picks=None,status=False):
        if picks is not None:
           if picks.shape[0]:
              self.selected[picks]=status
              return
        self.selected[:]=status
           
     
    def GetInfo(self):
        """
        write info to logger
        :return:
        """
        msg=[]
        msg.append(" {} {} {} {} {} {} {}".format("index","label","selected","colour","colour_index","scale","group_index","bads","dcoffset"))
        
        if not self.debug: return
        
        for idx in range(self.n_channels):
            msg.append(" {} {} {} {} {} {} {} {}".format(idx,self.labels[idx],self.selected[idx],self.colour[idx],self.colour_index[idx],self.scale[idx],self.group_index[idx],self.bads[idx],self.dcoffset[idx]))
        
        logger.debug("---> Channel INFO:\n"+ "\n  -> ".join(msg))
        
        #--- working with numpy arrays in pl_channels class
        if self.debug:
           logger.debug("---> channel.colour_index: {}".format(self.colour_index))
           logger.debug("---> channel.selected: {}".format(self.selected))
           logger.debug("---> channel.label: {}".format(self.labels))
           logger.debug("---> channel.scale: {}".format(self.scale))
           logger.debug("---> channel.group_index: {}".format(self.group_index))
           logger.debug("---> channel bit index: {}".format(self.Channel.bit_index))
   
class JuMEG_TSV_PLOT2D_DATA_SETTINGS(object):
      __slots__=["raw","verbose","debug","_Group","_Channel"]
      def __init__ (self,**kwargs):
          self._Group   = GroupSettings()
          self._Channel = ChannelSettings()
    
          self.raw          = None
          self.verbose      = False
          self.debug        = False
       
          self._init(**kwargs)
   
   
      
      @property
      def Group(self): return self._Group
      @Group.setter
      def Group(self,v):
          self._Group=v

      @property
      def Channel(self):return self._Channel
      @Channel.setter
      def Channel(self,v):
          self._Channel=v

      def _update_from_kwargs(self,**kwargs):
          self.verbose    = kwargs.get("verbose",self.verbose)
          self.debug      = kwargs.get("debug",self.debug)
          self.raw        = kwargs.get("raw",self.raw)
          
      def _init(self,**kwargs):
          self._update_from_kwargs(**kwargs)
          #self._init_pubsub()
          self.update()
          
      def _init_pubsub(self):
         """ init pubsub call and messages"""
        #--- verbose debug
         pub.subscribe(self.SetVerbose,'MAIN_FRAME.VERBOSE')
         pub.subscribe(self.SetDebug,'MAIN_FRAME.DEBUG')

      def SetVerbose(self,value=False):
          self.verbose = value

      def SetDebug(self,value=False):
          self.debug = value
          
      def _remove_obsolete_groups(self):
          old_grps = self.Group.labels.copy()
          self.Group.labels = []
         #--- find obsolete grp labels
          raw_grps = dict()
          for g in old_grps:
              raw_grps[g] = False
    
          #--- set group in new raw true
          for idx in range(self.Channel.n_channels):
              g = mne.io.pick.channel_type(self.raw.info,idx)
              raw_grps[g] = True
         #--- delete obsolete groups not in new raw
          grps2del = []
          labels   = []
          for k,v in raw_grps.items():
              if v:
                 labels.append(k)
              else:
                 grps2del.append(k)
                 
          self.Group.DeleteNonExistingGroup(grps2del)
          self.Group.UpdateLabels(labels)

      def init_group_info(self):
          
          self.Channel.reset( raw=self.raw )
          
          self._remove_obsolete_groups()
          
          for idx in range(self.Channel.n_channels):
             #--- get group for idx meg eeg ...
              g = mne.io.pick.channel_type(self.raw.info,idx)
              self.Group.Add(g)
             #---  numpy arrays in ChannelOption class
              self.Channel.group_index[idx] = self.Group.GetIndex(g)
        
        # logger.info("RAW BADS: {} ".format(self.raw.info['bads']))
          
          bads_idx = jb.picks.bads2picks(self.raw)
          
          if isinstance(bads_idx,(np.ndarray)):
             if bads_idx.shape[0]:
               # logger.info("RAW BADS: {}".format(self.raw.info['bads']))
                self.Channel.bads[ bads_idx ] = True
             
              # colour_bads = self.Group.Colour.label2colourRBG("GREY70")
              # colour_idx  = self.Group.Colour.colour2index( self.Channel.ColourBads )
              # self.Channel.SetBadsColour(colour=colour_bads,index=colour_idx )
          
          bit_index=[]
          for grp in self.Group.labels:
              idx  = self.Group.GetIndex(grp)
              cidx = np.array(np.where(self.Channel.group_index == idx),dtype=np.uint32).flatten()
              self.Group.SetChannelIndex(grp,cidx)
              self.Group.SetStatus(grp,self.Group.Status.update)
              if self.Group.GetUnit(grp) == "bit":
                 bit_index.extend( cidx.tolist() )
          if bit_index:
             bit_index.sort()
             self.Channel.bit_index = bit_index
            
      def update_channel_options(self,**kwargs):
        
          for grp in self.Group.labels:
              if self.Group.GetStatus(grp):
                 picks = self.Group.GetChannelIndex(grp)
                 
                 self.Channel.dcoffset_mode[picks] = self.Group.GetDCOffsetMode(grp)
                 
                 if self.Group.GetStatusSelected(grp):
                     self.Channel.selected[picks] = self.Group.GetSelected(grp)
   
                 if self.Group.GetStatusScale(grp):
                    """
                    ck scale mode:
                    division: copy scaling to channels
                    in plot set trafo matrix
                    
                    """
                    # logger.info( "GRP:{}".format(grp))
                    self.Channel.scale[picks]      = self.Group.GetScaling(grp)
                    self.Channel.scale_mode[picks] = self.Group.GetScaleMode(grp)
                    
                 if self.Group.GetStatusColour(grp):
                    colour_idx = self.Group.GetColourIndex(grp)
                    self.Channel.colour_index[picks] = colour_idx
                    self.Channel.colour[picks]       = self.Group.Colour.index2colour(colour_idx)
                
                 self.Group.ResetStatus(grp)
          self.Channel.SetBadsColour()
     
      def update(self,**kwargs):
          self._update_from_kwargs(**kwargs)
          
          if "raw" in kwargs.keys():
             self.init_group_info()
          
          self.update_channel_options(**kwargs)
        
          #if self.debug:
          #   self.GetInfo()
         
    
      def GetInfo(self):
          self.Group.GetInfo()
          self.Channel.GetInfo()
          pass
      #---
      def ToggleBads(self,idx):
      
      #--- no bads, reset colour
          if self.Channel.bads[idx]:
             grp  = self.Group.GetGroupNameFromIndex(self.Channel.group_index[idx])
             cidx = self.Group.GetColourIndex(grp)
             self.Channel.colour[idx] = self.Group.Colour.index2RGB(cidx)
             self.Channel.bads[idx]   = False
          else:
             self.Channel.bads[idx] = True
             self.Channel.SetBadsColour(idx)
          
          self.update_bads_in_raw()
         #logger.info("TGbads: idx: {} => {}  raw bads:".format(idx,self.Channel.bads[idx],self.raw.info["bads"] ))
      
      def update_bads_in_raw(self):
          self.raw.info["bads"] = self.Channel.GetBadsLabels()
        