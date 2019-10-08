# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 09:19:26 2015

@author: fboers
"""
import re
import numpy as np
import mne # groups channel info

# import wx.lib.colourdb as WX_CDB

import logging
logger = logging.getLogger("JuMEG")

class ColourOptions( object ):
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
          return self._colour[index][0]/ 255.0,self._colour[index][1]/ 255.0,self._colour[index][2]/ 255.0

      def label2colour(self,label):
          return self._colours[ self.label2index( label) ]

      def label2colourRBG(self,label):
          return self.index2RGB( self._label2index( label) )
      
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
class UnitOptions(object):
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
      
         self._prescales = ["1","2","3","4","5","10","15","20","25","30","40","50","75","100","150","200","250","400","500","750","1000"]
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
      
   

class GroupOptionsBase(object):
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
  
          self.Status = STATUS()
          self._labels = ['grad','mag','ref_meg','eog','emg','ecg','stim','eeg'] #,'resp'] #,'exci','ias','syst','misc','seeg','chpi']
          
          
          self._grp={
                       'mag':    {"index":0,"selected":True,"colour":"RED",       "prescale":200,"unit":"fT"},
                       'grad':   {"index":0,"selected":True,"colour":"BLUE",      "prescale":200,"unit":"fT"},
                       'ref_meg':{"index":0,"selected":True,"colour":"GREEN",     "prescale":2,  "unit":"pT"},
                       'eeg':    {"index":0,"selected":True,"colour":"BLUE",      "prescale":1,  "unit":"uV"},
                       'eog':    {"index":0,"selected":True,"colour":"PURPLE",    "prescale":100,"unit":"uV"},
                       'emg':    {"index":0,"selected":True,"colour":"DARKORANGE","prescale":100,"unit":"uV"},
                       'ecg':    {"index":0,"selected":True,"colour":"DARKGREEN", "prescale":100,"unit":"mV"},
                       'stim':   {"index":0,"selected":True,"colour":"CYAN",      "prescale":1,  "unit":"bits"}
                       }
          for g in self._labels:
            # self._grp[g]["bads"]       = None
              self._grp[g]["status"]     = 0
              self._grp[g]["DCoffset"]   = False
              self._grp[g]["scale_mode"] = 0
              
          self._default_grp = self._grp["stim"].copy()
          self._default_grp["unit"] = "AU"
          
   
      @property
      def labels(self): return self._labels
      @labels.setter
      def labels(self,v):
          self._labels=v
      
        
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
             self._grp[g]["status"] = self.Status.allkeys
          if grp not in self._labels:
             self._labels.append(grp)

      def GetSelected(self,grp):
          return self._get_grp_key(grp,"selected")
      def SetSelected(self,grp,v):
          self._set_grp_key(grp,"selected",v)
          self._grp[grp]["status"] = self._grp[g]["status"] | self.Status.selected
      def GetStatusSelected(self,grp):
          return self._grp[grp]["status"] & self.Status.selected
          
      def GetIndex(self,grp):
          return self._get_grp_key(grp,"index")
      def SetIndex(self,grp,v):
          self._set_grp_key(grp,"index",v)
          
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
 
      def GetDCoffset(self,grp):
          return self._get_grp_key(grp,"DCoffset")
      def SetDCoffset(self,grp,v):
          self._set_grp_key(grp,"DCoffset",v)
          self._grp[grp]["status"] = self._grp[grp]["status"] | self.Status.dcoffset
      def GetStatusDCoffset(self,grp):
          return self._grp[grp]["status"] & self.Status.dcoffset
        
      def GetStatusScale(self,grp):
          return self._grp[grp]["status"] & self.Status.scale
      
      def GetStatus(self,grp):
          return self._grp[grp]["status"]
      def SetStatus(self,grp,v):
          self._grp[grp]["status"] = v
          
      def ResetStatus(self,grp,status=None):
          """
          
          :param status:
          :return:
          """
          if status:
             self._grp[grp]["status"] = self._grp[grp]["status"] ^ status
          else:
             self._grp[grp]["status"] = 0
          
class GroupOptions(GroupOptionsBase):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self._defaults = GroupOptionsBase(**kwargs)
        self.Colour    = ColourOptions()
        self.Unit      = UnitOptions()
        
        self.__SCALE_MODE_DIVISION = 0
        self.__SCALE_MODE_GLOBAL   = 1
        self.__SCALE_MODE_WINDOW   = 2
        self._scale_modes = ["Divsion","Global","Window"]
        
    @property
    def ScaleModes(self): return self._scale_modes
    
    @property
    def defaults(self): return self._defaults

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
           # logger.info("---> Add group: {}\n  -> groups: {}".format(grp,self.labels))
           
        #if kwargs:
        #   self._grp[grp]={"selected": kwargs.get("selected",True),
        #                   "colour"  : kwargs.get("colour",self._default_grp["colour"]),
        #                   "prescale": kwargs.get("prescal",1),
        #                   "unit"    : kwargs.get("unit","AU"),
        #                   "index"   : kwargs.get("index",None)
        ##                 }
        #else:
        if grp in self._defaults._labels:
           self._grp[grp] = self._defaults._grp[grp].copy()
        else:
           self._grp[grp] = self._defaults._default_grp.copy()
             
        self._grp[grp]["index"]         = self._labels.index(grp)
        self._grp[grp]["channels"]      = []
        self._grp[grp]["channel_index"] = np.array([],dtype=np.uint32)
        self._grp[grp]["scale_mode"]    = self.__SCALE_MODE_DIVISION
        
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

class ChannelOptions(object):
    __slots__ = ["colour_index","colour","selected","scale","group_index","bads","labels","verbose","debug","_colour_bads","scale_mode",
                 "dcoffset","mean","minmax_global","minmax_window"]
    
    def __init__(self,**kwargs):
        self._init(**kwargs)
    
    def _init(self,**kwargs):
        
        self.labels  = kwargs.get("labels",None)
        self.verbose = kwargs.get("verbose",False)
        self.debug   = kwargs.get("debug",False)
        self._colour_bads = kwargs.get("ColourBads","GREY70")
        
        n_chan = self.n_channels
        
        self.selected     = np.zeros(n_chan,dtype=np.bool)
        self.bads         = np.zeros(n_chan,dtype=np.bool)
        self.scale        = np.zeros(n_chan,dtype=np.float32)
        self.colour_index = np.zeros(n_chan,dtype=np.uint32)
        self.group_index  = np.zeros(n_chan,dtype=np.uint32)
        self.scale_mode   = np.zeros(n_chan,dtype=np.uint32)
        
        self.dcoffset      = np.zeros(n_chan,dtype=np.bool)
        self.mean          = np.zeros(n_chan,dtype=np.float32)
        
        if n_chan:
           self.colour = np.zeros([n_chan,3],dtype=np.float32)
           self.minmax_global = np.zeros([n_chan,2],dtype=np.float32)
           self.minmax_window = np.zeros([n_chan,2],dtype=np.float32)
        else:
           self.colour        = np.zeros([],dtype=np.float32)
           self.minmax_global = np.zeros([],dtype=np.float32)
           self.minmax_window = np.zeros([],dtype=np.float32)

    @property
    def n_channels(self):
        if self.labels:
           return len(self.labels)
        return None
    @property
    def ColourBads(self): return self._colour_bads
    
    def reset(self,**kwargs):
        """
        :param labels: pointer to  raw.info['ch_names']
        :param verbose
        :param debug
        :return:
        """
        self._init(**kwargs)
    
    #---
    def GetBads(self):
        return np.where(self.bads)
    
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
        picks = self.bads
        self.colour_index[picks] = index
        self.colour[picks]       = self._colour_bads
        
    def GetSelected(self):
        return np.where(self.selected)[0]
    
    def GetDeselected(self):
        return np.where(self.selected == False)[0]
    
    #---
    def GetScale(self,idx):
        return self.scale[idx]
    
    def SetScale(self,idx,v):
        self.scale[idx] = v
    
    #---
    def GetGroupIndex(self,idx):
        return self.group_index[idx]
    
    def SetGroupIndex(self,idx,v):
        self.group_index[idx] = v
    
    def GetInfo(self):
        """
        write info to logger
        :return:
        """
        msg=[]
        msg.append(" {} {} {} {} {} {} {}".format("index","label","selected","colour","colour_index","scale","group_index","bads"))
        
        for idx in range(self.n_channels):
            msg.append(" {} {} {} {} {} {} {} {}".format(idx,self.labels[idx],self.selected[idx],self.colour[idx],self.colour_index[idx],self.scale[idx],self.group_index[idx],self.bads[idx]))
        
        logger.info("---> Channel INFO:\n"+ "\n  -> ".join(msg))
        
        #--- working with numpy arrays in pl_channels class
        if self.debug:
           logger.debug("---> channel.colour_index: {}".format(self.colour_index))
           logger.debug("---> channel.selected: {}".format(self.selected))
           logger.debug("---> channel.label: {}".format(self.labels))
           logger.debug("---> channel.scale: {}".format(self.scale))
           logger.debug("---> channel.group_index: {}".format(self.group_index))
           
   
class JuMEG_TSV_PLOT2D_DATA_OPTIONS(object):
      __slots__=["raw","verbose","debug","_Group","_Channel"]
      def __init__ (self,**kwargs):
          self._Group   = GroupOptions()
          self._Channel = ChannelOptions()
    
          self.raw          = None
          self.verbose      = False
          self.debug        = False
       
          self._init(**kwargs)
      
      @property
      def Group(self): return self._Group

      @property
      def Channel(self):return self._Channel

      def _update_from_kwargs(self,**kwargs):
          self.verbose    = kwargs.get("verbose",self.verbose)
          self.debug      = kwargs.get("debug",self.debug)
          self.raw        = kwargs.get("raw",self.raw)
          
      def _init(self,**kwargs):
          self._update_from_kwargs(**kwargs)
          self.update()
   
      def init_group_info(self):
          self.Group.reset()
          self.Channel.reset( labels= self.raw.info['ch_names'] )

          for idx in range( self.Channel.n_channels ):
             #--- get group for idx meg eeg ... 
              g = mne.io.pick.channel_type(self.raw.info,idx)
              if g not in self.Group.labels:
                 logger.info("  -> adding new group to group list: {}".format(g))
                 self.Group.Add(g)
            #---  numpy arrays in ChannelOption class
              self.Channel.group_index[idx] = self.Group.GetIndex(g)
              
          if self.raw.info['bads']:
             self.Channel.bads[self.raw.info['bads']]  = True
             
             colour_bads = self.Group.Colour.label2colourRBG("GREY70")
             colour_idx  = self.Group.Colour.colour2index( self.Channel.ColourBads )
             self.Channel.SetBadsColour(colour=colour_bads,index=colour_idx )
             
          for grp in self.Group.labels:
              
              idx  = self.Group.GetIndex(grp)
              cidx = np.array(np.where(self.Channel.group_index == idx),dtype=np.uint32).flatten()
              self.Group.SetChannelIndex(grp,cidx)
              self.Group.SetStatus(grp,self.Group.Status.update)
             #--- set all channel in group
             # self.Channel.colour_index[cidx] = self.Group.GetColourIndex(grp)
             # self.Channel.selected[cidx]     = self.Group.GetSelected(grp)
              
             # self.Channel.scale[cidx]        = self.Group.GetScaling(grp)
             # self.Channel.dcoffset[cidx]     = self.Group.GetDCoffset(grp)
       
      
           #--- calc data  maybe in Shader uniform list?
           #--- min max in numba
             # self.Channel.mean           = self.raw._data.mean
             # self.Channel.dcoffset
             # self.Channel.minmax_global
             # Self.Channel.minmax_window

      
      def update_channel_options(self,**kwargs):
        
          for grp in self.Group.labels:
              if self.Group.GetStatus(grp):
                 picks = self.Group.GetChannelIndex(grp)

                 if self.Group.GetStatusSelected(grp):
                     self.Channel.selected[picks] = self.Group.GetSelected(grp)

                 if self.Group.GetStatusScale(grp):
                    """
                    ck scale mode:
                    division: copy scaling to channels
                    in plot set trafo matrix
                    
                    """
                    self.Channel.scale[picks] = self.Group.GetScaling(grp)
                    self.Channel.scale_mode   = self.Group.GetScaleMode(grp)
                    
                 if self.Group.GetStatusColour(grp):
                    colour_idx = self.Group.GetColourIndex(grp)
                    self.Channel.colour_index[picks] = colour_idx
                    self.Channel.colour[picks]       = self.Group.Colour.index2colour(colour_idx)
                 
                 if self.Group.GetStatusDCoffset(grp):
                    """ ToDO
                        apply/retain DC offset
                        mark/unmark dc orrection
                        choose mode:  global mean ,window mean, hp filter
                        substract in OGL shader? vertex[1] - dcoffset
                    """
                    self.Channel.dcoffset = self.Group.GetDCoffset(grp)
                    #self.Channel.dcoffset_mode = self.Group.GetDCoffsetMode(grp)
                    
                 self.Group.ResetStatus(grp)
          
          #--- Channels settings for raw data
          # calc dcoffset, mean
          # self.dcoffset = np.zeros(n_chan,dtype=np.bool)
          # self.mean = np.zeros(n_chan,dtype=np.float32)
          # self.min_global = np.zeros(n_chan,dtype=np.float32)
          # self.max_global = np.zeros(n_chan,dtype=np.float32)
          # self.min_window = np.zeros(n_chan,dtype=np.float32)
          # self.min_global = np.zeros(n_chan,dtype=np.float32)
          
          
          
      def update(self,**kwargs):
          self._update_from_kwargs(**kwargs)
          
          if "raw" in kwargs.keys():
             self.init_group_info()
          
          self.update_channel_options(**kwargs)
        
          if self.verbose:
             self.GetInfo()
         
    
      def GetInfo(self):
          self.Group.GetInfo()
          self.Channel.GetInfo()
       
      def selected_channel_index(self):
          return np.array( np.where(self.Channel.selected == True),dtype=np.uint32).flatten()
 
      def selected_channel_label(self):
          return self.Channel.label[ self.selected_channel_index() ]
  