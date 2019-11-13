#!/usr/bin/env python3
# -+-coding: utf-8 -+-

"""
"""

#--------------------------------------------
# Authors: Frank Boers <f.boers@fz-juelich.de> 
#
#-------------------------------------------- 
# Date: 12.04.19
#-------------------------------------------- 
# License: BSD (3-clause)
#--------------------------------------------
# Updates
#--------------------------------------------
import numpy as np
from pubsub import pub
import sys,logging
logger=logging.getLogger("jumeg")


import OpenGL

#OpenGL.ERROR_CHECKING = False
#OpenGL.ERROR_LOGGING = False
#--- OGL debug / logging SLOW !!!
#OpenGL.FULL_LOGGING = False  # True

#OpenGL.ERROR_ON_COPY = True

from OpenGL.GL import *
from OpenGL.GLUT import *

import numpy,math
#from PIL import Image

# import tsv.utils.jumeg_transforms as jtr
#from  tsv.ogl.jumeg_tsv_ogl_vbo import JuMEG_TSV_OGL_VBO

from jumeg.gui.tsv.ogl.jumeg_tsv_ogl_plot2d import GLPlotWidget

from jumeg.gui.tsv.plot.jumeg_tsv_plot2d_data_settings import JuMEG_TSV_PLOT2D_DATA_SETTINGS
from jumeg.gui.tsv.plot.jumeg_tsv_plot2d_options       import JuMEG_TSV_PLOT2D_OPTIONS

__version__="2019-09-13-001"

class JuMEG_TSV_OGL_Data(object):
    """
    cls handel dataIO
    subcls:
     settings => group settings
     options  => plot options
    """
    __slots__ = ["_raw","_isInit","_isUpdate","_info","_data_options","_plot_options","_settings"]
    def __init__(self,**kwargs):
        self._raw       = None
        self._isInit   = False
        self._isUpdate = False
        self._settings     = JuMEG_TSV_PLOT2D_DATA_SETTINGS() # JuMEG_TSV_PLOT2D_DATA_INFO()
        self._plot_options = JuMEG_TSV_PLOT2D_OPTIONS()
     
    @property
    def raw(self): return self._raw
   
    @property
    def data(self): return self._raw._data
   
    @property
    def opt(self):
        return self._plot_options
    @property
    def settings(self):
        return self._settings

    @property
    def isInit(self):
        return self._isInit
    @property
    def isUpdate(self):
        return self._isUpdate
    @property
    def timepoints(self):
        return self._raw.times
    @property
    def samples(self):
        return self._raw.n_times
    @property
    def sfreq(self):
        return self._raw.info.get("sfreq",1)
    @property
    def n_channels(self):
        return self._raw._data.shape[0]

    def GetTimeRange(self,tsl0,tsl1):
        return self.raw.times[tsl0:tsl1]

    def GetChannelData(self,picks=None,tsl0=0,tsl1=-1):
        """
        
        :param picks: <None> all channels
        :param tsl0:  0
        :param tsl1: -1
        :return:
        """
        if picks is not None:
          return self.raw._data[picks,tsl0:tsl1]
   
        return self.raw._data[:,tsl0:tsl1]

    def GetChannelNames(self): return self._raw.info['ch_names']

    def GetMin(self,axis=0):
        self._lastmin = self.raw._data.min(axis=axis)
        return self._lastmin
        #--- ToDo write MinMax with numba

    def GetMax(self,axis=0):
        self._lastmax = self.raw._data.max(axis=axis)
        return self._lastmax

    def GetMinMax(self,axis=0):
        return self.GetMin(axis=axis),self.GetMax(axis=axis)

    def _update_from_kwargs(self,**kwargs):
        #self.Info.verbose = kwargs.get("verbose",self.verbose)
        #self.Info.debug   = kwargs.get("debug",self.debug)
        
        raw = kwargs.get("raw")
        if raw:
           self._raw    = raw
           self._isInit = False
    
    def _update(self,**kwargs):
        """
        ToDO
        keep settings start,window,channels ...
        :param kwargs:
        :return:
        """
      #--- defaults
        tw=self.opt.time.window
        tsp=self.opt.time.scroll_step
        nc=self.opt.n_cols
        c2d=self.opt.channels.channels_to_display
        
      #--- update reset for new raw-obj
        self.opt.time.timepoints = self.timepoints
        self.opt.time.counts     = self.timepoints.shape[0]
        self.opt.time.sfreq      = self.sfreq
        self.opt.time.start      = 0.0

        self.opt.channels.idx_start = 0
        self.opt.channels.counts    = self.n_channels
        self.opt.GetInfo()
        
      #--- keep settings
        if self.opt.isInit:
           self.opt.time.window      = tw
           self.opt.time.scroll_step = tsp
           self.opt.n_cols           = nc
           self.opt.channels.channels_to_display=c2d
           
           self.opt.time.window      = kwargs.get("window",tw)
           self.opt.time.scroll_step = kwargs.get("scroll_step",tsp)
           self.opt.n_cols           = kwargs.get("n_cols",nc)
           self.opt.channels.channels_to_display = kwargs.get("channels_to_display",c2d)
           self.opt.isInit = True
           
        else:
           self.opt.time.window      = np.divmod(self.opt.time.timepoints[-1],4.0)[0]
           self.opt.time.scroll_step = np.divmod(self.opt.time.window,2.0)[0]
           self.opt.n_cols           = 1
           self.opt.channels.channels_to_display = 20
           self.opt.n_cols           = 1
           self.opt.isInit           = True
           
        self.opt.GetInfo()
        
        self.settings.update(raw=self.raw)
        self.settings.GetInfo()
        
        self.opt.GetInfo()
       
        self._isInit = True
        
    
    def update(self,**kwargs):
        self._update_from_kwargs(**kwargs)
        if not self.raw:
           return None
        
        if not self._isInit:
           self._update(**kwargs)
        return self._isInit
        


class JuMEG_TSV_OGLPlot2D(object):
    """
     """
    def __init__(self,size=None,n_channels=10,timepoints=20000,sfreq=1017.25,demo_mode=True):
        super().__init__()
       
      #--- data with RAW obj
        self.data = JuMEG_TSV_ICA_OGL_Data()
      #---
        self.GLPlot = GLPlotWidget()

        self._isInit   = False
        self._isOnDraw = False
        self._isUpdate = False
        self.verbose   = False
        self.debug     = False
        self._init_pubsub()
        
    def _init_pubsub(self):
        """ init pubsub call and messages"""
        #--- verbose debug
        pub.subscribe(self.SetVerbose,'MAIN_FRAME.VERBOSE')
        pub.subscribe(self.SetDebug,'MAIN_FRAME.DEBUG')

    def SetVerbose(self,value=False):
        self.verbose = value

    def SetDebug(self,value=False):
        self.debug = value
    
    @property
    def size_in_pixel(self): return  self.GLPlot.size_in_pixel

    @size_in_pixel.setter
    def size_in_pixel(self,v):
        self.GLPlot.size_in_pixel=v
    @property
    def BackGroundColour(self): return self.GLPlot.backgroundcolour
    @BackGroundColour.setter
    def BackGroundColour(self,v):
        self.GLPlot.backgroundcolour=v
    
    @property
    def timepoints(self): return self.GLPlot.signals.timepoints
    @timepoints.setter
    def timepoints(self,v):
        self.GLPlot.signals.timepoints=v

    @property
    def isIinit(self):
        return self._isInit

    @property
    def isUpdate(self):
        return self._isUpdate
    @property
    def isOnDraw(self):
        return self._isOnDraw
      
    def _update_from_kwargs(self,**kwargs):
        self.verbose = kwargs.get("verbose",self.verbose)
        self.debug   = kwargs.get("debug",self.debug)
        self.data.update(**kwargs)
   
    def ShowBads(self,status):
        """
        todo check for deselected channels
        :param status:
        :return:
        """
        if status:
           self.data.settings.Channel.SetSelected(picks=None,status=False)
           picks = self.data.settings.Channel.GetBadsPicks()
           self.data.settings.Channel.SetSelected(picks=picks,status=True)
        else:
           self.data.settings.Channel.SetSelected(picks=None,status=True)

        self.data.opt.channels.do_scroll = True
        #self.update_plot_data()
        #self.display()
        
    def update(self,**kwargs):
        """
         VBO if  time,channel size changed
         cp data to VBO array
        
        :param raw:
        :param n_channels:
        :param n_cols:
        :param kwargs:
        :return:
        """
        #logger.info("  -> start update data :  {}".format(kwargs))
     
       
    
      #--- ck init raw data
        self._isUpdate = False
        self.data_is_update=False
        
        self.data.opt.SetOptions(**kwargs)
       
       #--- ck new RawObj
        if not ( self.data.update(**kwargs) ):
           logger.error("  -> update data false")
           return False
       
       #--- ck for update
        if kwargs.get("raw",False):
           self.update_plot_data(init=True,**kwargs)
        if kwargs.get("ica",False):
           self.update_plot_data(init_ica=True,**kwargs)
        else:
          
          # logger.info(" ---> UPATE => update_plot_data")
           self.update_plot_data(**kwargs)
           
          # logger.info(" --->DONE UPATE => update_plot_data")
           
      
        self.data_is_update=True
        self._isUpdate=True
          
        #logger.info("  -> done update data")
          
        return self.data_is_update

    def ToggleBadsFromPosition(self,pos):
        """
        get De/selected channel from x,y pos, mouseclick
        toggle Bads selection in channles  True/False
        update GLPlot
        assuming: <first channel> on TOP LEFT and <last channel> is DOWN RIGHT corner
        e.g.:
        cols=4 rows=2
        | 1,2,3,4|
        | 5,6,7,8|
        
        :param x:
        :param y:
        :return: True/False
        if True: call in <parent GLCanvas> refresh() e.g. OGL swap buffer
        """
        m = self.GLPlot.VieportMatrix
        yidx  = np.where(m[:,1] < pos[1])[0]
        xidx  = np.where( np.logical_and( m[yidx,0] < pos[0], pos[0] < m[yidx,0] + m[0,2] ) )[0]
        chidx = self.GLPlot.signals.picks[xidx[-1]]  # MEG 001 is on TOP Left
     
        self.data.settings.ToggleBads(chidx)
      #--- ToDo refresch only subplot with VPM idx
        self.GLPlot.signals.colours = self.data.settings.Channel.colour
        self.GLPlot.plot()
        return True

  
    def update_plot_data(self,init=False,**kwargs):
        tsl0,tsl1 = self.data.opt.time.index_range()
        self.GLPlot.n_cols = self.data.opt.n_cols
        
        if "settings" in kwargs:
           self.data.settings.Group = kwargs["settings"] # overwrite Group obj
           self.data.settings.update_channel_options()
           
           #if self.debug:
           #   self.data.settings.GetInfo()
           
           self.data.opt.channels.do_scroll = True
           
        if init:
            #--- cp data pointer
            self.data.opt.time.do_scroll     = True
            self.data.opt.channels.do_scroll = True
            self.GLPlot.signals.labels       = self.data.settings.Channel.labels # GetChannelNames()
          
        if self.data.opt.time.do_scroll:
            self.GLPlot.signals.data       = self.data.GetChannelData(tsl0=tsl0,tsl1=tsl1)
            self.GLPlot.signals.timepoints = self.data.GetTimeRange(tsl0,tsl1)
            self.GLPlot.signals.dcoffset   = self.data.settings.Channel.GetDCoffset(raw=self.data.raw,tsls=self.data.opt.time.index_range())
            
        #--- toDo data vbo in cls
        if self.data.opt.channels.do_scroll:
            picks_selected = self.data.settings.Channel.GetSelected()
            
           #--- calc channel range  0 ,-1
            ch_start,ch_end_range = self.data.opt.channels.index_range(n_counts = picks_selected.shape[0] +1)
            
           #--- get selected picks
            self.GLPlot.signals.picks = picks_selected[ch_start:ch_end_range]
          
       #--- set the rest
   
        self.GLPlot.signals.scale = self.data.settings.Channel.GetMinMaxScale(raw=self.data.raw,tsls=self.data.opt.time.index_range(),
                                                                              picks= self.GLPlot.signals.picks, div=self.GLPlot.Grid.ydiv)
        
        #,picks=self.GLPlot.signals.picks)
        self.GLPlot.signals.colours  = self.data.settings.Channel.colour
     

    def display(self):
        #self.data.info.GetInfo()
        if self.data.isInit:
          # self.update_plot_data()
           self.GLPlot.plot()
    '''
    ToDo move to ogl plt
    '''
    def initGL(self):
        """
        :param size:
        :return:
        """
        #logger.debug("OGL -> start initGL window")
        
        glutInit(sys.argv)
        
        self.clear_display(splash=True)
        
        #logger.debug("done OGL -> initGL")
        return True
    
    
    def clear_display(self,splash=False):
        """
        :param size:
        :return:
        """
        #logger.debug("OGL -> start clear display")
        
        #-- ToDo ck for use display list
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        
        #logger.debug("clear display size: {}".format(self.size))
        
        glViewport(0,0,self.GLPlot.width,self.GLPlot.height)
        
        #logger.info(" display size: {}".format(self.GLPlot.size_in_pixel))
        
        r,g,b,a = self.BackGroundColour
        glClearColor(r,g,b,a)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glEnable(GL_LINE_SMOOTH)
        
        #glHint(GL_LINE_SMOOTH,GL_FASTEST)
        glDisable(GL_SCISSOR_TEST)
        glLineWidth(1)
      
        if splash or not self.Data.isInit:
           self.GLPlot.draw_splash_screen()
        return True
    
