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

import wx

import wx.lib.dragscroller #demos
from pubsub import pub

from jumeg.gui.tsv.wxutils.jumeg_tsv_wxutils import RULERS as TimeScaler

import sys
import logging
logger = logging.getLogger("jumeg")

try:
    from wx import glcanvas
    haveGLCanvas = True
except ImportError:
    haveGLCanvas = False

try:
    # The Python OpenGL package can be found at
    # http://PyOpenGL.sourceforge.net/
    from OpenGL.GL import *
    from OpenGL.GLUT import *
    haveOpenGL = True
except ImportError:
    haveOpenGL = False
    logger.exception("---> can not import OpenGL\n --> The Python OpenGL package can be found at\n -->http://PyOpenGL.sourceforge.net/")
    sys.exit()
    
from jumeg.gui.tsv.plot.jumeg_tsv_plot_plot2d import JuMEG_TSV_OGLPlot2D

__version__="2019-04-11-001"

class JuMEG_TSV_wxGLCanvasBase(glcanvas.GLCanvas):
    def __init__(self,parent):
        
        attribList = (glcanvas.WX_GL_RGBA,  # RGBA
                      glcanvas.WX_GL_DOUBLEBUFFER,  # Double Buffered
                      glcanvas.WX_GL_DEPTH_SIZE,16)  # 24 bit
        
        super().__init__(parent,-1,attribList=attribList,style=wx.DEFAULT_FRAME_STYLE)
        
        self.verbose    = False
        self.debug      = False
      
        self._isInit    = False
        self._isInitGL  = False
        self._isOnDraw  = False
        self._isOnPaint = False
        self._isOnSize  = False
        
        self._init_pubsub()
        
        self.context = glcanvas.GLContext(self)
      # Create graphics context from it
        #gc = wx.GraphicsContext.Create(self.context)

        
        self.SetMinSize((10,10))
        
        # initial mouse position
        self.lastx = self.x = 30
        self.lasty = self.y = 30
        self.size = None
        #self.Bind(wx.EVT_ERASE_BACKGROUND,self.OnEraseBackground)
        
        self.Bind(wx.EVT_SIZE,self.OnReSize)
        self.Bind(wx.EVT_PAINT,self.OnPaint)
        self.Bind(wx.EVT_KEY_DOWN,self.OnKeyDown)
        self.Bind(wx.EVT_CHAR,self.OnKeyDown)
       
       # self.Bind(wx.EVT_LEFT_DOWN,  self.OnMouseLeftDown)
       # self.Bind(wx.EVT_LEFT_UP,    self.OnMouseUp)
       # self.Bind(wx.EVT_MOUSEWHEEL, self.OnMouseWheel)
        self.Bind(wx.EVT_RIGHT_DOWN, self.OnMouseRightDown)
       # self.Bind(wx.EVT_RIGHT_UP,   self.OnMouseRightUp)
       # self.Bind(wx.EVT_MOTION,     self.OnMouseMotion)
    
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
    def isInitGL(self):   return self._isInitGL
    @property
    def isOnDraw(self):   return self._isOnDraw
    @property
    def isIOnPaint(self): return self._isOnPaint
    @property
    def isOnSize(self):   return self._isOnSize
    
    def OnEraseBackground(self,event):
        pass  # Do nothing, to avoid flashing on MSW.
    
    def OnReSize(self,event):
        pass
        # wx.CallAfter(self.DoSetViewport)
        #event.Skip()
    
    def DoSetViewport(self):
        if self.isInitGL:
           size = self.size = self.GetClientSize()
           self.SetCurrent(self.context)
           glViewport(0,0,size.width,size.height)
    
    def OnPaint(self,event):
        if self.isIOnPaint:
           return
        else:
           self._isOnPaint = True
         
        if self.isInitGL:
           wx.CallAfter(self.OnDraw) #(size_mm=dc.GetSizeMM())
        else:
           self.InitGL()
        
        self._isOnPaint = False
    
    def InitGL(self):
        """ initGL """
        pass
        
    def OnDraw(self):
        """ OnDraw do your drawing,paintinf,plotting here"""
    
    def OnKeyDown(self,evt):
        """   press <ESC> to exit pgr """
        #key = e.GetKeyCode()
        #---escape to quit
       # if key == wx.WXK_ESCAPE:
       #    pub.sendMessage("MAIN_FRAME.CLICK_ON_CLOSE")
           #self.click_on_exit(e)
        evt.Skip()
    
    def OnMouseLeftDown(self,evt):
        evt.Skip()
 
    def OnMouseUp(self,evt):
        evt.Skip()

    def OnMouseWheel(selfself,evt):
        evt.Skip()

    def OnMouseRightDown(self,evt):
        evt.Skip()
    def OnMouseRightUp(self,evt):
        evt.Skip()
    def OnMouseMotion(self,evt):
        evt.Skip()


class JuMEG_TSV_wxCanvas2D(JuMEG_TSV_wxGLCanvasBase):
    """
    """
    def __init__(self,parent=None,*args,**kwargs):
        super().__init__(parent)  #, *args, **kwargs)
        self._glplot2d = None
        #self.InitGL()
       
        self.duration   = 1.0
        self.start      = 0.0
        #self.n_channels = 10
        #self.n_cols     = 1
        self._update_from_kwargs(**kwargs)
    
    
        
        
    @property
    def plot(self): return self._glplot2d
    '''
    @property
    def plot_option(self):
        return self.plot.opt
    @plot_option.setter
    def plot_option(self,opt):
        self.plot.opt = opt

    @property
    def plot_info(self):
        return self.plot.info

    @plot_info.setter
    def plot_info(self,info):
        self.plot.info = info
   #---ToDo in subcls  glplot2d update
       # self.plot.data_channel_selection_is_update = False
       # self.plot2d.info.update()
    '''
    def _update_from_kwargs(self,**kwargs):
        #pass
        self.verbose    = kwargs.get("verbose",self.verbose)
        self.debug      = kwargs.get("debug",self.debug)
      
        self.duration   = kwargs.get("duration",self.duration)
        self.start      = kwargs.get("start",self.start)
        #self.n_channels = kwargs.get("n_channels",self.n_channels)
        #self.n_cols     = kwargs.get("n_cols",self.n_cols)
        
    def OnKeyDown(self,evt):
        action = None
        type   = None
        key = evt.GetKeyCode()
        
        if not self.isInitGL:
           pub.sendMessage("EVENT_KEY_DOWN",event=evt)
           return
     
        #key_list    = [wx.WXK_LEFT,wx.WXK_RIGHT,wx.WXK_HOME,wx.WXK_END,wx.WXK_UP,wx.WXK_DOWN,wx.WXK_PAGEUP,wx.WXK_PAGEDOWN]
        #action_ctrl = ["FAST_REWIND","FAST_FORWARD","START","END"]
   
        #if key in key_list:
        #   evt.Skip()
        #   return
           
        #--- scroll time fast by window
        if (wx.GetKeyState(wx.WXK_CONTROL) == True):
            
            if key == (wx.WXK_LEFT):
                action = "FAST_REWIND"
            elif key == (wx.WXK_RIGHT):
                action = "FAST_FORWARD"
            elif key == (wx.WXK_HOME):
                action = "START"
            elif key == (wx.WXK_END):
                action = "END"
       #----
        elif key == (wx.WXK_F11):
            action = "TIME_DISPLAY_ALL"
        elif key == (wx.WXK_F12):
            action = "CHANNELS_DISPLAY_ALL"
            
       #--- scroll time by scroll step
        elif key == wx.WXK_LEFT:
            action = "REWIND"
        elif key == wx.WXK_RIGHT:
            action = "FORWARD"
        
        #--- scroll channels
        elif key == wx.WXK_UP:
            action = "UP"
        elif key == wx.WXK_DOWN:
            action = "DOWN"
        elif key == wx.WXK_PAGEUP:
            action = "PAGEUP"
        elif key == wx.WXK_PAGEDOWN:
            action = "PAGEDOWN"
        elif key == wx.WXK_HOME:
            action = "TOP"
        elif key == wx.WXK_END:
            action = "BOTTOM"
      #---
        if action:
           self.plot.data.opt.action(action)
           self.update()
        
        else:
            pub.sendMessage("EVENT_KEY_DOWN",event=evt)
            #evt.Skip()
    
    def InitGL(self):
        self._isInitGL = False
        if not self.IsShown(): return
        self.SetCurrent(self.context)
      #---
        self._glplot2d = JuMEG_TSV_OGLPlot2D()
        self.plot.size_in_pixel = size = self.GetClientSize()
        self._isInitGL = self.plot.initGL()
        return self.isInitGL
 
    def OnDraw(self,size_mm=None):
        
        if self.isOnDraw:
           return
        
        self._isOnDraw = True
        
        if not self.isInitGL: return
        self.SetCurrent(self.context)
        
        w,h = size = self.GetClientSize()
        
        self.plot.size_in_pixel = [w,h] #np.array([ w,h ],dtype0np.float32)
        self.plot.display()
       
        self.SwapBuffers()
        self._isOnDraw = False
    
    def update(self,**kwargs):
        """
        
        :param raw:
        :param n_channels:
        :param n_cols:
        :return:
        """
        
    
        if not self.isInitGL:
           self.InitGL()
        else:
           self.SetCurrent(self.context)
        
        self._update_from_kwargs(**kwargs)
        
        self.plot.update(**kwargs)
       
        if self.plot.data.opt.do_scroll:
           if self.plot.data.opt.time.do_scroll:
              self.GetParent().OnScroll(start=self.plot.timepoints[0],end=self.plot.timepoints[-1]) #,n_cols=self.plot.data.opt.n_cols)
        self.Refresh()
        
    def OnMouseRightDown(self,evt):
        try: # self.CaptureMouse() !!! finaly release
           if self.plot.ToggleBadsFromPosition(evt.GetPosition()):
              self.Refresh()
             #--- send msg update BADS
              pub.sendMessage("MAIN_FRAME.UPDATE_BADS",status="CHANGED")
        except:
            logger.exception("---> ERROR in Mouse Right Down")
        
        evt.Skip()
        
    def GetNumberOfCols(self):
        return self.plot.data.opt.n_cols
 
class JuMEG_TSV_wxPlot2D(wx.Panel):
    """
       CLS container:
         - GLCanvas 2DPlot
         - time scale (ruler)
         - xyz stuff
    """
 
    def __init__(self,parent=None,*args,**kwargs):
        super().__init__(parent)  #, *args, **kwargs)
        self._init(**kwargs)
    
    def _init(self,**kwargs):
        self.verbose        = False
        self.debug          = False
        self._wxPlot2D      = None
        self._wxTimeScaler  = None
        
        self._update_from_kwargs(**kwargs)
        self._wx_init(**kwargs)
        self._ApplyLayout()
    
    @property
    def plot(self): return self._wxPlot2D
    
    @property
    def TimeScaler(self): return self._wxTimeScaler
    
    @property
    def n_cols(self): return self._wxTimeScaler.n_cols
    @n_cols.setter
    def n_cols(self,v):
        if self.n_cols != v:
           self._wxTimeScaler.update(n_cols=v)
    
    def ShowBads(self,status):
        self.plot.plot.ShowBads(status)
        self.plot.update()
        
    def GetPlotOptions(self):
        return self.plot.plot.data.opt
    
    def GetGroupSettings(self):
        return self.plot.plot.data.settings.Group
    
    def _update_from_kwargs(self,**kwargs):
        self.SetName(kwargs.get("name","PLOT_2D"))
        self.SetBackgroundColour(kwargs.get("bg",wx.BLUE))
        self.verbose = kwargs.get("verbose",self.verbose)
        self.debug   = kwargs.get("debug",self.debug)

    def OnScroll(self,**kwargs):
        """
        
        :param tmin:
        :param tmax:
        :return:
        """
        #logger.info("  -> scroll t: {} {}".format(tmin,tmax))
        self._wxTimeScaler.UpdateRange(**kwargs)

    def _wx_init(self,**kwargs):
        self._wxTimeScaler = TimeScaler(self)
        self._wxPlot2D     = JuMEG_TSV_wxCanvas2D(self,**kwargs)
        self.Bind(wx.EVT_CHAR,self.ClickOnKeyDown)
    
    def ClickOnKeyDown(self,evt):
        evt.Skip()
    
    def showbads(self,status):
        self.plot.showbads(status)
    
    def update(self,raw=None,**kwargs):
        """

        :param raw:
        :param n_channels:
        :param n_cols:
        :return:
        """
       # logger.info("PARAM: {}".format(kwargs))
        
        if self.plot:
           try:
               self.plot.update(raw=raw,**kwargs) # if raw reset data
               self.n_cols = self.plot.GetNumberOfCols() # update TimeScalers via n_cols property
           except:
               logger.exception("Error in update plot => kwargs:\n {}\n".format(kwargs))
               
    def ClickOnCtrls(self,evt):
        """ pass to parent event handlers """
        evt.Skip()

    def _ApplyLayout(self):
        """ default Layout Framework """
        vbox = wx.BoxSizer(wx.VERTICAL)
        if self.plot:
           vbox.Add(self.plot,1,wx.ALIGN_LEFT | wx.EXPAND | wx.ALL,1)
        vbox.Add(self._wxTimeScaler ,0,wx.ALIGN_LEFT | wx.EXPAND | wx.ALL,1)
      
        self.SetSizer(vbox)
        self.Fit()
        self.SetAutoLayout(1)
        self.GetParent().Layout()
        
    def _init_pubsub(self):
        """ init pubsub call and messages"""
      #--- verbose debug
        pub.subscribe(self.SetVerbose,'MAIN_FRAME.VERBOSE')
        pub.subscribe(self.SetDebug,'MAIN_FRAME.DEBUG')
        
    def SetVerboss(self,value=False):
        self.verbose=value
    def SetDebug(self,value=False):
        self.debug=value
'''
 def pixel_size2mm(self,w=1,h=1):
        (x_pix,y_pix) = wx.GetDisplaySize()
        (x_mm,y_mm )  = wx.GetDisplaySizeMM()
        return  x_mm/x_pix * w, y_mm/y_pix * h
   
   
    def mm_size2pixel(self,w=1.0,h=1.0):
        
        (x_pix,y_pix) = wx.GetDisplaySize()
        (x_mm,y_mm )  = wx.GetDisplaySizeMM()
        
        logger.info("---> diplay size pix: {} {}".format(x_pix,y_pix))
        logger.info("---> diplay size mm : {} {}".format(x_mm,y_mm))
       
        print("---> diplay size pix: {} {}".format(x_pix,y_pix))
        print("---> diplay size mm : {} {}".format(x_mm,y_mm))

        return  x_pix/x_mm *w, y_pix/y_mm * h
        # return y_pix/y_mm * mm


'''