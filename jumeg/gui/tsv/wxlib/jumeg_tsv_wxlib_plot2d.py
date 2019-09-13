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

try:
    from agw import rulerctrl as RC
except ImportError: # if it's not there locally, try the wxPython lib.
    import wx.lib.agw.rulerctrl as RC


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
        
        self._isInit    = False
        self._isInitGL  = False
        self._isOnDraw  = False
        self._isOnPaint = False
        self._isOnSize  = False
        
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
    
    def OnKeyDown(self,e):
        """   press <ESC> to exit pgr """
        key = e.GetKeyCode()
        #---escape to quit
        if key == wx.WXK_ESCAPE:
           pub.sendMessage("MAIN_FRAME.CLICK_ON_CLOSE")
           #self.click_on_exit(e)
    
    def OnMouseLeftDown(self,evt):
        pass
 
    def OnMouseUp(self,evt):
        pass

    def OnMouseWheel(selfself,evt):
        pass

    def OnMouseRightDown(self,evt):
        pass
    def OnMouseRightUp(self,evt):
        pass
    def OnMouseMotion(self,evt):
        pass


class JuMEG_TSV_wxCanvas2D(JuMEG_TSV_wxGLCanvasBase):
    """
    """
    def __init__(self,parent=None,*args,**kwargs):
        super().__init__(parent)  #, *args, **kwargs)
        self._glplot2d = None
        #self.InitGL()
        self.verbose    = False
        self.debug      = False
      
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
        if not self.isInitGL:
           evt.skip()  #---escape to quit
        
        key = evt.GetKeyCode()
        
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
            evt.Skip()
    
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
            w,h = size = self.GetClientSize()
            self.plot.GLPlot.size_in_pixel = [w,h]
            self.InitGL()
        else:
           self.SetCurrent(self.context)
        
        self._update_from_kwargs(**kwargs)
        
        self.plot.update(**kwargs)
       
        #if self.plot.data.opt.do_scroll_channels:
        
        if self.plot.data.opt.do_scroll:
           self.Refresh()
           if self.plot.data.opt.time.do_scroll:
              self.GetParent().OnScroll(tmin=self.plot.timepoints[0],tmax=self.plot.timepoints[-1])
    
    def OnMouseRightDown(self,evt):
        try: # self.CaptureMouse() !!! finaly release
           if self.plot.ToggleBadsFromPosition(evt.GetPosition()):
              self.Refresh()
             #--- send msg update BADS
              pub.sendMessage("MAIN_FRAME.UPDATE_BADS",value="CHANGED")
        except:
            logger.exception("---> ERROR in Mouse Right Down")
        
        evt.Skip()
        

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
        self.verbose      = False
        self.debug        = False
        self._wxPlot2D    = None
        self._wxTimeScale = None
   
        self._update_from_kwargs(**kwargs)
        self._wx_init(**kwargs)
        self._ApplyLayout()
    
    @property
    def plot(self): return self._wxPlot2D
    @property
    def TimeScale(self): return self._wxTimeScale
    
    def OnScroll(self,tmin=None,tmax=None):
        #logger.info("  -> scroll t: {} {}".format(tmin,tmax))
        self._wxTimeScale.SetRange(tmin,tmax)
        
    def GetPlotOptions(self):
        return self.plot.plot.data.opt
    
    def GetGroupSettings(self):
        return self.plot.plot.data.settings.Group
    
    def _update_from_kwargs(self,**kwargs):
        self.SetName(kwargs.get("name","PLOT_2D"))
        self.SetBackgroundColour(kwargs.get("bg",wx.BLUE))
        self.verbose = kwargs.get("verbose",self.verbose)
        self.debug   = kwargs.get("debug",self.debug)
    
    def _wx_init(self,**kwargs):
        self._wxTimeScale = RC.RulerCtrl(self,-1,orient=wx.HORIZONTAL,style=wx.SUNKEN_BORDER)
        self._wxTimeScale.TickMinor(tick=False)
        self._wxTimeScale.SetTimeFormat(3)
        self._wxPlot2D    = JuMEG_TSV_wxCanvas2D(self,**kwargs)

    def update(self,raw=None,**kwargs):
        """

        :param raw:
        :param n_channels:
        :param n_cols:
        :return:
        """
        if self.plot:
           try:
               self.plot.update(raw=raw,**kwargs) # if raw reset data
               
           except:
               logger.exception("Error in updating plot")
               
        #---
   
    def ClickOnCtrls(self,evt):
        """ pass to parent event handlers """
        evt.Skip()


    def _ApplyLayout(self):
        """ default Layout Framework """
        vbox = wx.BoxSizer(wx.VERTICAL)
        if self.plot:
           vbox.Add(self.plot,1,wx.ALIGN_LEFT | wx.EXPAND | wx.ALL,1)
        vbox.Add(self.TimeScale,0,wx.ALIGN_LEFT | wx.EXPAND | wx.ALL,1)
        self.SetSizer(vbox)
        self.Fit()
        self.SetAutoLayout(1)
        self.GetParent().Layout()
        
    
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