import wx
import sys

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

from jumeg.tsvgl.plot2d.jumeg_tsv_plot2d_ogl_axis  import JuMEG_TSV_PLOT2D_OGL
# from jumeg.tsvgl.plot2d.jumeg_tsv_plot2d_ogl_axis_viewport_ok  import JuMEG_TSV_PLOT2D_OGL

class JuMEG_TSV_PLOT2D_WX_GL_CANVAS_BASE(glcanvas.GLCanvas):
    def __init__(self, parent):

        attribList = (glcanvas.WX_GL_RGBA,          # RGBA
                      glcanvas.WX_GL_DOUBLEBUFFER,  # Double Buffered
                      glcanvas.WX_GL_DEPTH_SIZE,16) # 24 bit
                      
        glcanvas.GLCanvas.__init__(self, parent, -1,attribList=attribList,style = wx.DEFAULT_FRAME_STYLE)
       
        self.is_initGL   = False
        self.is_on_draw  = False
        self.is_on_paint = False
        self.is_on_size  = False

             
        self.init = False
        self.context = glcanvas.GLContext(self)
        
        # initial mouse position
        self.lastx = self.x = 30
        self.lasty = self.y = 30
        self.size = None
        self.Bind(wx.EVT_ERASE_BACKGROUND, self.OnEraseBackground)
        self.Bind(wx.EVT_SIZE, self.OnSize)
        self.Bind(wx.EVT_PAINT, self.OnPaint)
        #self.Bind(wx.EVT_LEFT_DOWN, self.OnMouseDown)
        #self.Bind(wx.EVT_LEFT_UP, self.OnMouseUp)
        #self.Bind(wx.EVT_MOTION, self.OnMouseMotion)
        
        self.Bind(wx.EVT_KEY_DOWN,  self.OnKeyDown)
        self.Bind(wx.EVT_CHAR,  self.OnKeyDown)

    def OnEraseBackground(self, event):
        pass # Do nothing, to avoid flashing on MSW.

    def OnSize(self, event):
        wx.CallAfter(self.DoSetViewport)
        event.Skip()

    def DoSetViewport(self):
        size = self.size = self.GetClientSize()
        self.SetCurrent(self.context)
        glViewport(0, 0, size.width, size.height)
      
    def OnPaint(self, event):
        if self.is_on_paint:
           return     
        else: 
           self.is_on_paint = True 
        
        dc = wx.PaintDC(self)
        self.SetCurrent(self.context)
        
        if self.is_initGL:
           self.OnDraw( size_mm=dc.GetSizeMM() )            
        else:   
           self.InitGL()           
           
        self.is_on_paint = False
        
    
    def initGL(self):
        print" ToDo dummy def initGL overwrite"
        
    def OnDraw(self):
        print" ToDo dummy def OnDraw overwrite"
        
    def OnKeyDown(self, e):

        key = e.GetKeyCode()
      #---escape to quit
        if key == wx.WXK_ESCAPE:
           self.click_on_exit(e)



#opt
#info
#update

class JuMEG_TSV_PLOT2D_WX_GL_CANVAS(JuMEG_TSV_PLOT2D_WX_GL_CANVAS_BASE):
      def __init__(self, parent=None, *args, **kwargs):
          super(JuMEG_TSV_PLOT2D_WX_GL_CANVAS,self).__init__(parent) #, *args, **kwargs)
          
          self.plot2d = None
          self.InitGL()
      
  
      def __get_plot_option(self):
          return self.plot2d.opt
      def __set_plot_option(self,opt):
          self.plot2d.opt = opt
      option = property(__get_plot_option,__set_plot_option)    
     
      def __get_plot_info(self):
          return self.plot2d.info
      def __set_plot_info(self,info):
          self.plot2d.info = info
          self.plot2d.data_channel_selection_is_update = False
         #--- update e.g. channel selction
          self.plot2d.info.update()        
      info = property(__get_plot_info,__set_plot_info)    
            

      def OnKeyDown(self, evt):
          action = None
          
          if not self.is_initGL :
             evt.skip()  #---escape to quit
             
          key = evt.GetKeyCode()      
                  
         #--- scroll time fast by window
          if (wx.GetKeyState(wx.WXK_CONTROL) == True):
             
             if key == (wx.WXK_LEFT):
                #print"FAST REW"               
                action = "FAST_REWIND"      
             elif key == (wx.WXK_RIGHT):
                action = "FAST_FORWARD" 
             elif key == (wx.WXK_HOME):
                action ="START"      
             elif key == (wx.WXK_END):
                action = "END" 
         #----
          elif key == (wx.WXK_F11): 
               action = "TIME_DISPLAY_ALL" 
          elif key ==(wx.WXK_F12): 
               action = "CHANNELS_DISPLAY_ALL" 
                
         #--- scroll time by scroll step 
          elif key == wx.WXK_LEFT:
               #print"LEFT"
               action = "REWIND" 
          elif key == wx.WXK_RIGHT:
               #print "RIGHT"
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
             self.plot2d.opt.action(action)
             self.update()  
          
          else:
             evt.Skip()

      def InitGL(self):
          
          self.is_initGL = False
         
          dc = wx.PaintDC(self)
          self.SetCurrent(self.context)
          size = self.size = self.GetClientSize()   
          self.plot2d      = JuMEG_TSV_PLOT2D_OGL(size=size)
          self.is_initGL   = self.plot2d.initGL()
                   
          return self.is_initGL

      def OnDraw(self,size_mm=None):

          if self.is_on_draw:
             return
             
          self.is_on_draw  = True

          if self.is_initGL:
             self.SetCurrent()
          else:
             self.InitGL()


          self.plot2d.size_in_pixel = self.GetClientSize()
          self.plot2d.size_in_mm    = size_mm

          print " ---> "+self.__class__.__name__ +" OnDraw -> plot size"
          print self.plot2d.size_in_pixel 
          print self.plot2d.size_in_mm 

          self.plot2d.display()

          self.SwapBuffers()
          self.is_on_draw  = False
      
      def update(self,raw=None,channels2plot=None,cols=None): #,do_scroll_channels=True,do_scroll_time=True):
       
          if self.is_initGL :
             self.SetCurrent()
             if raw :
                self.plot2d.init_raw_data(raw=raw,channels2plot=channels2plot,cols=cols) 
                
             elif self.plot2d.data_is_init: 
                  self.plot2d.update_data() #do_scroll_channels=True,do_scroll_time=True,)
                  
                  #self.plot_axis.range_max = self.plot2d.timepoints[-1]
                  #self.plot_axis.range_min = self.plot2d.timepoints[0]
       
             if self.plot2d.opt.do_scroll:
                self.Refresh()
               # self.plot_axis.range_max = self.plot2d.timepoints[-1]
                #self.plot_axis.range_min = self.plot2d.timepoints[0]
                
                #self.plot_axis.Refresh()
                    
          else: self.InitGL()
          