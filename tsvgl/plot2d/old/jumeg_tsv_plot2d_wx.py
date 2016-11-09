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

from jumeg.tsvgl.plot2d.jumeg_tsv_plot2d_ogl import JuMEG_TSV_PLOT2D_OGL
#from jumeg.tsv.test.axis01 import JuMEG_TSV_AXIS


attribList = (glcanvas.WX_GL_RGBA,          # RGBA
              glcanvas.WX_GL_DOUBLEBUFFER,  # Double Buffered
              glcanvas.WX_GL_DEPTH_SIZE,24) # 24 bit
              
style = wx.DEFAULT_FRAME_STYLE # | wx.NO_FULL_REPAINT_ON_RESIZE


'''
!!! working needs glutInit

import OpenGL
OpenGL.ERROR_ON_COPY = True
from OpenGL.GLUT import *
from OpenGL.GLU import *
from OpenGL.GL import *
glutInit(sys.argv)
glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
glutInitWindowSize(640, 480)
glutInitWindowPosition(0, 0)
 window = glutCreateWindow("Jeff Molofee's GL Code Tutorial ... NeHe '99")
glShadeModel(GL_SMOOTH)
pid=glCreateProgram()

'''


class JuMEG_TSV_PLOT2D_CanvasBase(glcanvas.GLCanvas):

    def __init__(self, parent):
        glcanvas.GLCanvas.__init__(self, parent, -1,attribList=attribList,style=style)
        

        
        self.is_initGL   = False
        self.is_on_draw  = False
        self.is_on_paint = False
        self.is_on_size  = False

        self.size    = None
        self.context = glcanvas.GLContext(self)
             
        self.plot2d    = None
        self.count     = 0
        self.rezise_cnt= 0

        self.LeftDown     = False
 
 
        # initial mouse position
        self.lastx = self.x = 30
        self.lasty = self.y = 30
       
        self.Bind(wx.EVT_ERASE_BACKGROUND, self.OnEraseBackground)
        self.Bind(wx.EVT_SIZE,      self.OnSize)
        self.Bind(wx.EVT_PAINT,     self.OnPaint)
         # self.Bind(wx.EVT_LEFT_DOWN, self.OnMouseLeftDown)
         # self.Bind(wx.EVT_LEFT_UP,   self.OnMouseLeftUp)
          #self.Bind(wx.EVT_MOTION,    self.OnMouseMotion)
        self.Bind(wx.EVT_KEY_DOWN,  self.OnKeyDown)
        self.Bind(wx.EVT_CHAR,  self.OnKeyDown)

    def OnEraseBackground(self, event):
        pass # Do nothing, to avoid flashing on MSW.


    def OnSize(self, event):
        """Process the resize event."""

        if self.is_on_draw:
           return
                    
        self.is_on_size = True

        if self.LeftDown :
            # print"==========>SKIP REFRESH UPDATE"
           self.is_on_size  = False
            # event.Skip()
           return

        if self.GetContext():
            # Make sure the frame is shown before calling SetCurrent.
           self.Show()
           self.SetCurrent()
           self.Refresh()

           event.Skip()
        self.is_on_size  = False

    def OnPaint(self, evt):
        if self.is_on_draw:
           return
        
        if not self.is_initGL:
           self.is_initGL = self.InitGL()
       
        if self.LeftDown :
           # print"-----> ONDRAW  Mouse DOWN"
             #evt.Skip()
           self.is_on_paint = False
           return

            
        if not self.is_initGL:
           return False
           
        self.is_on_paint = True
         # dc = wx.PaintDC(self)
        dc = wx.ClientDC(self) # no border
        self.OnDraw( size_mm=dc.GetSizeMM() )
          # evt.Skip()
        self.is_on_paint = False

     # def OnMouseLeftDown(self, evt):
     #     self.LeftDown=True
          #self.CaptureMouse()
          #self.x, self.y = self.lastx, self.lasty = evt.GetPosition()
     #     evt.Skip()

     # def OnMouseLeftUp(self, evt):
     #     self.LeftDown=False
     #     evt.Skip()
          #self.Refresh(True)
          #print"MUP"
          #self.ReleaseMouse()



     # def OnMouseMotion(self, evt):
        #  if evt.Dragging() and evt.LeftIsDown():
        #     self.lastx, self.lasty = self.x, self.y
        #     self.x, self.y = evt.GetPosition()
        #     print "on mouse drag LD"
            # self.Refresh(False)

    def OnKeyDown(self, e):

        key = e.GetKeyCode()
          # print"GLCanvas EVT OnKeyDown: " + str(key)
        #---escape to quit
        if key == wx.WXK_ESCAPE:
           self.click_on_exit(e)


class JuMEG_TSV_PLOT2D_WX(wx.Panel):   #JuMEG_TSV_PLOT2D_CanvasBase):
      def __init__(self, parent=None, *args, **kwargs):
          super(JuMEG_TSV_PLOT2D_WX,self).__init__(parent, *args, **kwargs)
          
          
           
          self.is_initGL   = False
          self.is_on_draw  = False
          self.is_on_paint = False
          self.is_on_size  = False

          self.size    = None
                    
          self.plot2d    = None
          self.count     = 0
          self.rezise_cnt= 0

          self.LeftDown     = False
 
          gl_canvas_attribs = [wx.glcanvas.WX_GL_RGBA,
                               wx.glcanvas.WX_GL_DOUBLEBUFFER,
                               wx.glcanvas.WX_GL_DEPTH_SIZE, 16]
          self.gl_canvas = wx.glcanvas.GLCanvasWithContext(self, attribList = gl_canvas_attribs)
          self.gl_canvas.SetMinSize((150,150))
          self.gl_canvas.Bind(wx.EVT_PAINT, self.OnPaint)
    
          self.Bind(wx.EVT_SIZE,      self.OnSize)
          self.Bind(wx.EVT_PAINT,     self.OnPaint)
          self.Bind(wx.EVT_KEY_DOWN,  self.OnKeyDown)
          self.Bind(wx.EVT_CHAR,      self.OnKeyDown)

      def OnSize(self, event):
          """Process the resize event."""

          if self.is_on_draw:
             return
                    
          self.is_on_size = True

          if self.LeftDown :
              # print"==========>SKIP REFRESH UPDATE"
             self.is_on_size  = False
            # event.Skip()
             return

          if self.gl_canvas.GetContext():
            # Make sure the frame is shown before calling SetCurrent.
             self.Show()
             self.gl_canvas.SetCurrent()
             self.Refresh()
  
             event.Skip()
          self.is_on_size  = False

      def OnPaint(self, evt):
          if self.is_on_draw:
             return
        
          if not self.is_initGL:
             self.is_initGL = self.InitGL()
       
          if self.LeftDown :
            # print"-----> ONDRAW  Mouse DOWN"
             #evt.Skip()
             self.is_on_paint = False
             return

            
          if not self.is_initGL:
             return False
           
          self.is_on_paint = True
         # dc = wx.PaintDC(self)
       # dc = wx.ClientDC(self) # no border
          #self.OnDraw( size_mm=dc.GetSizeMM() )
          print"TEST OnPaint"
          print  self.gl_canvas.wx.GetDisplaySizeMM()      
         # self.OnDraw( self.gl_canvas.GetDisplaySizeMM())
        # self.OnDraw( size_mm=dc.GetSizeMM() )
          # evt.Skip()
          self.is_on_paint = False

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
          
          self.gl_canvas.SetCurrent()
          glutInit(sys.argv)

          glShadeModel(GL_SMOOTH)

          #glutInit(sys.argv)
          self.plot2d = JuMEG_TSV_PLOT2D_OGL()
          self.plot2d.size_in_pixel = self.GetClientSize()
          self.plot2d.init_glwindow( )

          #glutInit(sys.argv)
          self.is_initGL = True
          
          return self.is_initGL

      def OnDraw(self,size_mm=None):

          if self.is_on_draw:
             return
             
          self.is_on_draw  = True

          if self.is_initGL:
             self.gl_canvas.SetCurrent()
          else:
             self.InitGL()

          self.plot2d.size_in_pixel = self.GetClientSize()
          self.plot2d.size_in_mm    = size_mm

          self.plot2d.display()

          self.gl_canvas.SwapBuffers()
          self.is_on_draw  = False
      
      def update(self,raw=None): #,do_scroll_channels=True,do_scroll_time=True):
       
          if self.is_initGL :
             self.gl_canvas.SetCurrent()
             if raw :
                self.plot2d.init_raw_data(raw=raw) 
             elif self.plot2d.data_is_init: 
                  self.plot2d.update_data() #do_scroll_channels=True,do_scroll_time=True,)
                  
                  #self.plot_axis.range_max = self.plot2d.timepoints[-1]
                  #self.plot_axis.range_min = self.plot2d.timepoints[0]
       
             if self.plot2d.opt.do_scroll:
                self.Refresh()
               # self.plot_axis.range_max = self.plot2d.timepoints[-1]
                #self.plot_axis.range_min = self.plot2d.timepoints[0]
                
                #self.plot_axis.Refresh()
                    