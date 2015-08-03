
import wx

#try:
from wx import glcanvas
#    haveGLCanvas = True
#except ImportError:
#    haveGLCanvas = False

#try:
    # The Python OpenGL package can be found at
    # http://PyOpenGL.sourceforge.net/
from OpenGL.GL import *
from OpenGL.GLUT import *
#    haveOpenGL = True
#except ImportError:
#    haveOpenGL = False

from jumeg.tsv.jumeg_tsv_ogl_plot2d import JuMEG_TSV_OGL_PLOT2D


attribList = (glcanvas.WX_GL_RGBA,          # RGBA
              glcanvas.WX_GL_DOUBLEBUFFER,  # Double Buffered
              glcanvas.WX_GL_DEPTH_SIZE,24) # 24 bit


#---------------------------------------------------------------------------------------
class JuMEG_TSV_CanvasBase(glcanvas.GLCanvas):
      def __init__(self, parent):
          style = wx.DEFAULT_FRAME_STYLE # | wx.NO_FULL_REPAINT_ON_RESIZE

          glcanvas.GLCanvas.__init__(self, parent, -1,attribList=attribList,style=style)
          self.context = glcanvas.GLContext(self)
          self.count=0
          self.rezise_cnt=0

          self.LeftDown     = False

          self._is_initGL   = True

          self._is_on_draw  = False
          self._is_on_paint = False
          self._is_on_size  = False

        # initial mouse position
          self.lastx = self.x = 30
          self.lasty = self.y = 30
          self.size  = None

          self.Bind(wx.EVT_ERASE_BACKGROUND, self.OnEraseBackground)
          self.Bind(wx.EVT_SIZE,      self.OnSize)
          self.Bind(wx.EVT_PAINT,     self.OnPaint)
          self.Bind(wx.EVT_LEFT_DOWN, self.OnMouseLeftDown)
          self.Bind(wx.EVT_LEFT_UP,   self.OnMouseLeftUp)
          #self.Bind(wx.EVT_MOTION,    self.OnMouseMotion)
        # self.Bind(wx.EVT_KEY_DOWN,  self.OnKeyDown)

          #self.Refresh()

      def OnEraseBackground(self, event):
          pass # Do nothing, to avoid flashing on MSW.


      def OnSize(self, event):
          """Process the resize event."""
          self._is_on_size = True

          if self.LeftDown :
             print"==========>SKIP REFRESH UPDATE"
             self._is_on_size  = False
            # event.Skip()
             return

          if self.GetContext():
            # Make sure the frame is shown before calling SetCurrent.
             self.Show()
             self.SetCurrent()
             self.Refresh()

             event.Skip()
          self._is_on_size  = False

      def OnPaint(self, event):
          if self._is_on_draw:
             return



          if self.LeftDown :
             print"-----> ONDRAW  Mouse DOWN"
             #event.Skip()
             self._is_on_paint = False
             return


          self._is_on_paint = True
         # dc = wx.PaintDC(self)
          dc = wx.ClientDC(self) # no border
          self.OnDraw( size_mm=dc.GetSizeMM() )
          event.Skip()
          self._is_on_paint = False




      def OnMouseLeftDown(self, evt):
          self.LeftDown=True
          #self.CaptureMouse()
          #self.x, self.y = self.lastx, self.lasty = evt.GetPosition()
          event.Skip()

      def OnMouseLeftUp(self, evt):
          self.LeftDown=False
          event.Skip()
          #self.Refresh(True)
          #print"MUP"
          #self.ReleaseMouse()



      def OnMouseMotion(self, evt):
          if evt.Dragging() and evt.LeftIsDown():
             self.lastx, self.lasty = self.x, self.y
             self.x, self.y = evt.GetPosition()
             print "on mouse drag LD"
            # self.Refresh(False)

#----------------------------------------------------------------------------------------
class JuMEG_TSV_Plot2D(JuMEG_TSV_CanvasBase):
      def __init__(self, parent):
          JuMEG_TSV_CanvasBase.__init__(self,parent)

         # self.plot2d = JuMEG_TSV_OGL_PLOT2D()

          self.InitGL()

      def InitGL(self):

          self.SetCurrent()

          self.plot2d = JuMEG_TSV_OGL_PLOT2D()
          self.plot2d.size_in_pixel = self.GetClientSize()
          self.plot2d.init_glwindow( )

          glutInit(sys.argv)

          self._is_initGL = True

      def OnDraw(self,size_mm=None):


          self._is_on_draw  = True

          if self._is_initGL:
             self.SetCurrent()
          else:
             self.InitGL()

          self.plot2d.size_in_pixel = self.GetClientSize()
          self.plot2d.size_in_mm    = size_mm

          self.plot2d.display()

          self.SwapBuffers()
          self._is_on_draw  = False

