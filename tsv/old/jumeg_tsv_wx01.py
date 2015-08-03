import sys
import numpy as np

#from OpenGL.GL import *
#from OpenGL.GLU import *

try:
    import wx
    from wx import glcanvas
except ImportError:
    raise ImportError, "Required dependency wx.glcanvas not present"

try:
    from OpenGL.GL import *
except ImportError:
    raise ImportError, "Required dependency OpenGL not present"





class GLFrame(wx.Panel):
    """A simple class for using OpenGL with wxPython."""

    def __init__(self, parent, *args, **kwargs):
        """Create the DemoPanel."""
        wx.Panel.__init__(self, parent, *args, **kwargs)

        self.parent = parent  # Sometimes one can use inline Comments

        self.is_on_draw =False

    #def __init__(self, parent, id, title, pos=wx.DefaultPosition,
    #             size=wx.DefaultSize, style=wx.DEFAULT_FRAME_STYLE,
    #             name='frame'):
        #
        # Forcing a specific style on the window.
        #   Should this include styles passed?
      #  style = wx.DEFAULT_FRAME_STYLE | wx.NO_FULL_REPAINT_ON_RESIZE

       # super(GLFrame, self).__init__(parent, id, title, pos, size, style, name)

        self.GLinitialized = False
        attribList = (glcanvas.WX_GL_RGBA, # RGBA
                      glcanvas.WX_GL_DOUBLEBUFFER, # Double Buffered
                      glcanvas.WX_GL_DEPTH_SIZE, 24) # 24 bit

        #
        # Create the canvas
        self.canvas = glcanvas.GLCanvas(self, attribList=attribList)

        #
        # Set the event handlers.
        self.canvas.Bind(wx.EVT_ERASE_BACKGROUND, self.processEraseBackgroundEvent)
        self.canvas.Bind(wx.EVT_SIZE, self.processSizeEvent)
        self.canvas.Bind(wx.EVT_PAINT, self.processPaintEvent)

    #
        Sizer = wx.BoxSizer(wx.VERTICAL)
        Sizer.Add(self.canvas, 1, wx.EXPAND|wx.ALL, 5)
        self.SetSizerAndFit(Sizer)

    # Canvas Proxy Methods

    def GetGLExtents(self):
        """Get the extents of the OpenGL canvas."""
        return self.canvas.GetClientSize()

    def SwapBuffers(self):
        """Swap the OpenGL buffers."""
        self.canvas.SwapBuffers()

    #
    # wxPython Window Handlers

    def processEraseBackgroundEvent(self, event):
        """Process the erase background event."""
        pass # Do nothing, to avoid flashing on MSWin

    def processSizeEvent(self, event):
        """Process the resize event."""
        if self.canvas.GetContext():
            # Make sure the frame is shown before calling SetCurrent.
            self.Show()
            self.canvas.SetCurrent()

            size = self.GetGLExtents()
            self.OnReshape(size.width, size.height)
            self.canvas.Refresh(False)
        event.Skip()

    def processPaintEvent(self, event):
        """Process the drawing event."""
        self.canvas.SetCurrent()

        # This is a 'perfect' time to initialize OpenGL ... only if we need to
        if not self.GLinitialized:
            self.OnInitGL()
            self.GLinitialized = True

        self.OnDraw()
        event.Skip()

    #
    # GLFrame OpenGL Event Handlers

    def OnInitGL(self):
        """Initialize OpenGL for use in the window."""
        glClearColor(1, 1, 1, 1)

    def OnReshape(self, width, height):
        """Reshape the OpenGL viewport based on the dimensions of the window."""
        glViewport(0, 0, width, height)

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(-0.5, 0.5, -0.5, 0.5, -1, 1)

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

    def OnDraw(self, *args, **kwargs):
        "Draw the window."

        if self.is_on_draw:
           return

        self.is_on_draw = True

        glClear(GL_COLOR_BUFFER_BIT)

        # Drawing an example triangle in the middle of the screen
        glBegin(GL_TRIANGLES)
        glColor(0, 0, 0)
        glVertex(-.25, -.25)
        glVertex(.25, -.25)
        glVertex(0, .25)
        glEnd()

        self.SwapBuffers()
        self.is_on_draw=False

class JuMEG_TSV_MainFrame(wx.Frame):
    """JuMEG TSV wxProject MainFrame."""
    def __init__(self, parent,title="JuMEG TSV",id=wx.ID_ANY,
                 pos=wx.DefaultPosition,size=wx.DefaultSize,style=wx.DEFAULT_FRAME_STYLE,name="MainWindow"):
        super(JuMEG_TSV_MainFrame, self).__init__(parent,id, title, pos, size, style, name)

   #---  Options Plot/Time/Channels
        self._ID_OPT_PLOT     = 10111
        self._ID_OPT_TIME     = 10112
        self._ID_OPT_CHANNELS = 10113


#--- init wx body
        self.wx_init_main_menu()
        #self.wx_init_toolbar()
        self.wx_init_statusbar()

#--- init wx panels
      # self.wx_init_panels()


      # Create the splitter window.
      # splitter = wx.SplitterWindow(self, style=wx.NO_3D|wx.SP_3D)
      # splitter.SetMinimumPaneSize(1)


      # Add the Widget Panel
       # self.Panel    = DemoPanel(self)
       # self.OGLFrame = GLFrame(self, -1, 'GL Window')
        self.OGLFrame = GLFrame(self)
#---


        Sizer = wx.BoxSizer(wx.VERTICAL)
        Sizer.Add(self.OGLFrame, 1, wx.EXPAND|wx.ALL, 5)
        #Sizer.Add(MsgBtn, 0, wx.ALIGN_CENTER|wx.ALL, 5)

        self.SetSizerAndFit(Sizer)
       # self.Fit()


#--- init stuff  I/O

#--- click on stuff

    def click_on_open(self,event=None):
        print"click_on_open"

    def click_on_save(self,event=None):
        print"click_on_save"

    def click_on_clear(self,event=None):
        print"click_on_clear"

    def click_on_exit(self,event=None):
        print"click_on_exit"
        self.Close()



    def click_on_plot(self,event=None):
        print"click_on_plot"

    def click_on_time(self,event=None):
        print"click_on_time"

    def click_on_channels(self,event=None):
        print"click_on_channels"

    def click_on_about(self,event=None):
        print"click_on_about"


#---  display
      # self.Show(True)


    def wx_init_main_menu(self):

        _menubar = wx.MenuBar()

       #--- File I/O
        _menu_file = wx.Menu()
        __id=_menu_file.Append(wx.ID_OPEN, '&Open')
        self.Bind(wx.EVT_MENU,self.click_on_open,__id )

        __id=_menu_file.Append(wx.ID_SAVE, '&Save')
        self.Bind(wx.EVT_MENU,self.click_on_save,__id)

        _menu_file.AppendSeparator()

        __idx=_menu_file.Append(wx.ID_CLEAR,'&Clear')
        self.Bind(wx.EVT_MENU,self.click_on_clear,__id)

        _menu_file.AppendSeparator()

        __id=_menu_file.Append(wx.ID_EXIT, '&Exit')
        self.Bind(wx.EVT_MENU,self.click_on_exit,__id)

        _menubar.Append(_menu_file, '&File')

       #--- Options
        _menu_opt = wx.Menu()
        __id=_menu_opt.Append(self._ID_OPT_PLOT,    '&Plot')
        self.Bind(wx.EVT_MENU,self.click_on_plot,__id)

        __id=_menu_opt.Append(self._ID_OPT_TIME,    '&Time')
        self.Bind(wx.EVT_MENU,self.click_on_time,__id)

        __id=_menu_opt.Append(self._ID_OPT_CHANNELS,'&Channels')
        self.Bind(wx.EVT_MENU,self.click_on_channels,__id)

        _menubar.Append(_menu_opt, '&Options')


       #--- About
        _menu_about = wx.Menu()
        __id=_menu_about.Append(wx.ID_ABOUT, '&About')
        self.Bind(wx.EVT_MENU,self.click_on_about,__id)

        _menubar.Append(_menu_about, '&About')


        self.SetMenuBar(_menubar)

    def wx_init_statusbar(self):
        self.statusbar = self.CreateStatusBar(3)
        self.statusbar.SetStatusWidths([-1, -1 , -1])
        #self.statusbar.SetStatusText('1', '2')



def wxTSV():
    app = wx.App()
    frame = JuMEG_TSV_MainFrame(None, title="JuMEG TSV")
    frame.Show()
    app.MainLoop()

if __name__ == '__main__':
    wxTSV()