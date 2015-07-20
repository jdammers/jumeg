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







 def plot(self, events=None, duration=10.0, start=0.0, n_channels=20,
bgcolor='w', color=None, bad_color=(0.8, 0.8, 0.8),
event_color='cyan', scalings=None, remove_dc=True, order='type',
show_options=False, title=None, show=True, block=False,
highpass=None, lowpass=None, filtorder=4, clipping=None):
"""Plot raw data
Parameters
----------
events : array | None
Events to show with vertical bars.
duration : float
Time window (sec) to plot in a given time.
start : float
Initial time to show (can be changed dynamically once plotted).
n_channels : int
Number of channels to plot at once.
bgcolor : color object
Color of the background.
color : dict | color object | None
Color for the data traces. If None, defaults to:
`dict(mag='darkblue', grad='b', eeg='k', eog='k', ecg='r', emg='k',
ref_meg='steelblue', misc='k', stim='k', resp='k', chpi='k')`
bad_color : color object
Color to make bad channels.
event_color : color object
Color to use for events.
scalings : dict | None
Scale factors for the traces. If None, defaults to:
`dict(mag=1e-12, grad=4e-11, eeg=20e-6,
eog=150e-6, ecg=5e-4, emg=1e-3,
ref_meg=1e-12, misc=1e-3, stim=1, resp=1, chpi=1e-4)`
remove_dc : bool
If True remove DC component when plotting data.
order : 'type' | 'original' | array
Order in which to plot data. 'type' groups by channel type,
'original' plots in the order of ch_names, array gives the
indices to use in plotting.
show_options : bool
If True, a dialog for options related to projection is shown.
title : str | None
The title of the window. If None, and either the filename of the
raw object or '<unknown>' will be displayed as title.
show : bool
Show figures if True
block : bool
Whether to halt program execution until the figure is closed.
Useful for setting bad channels on the fly (click on line).
highpass : float | None
Highpass to apply when displaying data.
lowpass : float | None
Lowpass to apply when displaying data.
filtorder : int
Filtering order. Note that for efficiency and simplicity,
filtering during plotting uses forward-backward IIR filtering,
so the effective filter order will be twice ``filtorder``.
Filtering the lines for display may also produce some edge
artifacts (at the left and right edges) of the signals
during display. Filtering requires scipy >= 0.10.
clipping : str | None
If None, channels are allowed to exceed their designated bounds in
the plot. If "clamp", then values are clamped to the appropriate
range for display, creating step-like artifacts. If "transparent",
then excessive values are not shown, creating gaps in the traces.
Returns
-------
fig : Instance of matplotlib.figure.Figure
Raw traces.
Notes
-----
The arrow keys (up/down/left/right) can typically be used to navigate
between channels and time ranges, but this depends on the backend
matplotlib is configured to use (e.g., mpl.use('TkAgg') should work).
To mark or un-mark a channel as bad, click on the rather flat segments
of a channel's time series. The changes will be reflected immediately
in the raw object's ``raw.info['bads']`` entry.
"""
return plot_raw(self, events, duration, start, n_channels, bgcolor,
color, bad_color, event_color, scalings, remove_dc,
order, show_options, title, show, block, highpass,
lowpass, filtorder, clipping)


class GLPlot2D(wx.glcanvas):




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