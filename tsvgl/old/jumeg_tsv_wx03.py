import sys
import numpy as np

#from OpenGL.GL import *
#from OpenGL.GLU import *

try:
    import wx
    from wx import glcanvas
    from wx.glcanvas import GLCanvas
except ImportError:
    raise ImportError, "Required dependency wx.glcanvas not present"

try:
    from OpenGL.GL import *
    from OpenGL.GLU import gluOrtho2D
except ImportError:
    raise ImportError, "Required dependency OpenGL not present"

import numpy as np

from jumeg.tsv.jumeg_tsv_gls_vob import JuMEG_GLS_Plotter


class JuMEGPlot2D(GLCanvas):
    def __init__(self, parent):

        attribList = (glcanvas.WX_GL_RGBA, # RGBA
                      glcanvas.WX_GL_DOUBLEBUFFER, # Double Buffered
                      glcanvas.WX_GL_DEPTH_SIZE, 24) # 24 bit

        GLCanvas.__init__(self, parent, -1, attribList=attribList)


        wx.EVT_PAINT(self, self.OnPaint)
        wx.EVT_SIZE(self, self.OnSize)

        #wx.EVT_MOTION(self, self.OnMouseMotion)
        wx.EVT_LEFT_DOWN(self, self.OnMouseLeftDown)
        wx.EVT_LEFT_UP(self, self.OnMouseLeftUp)
        wx.EVT_ERASE_BACKGROUND(self, lambda e: None)
       # wx.EVT_CLOSE(self, self.OnClose)
       # wx.EVT_CHAR(self, self.OnKeyDown)

        self.PLT = None

        self.n_channels=300
        self.n_timepoints= 1000000

        self.SetFocus()

        self.GLinitialized = False

        self.init = False
        self.rotation_y = 0.0
        self.rotation_x = 0.0
        self.prev_y = 0
        self.prev_x = 0

        self.mouse_down = False
        self.is_on_draw =False

        self.width = 400
        self.height = 400

        #Sizer = wx.BoxSizer(wx.VERTICAL)
        #Sizer.Add(self.canvas, 1, wx.EXPAND|wx.ALL, 5)
        #self.SetSizerAndFit(Sizer)
        self.srate=1024.15
        self.data = None
        self.timepoints=None
        self.vbo_id=0

        self._init_data()
       # self.vbo = VertexBuffer(self.data_vbo)
       # print "OK"

       #  self.vbo = JuMEGVertexBuffer(self.data_vbo)

       # If everything went well the following calls
       # will display the version of opengl being used



    def _init_data(self):
        import numpy as np
        ch=self.n_channels
        n = self.n_timepoints
        self.timepoints = np.arange(n) / self.srate
        self.data = np.zeros((ch,n), dtype=np.float32)
        #self.plot_color=np.ones( (ch,4), dtype=np.float32)

        #self.data = np.sin( 2 *np.pi + self.timepoints)
        print"start calc"
        for i in range( ch ):
            #self.data[i,:] = np.sin(self.timepoints * (10.0 + i) + (10 *i*np.pi) ) / ( 1.0 + self.timepoints * self.timepoints ) +np.sin( self.timepoints * 0.2* 2*np.pi)
            self.data[i,:] = np.sin(self.timepoints * (2 * i+1) * 2* np.pi)

        self.plot_color = np.repeat(np.random.uniform( size=(ch,3) ,low=.5, high=.9),1,axis=0).astype(np.float32)
        self.plot_color[:,-1] =1.0

        self.data_4_vbo = np.zeros((n,2), dtype=np.float32).flatten()
        #self.data_vbo[:,0] = self.timepoints
        #self.data_vbo[:,1] = self.data[-1,:]

        print"done calc"

        self.data_4_vbo_tp  = self.data_4_vbo[0:-1:2]
        self.data_4_vbo_sig = self.data_4_vbo[1::2]

        self.data_4_vbo_sig[:] = self.data[0,:]
        self.data_4_vbo_tp[:]  = self.timepoints



      #  graph[i].x = x;
      #  graph[i].y = sin(x * 10.0) / (1.0 + x * x);
#my $data_4_vbo             = pdl( zeroes(2,$data->dim(-1) ) )->float();
#my $data_4_vbo_timepoints  = $data_4_vbo->slice("(0),:");
#my $data_4_vbo_signal      = $data_4_vbo->slice("(1),:");
#   $data_4_vbo_timepoints .= $datax; #$self->xdata();
#my $data_vbo               = $data_4_vbo->flat;

    def set_window(self,l,r,b,t):
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluOrtho2D(l,r,b,t)

    def set_viewport(self,l,r,b,t):
        glViewport(l,b,r-l,t-b)





    def OnMouseLeftDown(self, event):
        self.mouse_down = True
        self.prev_x = event.GetX()
        self.prev_y = event.GetY()
        print"MLD"
        print self.prev_x
        print self.prev_y


    def OnMouseLeftUp(self, event):
        self.mouse_down = False
        print"MLU"

    # Canvas Proxy Methods

    def GetGLExtents(self):
        """Get the extents of the OpenGL canvas."""
        return self.GetClientSize()

    #def SwapBuffers(self):
    #    """Swap the OpenGL buffers."""
    #    #self.canvas.SwapBuffers()
    #    self.SwapBuffers()

    #
    # wxPython Window Handlers

    #def processEraseBackgroundEvent(self, event):
    #    """Process the erase background event."""
    #    pass # Do nothing, to avoid flashing on MSWin

    def OnSize(self, event):
        """Process the resize event."""
        if self.GetContext():
            # Make sure the frame is shown before calling SetCurrent.
            #self.Show()
            self.SetCurrent()

            size = self.GetGLExtents()
            self.OnReshape(size.width, size.height)
            self.Refresh(False)
        event.Skip()

    def OnPaint(self, event):
        """Process the drawing event."""
        self.SetCurrent()

        # This is a 'perfect' time to initialize OpenGL ... only if we need to
        if not self.GLinitialized:
            self.OnInitGL()
            self.GLinitialized = True




        if not self.PLT:
           self.PLT = JuMEG_GLS_Plotter()

        self.OnDraw()
        event.Skip()

    #
    # GLFrame OpenGL Event Handlers

    def OnInitGL(self):
        """Initialize OpenGL for use in the window."""
        glClearColor(1, 1, 1, 1)

    def OnReshape(self, width, height):
        """Reshape the OpenGL viewport based on the dimensions of the window."""

        self.set_viewport(0,width,0, height)


    def OnDraw(self, *args, **kwargs):
        "Draw the window."

        if self.is_on_draw:
           return

        self.is_on_draw = True

       #self.vbo = VertexBuffer(self.data_vbo)


        size = self.GetGLExtents()
        #--- reshape
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glClearColor(1.0,1.0,1.0,0.0)
        glLineWidth(2)

        # self.set_viewport(0,width,0, height)

        #glMatrixMode(GL_MODELVIEW)
        #glLoadIdentity()

        xmin=self.timepoints[0]
        xmax=self.timepoints[-1]

       #---start sub plots
        w0 = 10
        w1 = size.width-10

        h0 = 0
        dh = int( size.height / self.data.shape[0] );
        h1 = dh

        ymin=-1.0
        ymax=1.0
        dpos = ymin + (ymax - ymin) / 2.0

        glColor3f(0.0,0.0,1.0)

        glLineWidth(2)
        glColor3f(0,0,1)

        self.data_4_vbo_sig[:] = self.data[0,:]

        #self.PLT.data = self.data_4_vbo

        self.PLT.init_vao_vbo(self.data_4_vbo)


        #print self.PLT.data_buffer_size

        #self.PLT.vbo_init()

        # self.PLT.color= np.array([0.0,0.0,1.0,1.0])

        for idx in range( self.n_channels ):


           # glColor3f(0.0,0.0,1.0)

            self.set_viewport(w0,w1,h0,h1)

            #ymin = self.data[idx,:].min()
            #ymax = self.data [idx,:].max()
            #dpos = ymin + (ymax - ymin) / 2.0

            self.set_window(xmin,xmax,ymin,ymax )

           #--- draw zero line
           # glLineWidth(1)
           # glColor3f(0,0,0)
           # glColor3f(0.4,0.4,0.4)

           # glBegin(GL_LINES)
           # glVertex2f(xmin,0.0)
           # glVertex2f(xmax,0.0)
           # glEnd()

           # glBegin(GL_LINES)
           # glVertex2f(xmin,dpos)
           # glVertex2f(xmax,dpos)
           # glEnd();

            #glRasterPos2f(xmin,dpos)

          #--- plot signal
          #  glLineWidth(2)
          #  glColor3f(0,0,1)

         #--- create OGL verts buffer
          # glDisableClientState(GL_VERTEX_ARRAY)
          # self.data_vbo[:,0] = self.timepoints

            self.data_4_vbo_sig[:] = self.data[idx,:]
            self.PLT.vao_update(self.data_4_vbo)

            self.PLT.plot_color = self.plot_color[idx]
            self.PLT.plot()

            #self.vbo.data          = self.data_4_vbo
            #self.vbo.vbo_update()

           # self.vbo.data      = self.data_vbo

            #self.vbo.vbo_draw()

            h0 += dh
            h1 += dh + 1
          # glBufferSubDataARB_p(GL_ARRAY_BUFFER_ARB,0,$ogl_array);


        glFlush();
        self.SwapBuffers()
        self.is_on_draw=False
        self.PLT.reset();

        #self.vbo.vbo_reset()


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
        self.OGLplot = JuMEGPlot2D(self)
#---



        Sizer = wx.BoxSizer(wx.VERTICAL)
        Sizer.Add(self.OGLplot, 1, wx.EXPAND|wx.ALL, 5)
        #Sizer.Add(MsgBtn, 0, wx.ALIGN_CENTER|wx.ALL, 5)

        self.SetSizerAndFit(Sizer)

        self.Bind(wx.EVT_LEFT_DOWN,self.click_OnMouseLeftDown) #,self.Id)

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

    def click_OnMouseLeftDown(self, event):
        self.mouse_down = True
        self.prev_x = event.GetX()
        self.prev_y = event.GetY()
        print"MLD Main"
        print self.prev_x
        print self.prev_y


def wxTSV():
    app = wx.App()
    frame = JuMEG_TSV_MainFrame(None, title="JuMEG TSV")
    frame.Show()
    app.MainLoop()

if __name__ == '__main__':
    wxTSV()