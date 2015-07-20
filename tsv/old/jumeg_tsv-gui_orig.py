import sys
import numpy as np

import wx
# from python praxis book
# import wx.lib.newevent s 1076
# from threading import Thread s1078


from jumeg.jumeg_base                import JuMEG_Base_Basic
from jumeg.tsv.jumeg_tsv_wx_glcanvas import JuMEG_TSV_Plot2D

import jumeg.tsv.jumeg_tsv_wx_utils as jwx_utils
from jumeg.tsv.jumeg_tsv_data import JuMEG_TSV_DATA


#----------------------------------------------------------------------------------------
class JuMEG_TSV_MainPanel(wx.Panel):
      def __init__(self, parent):
          wx.Panel.__init__(self, parent, -1)

          vbox = wx.BoxSizer(wx.VERTICAL)
          #vbox.Add((20, 30))

          self.plt2d = JuMEG_TSV_Plot2D(self)
          self.plt2d.SetMinSize((10, 50))

          vbox.Add(self.plt2d, 1, wx.ALIGN_CENTER|wx.ALL|wx.EXPAND,5)

          #self.plt2d2 = JuMEG_TSV_Plot2D(self)
          #self.plt2d2.SetMinSize((10, 50))
          #vbox.Add(self.plt2d2, 1, wx.ALIGN_CENTER|wx.ALL|wx.EXPAND,5)

          self.SetAutoLayout(True)
          self.SetSizer(vbox)



#----------------------------------------------------------------------
class JuMEG_TSV_APP(wx.App):

      def __init__(self,**kwargs):
          self.TSV_DATA = JuMEG_TSV_DATA(**kwargs)

         # redirect stout/err to wx console
          wx.App.__init__(self,redirect=True, filename=None)


          #self.settings={ 'subplot':{ 'MEG':{'rows':1,'cols':10},
          #                   'ICA':{'rows':1,'cols':10}},
          #         'time':{}
          #       }

      def OnInit(self):

          self._ID_OPT_PLOT     = wx.NewId()
          self._ID_OPT_TIME     = wx.NewId()
          self._ID_OPT_CHANNELS = wx.NewId()

          self.frame = wx.Frame(None, -1, "JuMEG TSV: ", pos=(0,0),
                       style=wx.DEFAULT_FRAME_STYLE|wx.FULL_REPAINT_ON_RESIZE, name="jumeg_tsv_main")
           # self.frame.CreateStatusBar()


          self.wx_init_statusbar()
          #self.wx_init_toolbar()
          self.wx_init_main_menu()

          self.frame.Show(True)
          self.frame.Bind(wx.EVT_CLOSE,   self.OnCloseFrame)

          self.Bind(wx.EVT_KEY_DOWN,self.OnKeyDown)

          self.plot_panel = JuMEG_TSV_MainPanel(self.frame)
          self.plot2d     = self.plot_panel.plt2d.plot2d

          # set the frame to a good size for showing the two buttons
          self.frame.SetSize((640,480))
          #self.main_window.SetFocus()
          #self.window = win
          frect = self.frame.GetRect()

          self.SetTopWindow(self.frame)


          if self.TSV_DATA.fname:
             self.TSV_DATA.load_raw()
             self.plot2d.update_data(raw=self.TSV_DATA.raw)

          print "DONE ON INIT"

          return True



#--- init stuff  I/O

#--- click on stuff

      def ClickOnOpen(self,event=None):

          f = jwx_utils.jumeg_wx_openfile(self.frame,path=self.TSV_DATA.path)

          if f:
             print f
             self.TSV_DATA.update(fname=f)
             self.plot2d.update_data(raw=self.TSV_DATA.raw)
             print"DONE update"



      def ClickOnSave(self,event=None):
          print"click_on_save"

      def ClickOnClear(self,event=None):
          print"click_on_clear"

      def ClickOnExit(self,event=None):
          ret = wx.MessageBox('Are you sure to quit?', 'Question',wx.YES_NO | wx.NO_DEFAULT, self.frame)
          if ret == wx.YES:
             self.frame.Close(True)


      def ClickOnPlotOpt(self,event=None):
          print"click_on_plot"
          #r,c =jwx_utils.jumeg_wx_dlg_plot_option( r=3,c=5,rmax=1000,cmax=1000) #,self.option['subplot'])
          update = jwx_utils.jumeg_wx_dlg_plot_option( **self.plot2d.plot_options ) #,self.option['subplot'])

          if update:
             self.plot2d.update_data(raw=self.TSV_DATA.raw)
          #   self.plot2d.Refresh()

      def ClickOnTimeOpt(self,event=None):
          print"click_on_time"

      def ClickOnChannelsOpt(self,event=None):
          print"click_on_channels"

      def ClickOnAbout(self,event=None):
          jutils.jumeg_wx_utils_about_box()



      def wx_init_main_menu(self):

          _menubar = wx.MenuBar()

       #--- File I/O
          _menu_file = wx.Menu()
          __id=_menu_file.Append(wx.ID_OPEN, '&Open')
          self.Bind(wx.EVT_MENU,self.ClickOnOpen,__id )

          __id=_menu_file.Append(wx.ID_SAVE, '&Save')
          self.Bind(wx.EVT_MENU,self.ClickOnSave,__id)

          _menu_file.AppendSeparator()

          __idx=_menu_file.Append(wx.ID_CLEAR,'&Clear')
          self.Bind(wx.EVT_MENU,self.ClickOnClear,__id)

          _menu_file.AppendSeparator()

          __id=_menu_file.Append(wx.ID_EXIT, '&Exit')
          self.Bind(wx.EVT_MENU,self.ClickOnExit,__id)

          _menubar.Append(_menu_file, '&File')

        #--- Options
          _menu_opt = wx.Menu()
          __id=_menu_opt.Append(self._ID_OPT_PLOT,    '&Plot')
          self.Bind(wx.EVT_MENU,self.ClickOnPlotOpt,__id)

          __id=_menu_opt.Append(self._ID_OPT_TIME,    '&Time')
          self.Bind(wx.EVT_MENU,self.ClickOnTimeOpt,__id)

          __id=_menu_opt.Append(self._ID_OPT_CHANNELS,'&Channels')
          self.Bind(wx.EVT_MENU,self.ClickOnChannelsOpt,__id)

          _menubar.Append(_menu_opt, '&Options')


       #--- About
          _menu_about = wx.Menu()
          __id=_menu_about.Append(wx.ID_ABOUT, '&About')
          self.Bind(wx.EVT_MENU,self.ClickOnAbout,__id)

          _menubar.Append(_menu_about, '&About')

          self.frame.SetMenuBar(_menubar)

      def wx_init_statusbar(self):
          self.statusbar = self.frame.CreateStatusBar(3)
         # self.statusbar.SetStatusWidths([-1, -1 , -1])
          #self.statusbar.SetStatusText('1', '2')

      def OnExitApp(self, evt):
          self.frame.Close(True)

      def OnCloseFrame(self, evt):
          if hasattr(self, "window") and hasattr(self.window, "ShutdownDemo"):
             self.window.ShutdownDemo()
          evt.Skip()


      def OnKeyDown(self, e):

          key = e.GetKeyCode()

        #---escape to quit
          if key == wx.WXK_ESCAPE:
             self.click_on_exit(e)




def jumeg_tsv_gui(**kwargs):

    app = JuMEG_TSV_APP(**kwargs)
    # fname=None,path=None,raw=None,experiment=None,verbose=False,debug=False,duration=None,start=None,n_channels=None,bads=None)
    app.MainLoop()

if __name__ == '__main__':
   jumeg_tsv_gui()
