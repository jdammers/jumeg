import sys
import numpy as np

import wx
# from python praxis book 
# import wx.lib.newevent s 1076
# from threading import Thread s1078


#from jumeg.jumeg_base               import JuMEG_Base_Basic
from jumeg.tsv.jumeg_tsv_plot2d_wx  import JuMEG_TSV_Plot2D

import jumeg.tsv.jumeg_tsv_wx_utils as jwx_utils
from jumeg.tsv.jumeg_tsv_data import JuMEG_TSV_DATA




#----------------------------------------------------------------------------------------
class JuMEG_TSV_Plot2DPanel(wx.Panel):
      def __init__(self, parent):
          wx.Panel.__init__(self,parent,-1)
          
         # __metaclass__= JuMEG_TSV_Plot2D(self)
          
          self.SetBackgroundColour("green")
          vbox = wx.BoxSizer(wx.VERTICAL)
          #vbox.Add((20, 30))

          self.plot = JuMEG_TSV_Plot2D(self)
          self.plot.SetMinSize((10, 50))

          vbox.Add(self.plot, 1, wx.ALIGN_CENTER|wx.ALL|wx.EXPAND,5)

          #self.plt2d2 = JuMEG_TSV_Plot2D(self)
          #self.plt2d2.SetMinSize((10, 50))
          #vbox.Add(self.plt2d2, 1, wx.ALIGN_CENTER|wx.ALL|wx.EXPAND,5)

          self.SetAutoLayout(True)
          self.SetSizer(vbox)


#----------------------------------------------------------------------------------------
class JuMEG_TSV_MainPanel(wx.Panel):
      def __init__(self, parent):
          wx.Panel.__init__(self, parent, -1)

          vbox = wx.BoxSizer(wx.VERTICAL)
          #vbox.Add((20, 30))

          self.plot2d = JuMEG_TSV_Plot2D(self)
          self.plot2d.SetMinSize((10, 50))

          vbox.Add(self.plot2d, 1, wx.ALIGN_CENTER|wx.ALL|wx.EXPAND,5)

          #self.plt2d = JuMEG_TSV_Plot2D(self)
          #self.plt2d.SetMinSize((10, 50))

          #vbox.Add(self.plt2d, 1, wx.ALIGN_CENTER|wx.ALL|wx.EXPAND,5)

          #self.plt2d2 = JuMEG_TSV_Plot2D(self)
          #self.plt2d2.SetMinSize((10, 50))
          #vbox.Add(self.plt2d2, 1, wx.ALIGN_CENTER|wx.ALL|wx.EXPAND,5)

          self.SetAutoLayout(True)
          self.SetSizer(vbox)


#----------------------------------------------------------------------------------------
class JuMEG_TSV_App(wx.App):

      def __init__(self,**kwargs):

          self.TSV_DATA = JuMEG_TSV_DATA(**kwargs)

         # redirect stout/err to wx console
          wx.App.__init__(self,redirect=False, filename=None)


          #self.settings={ 'subplot':{ 'MEG':{'rows':1,'cols':10},
          #                   'ICA':{'rows':1,'cols':10}},
          #         'time':{}
          #       }

      def OnInit(self):

          self.frame = wx.Frame(None, -1, " JuMEG TSV  ", pos=(0,0),
                       style=wx.DEFAULT_FRAME_STYLE|wx.FULL_REPAINT_ON_RESIZE, name="jumeg_tsv_main")

          #self.main_panel = wx.Panel(self, -1)
          #self.main_panel.SetBackgroundColour("blue")
         # vbox = wx.BoxSizer(wx.VERTICAL)

          self.wx_InitStatusbar()
          #self.wx_init_toolbar()
          self.wx_InitMainMenu()

          #self.Bind(wx.EVT_CLOSE,   self.OnCloseWindow)
          self.Bind(wx.EVT_KEY_DOWN,self.OnKeyDown)


          self.frame.Show(True)


         # vbox.Add(self.main_panel, 1, wx.ALIGN_CENTER|wx.ALL|wx.EXPAND,5)

          self.plt2d = JuMEG_TSV_Plot2DPanel(self.frame)
          #self.plt2d.plot.raw = self.TSV_DATA

          self.frame.SetSize((640,480))
          self.SetTopWindow(self.frame)

          if self.TSV_DATA.fname:
             self.TSV_DATA.load_raw()
             #self.plt2d.plot.plot2d.opt.plots     = 10
             #self.plt2d.plot.plot2d.opt.plot_cols = 2
                         
             if self.TSV_DATA.raw_is_loaded:
                self.plt2d.plot.update(raw=self.TSV_DATA.raw)
                self.frame.SetTitle(' JuMEG TSV  '+self.TSV_DATA.fname)
         # self.SetAutoLayout(True)
         # self.SetSizer(vbox)

         # self.SetFocus()


          return True



      def __menu_data(self):
          return (("&File",
                  ("&Open", "Open in status bar", self.ClickOnOpen),
                  ("&Save", "Save", self.ClickOnSave),
                  ("", "", ""),
                  ("&Clear", "Clear", self.ClickOnClear),
                  ("", "", ""),
                  ("&Exit", "Exit", self.ClickOnExit)),
                  ("&Option",
                  ("&Plot","Subplot",self.ClickOnPlotSubplot),
                  ("&Groups","Group",self.ClickOnPlotGroup)),
                  ("&About",
                  ("&About", "About", self.ClickOnAbout)))


      def wx_InitMainMenu(self):

          menuBar = wx.MenuBar()

          for eachMenuData in self.__menu_data():
              menuLabel = eachMenuData[0]
              menuItems = eachMenuData[1:]
              menuBar.Append(self.createMenu(menuItems), menuLabel)
              
         #--- plot
         # mu_sub = wx.Menu()    
         # mu_sub.AppendCheckItem(-1,'Sub Plot',self.ClickOnSubPlotOption)
         # menuBar.InsertMenu("Plot","Sub Plot", submenu,"Subplotoptions")
         #  Append("Sub Plot","Plot")   
          self.frame.SetMenuBar(menuBar)

      def createMenu(self, md):
          menu = wx.Menu()
          for eachLabel, eachStatus, eachHandler in md:
              if not eachLabel:
                 menu.AppendSeparator()
                 continue

              menuItem = menu.Append(-1, eachLabel, eachStatus)
              self.Bind(wx.EVT_MENU, eachHandler, menuItem)

          return menu


#--- init stuff  I/O

#--- click on stuff

      def ClickOnOpen(self,event=None):

          f = jwx_utils.jumeg_wx_openfile(self.frame,path=self.TSV_DATA.path)

          if f:
             print f
             self.TSV_DATA.update(fname=f)
             # self.plt2d.plot.update(raw=self.TSV_DATA.raw)
             self.plt2d.plot.update(raw=self.TSV_DATA.raw)
             self.frame.SetTitle('JuMEG-TSV: '+self.TSV_DATA.fname)
             print"DONE update"


      def ClickOnSave(self,event=None):
          print"click_on_save"

      def ClickOnClear(self,event=None):
          print"click_on_clear"

      def ClickOnExit(self,event=None):
         # msg = wx.MessageDialog('Are you sure to quit?', 'Question',wx.YES_NO | wx.NO_DEFAULT, self.frame)
          dlg = wx.MessageDialog(self.frame, 'Are you sure to quit?', 'Question', wx.YES_NO | wx.NO_DEFAULT | wx.ICON_QUESTION)
          
          if (dlg.ShowModal() == wx.ID_YES):
             self.frame.Close(True)
            # wx.Exit()
             

#---TODO self.plt2d.plot... as short cut; self.plt2d.plot._is_on_draw-> property

      def ClickOnPlotSubplot(self,event=None):
          self.plt2d.plot.is_on_draw=True           
          opt = jwx_utils.jumeg_wx_dlg_subplot(opt=self.plt2d.plot.plot2d.opt)#self.plt2d.plot.plot_options ) #,self.option['subplot'])
          self.plt2d.plot.is_on_draw=False       
          if opt:
             self.plt2d.plot.plot2d.opt = opt
             if self.TSV_DATA.raw_is_loaded:
                self.plt2d.plot.update()
             
             #self.plt2d.Refresh()

      def ClickOnPlotGroup(self,evt=None):
          print "Click on Plot -> Groups"
          self.plt2d.plot.is_on_draw = True           
          info = jwx_utils.jumeg_wx_dlg_group(opt=self.plt2d.plot.plot2d.info)#self.plt2d.plot.plot_options ) #,self.option['subplot'])
          self.plt2d.plot.is_on_draw = False       
          if info:
             self.plt2d.plot.plot2d.info = info
             if self.TSV_DATA.raw_is_loaded:
                self.plt2d.plot.update()
        

          
    
      def ClickOnChannelsOpt(self,event=None):
          print"click_on_channels"

      def ClickOnAbout(self,event=None):
          jwx_utils.jumeg_wx_utils_about_box()


      def wx_InitStatusbar(self):
          self.statusbar = self.frame.CreateStatusBar(3)
         # self.statusbar.SetStatusWidths([-1, -1 , -1])
          #self.statusbar.SetStatusText('1', '2')


# statusbar_fields = [("ButtonPanel wxPython Demo, Andrea Gavana @ 02 Oct 2006"),
#                            ("Welcome To wxPython!")]

#        for i in range(len(statusbar_fields)):
#            self.statusbar.SetStatusText(statusbar_fields[i], i)


      def OnExitApp(self, evt):
          self.Close(True)

      #def OnCloseWindow(self, evt):
          print "ClickOnCloseWindow"
         # if hasattr(self, "window") and hasattr(self.window, "ShutdownDemo"):
         #    self.window.ShutdownDemo()
         # evt.Skip()

      def OnKeyDown(self, e):

          key = e.GetKeyCode()
          #print"EVT OnKeyDown: " + key
        #---escape to quit
          if key == wx.WXK_ESCAPE:
             self.ClickOnExit(e)
          e.Skip()

#----------------------------------------------------------------------
class __JuMEG_TSV_App(wx.App):

      def __init__(self,**kwargs):

          wx.App.__init__(self, redirect=True, filename=None)


      def OnInit(self,**kwargs):

          self.main_frame= JuMEG_TSV_MainFrame(self,**kwargs)
                           #-1,"JuMEG TSV: ", pos=(0,0),
                           #         style=wx.DEFAULT_FRAME_STYLE|wx.FULL_REPAINT_ON_RESIZE, name="jumeg_tsv_main")

          self.main_frame.Show()
          self.SetTopWindow(self.main_frame)
          return True

#-------------------------------------------------------------------------
# fname=opt.fname,path=opt.path,verbose=opt.v,debug=opt.d,experiment=opt.exp,
# duration=opt.duration,start=opt.start,n_channels=opt.n_channels,bads=opt.bads
def jumeg_tsv_gui(**kwargs):
     app = JuMEG_TSV_App(**kwargs)
     app.MainLoop()



if __name__ == '__main__':
   jumeg_tsv_gui()

