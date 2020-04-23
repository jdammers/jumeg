#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JuMEG GUI Main Frame Class
frame work
"""
#--------------------------------------------
# Authors: Frank Boers <f.boers@fz-juelich.de>
#
#--------------------------------------------
# Date: 21.11.18
#--------------------------------------------
# License: BSD (3-clause)
#--------------------------------------------
# Updates
#--------------------------------------------


#--- wx Phoenix Version 4.03.01
import warnings
import os,wx
from pubsub        import pub
from wx.lib.scrolledpanel import ScrolledPanel

try:
   from wx.adv import AboutBox,AboutDialogInfo
except:
   from wx import AboutBox,AboutDialogInfo
   
__version__= '2019.05.14.001'

class JuMEG_wxAboutBox(object):
    def __init__(self):
        """   
        wrapper of wx.AboutBox
        """
        self.name        = "JuMEG"
        self.version     = "0.007"
        self.description = "JuMEG MEG Data Analysis at INM4-MEG-FZJ   user: " +os.getlogin()
        self.licence     = "License: BSD (3-clause)"
        self.copyright   = "Copyright, authors of JuMEG"
        self.website     = 'https://github.com/jdammers/jumeg'
        self.developer   = "JuMEG Team"
        self.docwriter   = None
        self.artist      = "JuMEG"
   #---     
    def show(self,parent):
        
        if parent:
           self.parent = parent
        if not self.docwriter:
           self.docwriter = self.developer
           
        info = AboutDialogInfo()
        info.SetName(self.name)
        info.SetVersion(self.version)
        info.SetDescription(self.description)
        info.SetCopyright(self.copyright)
        info.SetWebSite(self.website)
        info.SetLicence(self.licence)
        info.AddDeveloper(self.developer)
        info.AddDocWriter(self.docwriter)
        info.AddArtist(self.artist)
        AboutBox(info) #,self.parent)
        
class JuMEG_wxMainFrame(wx.Frame):
    """ 
    basic Main Frame CLS
    
    parent,id, title, pos, size, style, name
    style = wx.DEFAULT_FRAME_STYLE | wx.NO_FULL_REPAINT_ON_RESIZE):
        
    """ 
    def __init__(self,*kargs,**kwargs):
        super().__init__(*kargs)
        self.verbose    = kwargs.get("verbose",False)
        self.debug      = kwargs.get("debug",False)   
        self.use_pubsub = kwargs.get("pubsub",True)
        self.title_postfix = "   *** JuMEG MEG Data Analysis @ INM4-MEG-FZJ ***        user: " +os.getlogin()+"@"+os.uname()[1]
        self.isInit     = False
        self.StatusBar  = None 
        self.Sizer      = None
        self.AboutBox   = JuMEG_wxAboutBox()
        w,h = wx.GetDisplaySize()
        self.SetSize(w/1.1,h/1.3)
        self.Center()
       #--- Menue 
        self.MenuBar = wx.MenuBar()  
        self.init_frame(**kwargs) 
        
   #--- 
    def _init_AboutBox(self):
        self.AboutBox.name        = self.GetName() #"JuMEG MEEG Merger INM4-MEG-FZJ"
        self.AboutBox.description = self.GetName()#"JuMEG MEEG Merger"
        self.AboutBox.version     = __version__
        self.AboutBox.copyright   = '(C) 2018 JuMEGs FB'
        self.AboutBox.developer   = 'JuMEGs FB'
        self.AboutBox.docwriter   = 'JuMEGs FB'
    
   #---
    def _init_MenuDataListFileIO(self):
        """
        generate menu list for menu with File I/O items
        no used yet
        """
        self.menu_data_list = (("&File",
                             ("&Open", "Open in status bar",wx.ITEM_NORMAL, self.ClickOnOpen),
                             ("&Save", "Save", wx.ITEM_NORMAL,self.ClickOnSave),
                             ("", "", "",""),
                             ("&Exit", "Exit", wx.ITEM_NORMAL,self.ClickOnClose)),
                             ("&Settings",
                             ("&StatusBar","Status Bar On/Off", wx.ITEM_CHECK,self.ToggleCheckBox,True),
                             ("", "", "",""),
                             ("&Verbose","Verbose",wx.ITEM_CHECK,self.ToggleCheckBox,self.verbose),
                             ("&Debug","Debug",wx.ITEM_CHECK,self.ToggleCheckBox,self.debug)),
                             ("&Help",
                             ("&Help", "Help", wx.ITEM_NORMAL,self.ClickOnHelp)),
                             ("&About",
                             ("&About", "About", wx.ITEM_NORMAL,self.ClickOnAbout)))

    def _init_MenuDataList(self):
        self.menu_data_list = (("&File",
                               ("&Exit", "Exit", wx.ITEM_NORMAL,self.ClickOnClose)),
                               ("&Settings",
                               ("&Font","Font",wx.ITEM_NORMAL,self.ClickOnFont), 
                               ("&StatusBar","Status Bar On/Off", wx.ITEM_CHECK,self.ToggleCheckBox,True),
                               ("", "", "",""),
                               ("&Verbose","Verbose",wx.ITEM_CHECK,self.ToggleCheckBox,self.verbose),
                               ("&Debug","Debug",wx.ITEM_CHECK,self.ToggleCheckBox,self.debug)),
                               ("&Help",
                               ("&Help", "Help", wx.ITEM_NORMAL,self.ClickOnHelp)),
                               ("&About",
                               ("&About",  "About",  wx.ITEM_NORMAL,self.ClickOnAbout))
                               #("&MIN/MAX","MinMax", wx.ITEM_NORMAL,self.ClickOnMinMax))
                               )

    def msg_error(self,data="ERROR"):
        if isinstance(data,(list)):
           data= "\n".join(data)
        wx.MessageBox("Error: " + data, caption="ERROR  " + self.Name, style=wx.ICON_ERROR | wx.OK)

    def msg_warning(self,data="WARNING"):
        if isinstance(data, (list)):
           data = "\n".join(data)
        wx.MessageBox("Warning: "+data,caption="Warning  " +self.Name,style=wx.ICON_Warning|wx.OK)

    def msg_info(self,data="INFO"):
        if isinstance(data,(list)):
           msgtxt= "\n".join(data)
        wx.MessageBox("Info: "+data,caption="Info  " +self.Name,style=wx.ICON_INFORMATION|wx.OK)
   #---
    def wxInitToolBar(self,**kwargs):
        """
        overwrite  e.g.
        self.toolbar = self.CreateToolBar()
        self.toolbar.SetToolBitmapSize((16,16))  # sets icon size
        self.toolbar.Realize()
        """
        pass

    def _init_pubsub(self):
        """ init pubsub call and messages"""
       #---
        pub.subscribe(self.msg_error,  "MAIN_FRAME.MSG.ERROR")
        pub.subscribe(self.msg_warning,"MAIN_FRAME.MSG.WARNING")
        pub.subscribe(self.msg_info,   "MAIN_FRAME.MSG.INFO")
       #---
        pub.subscribe(self.SetStatusBarMSG, "MAIN_FRAME.STB.MSG")
        pub.subscribe(self.ClickOnClose, "MAIN_FRAME.CLICK_ON_CLOSE")
       # ---
        pub.sendMessage('MAIN_FRAME.VERBOSE', value=self.verbose)
        pub.sendMessage('MAIN_FRAME.DEBUG', value=self.debug)

    #---
    def init_frame(self,**kwargs):
        if self.isInit:
           self.DestroyChilderen() 
           self.MenuBar= wx.MenuBar()
           self.isInit = False

        self.update_from_kwargs(**kwargs) # overwrite
        self.onInit(**kwargs) # overwrite
        self._init_AboutBox()
        self.wxInitMainMenu()
        self.wxInitToolBar(**kwargs)
        self.wxInitStatusBar()
        self.UpdateDebugMode(self.debug)
        
        self.Sizer = wx.BoxSizer(wx.VERTICAL)
       #---
        self.UpdateAboutBox()
       #---
        self._init_MenuDataList()
       #---
        wxOBJ = self.update(**kwargs) # return wx.Panel or wx.XYZ-Sizer
        
      #--- use pubsub to close frame
        if self.use_pubsub: self._init_pubsub()

        self.Bind(wx.EVT_CLOSE,self.ClickOnClose)
        self.Sizer.Add( wxOBJ,1, wx.EXPAND|wx.ALL,1)
       
        self.SetSizer(self.Sizer)
        self.SetAutoLayout(True)
        self.SetTitle(self.GetTitle() + self.title_postfix)
        
        self.Show(True) 
        self.isInit = True
   #---
    def onInit(self,**kwargs):
        """
        function to setup and do changes Things in the GUI
        before  sohwing up
        overwrite with your own update function
        """
        pass
   #---
    def update_from_kwargs(self,**kwargs):
        """"""
        pass
   #---
    def update(self,**kwargs):
        """
        function to setup and do changes in the GUI
        overwrite with your own update function
        
        Parameters:
        -----------            
        **kwargs: all the key/value pairs you need to setup the GUI
        
        Results:
        ---------    
        wxWidged e.g kind of wx.Sizer or wx.Panel
        """
        pass

    def UpdateAboutBox(self):
        pass

    def UpdateDebugMode(self,v):
        """set debug flag and warning-mode True:default False:ignore"""
        self.debug = v
        if self.debug:
           warnings.filterwarnings("ignore")
        else:
           warnings.filterwarnings("default")
    #---
    def _update_menubar(self):
        for mdata in self.menu_data_list:
            mlabel = mdata[0]
            mitems = mdata[1:]
            self.MenuBar.Append(self.CreateMenu(mitems), mlabel)
        self.SetMenuBar(self.MenuBar)
   #---
    def wxInitMainMenu(self):
        self.MenuBar.DestroyChildren()
        self._init_MenuDataList()
        self._update_menubar()
        #if self.ShowLogger:
        self.AddLoggerMenu(pos=1,label="Logger")
   #---
    def CreateMenu(self,md):
        """
        create menues from  menu data list
        input: sub menu list 
                 label,status,kind,handler [True/False for kind]
          e.g.:
               ("&Open", "Open in status bar",wx.ITEM_NORMAL, self.ClickOnOpen)
               or as checkbox
               ("&StatusBar","Statusbar On/Off", wx.ITEM_CHECK,self.ToggleStatusBar,True)
           
        """
        menu = wx.Menu()

        for msub in md: #--- label,status,k,handler in sub_menu[0:3]:
            if not msub[0]:
               menu.AppendSeparator()
               continue

            menuItem = menu.Append(wx.ID_ANY,msub[0],msub[1],kind=msub[2])
            if (msub[2] == wx.ITEM_CHECK):
                menu.Check(menuItem.GetId(), msub[-1])
            #if (msub[2] == wx.ITEM_RADIO):
            #    if len(msub)>3:
            #       menu.Check(menuItem.GetId(), msub[-1])

               #self.Bind(wx.EVT_MENU,msub[3], menuItem,msub[4])
            self.Bind(wx.EVT_MENU,msub[3], menuItem)

        return menu
   #---
    def AddLoggerMenu(self,pos=1,label="Logger"):
        """
        adds a sub menu to flip a WindowSplitter
        call ClickOnFlipSplitter

        Parameters
        ----------
        pos  : menu position in menubar array <1>
        label: <Logger>

        """
        M = self.MenuBar.GetMenu(pos)
        subMenu = wx.Menu()

        #itm0 = subMenu.Append(wx.ID_ANY,"&Log Output",'logger on/off',wx.ITEM_CHECK)
        #subMenu.Check(itm0.GetId(),True)
        #subMenu.AppendSeparator()

        itm1 = subMenu.Append(wx.ID_ANY,"&Vertical",  'flip up/down',   wx.ITEM_NORMAL)
        itm2 = subMenu.Append(wx.ID_ANY,"&Horizontal",'flip left/right',wx.ITEM_NORMAL)
        M.AppendSubMenu(subMenu,label)
        l = self.GetName().replace(" ", "_").upper()
        self.Bind(wx.EVT_MENU,lambda evt,label=l:self.ClickOnLogger(evt,label),itm1)
        self.Bind(wx.EVT_MENU,lambda evt,label=l:self.ClickOnLogger(evt,label),itm2)
   #---
    def ClickOnLogger(self,evt,label="TEST"):
        if not self.use_pubsub: return
        menu     = evt.GetEventObject()
        menuItem = menu.FindItemById( evt.GetId() )
        if menuItem.GetItemLabelText() == "Log Output":
           pub.sendMessage(label.upper()+".SET_STATUS",   value= menuItem.IsChecked())
        elif menuItem.GetItemLabelText() == "Horizontal":
           pub.sendMessage(label.upper()+".FLIP_POSITION",value=wx.SPLIT_HORIZONTAL)
        else:
           pub.sendMessage(label.upper()+".FLIP_POSITION",value=wx.SPLIT_VERTICAL)

   #--- click on stuff
    def ClickOnFont(self,evt):
        dlg = wx.FontDialog(self,wx.FontData())
        if dlg.ShowModal() == wx.ID_OK:
           data   = dlg.GetFontData()
           #font   = data.GetChosenFont()
           #colour = data.GetColour()
           self.SetFont( data.GetChosenFont() ) 
        dlg.Destroy() 
        
    def ClickOnOpen(self,evt):
        pass
    
    def ClickOnSave(self,evt):
        pass

    def ClickOnClear(self,evt):
        pass

    def ClickOnClose(self,evt):
        if not self.debug:
           if wx.MessageBox('Are you sure to quit?',"Please confirm", wx.ICON_QUESTION | wx.YES_NO) != wx.YES:
              return
        self.Destroy() 
        evt.Skip()
           
        #   dlg = wx.MessageDialog(self, 'Are you sure to quit?', 'Question', wx.YES_NO | wx.NO_DEFAULT | wx.ICON_QUESTION)
        #if (dlg.ShowModal() == wx.ID_YES):
        #   self.Close(True)
  
    
    def ClickOnHelp(self,evt):
        pub.sendMessage("MAIN_FRAME.CLICK_ON_HELP")

    def ClickOnAbout(self,evt):
        self.AboutBox.show(self)

    def SetStatusBarMSG(self,data):
        '''
         call from pubsub
         "MAIN_FRAME.STB.MSG", value=["RUN", self._args[0], "PID", str(self.__proc.pid)]
        '''
        idx=0
        if not data: return
        if not isinstance(data,(list)): data=list(data)
        if data:
           for s in data:
               #self.msg_info("STB: "+s +"  "+str(idx))
               self.StatusBar.SetStatusText(s,i=idx)
               idx+=1
               if idx >= self.StatusBar.GetFieldsCount(): break


    def wxInitStatusBar(self,fields=4):
        if self.StatusBar:
           self.StatusBar.Destroy() 
        self.StatusBar = self.CreateStatusBar(fields,style=wx.STB_DEFAULT_STYLE)
        #self.StatusBar.SetStatusWidths([-1,-1,-1,-1])
        self.StatusBar.SetStatusStyles([wx.SB_SUNKEN,wx.SB_SUNKEN,wx.SB_SUNKEN,wx.SB_SUNKEN])
        
    def OnExitApp(self,evt):
        self.ClickOnClose(evt)

    def ToggleCheckBox(self,evt):
        v, label = self._ischecked(evt)
        
        if label=="StatusBar":
           if v: 
               self.StatusBar.Show()
           else:
               self.StatusBar.Hide()
           return    
        elif label=="Verbose":
             self.verbose = v
        elif label=="Debug":
             self.UpdateDebugMode(v)
        if self.use_pubsub:  # verbose,debug status
           pub.sendMessage('MAIN_FRAME.' + label.upper(), value=v)

    def _ischecked(self,evt):
        """ check checkbox for true/false """
        menu = evt.GetEventObject()
        menuItem = menu.FindItemById( evt.GetId() )
        return menuItem.IsChecked(),menuItem.GetItemLabelText()

#----    
class JuMEG_wxMainFrameDemo(JuMEG_wxMainFrame):
    """
    Example
    --------
    import wx
    from jumeg.gui.wxlib.jumeg_gui_wxlib_main_frame           import JuMEG_wxMainFrameDemo

    app    = wx.App()
    frame  = JuMEG_MainFrameDemo(None,-1,'JuMEG Main Frame Demo FZJ-INM4',debug=True)
    frame.AboutBox.name="JuMEG MAIN TEST"
    app.MainLoop()
    """
    def __init__(self,parent,id,title,pos=wx.DefaultPosition,size=wx.DefaultSize,name='JuMEG_MAIN_FRAME_DEMO',*kargs,**kwargs):
        style = wx.DEFAULT_FRAME_STYLE | wx.NO_FULL_REPAINT_ON_RESIZE
        super().__init__(parent,id, title, pos, size, style, name,**kwargs)
        self.SetBackgroundColour("blue")
        self.Center()
   #---
    def update(self,**kwargs):    
        """ 
        Results
        -------
        wxPanel obj
        https://stackoverflow.com/questions/3104323/getting-a-wxpython-panel-item-to-expand  
        """
        p1 = ScrolledPanel(self,-1,style = wx.TAB_TRAVERSAL|wx.SUNKEN_BORDER, name="panel1" )
        p1.SetBackgroundColour("White")
        vbox = wx.BoxSizer(wx.VERTICAL)
        
        txt_info = wx.StaticText(p1, -1,txt)
        txt_info.SetForegroundColour("Green")
        
        vbox.Add(txt_info, 0, wx.ALIGN_LEFT|wx.ALL, 5)
        vbox.Add(wx.StaticLine(p1, -1, size=(1024,-1)), 0, wx.ALL, 5)
        vbox.Add((20,20))
        
        p1.SetSizer(vbox)
        p1.SetAutoLayout(1)
        p1.SetupScrolling()
        return p1  
  
if __name__ == '__main__':
   
   app    = wx.App()
   frame  = JuMEG_MainFrameDemo(None,-1,'JuMEG Main Frame Demo FZJ-INM4',debug=True)
   frame.AboutBox.name="JuMEG MAIN TEST"
 
   app.MainLoop()   
