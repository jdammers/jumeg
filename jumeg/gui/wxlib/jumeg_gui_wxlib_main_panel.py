#!/usr/bin/envn python3
# -+-coding: utf-8 -+-
#----------------------------------------
# Created by fboers at 19.09.18
#----------------------------------------
# Update
#----------------------------------------

import wx
from pubsub import pub
from jumeg.gui.wxlib.jumeg_gui_wxlib_logger               import JuMEG_wxLogger
#from jumeg.gui.wxlib.jumeg_gui_wxlib_loglog               import JuMEG_wxLogger
from jumeg.gui.wxlib.utils.jumeg_gui_wxlib_utils_controls import JuMEG_wxSplitterWindow,JuMEG_wxCMDButtons


__version__= '2019.05.14.001'

class JuMEG_wxMainPanelBase(wx.Panel):
    '''
    JuMEG_wxPanel  base CLS for JuMEG wx.Panels
    with functions to ovewrite to avoid overhead

        PanelTop
    PanelA | PanelB
        Logger panel

    PanelTop: for comboboxes e.g. experiment selection
    PanelA  : left  container panel
    PanelB  : right container panel e.g. parameter for argparser
    Logger  : panel with TextCtrl and <Clear> and <MinMax> button on top right

    Paremeters:
    -----------
     bg     : panel backgroundcolor <grey88>
     bgA    : panel A" backgroundcolor <wx.Colour(132, 126, 238)>
     bgB    : panel B" backgroundcolor <wx.Colour(140, 233, 238)>
     labelA : panel A label
     labelB : panel B label

     Flags
      ShowLogger    : show logger window at bottom <False>
      ShowMinMaxBt  : show MinMax Button in Logger window, to min/max window <True>
      ShowCmdButtons: show command buttons <Cloas,Cancel,Apply> <True>

      verbose : <False>
      debug   : <False>


    Functions to overwrite:
    -----------------------
     update_from_kwargs(**kwargs)
     wx_init(**kwargs)
     init_pubsub(**kwargs)
     update(**kwargs)

    Layout function
    ----------------
    ApplyLayout()   use <self.MainPanel> to pack you controls

    Example:
    --------
    class JuMEG_wxMEEGMerger(JuMEG_wxPanel):
          def __init__(self, parent, **kwargs):
             super(JuMEG_wxMEEGMerger, self).__init__(parent)
             self._init(**kwargs) # here is the initialization

          def ApplyLayout(self):
              """ use PanelA and PanelB to put the contrls and to split between """
              ds = 4
             #--- fill PanelA with controls
              vboxA = wx.BoxSizer(wx.VERTICAL)
              vboxA.Add(self.CtrlA1, 0, wx.ALIGN_LEFT | wx.EXPAND | wx.ALL, ds)
              vboxA.Add((0, 0), 0, wx.ALIGN_RIGHT | wx.ALL)
              hboxA = wx.BoxSizer(wx.HORIZONTAL)
              hboxA.Add(self.CtrlA2, 0, wx.ALIGN_LEFT | wx.EXPAND | wx.ALL, ds)
              hboxA.Add(self.CtrlA3, 1, wx.ALIGN_LEFT | wx.EXPAND | wx.ALL, ds)
              vboxA.Add(hboxA, 1, wx.ALIGN_LEFT | wx.EXPAND | wx.ALL)
              self.PanelA.SetSizer(vboxA)
             #--- fill PanelB with controls
              vboxB = wx.BoxSizer(wx.VERTICAL)
              vboxB.Add(self.CtrlB1, 0, wx.ALIGN_LEFT | wx.EXPAND | wx.ALL, ds)
              self.PanelB.SetSizer(vboxB)

              self.SplitterAB.SplitVertically(self.PanelA,self.PanelB)

    '''
    def __init__(self,parent,name="MAIN_PANEL_BASE",**kwargs):
        super().__init__(parent,id=wx.ID_ANY,style=wx.SUNKEN_BORDER,name=name)
        self._splitter  = None
        self.use_pubsub = True
        self.debug      = False
        self.verbose    = False
        self._param     = {}

        self.__isInit       = False
        self.__isMinimize   = False
        self._show_logger   = False
        self._show_minmax_bt= False
        self._show_cmd_bts  = False
        self._show_top_pnl  = True
        self._show_titleA   = True
        self._show_titleB   = True
      # self._init(**kwargs) # the function to call in child class

    def _get_param(self,k1,k2):
        return self._param[k1][k2]
    def _set_param(self,k1,k2,v):
        self._param[k1][k2]=v

    @property
    def isInit(self): return self.__isInit

    @property
    def MainPanel(self):    return self._pnl_main

    @property
    def LoggerPanel(self):  return self._pnl_logger
  
    @property
    def TopPanel(self):  return self._pnl_top
  
    @property
    def CmdButtonPanel(self):  return self._pnl_cmd_buttons

    @property
    def WindowSplitter(self): return self._splitter

    @property
    def ShowLogger(self)  : return self._show_logger
    @ShowLogger.setter
    def ShowLogger(self,v): self._show_logger = v

    @property
    def ShowMinMaxBt(self)  : return self._show_minmax_bt
    @ShowMinMaxBt.setter
    def ShowMinMaxBt(self,v): self._show_minmax_bt = v

    @property
    def ShowCmdButtons(self)  : return self._show_cmd_bts
    @ShowCmdButtons.setter
    def ShowCmdButtons(self,v): self._show_cmd_bts = v

    @property
    def ShowTopPanel(self)  : return self._show_top_pnl
    @ShowTopPanel.setter
    def ShowTopPanel(self,v): self._show_top_pnl = v

    @property
    def ShowTitleA(self): return self._show_titleA
    @ShowTitleA.setter
    def ShowTitleA(self,v): self._show_titleA = v
    @property
    def ShowTitleB(self): return self._show_titleB
    @ShowTitleB.setter
    def ShowTitleB(self,v): self._show_titleB = v

    def SetVerbose(self,value=False):
        self.verbose=value

    def ShowHelp(self):
        """ show help __doc__string"""
        wx.MessageBox(self,self.__doc__ )
    
#--- default methods
    def _update_from_kwargs_default(self,**kwargs):
        """
        update kwargs like doc <JuMEG_wxMainPanelBase>

        :param kwargs:
        :return:
        """
        self.labelA         = kwargs.get("labelA","PANEL A")
        self.labelB         = kwargs.get("labelB","PANEL B")
        self.bgA            = kwargs.get("bgA",wx.Colour(132, 126, 238))
        self.bgB            = kwargs.get("bgB",wx.Colour(140, 233, 238))
        self._show_logger   = kwargs.get("ShowLogger",self._show_logger)
        self._show_minmax_bt= kwargs.get("ShowMinMaxBt",self._show_minmax_bt)
        self._show_cmd_bts  = kwargs.get("ShowCmdButtons",self._show_cmd_bts)
        self.verbose        = kwargs.get("verbose", False)
        self.debug          = kwargs.get("debug", False)
        self.SashPosition   = kwargs.get("SashPosition",-50)
        self.SetBackgroundColour(kwargs.get("bg", "grey88"))
        
    def FitBoxSizer(self,pnl,pos=wx.HORIZONTAL):
        """ fits a BoxSizer to a panel + AutoLayout
        pnl: wx:panel
        pos: sizer type <wx.HORIZONTAL>
        """
        pnl.SetSizer(wx.BoxSizer(pos))
        pnl.Fit()
        pnl.SetAutoLayout(True)

    def _wx_init_default(self, **kwargs):
        """ window default settings"""
        self.clear_children(self)
        # --- init splitter for controls and logger
        self._splitter   = None
        self._pnl_logger = None
        self._pnl_main   = None
        self._pnl_top    = None
        self._pnl_cmd_buttons = None
        
        # --- command logger
        if self.ShowLogger:
           self._splitter   = JuMEG_wxSplitterWindow(self,label="Logger",name=self.GetName() + ".SPLITTER")
           self._pnl_logger = JuMEG_wxLogger(self._splitter,name=self.GetName().upper()+".LOGGER") #listener=self.GetName())
           self._pnl_main   = wx.Panel(self._splitter)
           self._pnl_main.SetBackgroundColour(wx.Colour(0, 0, 128))
        else:  # --- only you controls
           self._pnl_main = wx.Panel(self)
           self._pnl_main.SetBackgroundColour(wx.RED)

    def _init_pubsub_default(self):
        pub.subscribe(self.SetVerbose,'MAIN_FRAME.VERBOSE')
        pub.subscribe(self.ShowHelp, self.GetName()+".SHOW_HELP")
        
#--- overwrite methods
    def _update_from_kwargs(self,**kwargs):
        """ pass """
        pass

    def wx_init(self, **kwargs):
        """ init WX controls """
        pass

    def init_pubsub(self, **kwargs):
        """"
        init pubsub call
        pub.subscribe(self.SetVerbose,'MAIN_FRAME.VERBOSE')
        """
        pass

    def update(self, **kwargs):
        pass

    def ApplyLayout(self):
        """ your layout stuff """
        pass

  #--- init all
    def _init(self,**kwargs):
        if self.isInit:
           self.clear()
      #---
        self._update_from_kwargs_default(**kwargs)
        self._update_from_kwargs(**kwargs)
      #---
        self._wx_init_default()
        self.wx_init(**kwargs)
      # ---
        self.update(**kwargs)
      #---
        self._init_pubsub_default()
        self.init_pubsub(**kwargs)
      #---
        self.__isInit=True
        self._ApplyLayout()
        self.update_on_display()
    
    def update_on_display(self):
        pass
    
    def _ApplyLayout(self):
        """ default Layout Framework """
        self.Sizer = wx.BoxSizer(wx.VERTICAL)

        self.ApplyLayout() #-- fill PanelA amd PanelB with controls
        self.MainPanel.Fit()
        self.MainPanel.SetAutoLayout(1)

        if self.ShowLogger:
           self._splitter.SplitHorizontally(self.MainPanel,self.LoggerPanel)
           self._splitter.SetMinimumPaneSize(50)
           self._splitter.SetSashGravity(1.0)
           self._splitter.SetSashPosition(self.SashPosition,redraw=True)
           self.Sizer.Add(self._splitter, 1, wx.ALIGN_CENTER | wx.EXPAND | wx.ALL, 5)
        else:
           self.Sizer.Add(self.MainPanel, 1, wx.ALIGN_CENTER | wx.EXPAND | wx.ALL, 5)

        self.SetSizer(self.Sizer)
        self.Fit()
        self.SetAutoLayout(1)
        self.GetParent().Layout()

   #--- clear
    def clear_parameter(self):
        """ clear parameter overwrite"""
        pass

    def clear_children(self,wxctrl):
        """ clear/delete wx childeren """
        for child in wxctrl.GetChildren():
            child.Destroy()
        self.Layout()
        self.Fit()

    def clear(self,wxctrl=None):
        """ clear parameter and delete wx childeren """
        self.__isInit = False
        self.clear_parameter()
       #--- clear wx stuff
        self.clear_children(self)

class JuMEG_wxMainPanel(JuMEG_wxMainPanelBase):
    """
     main panel with sub-panels A,B inside a SplitterPanel
     call self._init(**kwargs) in subclass

     Parameter:
     ----------
      labelA: title panel A
      labelB: title panel B
      bgA   : wx.Colour() <132, 126, 238>
      bgB   : wx.Colour() <140, 233, 238>

      ShowMinMAxBt: True

     Panels:
     ---------
                TopPanel
     ========================================
      PanelA Title  | PanelB Title + MinMaxBt
      PanelA Panles | PanelB Panels
     ========================================
               Button Panel
    """
    def __init__(self,*kargs,name="MAIN_PANEL",**kwargs):
        super(JuMEG_wxMainPanel,self).__init__(*kargs,name=name,**kwargs)
        #self._init(**kwargs)

    def wx_init(self, **kwargs):
        """ init WX controls """
        self._update_from_kwargs(**kwargs)
       #---
        if self.ShowTopPanel:
           self._pnl_top = wx.Panel(self.MainPanel)
           self.FitBoxSizer(self.TopPanel)
       #---
        self.SplitterAB = JuMEG_wxSplitterWindow(self.MainPanel,listener=self.GetName()+"_B")
        self.SplitterAB.SetSashGravity(1.0)
       #---
        self.PanelA     = JuMEG_wxPanelAB(self.SplitterAB,name=self.GetName()+"_A",bg=self.bgA,label=self.labelA,ShowTitle=self.ShowTitleA)
        self.PanelB     = JuMEG_wxPanelAB(self.SplitterAB,name=self.GetName()+"_B",bg=self.bgB,label=self.labelB,ShowMinMaxBt=self.ShowMinMaxBt,ShowTitle=self.ShowTitleB)
        self.SplitterAB.SplitVertically(self.PanelA, self.PanelB)
       #---
        if self.ShowCmdButtons:
           self._pnl_cmd_buttons = JuMEG_wxCMDButtons(self.MainPanel,prefix=self.GetName(),
                                                      ShowClose=True,ShowCancel=True,ShowApply=True)

       #--- make a BoxSizer to pack later CTRLs
        self.FitBoxSizer(self.PanelA.Panel)
        self.FitBoxSizer(self.PanelB.Panel)

    def ApplyLayout(self):
        """ your layout stuff """
        vbox = wx.BoxSizer(wx.VERTICAL)
        
        if self.TopPanel:
          # if self.TopPanel.GetChildren():
           vbox.Add(self.TopPanel, 0, wx.ALIGN_LEFT| wx.EXPAND | wx.ALL, 1)
           #else: self.TopPanel.Destroy()
              
        vbox.Add(self.SplitterAB, 1, wx.ALIGN_LEFT | wx.EXPAND | wx.ALL, 1)
        
        if self.ShowCmdButtons:
           vbox.Add(self.CmdButtonPanel,0,wx.ALIGN_BOTTOM|wx.EXPAND|wx.ALL,1 )

        self.MainPanel.SetSizer(vbox)

class JuMEG_wxTitlePanel(wx.Panel):
    """
     wx.Panel with TextCtrl and MinMax Button packed Horizontal

     Parameter:
     ----------
     parent: parent widget
     title : text <TEST>
     ShowMinMaxBt: will show MinMAx button <False>

     Example:
     --------
      pnl_title = JuMEG_wxTitlePanel( <parent pnl>,label="Parameter",ShowMinMaxBt=True)

    """
    def __init__(self,parent,*kargs,**kwargs):
        super(JuMEG_wxTitlePanel,self).__init__(parent,*kargs,id=wx.ID_ANY,style=wx.SUNKEN_BORDER)
        self._txt          = None
        self._ShowMinMaxBt = False
        self._font         = wx.Font(10, wx.MODERN, wx.NORMAL, wx.NORMAL, False, u'Arial', wx.FONTENCODING_ISO8859_1)
        self.update_from_kwargs(**kwargs)
        self._wx_init(**kwargs)
        self.ApplyLayout()

    @property
    def label(self): return self._label
    @label.setter
    def label(self,v):
        self._label=v
        self._txt.SetLabel(v)
    @property
    def TxtCtrl(self): return self._txt

    def update_from_kwargs(self,**kwargs):
        self._label        = kwargs.get("label",self.GetName())
        self._ShowMinMaxBt = kwargs.get("ShowMinMaxBt",False)

    def _wx_init(self,**kwargs):
        """ init Wx controls """

        self.Sizer = wx.BoxSizer(wx.HORIZONTAL)
        self._txt = wx.StaticText(self, wx.ID_ANY,self.label,style=wx.ALIGN_CENTRE)
        self._txt.SetBackgroundColour("grey70")
        self._txt.SetFont(self._font)
        self.Sizer.Add(self._txt, 1, wx.ALIGN_LEFT | wx.EXPAND | wx.ALL, 1)

        if self._ShowMinMaxBt:
           stl = wx.BU_EXACTFIT | wx.BU_NOTEXT
           self._MinMaxBt = wx.Button(self, -1,name=self.GetParent().GetName()+".SPLIT_MIN_MAX", style=stl)
           self._MinMaxBt.SetBitmapLabel(wx.ArtProvider.GetBitmap(wx.ART_CROSS_MARK, wx.ART_MENU, (12, 12)))
           self.Bind(wx.EVT_BUTTON, self.ToggleMinimize, self._MinMaxBt)
           self.Sizer.Add( self._MinMaxBt, 0, wx.ALIGN_RIGHT | wx.EXPAND | wx.ALL, 1)

    def ToggleMinimize(self, evt):
         """
         toggle min/max size of logger window
         send cmd to parent splitter window via pubsub
         """
         obj = evt.GetEventObject()
         #print("ToggleMinimize: {} name: {} ".format(obj.GetName().upper(),self.GetParent().GetName()))
         pub.sendMessage(obj.GetName().upper(),name=self.GetParent().GetName(),size=obj.GetSize())

    def ApplyLayout(self):
        self.SetSizer(self.Sizer)
        self.Fit()
        self.SetAutoLayout(True)

class JuMEG_wxPanelAB(wx.Panel):
    """
     wx.Panel PanelA or PanelB container panel

     Parameter:
     ----------
     parent: parent widget
     name  : text <TEST>
     bg    : wx.Colour()
     title : text for title-panel
     ShowMinMaxBt: will show MinMAx button <False>
     showTitle: <True>
     
     Example:
     --------
      PanelA = JuMEG_wxPanelAB(SplitterAB,name="MAIN_PANEL_A",bg=wx.Colour(132, 126, 238),label="PDFs")
      PanelB = JuMEG_wxPanelAB(SplitterAB,name="MAIN_PANEL_B",bg=wx.Colour(140, 233, 238),label="Parameter",ShowMinMaxBt=True)
    """
    def __init__(self,parent,*kargs,**kwargs):
        super().__init__(parent,*kargs,id=wx.ID_ANY,style=wx.SUNKEN_BORDER)
        self.update_from_kwargs(**kwargs)
        self.wx_init(**kwargs)
        self.ApplyLayout()

    def SetLabel(self,v):
        try:
            self.PanelHead.label= v
        except: pass

    def SetTitle(self, v):
        self.SetLabel(v)

    def update_from_kwargs(self,**kwargs):
        """  """
        self.SetName(kwargs.get("name","PANEL_AB"))
        self.SetBackgroundColour(kwargs.get("bg",wx.Colour(132, 126, 238)))
        self._show_title = kwargs.get("ShowTitle",True)
        
    def wx_init(self, **kwargs):
        """ init WX controls """
        if self._show_title:
           self.PanelHead = JuMEG_wxTitlePanel(self,**kwargs) # label="PDFs",ShowMinMaxBt=False
        self.Panel = wx.Panel(self)

    def ApplyLayout(self):
        self.Sizer  = wx.BoxSizer(wx.VERTICAL)
        if self._show_title:
           self.Sizer.Add(self.PanelHead,0, wx.ALIGN_LEFT | wx.EXPAND | wx.ALL, 1)
        self.Sizer.Add(self.Panel,1, wx.ALIGN_LEFT | wx.EXPAND | wx.ALL, 1)
        self.SetSizer(self.Sizer)
        self.Fit()
        self.SetAutoLayout(True)