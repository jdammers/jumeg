#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 22:35:34 2017

@author: fboers
"""

import wx
from pubsub import pub

#try:
#    from agw import floatspin as FS
#except ImportError: # if it's not there locally, try the wxPython lib.

import wx.lib.agw.floatspin as FS
from   wx.adv import EditableListBox

import logging
from jumeg.base import jumeg_logger
logger = logging.getLogger('jumeg')

__version__='2020.03.12.001'

LEA=wx.LEFT|wx.EXPAND|wx.ALL

class EditableListBoxPanel(wx.Panel):
    """
     wx.adv.EditableListBox within a in Panel in order to use
     with wx.lib.agw.customtreectrl
     
    """
    def __init__(self,parent,**kwargs):
        """
        
        :param parent:
        :param label: label
        :param name : name of ctrl
        :param bg   : background colour
       
       
        Example
        -------
        from jumeg.gui.wxlib.utils.jumeg_gui_wxlib_utils_controls import EditableListBoxPanel
       
        items = [ 1,2,"A","b","C","HELLO World"]
        ctrl = EditableListBoxPanel(self,label="TEST",name="WxTEST",size=(-1,150) )
        ctrl.Value = items
   
        items = ctrl.Value
   
        """
        super().__init__(parent)
        self._etb  = None
        self._lbox = None
        
        self._wx_init(**kwargs)
        self._ApplayLayout()
        
    @property
    def EditableListBox(self): return self._etb
    @property
    def ListBox(self): return self._lbox

    @property
    def Value(self): return self._etb.Strings
    @Value.setter
    def Value(self,v):
        self.SetValue(v)
        
    def GetValue(self): return self._etb.Strings

    def SetValue(self,v):
        if isinstance( v,(list) ):
           self._etb.Strings=[str(x) for x in v]
        else:
           self._etb.Strings = [v]

    def _wx_init(self,**kwargs):
        self.SetBackgroundColour(kwargs.get("bg","GREY95"))
        style = wx.adv.EL_ALLOW_NEW | wx.adv.EL_ALLOW_EDIT | wx.adv.EL_ALLOW_DELETE | wx.adv.EL_NO_REORDER
        self._etb  = EditableListBox(self,id=wx.ID_ANY,label=kwargs.get("label",""),style=style,size=kwargs.get("size",(-1,100 )) )
        self._lbox = self._etb.GetListCtrl()
    
    def _ApplayLayout(self):
        hbox = wx.BoxSizer(wx.HORIZONTAL)
        hbox.Add(self._etb,1,LEA,5)
        self.SetAutoLayout(True)
        self.SetSizer(hbox)
        self.Fit()
        self.Layout()

class SpinCtrlScientific(wx.Panel):
    """
    min,max,initial,inc,name
    Digits,Increment
    Min,Max,Range,Value
    """
    def __init__(self,parent,**kwargs):
        super().__init__(parent)
        self.SetName("SpinCtrlScientific")
        self._SpinMantisse = None
        self._SpinExponent = None
        
        self._min = [-10000,-100]
        self._max = [ 10000, 100]
        self._inc = 1
        self._initial = 0
        
        self._wx_init(**kwargs)
        self._ApplayLayout()
   
  
    @property
    def Mantisse(self): return self._SpinMantisse
    @property
    def Exponent(self): return self._SpinExponent

    @property
    def Digits(self): return self.Mantisse.Digits
    @Digits.setter
    def Digits(self,v):
        if self.Mantisse:
           self.Mantisse.Digits=v
        
    @property
    def Increment(self):
        return self._inc

    @Increment.setter
    def Increment(self,v):
        self.SetIncrement(v)

    def GetIncrement(self):
        return self._inc

    def SetIncrement(self,v):
        self._inc = v
        if self.Mantisse:
           self.Mantisse.Increment = self._inc
        
    @property
    def initial(self):
        return self._initial

    @initial.setter
    def initial(self,v):
        if isinstance(v,(list,tuple)):
            self._initial = v
        else:
            self._initial = [v,v]
    @property
    def Value(self): return self.GetValue()
    @Value.setter
    def Value(self,v):
        self.SetValue(v)
        
    def GetValue(self):
       # https://stackoverflow.com/questions/29849445/convert-scientific-notation-to-decimals
        return float( self.Mantisse.Value * 10.0 **  self.Exponent.Value )
        
   
    def _update_ctrl(self,ctrl,v,idx):
        ctrl.Min = self._min[idx]
        ctrl.Max = self._max[idx]
    
        if v < ctrl.Min:
            ctrl.Min = abs(v) * -2.0
        if v > ctrl.Max:
            ctrl.Max = abs(v) * 2.0
    
        ctrl.Value = v

    def SetValue(self,v):
        if not self._SpinMantisse: return
        
        v = str(v)
        if v.find("e")>0: # 5.123e-11
           m,ex = v.split("e")
        else:
           m,rest= v.split(".")
           m += rest
           ex = -len(rest)
        
        digs = 1
        if m.find(".") > -1:
           _,rest = m.split(".")
           if rest:
              digs: len(rest) + 1
              
        self.Mantisse.Digits = digs
        
        self._update_ctrl(self.Mantisse,float(m),0)
        self._update_ctrl(self.Exponent,int(ex), 1)

     
    def update(self,**kwargs):
        
        self._min      = kwargs.get("min",self._min)
        self._max      = kwargs.get("max",self._max)
        self.Digits    = kwargs.get("digits",self.Digits)
        self.Increment = kwargs.get("inc",self._inc)
        
        v = kwargs.get("value",0.0)
        self.Value = kwargs.get("Value",v)
        
    
    def _wx_init(self,**kwargs):
        """
        < spin ctrl double> <e> < spin ctrl double >
        :param kwargs:
        :return:
        """
        self.SetName(kwargs.get("name",self.GetName()))
        self.Increment = kwargs.get("inc",self._inc)
        
        style=kwargs.get("style",wx.SP_ARROW_KEYS)
        self.SetBackgroundColour(kwargs.get("bg","GREY90"))
       #---5.123 e-11  mantisse 5.123
        self._SpinMantisse = wx.SpinCtrlDouble(self,inc=self._inc,name="mantisse",style=style)
       #--- e
        self._txt= wx.StaticText(self,label="exp") #,style=wx.ALIGN_CENTRE_HORIZONTAL)
       #---5.123 e-11  exponent -11
        style = wx.SP_ARROW_KEYS | wx.SP_WRAP | wx.TE_PROCESS_ENTER | wx.ALIGN_RIGHT
        self._SpinExponent = wx.SpinCtrl(self,wx.ID_ANY,name="exponent",style=style)
       #---
        self.update(**kwargs)
        
        
    def _ApplayLayout(self):
        hbox = wx.BoxSizer(wx.HORIZONTAL)
        hbox.Add(self._SpinMantisse,1,LEA,2)
        hbox.Add(self._txt,0,wx.LEFT|wx.RIGHT,5)
        hbox.Add(self._SpinExponent,0,wx.LEFT|wx.ALL,2)
    
        self.SetAutoLayout(True)
        self.SetSizer(hbox)
        self.Fit()
        self.Layout()
        

class CheckBoxPopup(wx.PopupTransientWindow):
    """
    controls & flags e.g.: verbose,debug ...
    
    :param parent:
    :param kwargs:
       position: fixed position
       popup: do popup window
    
    Example:
    --------
     ctrls =[ ["verbose",self._verbose,self.ClickOnPopupCtrl],
              ["debug",  self._debug,  self.ClickOnPopupCtrl]
            ]
     win = MiscPopup(self,ctrls=ctrls)

    """
    
    def __init__(self,parent,**kwargs):
        """

        :param parent:
        :param kwargs:
        """
        wx.PopupTransientWindow.__init__(self,parent)
        mpos = wx.GetMousePosition()
      
        panel = wx.Panel(self)
        panel.SetBackgroundColour(kwargs.get("bg","GREY90"))
        sizer = wx.BoxSizer(wx.VERTICAL)
        
        for ctrl in kwargs.get("ctrls"):
            if len(ctrl):
                ckb = wx.CheckBox(self,wx.NewId(),ctrl[0])
                ckb.SetName(ctrl[0])
                ckb.SetValue(ctrl[1])
                ckb.Bind(wx.EVT_CHECKBOX,ctrl[2])
                sizer.Add(ckb,0,LEA,5)
            else:
                stl = wx.StaticLine(self,style=wx.LI_HORIZONTAL)
                stl.SetBackgroundColour("GREY60")
                sizer.Add(stl,0,LEA,5)
        
        panel.SetSizer(sizer)
        sizer.Fit(panel)
        sizer.Fit(self)
        self.Layout()

        if kwargs.get("position",None):
           pos = kwargs.get("position")
        else:
            pos = mpos
            pos[0] -= panel.GetSize()[0]
            pos[1] += 24

        self.Position(pos,(-1,-1))
        
        if kwargs.get("popup",False):
            self.Popup()


class JuMEG_wxSTXTBTCtrl(wx.Panel):
    """
    TextCtrl and Bitmap Button
    send command to show ListBox under TextCtrl
    :param   name: name of ctrl window
    :param   label: text in text ctrl
    :param   bg: background colour
    :param   textlength: number of chars to calculate textfield size
    :param   cmd: command to bind with buton press
    
    Example:
    --------
    from jumeg.gui.wxlib.utils.jumeg_gui_wxlib_utils_controls import JuMEG_wxPopUpListBox,JuMEG_wxSTXTBTCtrl
    
    
    data=["A","B","c"]
    txtlen=10 # 10 chars
    ctrl = JuMEG_wxSTXTBTCtrl(self,name="TEST",label=data[0],cmd=self.ClickOnShowList,textlength=txtlen)
    
    def ClickOnShowList(self,evt):
        try:
            obj = evt.GetEventObject()
        except:
            obj=evt
           
       #--- Lbox
        win = JuMEG_wxPopUpListBox(self,choices=data,caller=obj)
        win.Popup()
        
    """
    __slots__=["_STXT","_BT","_CMD"]
    def __init__(self,parent,**kwargs):
        super().__init__(parent)
        
        self._TXT = None
        self._BT   = None
        
        self._wx_init(**kwargs)
        self._ApplyLayout()
    
    @property
    def Text(self): return  self._TXT
    
    def _wx_init(self,**kwargs):
        style     = kwargs.get("style",wx.TE_RIGHT|wx.TE_READONLY)
        self._CMD = kwargs.get("cmd",None)
        self.SetName( kwargs.get("name","JuMEG_WX_STXT_BT"))
        
        txt = wx.TextCtrl(self,-1,style=style)
        txt.AppendText( str( kwargs.get("label","") ) )
        txt.SetBackgroundColour(wx.WHITE)
        txt.SetName( "TXT."+self.GetName() )
        txtlen = kwargs.get("textlength",10)
        
        sz = txt.GetSizeFromTextSize( txt.GetTextExtent("W" * txtlen ))
        txt.SetInitialSize(sz)
        self._TXT= txt
        
        bmp = wx.ArtProvider.GetBitmap(wx.ART_GO_DOWN,wx.ART_BUTTON)
        w=sz[1]
        self._BT = wx.BitmapButton( self, -1, bmp, size=(w,w),style=wx.BU_LEFT )
        self._BT.SetBackgroundColour( kwargs.get("bg","GREY80"))
        self._BT.SetName("BT." + self.GetName())
        self.Bind(wx.EVT_BUTTON,self.ClickOnButton)
        
    def _ApplyLayout(self):
        hbox = wx.BoxSizer(wx.HORIZONTAL)
        hbox.Add(self._TXT,1,wx.ALL,2)
        hbox.Add(self._BT,0,wx.ALL,1)

        self.SetAutoLayout(True)
        self.SetSizer(hbox)
        self.Fit()
        self.Layout()
        
    def SetValue(self,v):
        #self._STXT.Clear()
        self._STXT.SetValue( str(v))
    def GetValue(self,v):
        return self._STXT.GetValue()

    def ClickOnButton(self,evt):
        if self._CMD:
           self._CMD(self._TXT)
           return
        evt.Skip()

    
class JuMEG_wxPopUpListBox(wx.PopupTransientWindow):
    """
     :param parent:
     :param **kwargs:
            bg            : background colour
            size,position : wx parameter
            choices,style : listbox parameter
            caller        : calling ctrl
            cmd           : function to execute keys:
                            cmd( obj= caller,index= index of listbox selection)
      
    Example: show triggered by EVT
    from jumeg.gui.wxlib.utils.jumeg_gui_wxlib_utils_controls import JuMEG_wxPopUpListBox
    --------
    def ClickOnShowList(self,evt):
        obj = evt.GetEventObject()
        pos = obj.GetScreenPosition()
 
        win = JuMEG_wxPopUpCkBox(self,choices=data,position=None,size=None,caller=calling obj)
      
       #--- or set pos
        win.Position( wx.GetMousePosition(),(-1,-1))
        win.Position(pos,(-1,-1))
        
        win.Popup()
      """
    def __init__(self, parent, **kwargs):
        """
        
        :param parent:
        :param kwargs:
        bg,size,position,choices,style
        caller: calling ctrl
        """
        wx.PopupTransientWindow.__init__(self, parent)
        self._lb     = None
        self._caller = None
        self._cmd    = None
        self._wx_init(**kwargs)
     
    @property
    def ListBox(self):
        return self._lb
    
    def _wx_init(self,**kwargs):
        
        self._caller = kwargs.get("caller",None)
        self._cmd    = kwargs.get("cmd",None)
        try:
           sz    = self._caller.GetSize()
           sz[1] = 100
           self.SetSize(kwargs.get("size",sz))
        except:
           pass
       
        lb_idx = None
        choices = kwargs.get("choices",[])
        
        if self._caller:
           sz  = self._caller.GetSize()
           pos = self._caller.GetScreenPosition()
           pos[1] += sz[1]  # show LB at bottom of caller
           
           v = self._caller.GetValue()
           if v in choices:
              lb_idx = choices.index( v )
           
        if "position" in kwargs:
           pos = kwargs.get("position",wx.DefaultPosition)
        self.SetPosition(pos)

       #--- ListBox
        self._lb = wx.ListBox(self,-1,choices=choices,style=kwargs.get("style",wx.LB_SINGLE)) # |wx.TE_RIGHT))
        self._lb.SetSize(self.GetClientSize())
        if lb_idx != None:
           self._lb.SetSelection(lb_idx)
        
        self.Bind(wx.EVT_LISTBOX,self.ClickOnListBox)
        self.Bind(wx.EVT_LISTBOX_DCLICK,self.DoubleClickOnListBox)
        
    def ClickOnListBox(self,evt):
        #obj = evt.GetEventObject()
        #logger.info("OnListBox: %s\n" % obj)
        #logger.info('Selected: %s\n' % obj.GetString(evt.GetInt()))
        idx = self._lb.GetSelection()
        v   = self._lb.GetString(idx)
        try:
           self._caller.SetValue(v) # get index
        except:
           self._caller.SetLabel(v)
        if self._cmd:
           self._cmd(obj=self._caller,value=v,index=idx)

    def DoubleClickOnListBox(self,evt):
        self.Hide()
        self.Destroy()
        
    def SetValue(self,v):
        lb_idx = self._lb.GetItems().index(str(v))
        self._lb.SetSelection(lb_idx)
        
    def _ApplyLayout(self):

        vbox = wx.BoxSizer(wx.VERTICAL)
        vbox.Add(self._lb,1,wx.ALL,5)
        self.SetAutoLayout(True)
        self.SetSizer(vbox)
        self.Fit()
        self.Layout()
        self._lb.SetFocus()
        
class JuMEG_wxPopUpCkBox(wx.PopupTransientWindow):
   """
   
   Example:
   --------
   from jumeg.gui.wxlib.utils.jumeg_gui_wxlib_utils_controls import JuMEG_wxPopUpCkBox
   
   
   """
   def __init__(self,parent,**kwargs):
       super().__init__(parent)
       self._choices = []
       self._gap  = 4
     
       self._init(**kwargs)
 
   def SetSelection(self,idx,status=True):
       """
       
       :param idx: idx or list of idx
       :param status: <True>
       :return:
       """
       if not isinstance(idx,(list)):
          self._clb.Check(idx)
       else:
          for i in idx:
              self._clb.Check(i,status)
   
   def clear(self):
       self._clb.clear()
   
   def update_from_kwargs(self,**kwargs):
       self._choices = kwargs.get("choices",self._choices)
       self.SetName( kwargs.get("name",self.GetName()) )
       
   def _init(self,**kwargs):
       self.update_from_kwargs(**kwargs)
       self._wx_init()
       self._ApplyLayout()
       
   def _wx_init(self):
       self._clb = wx.CheckListBox(self,-1,choices=self._choices)
       self.Bind(wx.EVT_LISTBOX,self.ClickOnListBox,self._clb)
       
   def ClickOnListBox(self, evt):
       #print("TEST ClickOnListBox")
       #print(self._clb.GetSelections())
       evt.Skip()
       #pass
       # idx = evt.GetSelection()
       #label = self._lb.GetString(index)
       
   def _ApplyLayout(self):
       LEA = wx.LEFT | wx.EXPAND | wx.ALL
       vbox = wx.BoxSizer(wx.VERTICAL)
       vbox.Add(self._clb,1,LEA,self._gap)
       self.SetSizer(vbox)
       self.SetAutoLayout(True)
       self.Fit()
       self.Update()
       self.Refresh()


class JuMEG_wxMultiChoiceDialog(wx.Dialog):
    """
    :param: parent
    :param: message
    :param: caption
    :param: choices=[]
    
    code modified from:
    https://stackoverflow.com/questions/9361939/creating-a-custom-dialog-that-allows-me-to-select-all-files-at-once-or-a-single
    modify <update> with customize ctrls
    
    Example:
    ---------
    l = ['AND', 'OR', 'XOR', 'NOT']
    app = wx.PySimpleApp()
    dlg = JuMEG_wxMultiChoiceDialog(None, 'Choose as many as you wish', 'MCD Title', choices = l)
    if dlg.ShowModal() == wx.ID_OK:
       result = dlg.GetSelections()
       wx.MessageBox(str(result) + ' were chosen')
    dlg.Destroy()
    
    """
    def __init__(self, parent, message, caption, choices=[],**kwargs):
        super().__init__(parent, -1,size=(600,480),style=wx.DEFAULT_DIALOG_STYLE|wx.RESIZE_BORDER|wx.CLOSE_BOX|wx.STAY_ON_TOP)
        self._wx_init(message, caption, choices)
        self.update(**kwargs)
        self._ApplyLayout()
   
    @property
    def CtrlBox(self): return self._ctrlbox
    
    def _wx_init(self,message, caption, choices):
        self.SetTitle(caption)
        self.Sizer = wx.BoxSizer(wx.VERTICAL)
        self.message = wx.StaticText(self, -1, message)
        self.clb = wx.CheckListBox(self, -1,style=wx.LB_EXTENDED,choices = choices)
        
        self.tgbt = wx.ToggleButton(self, -1,label='De/Select All',name="BT.DESELECT_ALL")
        self.tgbt.SetValue(False)
        
      #--- horizontal box to add user ctrls used in update
        self._ctrlbox = wx.BoxSizer(wx.HORIZONTAL)
        self._ctrlbox.Add(self.tgbt,         0,wx.LEFT|wx.ALL|wx.EXPAND, 5)
     
        self.btns = self.CreateSeparatedButtonSizer(wx.OK | wx.CANCEL)
        self.Bind(wx.EVT_TOGGLEBUTTON,self.ClickOnCtrl)
    
    def update(self,**kwargs):
        """
        overwrite
        add user ctrls into self.CtrlBox (hbox sizer)
        
        :param kwargs:
        :return:
        """
        pass
    
    def GetSelections(self):
        return self.clb.GetCheckedItems() # GetChecked()
  
    def ClickOnCtrl(self,evt):
        obj = evt.GetEventObject()
        if obj.GetName() == "BT.DESELECT_ALL":
           for i in range(self.clb.GetCount()):
               self.clb.Check(i, self.tgbt.GetValue())
    
    def _ApplyLayout(self):
        self.Sizer.Add(self.message, 0, wx.ALL | wx.EXPAND, 5)
        self.Sizer.Add(self.clb, 1, wx.ALL | wx.EXPAND, 5)
        self.Sizer.Add(self._ctrlbox, 0, wx.ALL | wx.EXPAND, 5)
        self.Sizer.Add(self.btns, 0, wx.LEFT|wx.ALL | wx.EXPAND, 5 )
        self.SetSizer(self.Sizer)
       

class JuMEG_wxSplitterWindow(wx.SplitterWindow):
    def __init__(self, parent,**kwargs):
        """
        JuMEG wx.SplitterWindow with method to flip and min/max position

        Parameters
        ----------
         parent        : wx.Widged
         listener      : name of pubsub listener to subscribe <SPLITTER>
         flip_position : key word for pubsub listener to flip position <FLIP_POSITION>
         split_min_max : key word for pubsub listener to reise windows <SPLIT_MIN_MAX>
        
         Example:
         --------
           import wx
           from jumeg.gui.wxlib.utils.jumeg_gui_wxlib_utils_controls import JuMEG_wxSplitterWindow
        
           class Test( wx.Panel):
                def __init__(self,parent,**kwargs):
                    super().__init__(parent,**kwargs)
                    self.pnl_left = None
                    self.pnl_right= None
                    self._wx_init(**kwargs)
                    self._ApplyLayout()
                
                def _wx_init(self,**kwargs):
                    self._splitter = JuMEG_wxSplitterWindow(self,label="TEST",name=self.GetName() + ".SPLITTER")
                    self._pnl_left = wx.Panel(self._splitter,name="LEFT")
                    self._pnl_left.SetBackgroundColour("green")
                    self._pnl_right= wx.Panel(self._splitter,name="RIGHT")
                    self._pnl_right.SetBackgroundColour("blue")
               
                def _ApplyLayout(self):
                    LEA = wx.LEFT|wx.EXPAND|wx.ALL
                    vbox = wx.BoxSizer(wx.VERTICAL)
                    self._splitter.SplitVertically(self._pnl_left,self._pnl_right)
                    self._splitter.SetMinimumPaneSize(150)
                    self._splitter.SetSashGravity(1.0)
                    vbox.Add(self._splitter, 1, wx.ALIGN_CENTER | wx.EXPAND | wx.ALL, 5)
                   
                    self.SetAutoLayout(True)
                    self.SetSizer(vbox)
                    self.Fit()
                    self.Layout()
                    
           if __name__ == '__main__':
              app = wx.App()
              frame = wx.Frame(None, -1, "Test", size=(640, 480))
              win = Test(frame)
              frame.Show(True)
              app.MainLoop()
        """
        super().__init__(parent=parent, id=wx.ID_ANY,style=wx.SP_3DSASH|wx.SP_3DBORDER|wx.SUNKEN_BORDER)
        self._flip_position = "FLIP_POSITION"
        self._split_min_max = "SPLIT_MIN_MAX"
        self.__split_factor_list  = [-1.0,0.5,1.0]
        self.__split_position_idx = 1
        self.SetSashGravity(1.0) # expamd top/left
        self.SetMinimumPaneSize(100)
        self.update(**kwargs)
   #---
    def _update_from_kwargs(self,**kwargs):
        self.SetName(kwargs.get("name","SPLITTER").replace(" ", "_").upper())
        self.listener      = kwargs.get("listener",self.GetParent().GetName())
        self._flip_position= kwargs.get("flip_position", self._flip_position).replace(" ", "_").upper()
        self._split_min_max= kwargs.get("split_min_max", self._split_min_max).replace(" ", "_").upper()
   #---
    def _init_pubsub(self,**kwargs):
        #print("init pubsub " + self.GetName().upper()+"."+self._split_min_max)
        pub.subscribe(self.UpdateSplitPosition,self.listener+"."+self._split_min_max)
        
    def update(self,**kwargs):
        self._update_from_kwargs(**kwargs)
        self._init_pubsub(**kwargs)
        self._PanelA = None
        self._PanelB = None
        self.__v     = None
        self.__h     = None

    def UpdateSplitPosition(self,name=None,size=None):
        """
        update split position change window size to min,50%,max
        Parameter
        ---------
         name : name of the window eg: name of window 1 or 2 <None>
         size: wx.GetSize() to set MinimumPaneSize <None>
        """
        if not size: return
        if name not in [self.GetWindow1().GetName(), self.GetWindow2().GetName()]: return

        max_pos = 0

        if self.GetSplitMode() == wx.SPLIT_VERTICAL:
           pos     = size[0]
           max_pos = self.GetSize()[0]
        else:
           pos = size[1]
           max_pos = self.GetSize()[1]
        self.SetMinimumPaneSize(abs(pos) + 2)

        idx = self.__split_factor_list[self.__split_position_idx]
        self.__split_position_idx+=1
        self.__split_position_idx = self.__split_position_idx % len(self.__split_factor_list)

        if idx == -1:
           self.SetSashPosition(max_pos, redraw=True)
        elif idx == 0.5:
           self.SetSashPosition(max_pos * 0.5,redraw=True)
        else:
           self.SetSashPosition(1,redraw=True)

    def FlipPosition(self,value=wx.SPLIT_VERTICAL):
        """
        flips panels inside splitter

        Parameter
        ---------
        value: vertical,horizontal <vertical>
        """
        #--- todo check  keep sahposition
        # wx.SplitterWindow.GetSplitMode
        #spos = self.GetSize()[-1] - self.GetSashPosition()
        
        if not value: return
        w1 = self.GetWindow1()
        w2 = self.GetWindow2()
        self.Unsplit()

       # --- flip windows
        if w1 and w2:
           if not self.__v:
                  self.__v = [w1, w2]
           if value == wx.SPLIT_VERTICAL:
              if self.GetSplitMode() == wx.SPLIT_VERTICAL:
                  self.__v.reverse()
              self.SplitVertically(window1=self.__v[0], window2=self.__v[1],sashPosition=self.__v[0].GetSize()[-1] * 0.5)
           else:
              if self.GetSplitMode() == wx.SPLIT_HORIZONTAL:
                 self.__v.reverse()
              self.SplitHorizontally(window1=self.__v[0],window2=self.__v[1],sashPosition=self.__v[0].GetSize()[0] * 0.5)

           self.SetSashGravity(1.0)
           self.Fit()
           self.GetParent().Layout()


class JuMEG_wxCMDButtons(wx.Panel):
      ''' panel with Close,Cancel,Apply Buttons'''

      def __init__(self, parent,**kwargs):
          """

          :param parent:
          :param **kwargs:
           bg   : wx.Colour([230,230,230])
           bg_bt: wx.Colour([180,200,200])
           prefix:"CMD_BUTTONS"
           ShowClose:True,ShowCancel:True,ShowApply:True
           verbose: False, debug: False
                """
          super(JuMEG_wxCMDButtons,self).__init__(parent=parent, id=wx.ID_ANY,style=wx.SUNKEN_BORDER)
          self._init(**kwargs)

      @property
      def CallClickOnApply(self):
          return self.prefix.upper() + ".BT.APPLY".upper()
      @property
      def CallClickOnCancel(self):
          return self.prefix.upper() + ".BT.CANCEL".upper()

      def SetVerbose(self,value=False):
          self.verbose = value

      def update_from_kwargs(self,**kwargs):
          self.verbose = kwargs.get("verbose", False)
          self.debug   = kwargs.get("debug", False)
          self.prefix  = kwargs.get("prefix","CMD_BUTTONS")
          self.ShowClose  = kwargs.get("ShowClose" ,True)
          self.ShowCancel = kwargs.get("ShowCancel",True)
          self.ShowApply  = kwargs.get("ShowAppply",True)
          self._bg        = kwargs.get("bg",   wx.Colour([230,230,230]))
          self._bg_bt     = kwargs.get("bg_bt",wx.Colour([180,200,200]))

      def init_pubsub(self):
        """"
        init pubsub call
        """
        pub.subscribe(self.SetVerbose,'MAIN_FRAME.VERBOSE')

      def wx_init(self,**kwargs):
          """ init WX controls """
          #self.SetBackgroundColour(wx.Colour([200,200,200] ))
        #--- BTs
          ctrls = []
          if self.ShowClose:
             ctrls.append(["CLOSE" ,self.prefix+".BT.CLOSE", wx.ALIGN_LEFT,"Close program",  None])
          if self.ShowCancel:
             ctrls.append(["CANCEL",self.prefix+".BT.CANCEL",wx.ALIGN_LEFT,"Cancel progress",None])
          if self.ShowApply:
             ctrls.append(["APPLY",self.prefix+".BT.APPLY",  wx.ALIGN_RIGHT,"Apply command", None])
          self._PNL_BTN = JuMEG_wxControlButtonPanel(self,label=None,control_list=ctrls)
          self.SetBackgroundColour(self._bg)
          self._PNL_BTN.SetBackgroundColour(self._bg_bt)

          self.Bind(wx.EVT_BUTTON, self.ClickOnCtrl)

      def update(self, **kwargs):
          pass

      def _init(self,**kwargs):
          """" init """
          self.update_from_kwargs(**kwargs)
          self.wx_init()
          self.init_pubsub()
          self.update(**kwargs)
          self._ApplyLayout()
     #---
      def ClickOnCtrl(self,evt):
          obj=evt.GetEventObject()
          if obj.GetName().upper().startswith("BT.CLOSE"):
             pub.sendMessage("MAIN_FRAME.CLICK_ON_CLOSE",evt=evt)
          else:
             evt.Skip()

      def _ApplyLayout(self):
          """" default Layout Framework """
          ds=1
          self.Sizer = wx.BoxSizer(wx.HORIZONTAL)
          self.Sizer.Add(self._PNL_BTN, 1, wx.ALIGN_RIGHT | wx.EXPAND | wx.ALL, ds)
          self.SetSizer(self.Sizer)
          self.Fit()
          self.SetAutoLayout(1)
          self.GetParent().Layout()


class JuMEG_wxControlUtils(object):
    def __init__(self, parent):
        super (JuMEG_wxControlUtils,self).__init__()
        
        # self._init()

    #def __init(self):
    #    self.separator="."
    # ---
    def get_button_txtctrl(self, obj):
        """ finds the textctr related to the button event
        Parameters:
        -----------
        wx control button

        Results:
        ---------
        wx textctrl obj

        """
        return self.FindWindowByName(obj.GetName().replace("BT","TXT",1))
     
    def FindObjIndexByObjID(self,id,objs=None):
        """
        find ctrl by ID from objs list
        :param objs: the <objs> to serach
        :return: index
        """
        
        for idx in range(len(objs)):
            try:
                if id == objs[idx].GetId():
                   return idx
            except:
                pass
        
   # ---
    def FindControlByObjName(self, obj, prefix="", sep=None):
        """
        finds a obj ctrl by an other object and prefix to change

        Parameters:
        -----------
        obj to get the name from
        prefix: string to replace the input obj.Name startswith <"">

        Results:
        ---------
        wx ctrl obj

        Example:
        --------
        input obj with name: BT_PATH
        prefix: "COMBO"
        return ComboBox obj with name "COMBO_PATH"

        FindControlByObjName(obj_bt,"COMBO")

        """
        if not sep:
           sep = self.separator
        s = obj.GetName().split(sep)[0]
        return self.FindWindowByName(obj.GetName().replace(s, prefix, 1))


    def EnableDisableCtrlsByName( self,names,status,prefix=None ):
        """
        enabling or disabling wx controls like buttons or comboboxes
        :param names: list or string of Ctrl.name
        :param status: enable or disabled True/False
        :param prefix: list  <None>
               e.g. ["BT.","CB."] to find "BT.UPDATE"
        :return:
        """
        if not isinstance(names,(list)):
           names=[names]
        for n1 in names:
            try:
                if isinstance(prefix,(list)):
                   for prf in prefix:
                       n0 = prf if prf else ""
                       self.FindWindowByName(n0+n1).Enable(status)
                else:
                    n0 = prefix if prefix else ""
                    self.FindWindowByName(n0 + n1).Enable(status)

            except: pass
   # ---
    def UpdateComBox(self, cb, vlist, sort=True):
        """
        update wx.ComboBox: clear,SetItems,SetValue first element in list
        remove double items and sort list

        Parameters:
        -----------
        combobox obj
        list to insert
        sort : will sort list <True>
        """
        # if not cb: return
        if not cb: return
        cb.Enable(False)
        cb.Clear()
        if vlist:
            if isinstance(vlist, (list)):
                # avoid repetitios in list, make list from set
                if sort:
                    cb.SetItems(sorted(list(set(vlist))))
                else:
                    cb.SetItems(list(set(vlist)))
            else:
                cb.SetItems([vlist])
            cb.SetSelection(0)
            cb.SetValue(vlist[0])
        else:
            cb.Append('')
            cb.SetValue('')
            cb.Update()
        cb.Enable(True)

class JuMEG_wxControlBase(wx.Panel,JuMEG_wxControlUtils):
    """
    generate button, text,spin/floatspin combobox controls
     
    Parameters:
    ---------- 
     parent widged
     control_list: list of parameter and definition for widged controls
     label       : headline text
     drawline    : draw a line on top as separator <True>
     bg          : background color <"grey90">
     boxsizer    : [wx.VERTICAL,wx.HORIZONTAL] <wx.VERTICAL>
     separator   : "."
     set_ctrl_prefix : adds prefix [BT,COMBO,TXT,...] to  ctrl name <True>

     FlexGridSizer parameter:
      AddGrowableCol : int or list    <None> 
      AddGrowableRow : int or list    <None>
      -> col[i] or row[i] will grow/expand  use full to display long textfields
      
      cols  : number of cols in FGS <4>, may will be ignored due to task
      rows  : number of rows in FGS <1>, may will be ignored due to task
     
        
     Results:
     --------
     wx.Panel with sub controls group by wx.FlexGridSizer    

     Example:   
     --------
       Button  : ( control type,name,label,help,callback) 
                 ("BT","START","Start","press start to start",ClickOnButtton)
       
       CheckBox: ( control type,name,label,value,help.callback) 
                 ("CK","VERBOSE","verbose",True,'tell me more',ClickOnCheckBox)
       
       ComboBox: ( control type,name,value,choices,help,callback) 
                 ("COMBO","COLOR","red",['red','green'],'select a color',ClickOnComboBox)
        
       TxtCtrl : ( control type,name,label,help,callback) 
                 ("TXT","INFO","info","read carefully",ClickOnTxt)
       
       StaticText : ( control type,name,label,help,callback) 
                 ("STXT","INFO","info","read carefully",NoneClickOnTxt)
      
           
       SpinCtrl     : ( control type,label,[min,max,increment],value,help,callback) 
                      ("SP","Xpos",[-500,500,10],11,"change x position",ClickOnSpin)
                 
       FloatSpinCtrl: ( control type,label,[min,max,increment],value,help,callback) 
                      ("SPF","Xpos",[-500.0,500.0,10.0],11,"change x position",ClickOnFloatSpin)
                      
      
     in sub class __init__ call to add controls 
     self.add_controls(vbox)  
     
    """
    def __init__(self, parent,control_list=None,label='TEST',drawline=True,bg="grey90",boxsizer=wx.VERTICAL,init=True,
                 AddGrowableCol=None,AddGrowableRow=None,cols=4,rows=1,separator = ".",set_ctrl_prefix=True):
        """
        
        :param parent:
        :param control_list:
        :param label:
        :param drawline:
        :param bg:
        :param boxsizer:
        :param AddGrowableCol:
        :param AddGrowableRow:
        :param cols:
        :param rows:
        :param separator:
        :param set_ctrl_prefix:
        """
        super ().__init__(parent,id=wx.ID_ANY,style=wx.SUNKEN_BORDER)
        self.box = wx.BoxSizer(boxsizer)
        self.SetBackgroundColour(bg)
        self.separator = separator
        self.set_ctrl_prefix = set_ctrl_prefix

        self._obj   = None   
        self._rows  = rows
        self._cols  = cols
        self.label  = label
        self.drawline = drawline
        
        self.gap   = 4
        self.gap_minmax = 1
        self.vgap  = 4
        self.hgap  = 4
        self.LE    = wx.LEFT |wx.EXPAND
        self.RE    = wx.RIGHT|wx.EXPAND
        self.GS    = None
        
        self.list_of_controls = control_list
        self.AddGrowableCol   = AddGrowableCol
        self.AddGrowableRow   = AddGrowableRow
        
        self.boxsizer = wx.BoxSizer(boxsizer)
        
        self.SetBackgroundColour(bg)

       #--- in sub class call to add controls 
       # vbox = self.add_controls(vbox) 
             
    @property
    def cols(self): return self._cols
    @cols.setter
    def cols(self,v): self._cols=v
    
    @property
    def rows(self): return self._rows
    @rows.setter
    def rows(self,v): self._rows=v
     
    @property
    def controls(self): return self._obj
    @property
    def objects(self) : return self._obj

    def init_growable(self):
        """ expand / grow col row """
        if self.AddGrowableCol:
           if isinstance(self.AddGrowableCol,(list,tuple)):
              for c in self.AddGrowableCol:
                  if c < self.cols: self.GS.AddGrowableCol(c)
           else:
              if self.AddGrowableCol < self.cols: self.GS.AddGrowableCol(self.AddGrowableCol)

       #--- epand / grow row
        if self.AddGrowableRow:
           if isinstance(self.AddGrowableRow,(list,tuple)):
              for r in self.AddGrowableRow:
                  if r < self.rows:
                     self.FSG.AddGrowableRow(r)
           else:
              if self.AddGrowableRow < self.rows: self.GS.AddGrowableRow(self.AddGrowableRow)
    
    def _gs_add_ctrl(self,d):
        self.GS.Add(d[1],0,self.LE,self.gap)
        
    def _gs_add_BitmapButton(self,d):
        """ adds  BitmapButton
         ctrls.append( ["BTBMP","MY_COLOR",bmp,(w,w),wx.NO_BORDER|wx.BU_NOTEXT,wx.BLUE ] )
         type,name,bitmap,size,style,color
        """
        self._obj.append( wx.BitmapButton(self, -1, bitmap=d[2], size=d[3], style=d[4]) )
        if len(d)>5:
           self._obj[-1].SetBackgroundColour(d[5])
        
        # self.GS.Add(self._obj[-1],0,self.LE,self.gap)
        self._gs_add_init_last_obj(d,evt=wx.EVT_BUTTON)
        
        #self._gs_add_init_last_obj(d)
    
    def _gs_add_empty_cell(self):
        """ adds  empty cell / space """
        self.GS.Add( (0,0),0,self.LE,self.gap)          
     
    def _gs_add_add_first_text_info(self,d):
        """ add text in front of control """
        if len(d) >1:
           self.GS.Add( wx.StaticText(self,-1,label=d[1],style=wx.ALIGN_LEFT),0,wx.LEFT,self.gap)
   
    def _gs_add_add_last_obj(self):
        """ helper fct: add last obj to add """
        self.GS.Add(self._obj[-1],0,self.LE,self.gap)        
    
    def _gs_add_init_last_obj(self,d,evt=None):
        """ helper fct: init last obj to add """
        if self.set_ctrl_prefix:
           n = (d[0]+self.separator+d[1]).replace(" ","-").upper()
        else:
           n = d[1].replace(" ", "-").upper()
        self._obj[-1].SetName( n )
        
        if isinstance(d[-2],str):
           self._obj[-1].SetToolTip(wx.ToolTip(d[-2]))
        if evt:
           if callable(d[-1]) : self._obj[-1].Bind(evt,d[-1])
        self._gs_add_add_last_obj()
    
    def _gs_add_Button(self,d):
        """ add Button"""
        try:
           style = d[5]
        except:
           style=wx.BU_EXACTFIT
        self._obj.append(wx.Button(self,wx.ID_ANY,label=d[2],style=style))
        self._gs_add_init_last_obj(d,evt=wx.EVT_BUTTON)

    def _gs_add_FlatButton(self,d):
        """ add FlatButton"""
        try:
           style= d[5]
        except:
            style=wx.BU_EXACTFIT|wx.NO_BORDER

        self._obj.append(wx.Button(self,wx.ID_ANY,label=d[2],style=style))
        self._gs_add_init_last_obj(d,evt=wx.EVT_BUTTON)

    def _gs_add_CheckBox(self,d):
        """ add wx.CheckBox"""
        self._obj.append( wx.CheckBox(self,wx.ID_ANY,label=d[2] ) )  
        self._obj[-1].SetValue(d[3])
        self._gs_add_init_last_obj(d,evt=wx.EVT_CHECKBOX)             
        
    def _gs_add_ComboBox(self,d):
        """ add wx.ComboBox """
        #if len(d)>4:
        #   print(type(d[4]))
          # if isinstance( d[4],(list,tuple) ):
        #   s = d[4]
        #else:
        #   s=(-1,-1)
        
        # self._obj.append(wx.ComboBox(self,wx.ID_ANY,choices=d[3],size=s,style=wx.CB_READONLY))
        cb = wx.ComboBox(self,wx.ID_ANY,choices=d[3],style=wx.CB_READONLY)
        cb.SetValue(d[2])
        if d[3]:
           cnts = len( max(d[3],key=len))  # fix text length for max choices
           sz = cb.GetSizeFromTextSize(cb.GetTextExtent("W" * cnts))
           cb.SetInitialSize(sz)
        
        self._obj.append(cb)
        self._gs_add_init_last_obj(d,evt=wx.EVT_COMBOBOX)
        
    def _gs_add_TextCtrl(self,d):
        """ add wx.TextCrtl """
        self._obj.append( wx.TextCtrl(self,wx.ID_ANY ))   
        self._obj[-1].SetValue(d[2])
        self._gs_add_init_last_obj(d,evt=wx.EVT_TEXT)
            
    def _gs_add_StaticText(self,d):
        """ add wx.StaticText """
        self._obj.append(wx.StaticText(self,wx.ID_ANY,style=wx.ALIGN_LEFT|wx.ST_ELLIPSIZE_MIDDLE))
        self._obj[-1].SetLabel(d[2])
        self._gs_add_init_last_obj(d)
        
    def _gs_add_min_max_button(self,l,n):  
        """ add button MIN or MAX set to min or max in range of SpinCtrl"""
        self._obj.append( wx.Button(self,wx.ID_ANY,label=l,style=wx.BU_EXACTFIT,name=n) )
        self._obj[-1].Bind(wx.EVT_BUTTON,self.OnClickMinMax)
        self.GS.Add(self._obj[-1],0,wx.LEFT,self.gap_minmax)
            
    def _gs_add_SpinCtrl(self,d):
        """ add wx.SpinCtrl """
        self._obj.append( wx.SpinCtrl(self,wx.ID_ANY,style=wx.SP_ARROW_KEYS|wx.SP_WRAP|wx.TE_PROCESS_ENTER|wx.ALIGN_RIGHT))
        self._obj[-1].SetLabel(d[1])
        self._obj[-1].SetName("SP"+self.separator+d[1].upper()) if self.set_ctrl_prefix else self._obj[-1].SetName(d[1].upper())
        self._obj[-1].SetToolTip("Min: " + str(d[2][0]) +"  Max: " + str(d[2][1]) )
        self._obj[-1].SetRange(d[2][0],d[2][1])
        self._obj[-1].SetValue(d[3])
        self._obj[-1].Bind(wx.EVT_SPINCTRL, d[-1])  #5

       #self.GS.Add(self._obj[-1],0,wx.LEFT|wx.EXPAND,self.gap)
        self.GS.Add(self._obj[-1],0,wx.LEFT,self.gap)

    def _gs_add_FloatSpin(self,d):
        """ add wx.FloatSpin """
        self._obj.append( FS.FloatSpin(self,wx.ID_ANY,min_val=d[2][0],max_val=d[2][1],increment=d[2][2],value=1.0,agwStyle=FS.FS_RIGHT) )   
        self._obj[-1].SetLabel(d[1])
        self._obj[-1].SetName("FSP"+self.seperator+d[1].upper()) if self.set_ctrl_prefix else self._obj[-1].SetName(d[1].upper())
        self._obj[-1].SetFormat("%f")
        self._obj[-1].SetDigits(3)
        self._obj[-1].SetValue(d[3])
        self._obj[-1].Bind(wx.EVT_SPINCTRL, d[-1])  #6 
        #self.GS.Add(self._obj[-1],0,wx.LEFT|wx.EXPAND,self.gap)
        self.GS.Add(self._obj[-1],0,wx.LEFT,self.gap)
    
    def _gs_add_Ctrl(self,d):
        self._obj.append(d.pop())
        self.GS.Add(self._obj[-1],0,wx.LEFT,self.gap)

    def add_controls(self,**kwargs):
       #--- Label + line 
        gap=1
        self.Sizer = wx.BoxSizer(wx.VERTICAL)

        if self.drawline:
           gap=15
           self.Sizer.Add( wx.StaticLine(self),0, wx.EXPAND|wx.ALL,5 )
        if self.label:
           gap=15
           self.Sizer.Add( wx.StaticText(self,0,label=self.label),0,wx.LEFT,5)

        self.Sizer.Add( self.initControl(**kwargs),0,wx.EXPAND|wx.ALL,gap)
        
        self.SetSizer(self.Sizer)   
        self.SetAutoLayout(1)
        
        return self
   
    def init_number_of_rows_cols(self):
        """ cal number of rows  and cols
        
        Results:
        ----------
        number of rows and cols
        """
              
        if self.boxsizer.Orientation == wx.VERTICAL:
           self.rows = len( self.list_of_controls )
        else: # horizontal 
           self.cols = self._cols * len(self.list_of_controls)
            
        return self.rows,self.cols 
    
    def init_GridSizer(self):
        """ init wx.FelxGridSizer """
        
        self.init_number_of_rows_cols()
        
        self.GS = wx.FlexGridSizer(self.rows,self.cols,self.vgap,self.hgap)

        self.init_growable()
  
    def initControl(self,**kwargs):
        """
        generate and pack wx controls into a wx.FlexGridSizer via list of controls
        :param control_list: list of list with controls or [list of args]
        :param cols = None
        :param rows = None
      
        arg[0] <str>
         BT   : Button
         FLBT : FlatButton
         CK   : CheckBox
         COMBO: ComboBox
         TXT  : TextCtrl
         STXT : StaticText
         BTBMP: BitmapButton
         SP   : SpinCtrl
         SPF  : FloatSpin(
         OBJ, CTRL: add Ctrl Obj
    
        
        Results:
        ---------
         wx.FlexGridSizer with controls
         
        """
        if kwargs.get("control_list"):
           self.list_of_controls = kwargs.get("control_list")
        if kwargs.get("cols",None):
           self._cols=kwargs.get("cols")
        if kwargs.get("rows",None):
           self.rows=kwargs.get("rows")
           
        self._obj=[]
        self.init_GridSizer()
        
        for d in self.list_of_controls:
            if d:
               self._gs_add_add_first_text_info(d)
               
               if isinstance( d[0],(str) ):
                 #--- Button
                   if   d[0] == "BT"   : self._gs_add_Button(d)
                 #--- FlatButton
                   elif d[0] == "FLBT" : self._gs_add_FlatButton(d)
                 #--- CheckButton
                   elif d[0] == 'CK'   : self._gs_add_CheckBox(d)
                 #---ComboBox
                   elif d[0] == "COMBO": self._gs_add_ComboBox(d)
                  #---wx.TextCtrl
                   elif d[0] == 'TXT'  : self._gs_add_TextCtrl(d)
                 #---wx.StaticText
                   elif d[0] == 'STXT' : self._gs_add_StaticText(d)
                 #--- wx BitmapButton
                   elif d[0] == 'BTBMP': self._gs_add_BitmapButton(d)
                 #---  MIN/MAXSpin Buttons
                   elif d[0].startswith('SP'):
                      #--- min button
                       self._gs_add_min_max_button("|<","MIN")
                      #---SpinCrtl
                       if d[0] == 'SP'   : self._gs_add_SpinCtrl(d)
                      #---FloatSpinCrtl
                       elif d[0] == "SPF": self._gs_add_FloatSpin(d)
                      #--- max button
                       self._gs_add_min_max_button(">|","MAX")
                   elif d[0] == "OBJ":  self._gs_add_Ctrl(d)
                   elif d[0] == "CTRL": self._gs_add_Ctrl(d)
    
                   else:
                       self._gs_add_empty_cell()
               else:
                    self._gs_add_Ctrl(d)
            else:
               self._gs_add_empty_cell()
               
               #self.GS.Add(wx.StaticText(self,-1,label="NOT A CONTROLL"),wx.EXPAND,self.gap)
               #self.gs_add_empty_cell()
          # print(self._obj[-1].GetName())
        return self.GS                    

    def OnClickMinMax(self,evt): 
        obj = evt.GetEventObject()
          
    #--- get obj SpinCtrl   
        if   obj.GetName() == "MIN":
             idx = self.FindObjIndexByObjID(obj.GetId(),objs=self._obj)
             if idx != None:
                obj_sp = self._obj[idx+1] #self.FindWindowById( obj.GetId()+1)
                obj_sp.SetValue( obj_sp.GetMin() )
        elif obj.GetName() == "MAX":
             idx = self.FindObjIndexByObjID(obj.GetId(),objs=self._obj)
             if idx != None:
                obj_sp = self._obj[idx -1]  #self.FindWindowById( obj.GetId()-1)
                obj_sp.SetValue( obj_sp.GetMax() )
        else:
            evt.Skip()
            return    
        evtsp = wx.CommandEvent(wx.EVT_SPINCTRL.typeId, obj_sp.GetId()) 
        evtsp.SetEventObject(obj_sp)
        obj_sp.ProcessEvent(evtsp)   
    
      
class JuMEG_wxControls(JuMEG_wxControlBase):
    """
    generate button, text,spin/floatspin combobox controls
     
    Parameters:
    ---------- 
     parent widged
     control_list: list of parameter and definition for widged controls
     label       : headline text
     bg          : background color <"grey90">
     boxsizer    : [wx.VERTICAL,wx.HORIZONTAL] <wx.VERTICAL>
        
     FlexGridSizer parameter
      AddGrowableCol : int or list    <None> 
      AddGrowableRow : int or list    <None>
      -> col[i] or row[i] will grow/expand  use full to display long textfields
      
     Results:
     --------
     wx.Panel with sub controls group by wx.FlexGridSizer    

     Example:
     --------
       control_list: array of( ( type,label,min,max,value,callback fct) )
        control_list = ( ("SP","Xpos",-500,500,0,self.OnSpinXYpos),
                         ("SP","Ypos",-500,500,0,self.OnSpinXYpos),
                         ("SP","Width",1,5000,640,self.OnSpinSize),
                         ("SP","Height",1,5000,480,self.OnSpinSize)
                        )   
        
        
    """
    def __init__(self,parent,**kwargs):
        super(JuMEG_wxControls, self).__init__(parent,**kwargs) 
        
       #--- add controls 
        self.add_controls(**kwargs)
             
    def _gs_add_add_last_obj(self):
        """ helper fct: add last obj to add """
        self._gs_add_empty_cell()
        self.GS.Add(self._obj[-1],0,self.LE,self.gap)
        self._gs_add_empty_cell()
        
    
class JuMEG_wxControlGrid(JuMEG_wxControlBase):
    """
     generate controls in a grid e.g. for CheckButton
     
     Parameters:
     -----------    
      title   : text
      bg      : background colo
      boxsizer: sizer  [<wx.VERTICAL>,wx.HORIZONTAL]
      control_list :

     Results:
     --------
     wx.Panel with sub controls group by wx.FlexGridSizer    


    """
    def __init__(self,parent,**kwargs):
       super().__init__(parent,**kwargs)
       self._do_add_controls = kwargs.get("init",True)

      #--- add controls
       if self._do_add_controls:
          self.add_controls(**kwargs)
    
    def update(self,**kwargs):
        self.add_controls(**kwargs)
        
    def _gs_add_add_first_text_info(self,d):
        """ add text in front of control """
        pass
         
    def init_number_of_rows_cols(self):
        """ cal number of rows  and cols
        
        Results:
        ----------
        number of rows and cols
        """
        if self.cols == 1:
           self.rows = len( self.list_of_controls )
        else:
           self.rows,rest = divmod( len( self.list_of_controls ),self.cols)
           if rest:
              self.rows += 1

        return self.rows,self.cols

class JuMEG_wxControlIoDLGButtons(JuMEG_wxControlBase):
    """
    generate Buttons and TextCtrl to show File/Dir dialog and display filename or path in textfield
   
    """
    def __init__(self, parent,**kwargs):
        super (JuMEG_wxControlIoDLGButtons,self).__init__(parent,**kwargs)
      #--- add controls 
        self.cols = 2
        self.add_controls(**kwargs)
    
    def _gs_add_add_first_text_info(self,d):
        """ add text in front of control """
        pass    
    
    def init_number_of_rows_cols(self):
        """ cal number of rows  and cols
        
        Results:
        ----------
        number of rows and cols
        """
       
        self.rows = len( self.list_of_controls )
        return self.rows,self.cols 

class JuMEG_wxControlButtonPanel(JuMEG_wxControlBase):
    """
     generate controls in a grid e.g. for CheckButton
     
     Parameters:
     -----------    
      title   : text
      bg      : background colo
      boxsizer: sizer  [<wx.VERTICAL>,wx.HORIZONTAL]
      control_list :
     
     Results:
     --------
     wx.Panel with sub controls group by wx.FlexGridSizer    
     
     Example:   
     --------
       Button  : (label,name,pack option,wx-event,help,callback) 
       control_list = (["CLOSE","Close",wx.ALIGN_LEFT,"Close program",self.ClickOnButton ],
                       ["CANCEL","Cancel",wx.ALIGN_LEFT,"Cancel progress",self.ClickOnButton],
                       ["APPLY","Apply",wx.ALIGN_RIGHT,"Apply command",self.ClickOnButton])
              
    """
    def __init__(self,parent,**kwargs):
       super().__init__(parent,**kwargs)
       self.gap   = 3
       self.vgap  = 5
       self.hgap  = 4
       self.RE    = wx.RIGHT|wx.EXPAND
       self.SetBackgroundColour("grey40")      

      #--- add controls 
       self.add_controls(**kwargs)
    
    def add_controls(self,**kwargs):
       #--- Label + line 
        sbox = wx.BoxSizer(wx.VERTICAL)
        if self.label:
           sbox.Add( wx.StaticText(self,0,label=self.label),0,wx.LEFT,2)
           sbox.Add( wx.StaticLine(self),0, wx.EXPAND|wx.ALL,1 ) 
           
        sbox.Add( self.initControl(),0,wx.EXPAND|wx.ALL|wx.ALIGN_CENTER,self.hgap )
        self.SetAutoLayout(True)
        self.SetSizer(sbox)   
        
        return sbox
    
    def _gs_add_add_last_obj(self,d):
        """ helper fct: add last obj to add """
        if d[2] & (wx.ALIGN_RIGHT|wx.EXPAND):
           self.GS.AddStretchSpacer(prop=1)
        self.GS.Add(self._obj[-1],0,d[2]|wx.ALL,self.gap)
            
    def _gs_add_init_last_obj(self,d,evt=None):
        """ helper fct: init last obj to add """
        self._obj[-1].SetName(d[1])
        
        if isinstance(d[-2],str):
           self._obj[-1].SetToolTip(wx.ToolTip(d[-2]))
        if evt:
           if callable(d[-1]) : self._obj[-1].Bind(evt,d[-1])
        self._gs_add_add_last_obj(d)   
        
    def _gs_add_Button(self,d):
        """ add Button"""
        if d[0]:
           self._obj.append(wx.Button(self,wx.ID_ANY,label=d[0]))
           self._gs_add_init_last_obj(d,evt=wx.EVT_BUTTON)
        else: # add separator
           self.GS.Add(0,0,d[2]|wx.ALL,self.gap)

    def init_GridSizer(self):
        """ 
        
        Results:
        --------
        wx.GridSizer
        """
        self.GS = wx.BoxSizer(wx.HORIZONTAL)

    def initControl(self):
        """generate and pack wx controls into a wx.FlexGridSizer
        
        Results:
        ---------
         wx.FlexGridSizer with controls
         
        """
        self._obj=[]
        self.init_GridSizer() 
        
        for d in self.list_of_controls:
            self._gs_add_Button(d)
      
        return self.GS

