#!/usr/bin/env python3
# -+-coding: utf-8 -+-

"""
"""

#--------------------------------------------
# Authors: Frank Boers <f.boers@fz-juelich.de> 
#
#-------------------------------------------- 
# Date: 28.08.19
#-------------------------------------------- 
# License: BSD (3-clause)
#--------------------------------------------
# Updates
#--------------------------------------------
import wx

class JuMEG_wxDLGCtrls(wx.Dialog):
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
    
    
    
    hbox2 = wx.BoxSizer(wx.HORIZONTAL)
        okButton = wx.Button(self, label='Ok')
        closeButton = wx.Button(self, label='Close')
        hbox2.Add(okButton)
        hbox2.Add(closeButton, flag=wx.LEFT, border=5)

        vbox.Add(pnl, proportion=1,
            flag=wx.ALL|wx.EXPAND, border=5)
        vbox.Add(hbox2, flag=wx.ALIGN_CENTER|wx.TOP|wx.BOTTOM, border=10)

        self.SetSizer(vbox)

        okButton.Bind(wx.EVT_BUTTON, self.OnClose)
        closeButton.Bind(wx.EVT_BUTTON, self.OnClose)


    def OnClose(self, e):

        self.Destroy()
 
    
    
    
    
    
    """
    def __init__(self, parent, message, caption,size=(600,480),**kwargs):
        super().__init__(parent, -1,style=wx.DEFAULT_DIALOG_STYLE|wx.RESIZE_BORDER|wx.CLOSE_BOX|wx.STAY_ON_TOP)
        self._wx_init(message, caption)
        self.update(**kwargs)
        self._ApplyLayout()
        self.SetSize(size)
        #self.SetTitle(kwargs.get("title","JuMEG")
        
    @property
    def Ctrls(self): return self._ctrl_pnl
    
    def _wx_init(self,message, caption):
        self.SetTitle(caption)
        self.Sizer     = wx.BoxSizer(wx.VERTICAL)
        self.message   = wx.StaticText(self, -1, message)
        self._ctrl_pnl = wx.Panel(self)
      
      #--- VERTICA;l box to add user ctrls used in update
      #  self._ctrlbox = wx.BoxSizer(wx.HORIZONTAL)
      #  self._ctrlbox.Add(self.tgbt,         0,wx.LEFT|wx.ALL|wx.EXPAND, 5)
     
      #  self.btns = self.CreateSeparatedButtonSizer(wx.OK | wx.CANCEL)
      #  self.Bind(wx.EVT_TOGGLEBUTTON,self.ClickOnCtrl)
    
    def update(self,**kwargs):
        """
        overwrite
        add user ctrls into self.CtrlBox (hbox sizer)
        
        :param kwargs:
        :return:
        """
        pass
    
   
    def ClickOnCtrl(self,evt):
        pass
        #obj = evt.GetEventObject()
        #if obj.GetName() == "BT.DESELECT_ALL":
        #   for i in range(self.clb.GetCount()):
        #       self.clb.Check(i, self.tgbt.GetValue())
    
    def _ApplyLayout(self):
        bw = 2
        stl1 = wx.StaticLine(self,style=wx.LI_HORIZONTAL)
        stl2 = wx.StaticLine(self,style=wx.LI_HORIZONTAL)
        
        self.Sizer.Add(self.message, 0, wx.ALL | wx.EXPAND, bw)
        
        self.Sizer.Add(stl1, 0, wx.ALL | wx.EXPAND, bw)
        self.Sizer.Add(self._ctrl_pnl, 1, wx.ALL | wx.EXPAND, bw)
        self.Sizer.Add(stl2,0,wx.ALL | wx.EXPAND,bw)
        #self.Sizer.Add(self.btns, 0, wx.LEFT|wx.ALL | wx.EXPAND, 5 )
        self.SetSizer(self.Sizer)
        self.SetAutoLayout(True)
        self.Fit()
        

  

class TestFrame(wx.Frame):
    def __init__(self, *args, **kwargs):
        kwargs['style'] = wx.DEFAULT_FRAME_STYLE
        super(TestFrame, self).__init__(*args, **kwargs)
    
        
        sizer = wx.BoxSizer(wx.VERTICAL)
        pnl = wx.Panel(self)
        bt  = wx.Button(pnl,-1,"TEST")
        sizer.Add(bt)
        
        self._init_dlg()
        
        pnl.SetSizerAndFit(sizer)
        self.Bind(wx.EVT_BUTTON,self.ClickOnBt)
    
    def _init_dlg(self):
        message="JuMEG TSV TEST"
        caption="JuMEG TSV"
        ctrls = None
        self.DLG = JuMEG_wxDLGCtrls(self,message, caption)
        
    def ClickOnBt(self,evt):
        self.bt1=wx.Button(self.DLG._ctrl_pnl,1,"LAALA")
        
        self.DLG.Show()


if __name__ == "__main__":
    app = wx.App(False)
    TestFrame(None).Show()
    app.MainLoop()