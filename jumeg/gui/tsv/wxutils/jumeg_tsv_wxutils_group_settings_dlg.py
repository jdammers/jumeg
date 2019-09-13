#!/usr/bin/env python3
# -+-coding: utf-8 -+-

#--------------------------------------------
# Authors: Frank Boers <f.boers@fz-juelich.de>
#
#--------------------------------------------
# Date: 05.09.19
#--------------------------------------------
# License: BSD (3-clause)
#--------------------------------------------
# Updates
#--------------------------------------------

import numpy as np
import wx
import sys,logging

from jumeg.base import jumeg_logger
logger = logging.getLogger('JuMEG')


from jumeg.gui.wxlib.utils.jumeg_gui_wxlib_utils_controls import JuMEG_wxControlGrid
from tsv.wxutils.jumeg_tsv_wxutils                        import DLGButtonPanel

__version__="2019-09-05-001"

LEA = wx.ALIGN_LEFT | wx.EXPAND | wx.ALL

class PopupColourTable(wx.PopupTransientWindow):
    def __init__(self, parent,**kwargs):
        super().__init__(parent,wx.NO_BORDER)
        self._wx_init(**kwargs)
        self._ApplyLayout()

    def _wx_init(self,**kwargs):
        self.SetBackgroundColour(kwargs.get("bg","GREY60"))
        w = kwargs.get("w",24)
        self._C        = kwargs.get("colours")
        self._caller   = kwargs.get("caller")
        self._callback = kwargs.get("callback")
        self._title    = kwargs.get("title","JuMEG Select Colour")
        self._colour_text = None
        
        bmp   = wx.Bitmap()
        ctrls = []
        
        for i in range( self._C.n_colours ):
           #--- select colour
            ctrls.append( ["BTBMP",self._C.labels[i],bmp,(w,w),wx.NO_BORDER|wx.BU_NOTEXT,self._C.colours[i] ,self._C.labels[i],self.ClickOnCtrls] )
     
       #--- calc cols for grid
        n_cols,rest = divmod( len(ctrls),4)
        if rest: n_cols += 1
        
        self._pnl = JuMEG_wxControlGrid(self,label= self._title,control_list=ctrls,cols=n_cols,set_ctrl_prefix=False)
        
        
    def ClickOnCtrls(self,evt):
        obj = evt.GetEventObject()
        
        try:
           if obj.GetToolTip():
             self._colour_text = obj.GetToolTipText()
           #--- call the caller-function in parent
             if self._colour_text:
                self._callback( self._caller,self._colour_text )
           #--- close
             self.Dismiss() # close it
        except:
            logger.exception("---> ERROR can not set ToolTip".format(self._colour_text))
        
    def _ApplyLayout(self):
        vbox = wx.BoxSizer(wx.VERTICAL)
        vbox.Add(self._pnl,1,LEA,2)
     
        self.SetAutoLayout(True)
        self.SetSizer(vbox)
        self.Fit()
        self.Layout()
        

class GroupDLG(wx.Dialog):
    """
    
    Example:
    ---------
    input group dict
    
    labels = ['grad','mag','eeg','stim','eog','emg','ecg','ref_meg']
    grp={
                       'mag':    {"selected":True,"colour":"RED",       "prescale":200,"unit":"fT"},
                       'grad':   {"selected":True,"colour":"BLUE",      "prescale":200,"unit":"fT"},
                       'ref_meg':{"selected":True,"colour":"GREEN",     "prescale":2,  "unit":"pT"},
                       'eeg':    {"selected":True,"colour":"BLUE",      "prescale":1,  "unit":"uV"},
                       'eog':    {"selected":True,"colour":"PURPLE",    "prescale":100,"unit":"uV"},
                       'emg':    {"selected":True,"colour":"DARKORANGE","prescale":100,"unit":"uV"},
                       'ecg':    {"selected":True,"colour":"DARKGREEN", "prescale":100,"unit":"mV"},
                       'stim':   {"selected":True,"colour":"CYAN",      "prescale":1,  "unit":"bits"}
                       }
      
    
    """
    @property
    def Group(self): return self._GRP
    
    def __init__(self,parent,**kwargs):
        style=wx.CLOSE_BOX|wx.MAXIMIZE_BOX|wx.MINIMIZE_BOX #|wx.RESIZE_BORDER
        super().__init__(parent,title="JuMEG Group Settings",style=style)
        self._init(**kwargs)

    def _init(self,**kwargs):
        self._GRP = kwargs.get("grp")
        
        self._wx_init(**kwargs)
        self._wx_button_box()
        self._ApplyLayout()
        
    def _wx_button_box(self):
        """
        show DLG cancel apply bts
        :return:
        """
        self._pnl_button_box = DLGButtonPanel(self,style=wx.SUNKEN_BORDER)
    
    def _wx_init(self,**kwargs):
        w = kwargs.get("w",24)
        self.SetBackgroundColour(kwargs.get("bg","GREY90"))
        # self._bt=wx.Button(self,-1,"TEST")
        
        bmp   = wx.Bitmap()
        ctrls = [ ["STXT","Groups","Groups"], ["STXT","Colour","Colour"], ["STXT","Scale","Scale"],[ "STXT","Unit","Unit"]]
        n_cols= len(ctrls)
        
        for grp in self._GRP.labels:
            g = grp.upper()
            
          #--- ckbutton select group
            ctrls.append( ["CK",  g+".CKBOX",g,self._GRP.GetSelected(grp),'de/select group',self.ClickOnCtrls] )
          #--- select colour
            label = self._GRP.GetColour(grp)
            ctrls.append( ["BTBMP",g+".colour",bmp,(w,w),wx.NO_BORDER|wx.BU_NOTEXT,label,label +"\nclick to change",self.ClickOnCtrls] )
          #--- grp prescale
            ctrls.append(["COMBO",g.upper()+".PRESCALE",str(self._GRP.GetPreScale(grp)),self._GRP.Unit.prescales,'set scaling',self.ClickOnCtrls])
          #--- grp unit wit  prefix e.g. m,u,n,p,f,a => mT,uT,...
            self._GRP.Unit.unit = self._GRP.GetUnit(grp) # update Unit CLS
            ctrls.append(["COMBO",g+".UNIT",self._GRP.Unit.unit,self._GRP.Unit.GetUnits(),'set scaling unit',self.ClickOnCtrls])
        
        self._pnl_groups = JuMEG_wxControlGrid(self,label="Group Parameter",control_list=ctrls,cols=n_cols,set_ctrl_prefix=False,AddGrowableCol=[2,3])
     
        #self._bt.Bind(wx.EVT_BUTTON,self.ClickOnButton)
   
    def _popup_colour_table(self,obj,grp):
        PCT = PopupColourTable(self,title="Group: "+obj.GetName().split(".")[0],caller=obj,colours=self._GRP.Colour,callback=self.update_colour)
        pos = wx.GetMousePosition()
        PCT.Position(pos,(0,0))
        PCT.Popup()
    
    def update_colour(self,obj,label):
        """
        sets the group colour
        this is the callback executed from PopupColourTable
        may use wx.EVENTS or pubsub
        :param obj:  colour  bitmapbutton
        :param label:colour label: RED,GREEN has to be in colour-object label list ...
        :return:
        """
        grp,key = obj.GetName().lower().split(".")
        c = self._GRP.Colour.label2colour(label)
        obj.SetBackgroundColour( c )
        obj.SetToolTipString( label+"\nclick to change")
        self._GRP.SetColour( grp, label )
        
        # logger.info("set group: {} colour: {}".format(grp,self._GRP.GetGroup(grp) ))
        
    def ClickOnCtrls(self,evt):
        obj     = evt.GetEventObject()
        grp,key = obj.GetName().lower().split(".")
   
        if key == "colour":
           self._popup_colour_table(obj,grp)
           return
        
        v = obj.GetValue()
        if key == "selected":
           self._GRP.SetSelected(grp,v )
        elif key == "prescale":
             self._GRP.SetPreScale(grp,v)
        elif key == "unit":
            self._GRP.SetUnit(grp,v)
    
    def ClickOnButton(self,evt):
        pass
   
    def _ApplyLayout(self):
        vbox = wx.BoxSizer(wx.VERTICAL)
        #vbox.Add(self._bt,0,LEA,2)
        vbox.Add(self._pnl_groups,1,LEA,2)
    
       #--- fix size to show combos of scale and unit
        stl = wx.StaticLine(self,size=(350,2) )
        stl.SetBackgroundColour("GREY85")
        vbox.Add(stl,0,LEA,1)
        vbox.Add(self._pnl_button_box,0,LEA,2)
    
        self.SetAutoLayout(True)
        self.SetSizer(vbox)
        self.Fit()
        self.Layout()

class ChannelDLG(GroupDLG):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        
        
class MainFrame(wx.Frame):
   
   def __init__(self, parent, title,**kwargs):
      super().__init__(parent, title = title)
      self._init(**kwargs)
      self._ApplyLayout()
   
   def _init(self,**kwargs):
       self.test = wx.Panel(self)
       self.show_group_dialog(**kwargs)
       self.Close()
       
   def _ApplyLayout(self):
       vbox = wx.BoxSizer(wx.VERTICAL)
       vbox.Add(self.test,1,LEA,4)
       self.SetSizer(vbox)
       self.SetAutoLayout(True)
       self.Fit()
       self.Show(True)
       
   def show_group_dialog(self,**kwargs):
       dlg = GroupDLG(self,**kwargs)
       out = dlg.ShowModal()
    
       if out == wx.ID_APPLY:
          grp = dlg.Group.GetGroup(None)
          for g in grp.keys():
              logger.info("OUTPUT: {} => {} \n {}".format(g,dlg.Group.GetScaling(g),grp.get(g)))
    
       dlg.Destroy()



if __name__ == "__main__":
    opt=None

    #---  Testing DEBUG
    from tsv.utils.jumeg_tsv_utils_io_data       import JuMEG_TSV_Utils_IO_Data
    from tsv.plot.jumeg_tsv_plot2d_data_options  import GroupOptions,JuMEG_TSV_PLOT2D_DATA_OPTIONS

    
    verbose = True
    path  = "data"
    path  = "~/MEGBoers/programs/JuMEG/jumeg-py/jumeg-py-git-fboers-2019-08-21/projects/JuMEGTSV/data"
    fname = '200098_leda_test_10_raw.fif'
    raw   = None

    jumeg_logger.setup_script_logging(name="JuMEG",opt=opt,logger=logger)
    
    IO       = JuMEG_TSV_Utils_IO_Data()
    raw,bads = IO.load_data(raw,fname,path)
   
    DataOpt  = JuMEG_TSV_PLOT2D_DATA_OPTIONS()
    DataOpt.update(raw=raw)
    
   #--- ToDo use only groups from raw
    GRP      = GroupOptions()
    
    app = wx.App()
    MainFrame(None,'JuMEG demo',grp=GRP)
    app.MainLoop()
