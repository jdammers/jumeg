import sys,os
import wx
import wx.lib.scrolledpanel as scrolled

from jumeg.gui.tsv.wxutils.jumeg_tsv_wxutils import DLGButtonPanel #,colourComboBoxDialog


try:
    from agw import floatspin as FS
except ImportError: # if it's not there locally, try the wxPython lib.
    import wx.lib.agw.floatspin as FS

__version__ = "2019-09-18-001"

class ScrolledWrapper(scrolled.ScrolledPanel):
   def __init__(self, parent,w,opt,**kwargs):
       """Wrap the given window in a scrolled panel"""
       scrolled.ScrolledPanel.__init__(self, parent,-1)
       vbox = wx.BoxSizer(wx.VERTICAL)
       # Attributes
       pnl = w(self, opt,**kwargs)

        # Layout
       vbox.Add(pnl, 1,wx.ALL|wx.EXPAND, 5)
       vbox.Add(wx.StaticLine(self, -1, size=(1024, -1)), 0, wx.ALL, 5)
       vbox.Add((20, 20))

       self.SetSizer(vbox)
       self.SetupScrolling()


#----------------------------------------------
class CtrlGroup(wx.Panel):
   def __init__(self, parent,opt,label="Test"):
       wx.Panel.__init__(self,parent,-1,style=wx.SUNKEN_BORDER)
       self.vbox = wx.BoxSizer(wx.VERTICAL)
       self.SetBackgroundColour("grey90")
       self.opt = opt
       self.bt  = []
       
       self.obj = []
       self.obj_ckbt  = []
       self.obj_colour = []
       self.obj_unit  = []
      
       self.label = label
         
       stbox = self.InitGroupControls()
      
       self.vbox.Add(stbox, 0, wx.EXPAND|wx.ALL,15)
       
       self.SetSizer(self.vbox)
       
   def InitGroupControls(self):
       
       self.obj = []
       self.obj_ckbt  = []
       self.obj_colour = []
       self.obj_unit  = []
      
       head="Group <Select colour> Unit".split()
       sizer = wx.FlexGridSizer(rows= len(self.opt.group_list) +1, cols=len(head), hgap=5, vgap=5)
       for label in head:
           sizer.Add( wx.StaticText(self, label=label),0,0 )
       gidx = 0
       size_h_w=16
       for g in self.opt.group_list: 
           grp = self.opt.group[g]
           self.obj.append( wx.CheckBox(self,wx.NewId(),g.upper()) )
           self.obj[-1].SetName(g)           
           self.obj[-1].SetValue(grp.selected)
           self.obj[-1].Bind(wx.EVT_CHECKBOX,self.OnClickCheck)
           
           self.obj_ckbt.append(self.obj[-1])
         #--- 
           self.obj.append( wx.Button(self,wx.NewId(),label=' ') )
           self.obj[-1].SetName(g)           
           self.obj[-1].SetBackgroundColour(grp.colour)
           self.obj[-1].SetForegroundColour(grp.colour)
           self.obj[-1].Bind(wx.EVT_BUTTON,self.OnClickcolour)
           
           self.obj.append(wx.StaticText(self,wx.NewId(),grp.colour))
           self.obj[-1].SetBackgroundColour(grp.colour)
           
           self.obj.append(wx.ComboBox(self,choices=grp.unit.choices,style=wx.CB_READONLY))
           self.obj[-1].SetName(g)           
           self.obj[-1].SetValue(grp.unit.unit)
           self.obj[-1].Bind(wx.EVT_COMBOBOX,self.OnSelectUnit)
        
       sizer.AddMany(self.obj)
       return sizer    
       
   def OnClickCheck(self,evt):
       obj = evt.GetEventObject()
       self.opt.group[ obj.Name ].selected=obj.GetValue()
       
   def OnClickcolour(self, evt):
       """
       This is mostly from the wxPython Demo!
       """
       obj = evt.GetEventObject()
       evt.Skip()
       
       dlg = colourComboBoxDialog(colour_label = obj.GetBackgroundColour() , colour_list=self.opt.plt.colour.label_list)
      
       if dlg.ShowModal() == wx.ID_OK:
          c = dlg.selected_colour_label
          obj.SetBackgroundColour(c)
          self.opt.group[ obj.Name ].colour = c
       
       dlg.Destroy()
   
   def OnClose(self, event):
        print ('In OnClose')
        event.Skip()
       
#------------------------------------------------------- 
class TSVGroupDialog(wx.Dialog):
      """
      TSVGroupDialog 
        input : opt <obj>        
                   
        return: opt
      """
      def __init__(self,title="JuMEG TSV Groups Option",opt=None): 
          super(TSVGroupDialog,self).__init__(None,title=title) 
          self.opt = opt
        
          self.SetBackgroundColour("grey90")

          self._nb = wx.Notebook(self)
                    
          self.InitDlg()   
          self.__DoLayout() 
        
      def InitDlg(self):
          self._nb = wx.Notebook(self)
          self._ctrl_groups = CtrlGroup(self,self.opt,label="GROUPS")
         #--- button
          self._btbox = DLGButtonPanel(self,style=wx.SUNKEN_BORDER)
          
     
#---
      def __DoLayout(self):
          """Layout the panel"""
          vbox = wx.BoxSizer(wx.VERTICAL)
          
          vbox.Add(self._ctrl_groups,1,wx.ALL|wx.EXPAND,border=5)
        
          vbox.Add(self._btbox,0,wx.EXPAND,border=5)
        
          self.SetSizerAndFit(vbox)
       
     #---
     