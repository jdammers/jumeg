import sys,os
import wx
import wx.lib.scrolledpanel as scrolled

from jumeg.tsv.wxutils.jumeg_tsv_wxutils import DLGButtonPanel

try:
    from agw import floatspin as FS
except ImportError: # if it's not there locally, try the wxPython lib.
    import wx.lib.agw.floatspin as FS

"""
 self.mvp[:,0] = dborder
          if (self.opt.plot.cols >1):      
             mat = np.zeros( (self.opt.plot.rows,self.opt.plot.cols) )
             mat += np.arange(self.opt.plot.cols)
             self.mvp[:,0] +=  mat.T.flatten() * ( dw + 2 *dborder)          
"""         

#from jumeg.jumeg_base import AccessorType


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
       #ScrPanel.__init__(self,parent,-1,style=wx.SUNKEN_BORDER,)
       self.vbox = wx.BoxSizer(wx.VERTICAL)
       self.SetBackgroundColour("grey90")
       self.opt = opt
       self.bt  = []
       
       self.obj = []
       self.obj_ckbt  = []
       self.obj_color = []
       self.obj_unit  = []
      
       self.label = label
         
      # stbox = self.initControl()
       stbox = self.InitGroupControls()
      
       self.vbox.Add(stbox, 0, wx.EXPAND|wx.ALL,15)
       
       #self.SetAutoLayout(True)
       self.SetSizer(self.vbox)
       
   def InitGroupControls(self):
       head="Group <Select Color> Unit".split()
       sizer = wx.FlexGridSizer(rows= len(self.opt.group_list) +1, cols=len(head), hgap=5, vgap=5)
       for label in head:
           sizer.Add( wx.StaticText(self, label=label),0,0 )
       gidx = 0
       
       for g in self.opt.group_list: 
           grp = self.opt.group[g]
           self.obj.append( wx.CheckBox(self,wx.NewId(),g.upper()) )
           self.obj[-1].SetValue(grp.enabled)
           self.obj[-1].Bind(wx.EVT_CHECKBOX,self.OnClickCheck)
           self.obj_ckbt.append(self.obj[-1])
         #--- 
           self.obj.append( wx.Button(self,wx.NewId(),label='   ') )
           self.obj[-1].SetBackgroundColour(grp.color)
           self.obj[-1].SetForegroundColour(grp.color)
           self.obj[-1].Bind(wx.EVT_BUTTON,self.OnClickColor)
           
           self.obj.append(wx.StaticText(self,wx.NewId(),grp.color))
           self.obj[-1].SetBackgroundColour(grp.color)
          
           # wx.Choice(panel, -1, (85, 18), choices=sampleList)

           #self.obj.append(wx.Choice(self,choices=self.opt.plt_color.label_list)) #,style=wx.CB_READONLY))
           #self.obj[-1].SetValue(grp.color)
           #self.obj[-1].Bind(wx.EVT_COMBOBOX,self.OnSelectColor)
         
           #self.obj.append(wx.ComboBox(self,choices=self.opt.plt_color.label_list,style=wx.CB_READONLY))
           #self.obj[-1].SetValue(grp.color)
           #self.obj[-1].Bind(wx.EVT_COMBOBOX,self.OnSelectColor)
           
           self.obj.append(wx.ComboBox(self,choices=grp.unit.choices,style=wx.CB_READONLY))
           self.obj[-1].SetValue(grp.unit.unit)
           self.obj[-1].Bind(wx.EVT_COMBOBOX,self.OnSelectUnit)
        
       sizer.AddMany(self.obj)
       return sizer    
       #self.SetSizer(sizer)
       #self.Fit()
 
            
       #---  scaling auto min/max
       # self.ctrl_scaling = JuMEG_TSV_Utils_SpinControl(self,self.obj_list_time,label="Time")         
       # vbox.Add(self.ctrl_time, 0,flag=wx.ALL|wx.EXPAND, border=10)
      
       return stbox
   
   def OnClickCheck(self,evt):
       obj = evt.GetEventObject()
       v=obj.GetValue()
       #obj_bt = self.FindWindowById( obj.GetId()-1)
       
       #c = dlg.GetStringSelection()
       #   obj.SetBackgroundColour(c)
       #   obj.SetForegroundColour(c)
         
       print "You selected: %s\n" % v
    

   def _OnClickColor(self,evt):
       obj = evt.GetEventObject()
      #wx.ComboBox(self,choices=self.opt.plt_color.label_list,style=wx.CB_READONLY))
       dlg = wx.SingleChoiceDialog(None, "Select Group Color", "Colors",self.opt.plt_color.label_list)
       if dlg.ShowModal() == wx.ID_OK:
          c = dlg.GetStringSelection()
          obj.SetBackgroundColour(c)         
          obj.SetForegroundColour(c) 
                    
          obj_txt = self.FindWindowById( obj.GetId()+1)
          obj_txt.SetLabel(c)
          
         
          print "You selected: %s\n" % c
       dlg.Destroy()

        
   def __OnSelectColor(self,evt):
       obj = evt.GetEventObject()
       c = obj.GetValue()
       print "TEST CB color"
       print c
       obj_bt = self.FindWindowById( obj.GetId()-1)
       obj_bt.SetBackgroundColour(c)         
       obj_bt.SetForegroundColour(c)         
    
   def OnSelectUnit(self,evt):
       obj = evt.GetEventObject()
       v = obj.GetValue()
       print "TEST Unit"
       print v
       obj_bt = self.FindWindowById( obj.GetId()-1)
       #obj_bt.SetBackgroundColour(c)         
       #obj_bt.SetForegroundColour(c)         
     
    
   def OnClickColor(self, evt):
       """
       This is mostly from the wxPython Demo!
       """
       obj = evt.GetEventObject()
       dlg = wx.ColourDialog(self)
 
       # Ensure the full colour dialog is displayed, 
       # not the abbreviated version.
       dlg.GetColourData().SetChooseFull(True)
 
       if dlg.ShowModal() == wx.ID_OK:
          data = dlg.GetColourData()
         # obj.SetBackgroundColour(c)         
         # obj.SetForegroundColour(c) 
                    
         # obj_txt = self.FindWindowById( obj.GetId()+1)
         # obj_txt.SetLabel(c)
         # obj_bt = self.FindWindowById( obj.GetId()-1)
          obj.SetBackgroundColour(c)         
          obj.SetForegroundColour(c)         
          print 'You selected: %s\n' % str(data.GetColour().Get())
 
       dlg.Destroy()
        
        
        
        
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
        
          #self.vbox = wx.BoxSizer(wx.VERTICAL)      
          self.SetBackgroundColour("grey90")

          self._nb = wx.Notebook(self)
                    
          self.InitDlg()   
          self.__ApplyLayout() 
        
      def InitDlg(self):
          self._nb = wx.Notebook(self)
          #vbox = wx.BoxSizer(wx.VERTICAL)
         # hbox = wx.BoxSizer(wx.HORIZONTAL)
       #---global group optin
          #self._ctrl_groups = ScrolledWrapper(self,CtrlGroup,self.opt,label="GROUPS")        
          self._ctrl_groups = CtrlGroup(self,self.opt,label="GROUPS")        
           
         #hbox.Add(self._ctrl_group,0,flag=wx.ALL|wx.EXPAND, border=10)
          #self._ctrl_nb_groups = ScrolledWrapper(self,CtrlGroup,self.opt,label="GROUPS")        
        
       #--- group / channwel option  
          #for g in self.opt.group_list: 
          #    grp = self.opt.group[g]
          #    pnl = wx.Panel(self._nb,-1,style=wx.SUNKEN_BORDER)
          #    self._nb.AddPage(pnl, g)
       
         # hbox.Add(self._nb, 0,flag=wx.ALL|wx.EXPAND, border=10)    
         
          #vbox.Add(hbox, 1, flag=wx.ALL|wx.EXPAND, border=5)
          #return vbox
        #--- button
          self._btbox = DLGButtonPanel(self,style=wx.SUNKEN_BORDER)
          
      #def __DoLayout(self):
      #    """Layout the panel"""
      #    vbox = wx.BoxSizer(wx.VERTICAL)
      #    #vbox.Add(self._nb,   1,wx.EXPAND,border=5)
      #    vbox.Add(self._vbox, 1,wx.EXPAND,border=5)
      #    vbox.Add(self._btbox,0,wx.EXPAND,border=5)
      #  
      #    self.SetSizer(vbox)
      #    self.SetAutoLayout(True)
  
#---
      def __ApplyLayout(self):
          """Layout the panel"""
          vbox = wx.BoxSizer(wx.VERTICAL)
         # hbox = wx.BoxSizer(wx.HORIZONTAL)
         # hbox.Add(self._ctrl_groups, 0,flag=wx.ALL|wx.EXPAND, border=10)  
          #hbox.Add(self._ctrl_nb_groubs, 1,flag=wx.ALL|wx.EXPAND, border=10)
           
         # vbox.Add(hbox,1,wx.ALL|wx.EXPAND,border=5)
          
          vbox.Add(self._ctrl_groups,1,wx.ALL|wx.EXPAND,border=5)
        
          vbox.Add(self._btbox,0,wx.EXPAND,border=5)
        
          self.SetSizerAndFit(vbox)
       
     #---
     