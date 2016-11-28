import sys,os
import wx

from jumeg.tsvgl.wxutils.jumeg_tsv_wxutils import DLGButtonPanel

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


class JuMEG_TSV_Utils_SpinControl(wx.Panel):
   def __init__(self, parent,spin_control_list,label=None):
       wx.Panel.__init__(self,parent,-1,style=wx.SUNKEN_BORDER,)
       self.vbox = wx.BoxSizer(wx.VERTICAL)
       
       self.SetBackgroundColour("grey90")

       self.__obj     = []   
       self.__obj_spf = []   
       
       self.label = label
         
       stbox = self.initSpinControl(spin_control_list)
       self.vbox.Add(stbox, 0, wx.EXPAND|wx.ALL,15)
       
       self.SetAutoLayout(True)
       self.SetSizer(self.vbox)
       
   def initSpinControl(self,splist):
       combo_box = None
       sb  = wx.StaticBox(self,label=self.label)
       self.stbox = wx.StaticBoxSizer(sb,orient=wx.VERTICAL) 
       for d in splist:
           sb1  = wx.StaticBox(self,label=d[1])
           sb1.SetBackgroundColour("green")
           hbs1 = wx.StaticBoxSizer(sb1,orient=wx.HORIZONTAL)
           
           if d[0] == 'CB':
              self.__obj.append(wx.ComboBox(self,choices=[ str(x) for x in d[2] ],style=wx.CB_READONLY))
              self.__obj[-1].SetValue( str( d[3] ))
              self.__obj[-1].Bind(wx.EVT_COMBOBOX,d[4])
              hbs1.Add(self.__obj[-1],1,wx.EXPAND,10) 
              combo_box = self.__obj[-1]
                    
           else:
          #--- min button 
              self.__obj.append( wx.Button(self,wx.NewId(),label="|<",style=wx.BU_EXACTFIT,name="MIN") ) 
              self.__obj[-1].Bind(wx.EVT_BUTTON,self.OnClickMinMax)  
              hbs1.Add(self.__obj[-1],0,wx.LEFT,5)
          #---SpinCrtl     
              if d[0] == 'SP':   
                 self.__obj.append( wx.SpinCtrl(self,wx.NewId(),style=wx.SP_ARROW_KEYS|wx.SP_WRAP|wx.TE_PROCESS_ENTER|wx.ALIGN_RIGHT)) 
                 self.__obj[-1].SetToolTipString("Min: " + str(d[1]) +"  Max: " + str(d[2]) )
                 self.__obj[-1].SetRange(d[2],d[3])
                 self.__obj[-1].SetValue(d[4])
                 self.__obj[-1].Bind(wx.EVT_SPINCTRL,  d[5])  
                 hbs1.Add(self.__obj[-1],1,wx.EXPAND,10)
          #---FloatSpinCrtl      
              elif d[0] == "SPF":
                   self.__obj.append( FS.FloatSpin(self,wx.NewId(),min_val=d[2],max_val=d[3],increment=d[4],value=1.0,agwStyle=FS.FS_RIGHT) )   
                   self.__obj[-1].SetFormat("%f")
                   self.__obj[-1].SetDigits(3)
                   self.__obj[-1].SetValue(d[4])
                   self.__obj[-1].Bind(FS.EVT_FLOATSPIN, d[6])   
                   self.__obj_spf.append( self.__obj[-1] )
                   hbs1.Add(self.__obj[-1],1,wx.EXPAND,10)                  
          #--- max button   
              self.__obj.append( wx.Button(self,wx.NewId(),label=">|",style=wx.BU_EXACTFIT,name="MAX") )  
              self.__obj[-1].Bind(wx.EVT_BUTTON,self.OnClickMinMax)  
              hbs1.Add(self.__obj[-1],0,wx.RIGHT,5)
                  
           self.stbox.Add(hbs1,1,wx.EXPAND,5) 
         
         #--- update SPFs increment factor  
           if combo_box:
              self.spf_set_inc_factor( combo_box.GetValue() )                  
             
        
       return self.stbox
               
   def OnClickMinMax(self,evt): 
        obj = evt.GetEventObject()
          
    #--- get obj SpinCtrl   
        if   obj.GetName() == "MIN":
             obj_sp = self.FindWindowById( obj.GetId()+1)
             obj_sp.SetValue( obj_sp.GetMin() )
        elif obj.GetName() == "MAX": 
             obj_sp = self.FindWindowById( obj.GetId()-1)
             obj_sp.SetValue( obj_sp.GetMax() )
        else:
            evt.Skip()
            return            
      #--- update opt value  -> call obj SpinCtrl event manually   
        evt = wx.CommandEvent(wx.EVT_SPINCTRL.typeId, obj_sp.GetId()) 
        evt.SetEventObject(obj_sp)
        obj_sp.ProcessEvent(evt)
  
   def spf_set_inc_factor( self,f ):
           
       for spf in self.__obj_spf:     
           v = spf.GetValue()
           try:
               df = float(f)
           except ValueError:
               df= 1.0/10.0 ** 2 #spf.FloatSpin.getDigits()
                  
           spf.SetIncrement( df )
           spf.SetValue(v)               

               
class TSVSubPlotDialog(wx.Dialog):
    """
      TSVSubPlotDialog nedds 
       input: opt <obj>        
              opt.plot_counts
              opt.plot_cols
              opt.plot_start
       
       
       return: opt
    """
    #__metaclass__=AccessorType()

    def __init__(self,title="JuMEG TSV",opt=None):   #r=1,c=1,rmax=1000,cmax=1000):
        super(TSVSubPlotDialog,self).__init__(None,title=title) 

        self.obj_s = []        
        self.obj_t = []
        
        self.opt = opt
        self.inc_factor_list=[0.001,0.01,0.05,0.1,0.2,0.5,1.0,2,5,10,20,30,50,60,100,150,200,300,400,500,
                              1000,2000,3000,5000,10000,20000,30000,50000,60000,100000,120000,150000,200000]
       #--- subplot                         
        self.obj_list_subplot= (
               ("SP","Number of Plots",1,self.opt.channels.counts,self.opt.plots,self.OnSpinPlots),
               ("SP","Cols",1,self.opt.channels.counts,self.opt.plot_cols,self.OnSpinCols),
               ("SP","GoTo Channel",1,self.opt.channels.counts,self.opt.plot_start,self.OnSpinGotoChannel)
              )  
             #  ("SP","Display Channels",1,self.opt.channels.counts,self.opt.channels_to_display,self.OnSpinChannels),
             #  ("SP","Cols",1,self.opt.channels.counts,self.opt.plot.cols,self.OnSpinCols),
             #  ("SP","GoTo Channel",1,self.opt.channels.counts,self.opt.channels.start,self.OnSpinGotoChannel)
             
      #--- time
        self.obj_list_time=( 
               ("SPF","GoTo Times [s]",self.opt.time.pretime,self.opt.time.end,self.opt.time.start,self.opt.time.inc_factor,self.OnSpinTstart),
               ("SPF","Window [s]",0.0,self.opt.time.end,self.opt.time.window,self.opt.time.inc_factor,self.OnSpinTwindow),
               ("SPF","Scroll Step[s]",0.0,self.opt.time.end,self.opt.time.scroll_step,self.opt.time.inc_factor,self.OnSpinTscrollstep),
               ("CB","Increment",self.inc_factor_list,self.opt.time.inc_factor,self.OnSelectSPFloatInc)           
              )
                
        self.InitDlg()    
        self.__ApplyLayout()
        
    def InitDlg(self):
        
       #--- subplot
        self._ctrl_subplot = JuMEG_TSV_Utils_SpinControl(self,self.obj_list_subplot,label="Sub Plot")                 
       #--- time
        self._ctrl_time = JuMEG_TSV_Utils_SpinControl(self,self.obj_list_time,label="Time")            
       #--- button
        self._btbox = DLGButtonPanel(self,style=wx.SUNKEN_BORDER)
#---
    def __ApplyLayout(self):
          """Layout the panel"""
          vbox = wx.BoxSizer(wx.VERTICAL)
          vbox.Add(self._ctrl_subplot, 0,flag=wx.ALL|wx.EXPAND, border=10)  
          vbox.Add(self._ctrl_time, 0,flag=wx.ALL|wx.EXPAND, border=10)
          vbox.Add(self._btbox,0,wx.EXPAND,border=5)
        
          self.SetSizerAndFit(vbox)
          #self.SetAutoLayout(True)
            
        
#---    
    def OnSelectSPFloatInc(self,evt):
        obj = evt.GetEventObject()
        self.opt.time.inc_factor = float( obj.GetValue() )
        self._ctrl_time.spf_set_inc_factor( self.opt.time.inc_factor )

#--- opt.channels      
    def OnSpinPlots(self,evt):
        obj = evt.GetEventObject()
        self.opt.plots = obj.GetValue()      
   #---
    def OnSpinCols(self,evt):
        obj = evt.GetEventObject()
        self.opt.plot_cols = obj.GetValue()
   #---  
    def OnSpinGotoChannel(self,evt):
        obj = evt.GetEventObject()
        self.opt.plot_start = obj.GetValue()

#--- opt.time
    def OnSpinTstart(self,evt):
        obj = evt.GetEventObject()
        self.opt.time.start = obj.GetValue()     
   #---       
    def OnSpinTwindow(self,evt):
        obj = evt.GetEventObject()
        self.opt.time.window = obj.GetValue()  
   #---
    def OnSpinTscrollstep(self,evt):
        obj = evt.GetEventObject()
        self.opt.time.scroll_step = obj.GetValue()  
