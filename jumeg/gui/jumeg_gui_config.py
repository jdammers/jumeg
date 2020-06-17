#!/usr/bin/env python3
# coding: utf-8


#--------------------------------------------
# Authors:
# Lukas Kurth <l.kurth@fz-juelich.de>
# Frank Boers <f.boers@fz-juelich.de>,
#
#-------------------------------------------- 
# Date: 10.03.20
#-------------------------------------------- 
# License: BSD (3-clause)
#--------------------------------------------
# Updates
#--------------------------------------------

import getpass,datetime,platform
import os,sys,argparse,pprint
# from ruamel.ordereddict import ordereddict
from collections import OrderedDict
from copy import  deepcopy

import wx
import wx.lib.agw.customtreectrl as CT
from   wx.lib.agw.customtreectrl import CustomTreeCtrl

from jumeg.gui.wxlib.utils.jumeg_gui_wxlib_utils_controls import SpinCtrlScientific,EditableListBoxPanel,JuMEG_wxSTXTBTCtrl
from jumeg.base.jumeg_base import jumeg_base as jb
from jumeg.base.jumeg_base_config import JuMEG_CONFIG

from jumeg.base import jumeg_logger
logger = jumeg_logger.get_logger()

__version__= "2020.04.28.001" # platform.python_version()

class JuMEG_ConfigTreeCtrl(CustomTreeCtrl):
   def __init__(self,parent,**kwargs):
       style = (CT.TR_DEFAULT_STYLE| CT.TR_SINGLE
               |CT.TR_HAS_VARIABLE_ROW_HEIGHT
               |CT.TR_ELLIPSIZE_LONG_ITEMS|CT.TR_TOOLTIP_ON_LONG_ITEMS
               |CT.TR_ALIGN_WINDOWS)
       
       super().__init__(parent,agwStyle=style)
       
       self.root_name       = "jumeg"
       self.verbose         = False
      
      #--- float settings
       self.float_digits = 3
       self.float_min    = -1000.0
       self.float_max    = 1000.0
       self.float_inc    = 0.1

      #---
       self._root_key       = "_root_keys"
       self._sorted_key     = "_sorted_keys"
       self._sorted_keys    = []
       self._list_seperator = " "
       self._data           = dict()
      
       self._wx_init(**kwargs)
       
       self._info      = dict()
       self._used_dict = dict()
       
   def update(self,data=None,root_name=None,item_data=None):
      '''
      initialises a new TreeCtrl
      '''
      self._clear()
      if item_data==None:
         item_data=dict()
      if root_name:
         self.root_name=root_name
      self._wx_init(data=data,root_name=self.root_name,item_data=item_data)
      
   def sort(self,keys):
       pass
    
  # def _get_item_data(self,data,item_data):
  #    self.__get_item_data(data,item_data)
   
   def _get_item_data(self,data,item_data):
       
        if data==None:
           logger.exception("data is None")
           return
        keys = list(data.keys())
        keys.sort()
        #item_data = OrderedDict.fromkeys( keys )
        
        '''
        ToDo
        from collections import OrderedDict
        personA = OrderedDict([
            (u'score',
             OrderedDict([ (u'2015-09-09 03:40:33 +0100', 2646),
                        (u'2015-09-10 03:35:34 +0100', 2646),
                         ])),
            (u'adjusted_score',
             OrderedDict([ (u'2015-09-09 03:40:33 +0100', 3646),
                           (u'2015-09-10 03:35:34 +0100', 3646),
                         ]))
            ])
   
        '''
        for k in keys:
           if k.startswith("_"):  #_keys
              item_data[k] = deepcopy( data[k] )
              continue
               
           v = data[k]
           
           if isinstance(v,(dict)):
               item_data[k] = dict()
               self._get_item_data(data[k],item_data[k])
           else:
               try:
                  #--- ck for list as data type and convert str in list to orig data types
                   if v.GetName().startswith("list"):
                      dtype = v.GetName().split("_")[1]
                      # d = v.GetLineText(lineNo=0).split(self._list_seperator)
                      d = v.GetValue()
                      if (d):
                         if dtype == "float":
                            item_data[k] = [float(x) for x in d]
                         elif dtype == "int":
                            item_data[k] = [int(x) for x in d]
                         else:
                            item_data[k]=d
                      else: # str
                         item_data[k]=list()
                 #--- None        
                   elif v.GetName().startswith("NoneStr"):# check for "None" or "None,0"
                   
                      d = v.GetValue().strip()
                      if d.upper() == "NONE" or d.upper()=="NULL":
                         item_data[k] = None
                      else:
                         d = [ x.strip() for x in d.split(",") ]
                         for i in range(len(d)):
                             if (d[i].upper() == "NONE") or (d[i].upper()=="NULL"):
                                 d[i] = None
                         item_data[k] = d
                             
                   else:
                      item_data[k]=v.GetValue()
                      
               except:
                   logger.exception("ERROR")
                   continue # info, _keys
                   
        return item_data
  
   def GetData(self):
       data = self._item_data
       keys = list(data.keys())
       item_data = dict() #OrderedDict.fromkeys( ["info",*keys] )
       item_data["info"] = self.update_info()
       for k in keys:
           item_data[k] = dict()
           self._get_item_data(data[k],item_data[k])
          
       return item_data
   
   def _clear(self):
      '''
      deletes the actual TreeCtrl
      '''
      self.DeleteAllItems()
      self._data=None
    
   def _init_tree_ctrl(self,data=None,root=None,item_data=None):
       '''
       builds a new TreeCtrl recursively based on the data which is given as a dict
       '''
       if data==None:
           logger.exception("data is None")
           return
       
       txt_size = 30
       style    = wx.TE_RIGHT
             
       if not root:
           root = self.root
       
       klist = []
       dlist = []
       
       keys  = list(data.keys())
       keys.sort()
      
      #--- sort keys:
       # global sortred keys, lokal sorted keys, keys,dict-keys
      #--- global sorted keys
       skeys = [ *self._sorted_keys ]
      #--- extend with lokal sorted keys
       if self._sorted_key in keys:
          skeys.extend( data.get( self._sorted_key,[] ) )
          
       for k in keys:
           if k in skeys: continue
           if isinstance( data[k],(dict) ):
              dlist.append(k)
           else:
              klist.append(k)
      
       keys =[*skeys, *klist, *dlist]
       
       for k in keys:
           if k.startswith("_")    :
              item_data[k] = data[k]
              continue
           if not k in data.keys() : continue
           # if item_data.get(k,None): continue
           v = data[k]
           child= None
           ctrl = None
        
          #--- type dict recursive
           if isinstance(v,(dict)):
              item_data[k] = dict()
              child = self.AppendItem(root,"{}".format(k),ct_type=0)
              self._init_tree_ctrl(data=data[k],root=child,item_data=item_data[k])
              continue
           
           elif isinstance(v,(bool)):
               ctrl=wx.CheckBox(self,-1,name="bool") #label=k
               ctrl.SetValue(v)
               child = self.AppendItem(root,"{}".format(k),wnd=ctrl)
               self.SetItemBold(child,True)
        
           elif isinstance(v,(str)):
                if os.path.dirname(v):
                   ctrl = JuMEG_wxSTXTBTCtrl(self,name="TEST",label=v,cmd=self.ClickOnShowDLG,textlength=txt_size,style=style)
                else:
                   ctrl = wx.TextCtrl(self,-1,style=wx.TE_LEFT,value=v,name="str")
                   sz = ctrl.GetSizeFromTextSize(ctrl.GetTextExtent("W" * txt_size))
                   ctrl.SetInitialSize(sz)
            
                child = self.AppendItem(root,"{}".format(k),wnd=ctrl,ct_type=0)
        
           elif isinstance(v,(list)):
                ctrl = EditableListBoxPanel(self,label=k.upper())
              #--- ck data type for later reconstruction from listbox (string)
                dtype = str( type( v[0] ) ).lower()
                name = "list"
                if dtype.find("float")>-1:
                   name+= "_float"
                elif dtype.find("int") > -1:
                    name += "_int"
                else:
                    name +="_str"
                
                ctrl.SetName(name)
                ctrl.Value = v
                child = self.AppendItem(root,"{}".format(k),wnd=ctrl)
        
           elif isinstance(v,(int)):
               ctrl=wx.SpinCtrl(self,-1,"",(30,50),name="int")
               ctrl.SetRange(0,10000)
               ctrl.SetValue(v)
               child=self.AppendItem(root,"{}".format(k),wnd=ctrl)
               self.SetItemBold(child,True)
        
           elif isinstance(v,(float)):
              # v = float(v)  # e.g.: 1.123456 or  5.123e-11 convert to float
               
               if str(v).find("e")>0:
                  ctrl = SpinCtrlScientific(self,name="float")
                  # ctrl = wx.SpinCtrlDouble(self,inc=self.float_inc,name="float",style=wx.SP_ARROW_KEYS)
               else:
                  ctrl = wx.SpinCtrlDouble(self,inc=self.float_inc,name="float",style=wx.SP_ARROW_KEYS)
                  ctrl.Digits = self.float_digits
                  ctrl.Min    = self.float_min
                  ctrl.Max    = self.float_max
                  if v < ctrl.Min:
                     ctrl.Min = abs(v) * -2.0
                  if v > ctrl.Max:
                     ctrl.Max = abs(v) * 2.0
               
               ctrl.Value = v
               child = self.AppendItem(root,"{}".format(k),wnd=ctrl)
           
           else: # None => txt
               ctrl = wx.TextCtrl(self,-1,style=wx.TE_LEFT,value="None",name="NoneStr")
               sz = ctrl.GetSizeFromTextSize(ctrl.GetTextExtent("W" * txt_size))
               ctrl.SetInitialSize(sz)
               child = self.AppendItem(root,"{}".format(k),wnd=ctrl,ct_type=0)     
        
           item_data[k]=ctrl
           try:
              self.SetPyData(child,data[k])
           except:
              logger.exception("key: {}\n -> data: {}".format(k,data.get(k))) 
  
   def ClickOnShowDLG(self,evt):
       """
       shows File, DirDialog depending on file extention
       :param evt:
       :return:
       """
       try:
           obj = evt.GetEventObject()
       except:
           obj = evt
     # txt ctrl
       p = jb.expandvars( obj.GetValue() )
       if os.path.isdir(p):
          with wx.DirDialog(self,message=obj.GetName(),defaultPath=p,style=wx.DD_DEFAULT_STYLE,name=obj.GetName() + "_DLG") as DLG:
               DLG.SetPath( p )
               if DLG.ShowModal() == wx.ID_CANCEL:
                  return     # the user changed their mind
               obj.SetValue( DLG.GetPath() )
       else:
           fext = p.rsplit(".",1)[-1]
           wc = "files (*." +fext+",*.*)|*."+fext+";*.all"
           with wx.FileDialog(self,"{} => Select File Name".format(obj.GetName()),wildcard=wc,
                              style=wx.DD_DEFAULT_STYLE,
                              name=obj.GetName() + "_DLG") as DLG:
               DLG.SetPath(p)
               if DLG.ShowModal() == wx.ID_CANCEL:
                   return  # the user changed their mind
               obj.SetValue( DLG.GetPath() )
   
   def update_info(self):
       '''
       updates the time,version and user
       '''
       now = datetime.datetime.now()
       dt  = now.strftime('%Y-%m-%d')+" "+now.strftime('%H:%M')
       
       self._info={
                   "user": getpass.getuser(),
                   "time": dt,
                   "gui-version": __version__,
                   "python-version": platform.python_version()
                  }
       
       return self._info
       
   def update_used_dict(self):
      '''
      updates the used_dict i.e. the dict used for process
      '''
      self._used_dict=self.GetData()
      
   
   def info(self):
       logger.info("config info:\n {}\n".format(pprint.pformat(self.GetData(),indent=4)))
       
   def _wx_init(self,**kwargs):
    
       data = kwargs.get("data",{ })
       if not data: return
  
       item_data = dict()
      
      #--- get default sorted keys
       '''
       _keys:
         _root_keys: ["info","global","noise_reducer","suggest_bads","interpolate_bads","ica","report"]
         _sorted_keys: ["run","overwrite","save","plot","plot_show","plot_dir"]
       '''
       wxkeys = data.get("_keys")
       if wxkeys:
          self._sorted_keys = wxkeys.get(self._sorted_key,[])
        #--- get a sorted root key list avoid double items
          skeys = wxkeys.get(self._root_key,[] )
          keys  = skeys + [ k for k in list( data ) if k not in skeys ]
          item_data["_keys"]= deepcopy( wxkeys )
       else:
          keys = list( data ) #data.keys() => view of obj !!!
       
       if "info" in keys:
          keys.remove("info")
       
      #---
       self.root_name    = kwargs.get("root_name", self.root_name)
       self.root         = self.AddRoot(kwargs.get("root",self.root_name))
       self._info        = data.get("info")
      
       for k in keys:
           if k.startswith("_"): continue
           d = data.get(k,None)
           if isinstance(d,(dict)):
               item_data[k]=dict()
               child = self.AppendItem(self.root,"{}".format(k))
               self._init_tree_ctrl( data=d ,root=child,item_data=item_data[k])
               self.AppendSeparator(self.root)
               
       self._item_data = item_data
       
       self.Expand(self.root)

class JuMEG_wxConfig(wx.Panel):
    def __init__(self,parent,**kwargs):
        super().__init__(parent)
        self.root_name="jumeg"
        self.SetName(kwargs.get("name","test"))
        self._CfgTreeCtrl = None
        self._CfgTreeCtrlPNL = None
        self._wx_init(**kwargs)
        self._ApplyLayout()
    
    @property
    def verbose(self): return self.CFG.verbose

    @verbose.setter
    def verbose(self,v):
        self.CFG.verbose = v
        if self.CfgTreeCtrl:
           self.CfgTreeCtrl.verbose = v
           
    @property
    def CFG(self): return self._CFG
    
    @property
    def CfgTreeCtrl(self): return self._CfgTreeCtrl
    
    def _wx_init(self,**kwargs):
        self.SetBackgroundColour(wx.GREEN)
        self._CfgTreeCtrlPNL = wx.Panel(self)

        self._init_cfg(**kwargs)

        #--- init buttons
        #fehlerhaft show button
        self._bt_open = wx.Button(self,label="Open",name=self.GetName()+".BT.OPEN")
        self._bt_info  = wx.Button(self,label="Show", name=self.GetName()+".BT.SHOW")
        self._bt_save  = wx.Button(self,label="Save", name=self.GetName()+".BT.SAVE")
        self._bt_update =wx.Button(self,label="Update", name=self.GetName()+".BT.UPDATE")
        self._bt_close = wx.Button(self,label="Close",name=self.GetName()+".BT.CLOSE")
        
        self.Bind(wx.EVT_BUTTON,self.ClickOnButton)
    
    def _update_TreeCtrl(self):
        if self._CfgTreeCtrl:
           self._CfgTreeCtrl._clear()
           self.CfgTreeCtrl.update(data=self.CFG.GetDataDict(),root_name=self.root_name)
        else:
         #--- init & pacl ctrl
           self._CfgTreeCtrl = JuMEG_ConfigTreeCtrl(self._CfgTreeCtrlPNL,root_name=self.root_name,data=self.CFG.GetDataDict())
           self.CfgTreeCtrl.verbose = self.verbose
           LEA = wx.LEFT | wx.EXPAND | wx.ALL
           vbox = wx.BoxSizer(wx.VERTICAL)
           vbox.Add(self._CfgTreeCtrl,1,LEA,4)
           self._CfgTreeCtrlPNL.SetSizer(vbox)
           self._CfgTreeCtrlPNL.SetAutoLayout(True)
           self._CfgTreeCtrlPNL.Fit()
           self._CfgTreeCtrlPNL.Layout()
           self.Layout()

    def _init_cfg(self, **kwargs):
        self._CFG = JuMEG_CONFIG(**kwargs)
        if self.CFG.update(**kwargs):
           self._update_TreeCtrl() 
        
    def FDLGSave(self, event=None):
        '''
        opens a menu to save the current data into a .yaml file
        '''
        with wx.FileDialog(self, "Save config file", wildcard="config files (*.yaml,*.json)|*.yaml;*.json",
                          style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT) as FDGL:
             FDGL.SetDirectory(os.path.dirname(self.CFG.filename))
             if FDGL.ShowModal() == wx.ID_CANCEL:
                return     # the user changed their mind
   
           # save the current contents in the file
             fname,ext = FDGL.GetPath().rsplit(".",1)
             
             if ext in ["yaml","json"]:
                pathname = FDGL.GetPath()
             else:
                pathname = fname+".yaml"
                         
             try:
                 data = self.CfgTreeCtrl._used_dict
                 self.CFG.save_cfg(fname=pathname,data=data)
             except IOError:
                 wx.LogError("ERROR Can not save current data in config file '%s'." % pathname)
                 
    def FDLGOpen(self, event=None):
       '''
       opens a dialogue to load a [.yaml|.json] file and build a tree out of it
       '''
       # otherwise ask the user what new file to open
       with wx.FileDialog(self, "Open config file", wildcard="config files (*.yaml,+.json)|*.yaml;*.json",
                          style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST) as FDGL:
           
           p = "."
           if self.CFG.filename:
              p = os.path.dirname(self.CFG.filename)
           FDGL.SetDirectory(p)
           
           if FDGL.ShowModal() == wx.ID_CANCEL:
               return None    # the user changed their mind
   
           # Proceed loading the file chosen by the user
           return FDGL.GetPath()
            
    def ClickOnOpenConfigFile(self):
        fcfg = self.FDLGOpen()
        if not fcfg: return False
      
        if self.CfgTreeCtrl: # one config file is loaded 
           if wx.MessageBox("Do you want to save?", "Please confirm",wx.ICON_QUESTION | wx.YES_NO, self) == wx.YES:
              self.ClickOnSaveConfigFile()
                   
        self.CFG.update(config=fcfg)
        self._update_TreeCtrl()
        
    def ClickOnSaveConfigFile(self):
        self._CfgTreeCtrl.update_info()
        self._CfgTreeCtrl.update_used_dict()
        self.FDLGSave()
        
    def ClickOnButton(self,evt):
        '''
        implements the show, save, update and open buttons
        :param evt: the button which has been clicked
        '''
        obj = evt.GetEventObject()
        if obj.GetName().endswith(".BT.SHOW"):
           self.CfgTreeCtrl.info()
        elif obj.GetName().endswith(".BT.SAVE"):
           self.ClickOnSaveConfigFile()
           evt.Skip()
        elif obj.GetName().endswith(".BT.UPDATE"):
           self._CfgTreeCtrl.update_used_dict()
        elif obj.GetName().endswith(".BT.OPEN"):
           self.ClickOnOpenConfigFile()
           evt.Skip()
        else:
           evt.Skip()

    def _ApplyLayout(self):
        LEA = wx.LEFT | wx.EXPAND | wx.ALL
        
        vbox = wx.BoxSizer(wx.VERTICAL)
        #---
        st1 = wx.StaticLine(self)
        st1.SetBackgroundColour("GREY85")
        st2 = wx.StaticLine(self)
        st2.SetBackgroundColour("GREY80")
        
        vbox.Add(st1,0,LEA,1)
        
        st = wx.StaticLine(self)
        st.SetBackgroundColour("GREY85")
  
        vbox.Add(self._CfgTreeCtrlPNL,1,LEA,1)
        vbox.Add(st,0,LEA,1)
          
       #--- buttons
        hbox= wx.BoxSizer(wx.HORIZONTAL)
        hbox.Add(self._bt_close,0,LEA,2)
        hbox.Add((0,0),1,LEA,2)
        hbox.Add(self._bt_update,0,LEA,2)
        hbox.Add(self._bt_info,0,LEA,2)
        hbox.Add(self._bt_save,0,LEA,2)
        hbox.Add(self._bt_open,0,LEA,2)
        vbox.Add(hbox,0,LEA,2)
      
        self.SetAutoLayout(True)
        self.SetSizer(vbox)
        self.Fit()
        self.Layout()

class MainWindow(wx.Frame):
  def __init__(self, parent, title, **kwargs):
    wx.Frame.__init__(self, parent, -1,title=title)
    self._wx_init(**kwargs)
   
  def _update_from_kwargs(self,**kwargs):
      pass
  
  def _wx_init(self,**kwargs):
        w,h = wx.GetDisplaySize()
        self.SetSize(w/4.0,h/3.0)
        self.Center()
        
        self._update_from_kwargs(**kwargs)
      #--- init STB in a new CLS
        self._STB = self.CreateStatusBar()
        self._STB.SetStatusStyles([wx.SB_SUNKEN])
        
        self._PNL = JuMEG_wxConfig(self,**kwargs)
        
        self.Bind(wx.EVT_BUTTON,self.ClickOnButton)
        self.Bind(wx.EVT_CLOSE,self.ClickOnClose)
      
  def ClickOnButton(self,evt):
      '''
      implements the close button event or skips the event
      '''
      obj = evt.GetEventObject()
      if obj.GetName().endswith("CLOSE"):
        self.Close()
      if obj.GetName().endswith("OPEN"):
        self._STB.SetStatusText(self._PNL.CFG.fname)
      if obj.GetName().endswith("SAVE"):
          self._STB.SetStatusText(self._PNL.CFG.fname)

      else:
          evt.Skip()

  def ClickOnClose(self,evt):
      '''
      implements the close button event
      '''
      #--- place to clean your stuff
      evt.Skip()
     
#---
def run(opt):
    '''
    runs the project
    '''
    if opt.debug:
        opt.verbose = True
        opt.debug = True
        opt.path = "./config/"
        #opt.config = "test_config.yaml"
        opt.config = "test_config.json"
    
    app = wx.App()
    
    if opt.path:
        cfg = os.path.join(opt.path,opt.config)
    else:
        cfg = opt.config
    
    frame = MainWindow(None,'JuMEG Config',config=cfg,verbose=opt.verbose,debug=opt.debug)
    frame.Show()
    
    app.MainLoop()

#----
def get_args(argv):
    info_global = """
     JuMEG Config GUI Start Parameter

     ---> view time series data FIF file
      jumeg_cfg_gui01.py --config=test_config.yaml --path=./config -v

    """
    
    parser = argparse.ArgumentParser(info_global)
    
    parser.add_argument("-p","--path",help="config file path")
    parser.add_argument("-cfg","--config",help="config file name")
    
    parser.add_argument("-v","--verbose",action="store_true",help="verbose mode")
    parser.add_argument("-d","--debug",action="store_true",help="debug mode")
    
    #--- init flags
    # ck if flag is set in argv as True
    # problem can not switch on/off flag via cmd call
    opt = parser.parse_args()
    for g in parser._action_groups:
        for obj in g._group_actions:
            if str(type(obj)).endswith('_StoreTrueAction\'>'):
                if vars(opt).get(obj.dest):
                    opt.__dict__[obj.dest] = False
                    for flg in argv:
                        if flg in obj.option_strings:
                            opt.__dict__[obj.dest] = True
                            break
    
    return opt,parser


#=========================================================================================
#==== MAIN
#=========================================================================================
if __name__ == "__main__":
    opt,parser = get_args(sys.argv)
    
    """if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(-1)"""
    
    jumeg_logger.setup_script_logging(name=sys.argv[0],opt=opt,logger=logger)
    
    run(opt)

