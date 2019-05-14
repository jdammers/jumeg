#!/usr/bin/envn python3
# -+-coding: utf-8 -+-
#----------------------------------------
# Created by fboers at 21.09.18
#----------------------------------------
# Update
#----------------------------------------

import os,json
import wx
from pubsub  import pub
from jumeg.gui.wxlib.utils.jumeg_gui_wxlib_utils_controls import JuMEG_wxControls,JuMEG_wxControlGrid,JuMEG_wxControlButtonPanel

__version__="2019.05.14.001"

class JuMEG_PBSHostsParameter(object):
   """

   class to acces host parameter:
   "name","nodes","maxnodes","kernels","maxkernels","python_version"

   """
   __slots__ =["name","nodes","maxnodes","kernels","maxkernels","python_version","cmd_prefix"]

   def __init__(self,**kwargs):
       super().__init__()
       self._init(**kwargs)

   @property
   def hostname(self):    return self.name
   @hostname.setter
   def hostname(self, v): self.name= v

   def get_parameter(self,key=None):
       """
       get  host parameter
       :param key:)
       :return: parameter dict or value of  parameter[key]
       """
       if key: return self.__getattribute__(key)
       return {slot: self.__getattribute__(slot) for slot in self.__slots__}
  
   def _init(self,**kwargs):
      #--- init slots
       for k in self.__slots__:
           self.__setattr__(k,None)
       self._update_from_kwargs(**kwargs)

   def update(self,**kwargs):
       self._update_from_kwargs(**kwargs)

   def _update_from_kwargs(self,**kwargs):
       for k in self.__slots__:
           self.__setattr__(k,kwargs.get(k,self.__getattribute__(k)))
          
       
   def info(self):
       wx.LogMessage(
                     " ==> PBS HOST Info: " + self.hostname +
                     "\nHOST Nodes  : {}  Max Nodes  : {}".format(self.nodes,self.maxnodes) +
                     "\nHOST Kernels: {}  Max Kernels: {}".format(self.kernels,self.maxkernels) +
                     "\nHost Python Version: {}".format(self.python_version) +
                     "\nHost cmd prefix    : {}".format(self.cmd_prefix)
                    )

class JuMEG_PBSHosts(object):
    def __init__(self,**kwargs):
        super().__init__()
        self._template_default = {
                "python_versions": [None, "python", "python2", "python3"],
                "hosts"          : {
                    "local"    : {"name":"local",    "nodes":1,"maxnodes":  1,"kernels":1,"maxkernels":1,"cmd_prefix":"/usr/bin/env"},
                    "mrcluster": {"name":"mrcluster","nodes":1,"maxnodes": 10,"kernels":1,"maxkernels":8,"cmd_prefix":"/usr/bin/env"}
                    }
                }


        self._host = "local"
        self._template_path = os.getenv("JUMEG_PATH_TEMPLATE",os.getcwd())
        self._template_name = "jumeg_host_template.json"
        self.update(**kwargs)

    def hostlist(self): return list(self._template["hosts"].keys())

    def GetParameter(self,key=None):
        if key:  return self._template["hosts"][self._host][key]
        return  self._template["hosts"][self._host]
    
    def GetObj(self,key=None):
        """ do not use for pubsub calls"""
        if key:  return self._template["hosts"][key]
        return  JuMEG_PBSHostsParameter(**self._template["hosts"][self._host])

    def _set_param( self, key, v ):
        self._template["hosts"][self._host][key] = v

    def _get_param( self, key ):
        return self._template["hosts"][self._host].get(key,None)

    @property
    def python_versions( self ): return self._template["python_versions"]
    @python_versions.setter
    def python_versions( self, v ):
        if isinstance(v, (list)):
           self._template["python_versions"] =v

    @property
    def python_version( self ): return str(self._get_param("python_version"))
    @python_version.setter
    def python_version( self,v ):self._set_param("python_version",v)

    @property
    def template_name(self):
        return self._template_name

    @template_name.setter
    def template_name(self, v):
        self._template_name = v

    @property
    def template_path(self):
        return self._template_path

    @template_path.setter
    def template_path(self, v):
        self._template_path = v
    @property
    def template_file(self): return self.template_path +"/"+self.template_name

    @property
    def hostname(self):    return self._host

    @hostname.setter
    def hostname(self, v): self._host = v

    @property
    def nodes(self):    return self._get_param("nodes")

    @nodes.setter
    def nodes(self, v): self._set_param("nodes", v)

    @property
    def maxnodes(self):    return self._get_param("maxnodes")

    @maxnodes.setter
    def maxnodes(self, v): self._set_param("maxnodes", v)

    @property
    def kernels(self):    return self._get_param("kernels")

    @kernels.setter
    def kernels(self, v): self._set_param("kernels", v)

    @property
    def maxkernels(self):    return self._get_param("maxkernels")

    @maxkernels.setter
    def maxkernels(self, v): self._set_param("maxkernels", v)
 
    @property
    def cmd_prefix(self):    return self._get_param("cmd_prefix")

    @maxkernels.setter
    def cmd_prefix(self, v): self._set_param("cmd_prefix", v)

    def _update_from_kwargs(self,**kwargs):
         self._template_path = kwargs.get("template_path",self._template_path)
         self._template_name = kwargs.get("template_name",self._template_name)

    def update(self,**kwargs):
        self._update_from_kwargs(**kwargs)
        self.load_host_template()

    def load_host_template(self):
        """ load host template from <JuMEG TEMPLATE PATH> """

        self._template = self._template_default.copy()

        if ( os.path.isfile( self.template_file ) ):
            with open(self.template_file, 'r') as FID:
                 try:
                     self._template = json.load(FID)
                 except:
                     self._template = self._template_default.copy()
                     wx.LogError("NO JSON File Format:\n  ---> " + self._template_file)

        if not self._template.get("python_versions",None):
           self._template["python_versions"] = self._template_default["python_versions"]

        tmp_hosts = self._template["hosts"]
        for h in tmp_hosts.keys():
            if not tmp_hosts.get("python_version",None):
               tmp_hosts[h]["python_version"]=self.python_versions[-1]

        return self._template


class JuMEG_wxPBSHosts_Parameter(wx.PopupTransientWindow):
    """shows spin buttons for nodes and kernels"""
    def __init__(self, parent, style=wx.SIMPLE_BORDER,host=None):
        super().__init__(parent, style)
        self.HOST = host
        self._wx_init()
        self._ApplyLayout()

    def _wx_init(self, **kwargs):
        ctrls = []
        ctrls.append(("SP", "Nodes",  [1,self.HOST.maxnodes, 1], self.HOST.nodes, "select number of nodes",None))
        ctrls.append(("SP", "Kernels",[1,self.HOST.maxkernels,1],self.HOST.kernels,"select number of kernels/cpus",None))
        ctrls.append(("COMBO","Python Version",self.HOST.python_version,self.HOST.python_versions,"select Python verision to execute",None))
        self.pnl = JuMEG_wxControls(self,label="--- H O S T :  " + self.HOST.hostname.capitalize() +" ---", drawline=True,control_list=ctrls)

    def OnDismiss(self):
        """ copy values to parent HOST obj and destroy window"""
        self.HOST.nodes          = self.pnl.FindWindowByName("SP.NODES").GetValue()
        self.HOST.kernels        = self.pnl.FindWindowByName("SP.KERNELS").GetValue()
        self.HOST.python_version = self.pnl.FindWindowByName("COMBO.PYTHON-VERSION").GetValue()
        #wx.LogMessage( "\nOnDismiss python version: "+ self.HOST.python_version )
        self.Destroy()

    def _ApplyLayout(self):
        """" default Layout Framework """
        self.Sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.Sizer.Add(self.pnl,1, wx.ALIGN_LEFT | wx.EXPAND| wx.ALL,2)
        self.SetSizer(self.Sizer)
        self.Fit()
        self.Layout()

class JuMEG_wxPBSHosts(wx.Panel):
    '''
    HOST Panel
    select Host from list of host e.g. local,cluster,
    select nodes and kernels
    '''
    def __init__(self, parent, **kwargs):
        super().__init__(parent=parent, id=wx.ID_ANY, style=wx.SUNKEN_BORDER)
        self.HOST = JuMEG_PBSHosts(**kwargs)
        self._init(**kwargs)

    def SetVerbose(self,value=False):
        self.verbose = value

    def _update_from_kwargs(self, **kwargs):
        self.verbose = kwargs.get("verbose", False)
        self.debug   = kwargs.get("debug", False)
        self.prefix  = kwargs.get("prefix", "PBS_HOSTS")
        self.bg      = kwargs.get("bg",    wx.Colour([230, 230, 230]))
        self.bg_pnl  = kwargs.get("bg_pnl",wx.Colour([240, 240, 240]))

    def _init_pubsub(self):
        """"
        init pubsub call
        """
        pub.subscribe(self.SetVerbose, 'MAIN_FRAME.VERBOSE')

    def _wx_init(self, **kwargs):
        """ init WX controls """
        self.SetBackgroundColour(self.bg)
       # --- PBS Hosts
        ctrls = []
        ctrls.append(("BT",   "HOST", "Hosts", "update host list",None))
        ctrls.append(("COMBO","HOST", self.HOST.hostlist()[0],  self.HOST.hostlist(), "select a host",None))
        ctrls.append(("BT", "PARAMETER", "Parameter", "change parameter", None,wx.BU_EXACTFIT|wx.BU_NOTEXT))

        self.pnl = JuMEG_wxControlGrid(self,label=None, drawline=False,control_list=ctrls, cols=len(ctrls),AddGrowableCol=[1])
        self.pnl.SetBackgroundColour(self.bg_pnl)
        self.FindWindowByName("COMBO.HOST").SetValue(self.HOST.hostname)

        bt=self.FindWindowByName("BT.PARAMETER")
        bt.SetBitmapLabel(wx.ArtProvider.GetBitmap(wx.ART_INFORMATION, wx.ART_MENU,(12,12)))

        self.Bind(wx.EVT_BUTTON,  self.ClickOnCtrl)
        self.Bind(wx.EVT_COMBOBOX,self.ClickOnCtrl)

    def update(self, **kwargs):
        pass

    def _init(self, **kwargs):
        """" init """
        self._update_from_kwargs(**kwargs)
        self._wx_init()
        self._init_pubsub()
        self.update()
        self._ApplyLayout()

    def OnShowParameter(self, evt):
        #wx.LogMessage(" ==> HOST name: " +self.HOST.hostname)
        wxNK = JuMEG_wxPBSHosts_Parameter(self,style=wx.SIMPLE_BORDER,host=self.HOST)
        btn = evt.GetEventObject()
        pos = btn.ClientToScreen( (0,0) )
        sz =  btn.GetSize()
        wxNK.Position(pos, (0, sz[1]))
        wxNK.Popup()

    def ClickOnCtrl(self, evt):
        obj = evt.GetEventObject()
        if obj.GetName().upper().startswith("COMBO.HOST"):
           self.HOST.hostname = obj.GetValue()
        elif obj.GetName().upper().startswith("BT.PARAMETER"):
             self.OnShowParameter(evt)
             # self.info()

    def _ApplyLayout(self):
        """" default Layout Framework """
        self.Sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.Sizer.Add(self.pnl,1, wx.ALIGN_LEFT | wx.EXPAND| wx.ALL,2)
        self.SetSizer(self.Sizer)
        self.Fit()
        self.SetAutoLayout(1)
        self.GetParent().Layout()
