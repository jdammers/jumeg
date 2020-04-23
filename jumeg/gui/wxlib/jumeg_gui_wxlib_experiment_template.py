#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 14:14:42 2018

@author: fboers
-------------------------------------------------------------------------------
Updates:
2018-08-27.001 new structure, refrac
    

"""
import wx
from pubsub import pub
from jumeg.base.template.jumeg_template_experiments  import JuMEG_ExpTemplate
from jumeg.base.ioutils.jumeg_ioutils_functions      import JuMEG_IOUtils
from jumeg.gui.wxlib.utils.jumeg_gui_wxlib_utils_controls import JuMEG_wxControlGrid

__version__= "2019.05.14.001"

class JuMEG_wxExpTemplate(wx.Panel):
    """
     JuMEG_wxExpTemplate
     select Experiment Template [M100] and stage directory [/data/xyz/exp/M100] from Exp Template folder
     
     Paremeters:
     -----------    
      parent widged 
      template_path: path to experiment templates
      pubsub       : use wx.pupsub msg systen <False>
                     example: pub.sendMessage('EXPERIMENT_TEMPLATE',stage=stage,experiment=experiment,TMP=template_data)
                     or button event from  <ExpTemplateApply> button for apply/update   
      verbose      : <False>
      bg           : backgroundcolor <grey90>

      ShowExp   : show Experiment combobox <True>
      ShowScan  : show Scan combobox       <True>
      ShowStage : show Stage combobox      <True>

    """
    def __init__(self,parent,name="JUMEG_WX_EXPERIMENT_TEMPLATE",**kwargs):
        super().__init__(parent,name=name)
        self.TMP = JuMEG_ExpTemplate(**kwargs)
        self._ctrl_names     = ["EXPERIMENT", "SCAN","STAGE","UPDATE"]
        self.prefixes        = ["BT.","COMBO."]
        self._pubsub_messages={"UPDATE":"UPDATE","SELECT_EXPERIMENT":"SELECT_EXPERIMENT"}
        self.IOUtils         = JuMEG_IOUtils()
        
        self.ShowExp   = True
        self.ShowScan  = True
        self.ShowStage = True
        self._init(**kwargs)

    def _find_obj(self,prefix,postfix):
        """
        find obj by name like <pref>.<self.GetName()>.<postfix>
        :param prefix:
        :param postfix:
        :return: obj
        
        Example:
        --------
        self.SetName("TEST01")
        obj=self._fwn("BT","EXPERIMENT")
        obj.GetName()
        BT.TEST01.EXPERIMENT
        
        """
        return self.FindWindowByName(self.GetName().upper()+"."+prefix+"_"+postfix)
    
    @property
    def wxExpBt(self): return self._find_obj("BT","EXPERIMENT")
    @property
    def wxExpCb(self): return self._find_obj("COMBO","EXPERIMENT")

    @property
    def wxExpScanBt(self): return self._find_obj("FLBT","SCAN")
    @property
    def wxExpScanCb(self): return self._find_obj("COMBO","SCAN")

    @property
    def wxExpStageBt(self):  return self._find_obj("FLBT","STAGE")
    @property
    def wxExpStageCb(self):  return self._find_obj("COMBO","STAGE")

    @property
    def wxExpUpdateBt(self): return self._find_obj("BT","UPDATE")

    @property
    def verbose(self): return self.TMP.verbose
    @verbose.setter
    def verbose(self,v): self.TMP.verbose = v

    def GetExperiment(self):
        return self.wxExpCb.GetValue()

    def GetScan(self):
        if self.wxExpScanCb:
           return self.wxExpScanCb.GetValue()
        return None

    def GetStage( self ):
        if self.wxExpStageCb:
           p = self.TMP.isPath(self.wxExpStageCb.GetValue())
           if p : return p
           pub.sendMessage("MAIN_FRAME.MSG.INFO",data="<Experiment Template GetStage> <stage> no such file or directory:\n" + self.wxExpStageCb.GetValue())
        return
    
    def GetExperimentPath( self ):
        try:
            return self.GetStage() + "/"+ self.GetExperiment()
        except:
            pass

  # --- pubsub msg
  #--- ToDO new CLS
    def GetMessageKey( self, msg ):    return self._pubsub_messages.get(msg.upper())
    def SetMessageKey( self, msg, v ): self._pubsub_messages[msg] = v.upper()

    def GetMessage( self, msg ): return self.GetName()+ "." +self.GetMessageKey(msg)

    def send_message(self,msg,evt):
        """ sends a pubsub msg, can change the message via <MessageKey> but not the arguments
           "EXPERIMENT_TEMPLATE.UPDATE",stage=self.GetStage(),scan=self.GetScan(),data_type='mne'
        """
        if self.pubsub:
           #print("PUBSUB MSG: "+self.GetMessage(msg))
           pub.sendMessage(self.GetMessage(msg),stage=self.GetExperimentPath(),scan=self.GetScan(),data_type='mne')
        else: evt.Skip()

    def _init(self, **kwargs):
        """" init """
        self._update_from_kwargs(**kwargs)
        self._wx_init()
        self.update(**kwargs)
        self._ApplyLayout()

    def _wx_init(self):
        """ init WX controls """
        self.SetBackgroundColour(self.bg)
       # --- PBS Hosts
        ctrls = []
        if self.ShowExp:
           ctrls.append(["BT",   "EXPERIMENT", "Experiment", "update experiment template list",None])
           ctrls.append(["COMBO","EXPERIMENT", "COMBO_EXPERIMENT", [], "select experiment templatew",None])

        if self.ShowScan:
           ctrls.append(["FLBT", "SCAN", "SCAN", "select scan",None])
           ctrls.append(["COMBO","SCAN", "SCAN", [], "select experiment template",None])

        if self.ShowStage:
           ctrls.append(["FLBT", "STAGE", "Stage", "select stage", None])
           ctrls.append(["COMBO","STAGE", "Stage", [], "select experiment satge",None])
           ctrls.append(["BT",   "UPDATE","Update", "update",None])

        for i in range(len(ctrls)):
            ctrls[i][1] = self.GetName().upper()+"."+ctrls[i][0]+"_"+ctrls[i][1]
        
        self.CtrlGrid = JuMEG_wxControlGrid(self, label=None, drawline=False, control_list=ctrls, cols=len(ctrls) + 4,AddGrowableCol=[1,3,5],set_ctrl_prefix=False)
        self.CtrlGrid.SetBackgroundColour(self.bg_pnl)

        self.CtrlGrid.EnableDisableCtrlsByName(self._ctrl_names,False,prefix=self.prefixes)

       #--- bind CTRLs in class
        self.Bind(wx.EVT_BUTTON,  self.ClickOnCtrl)
        self.Bind(wx.EVT_COMBOBOX,self.ClickOnCtrl)

    def _update_from_kwargs(self,**kwargs):
        self.verbose       = kwargs.get("verbose",self.verbose)
        self.pubsub        = kwargs.get("pubsub",True)
        self.bg            = kwargs.get("bg",    wx.Colour([230, 230, 230]))
        self.bg_pnl        = kwargs.get("bg_pnl",wx.Colour([240, 240, 240]))
       #---
        self.ShowExp   = kwargs.get("ShowExp", self.ShowExp)
        self.ShowScan  = kwargs.get("ShowScan",self.ShowScan)
        self.ShowStage = kwargs.get("ShowStage",self.ShowStage)
       #---
       
    def update_template(self,name=None):
        """
        update template experiment name in combobox and template data
        :param name of experiment eg:default
        """
        if name:
           if name in self.TMP.get_sorted_experiments():
              #print("OK update_template: {}".format(name) )
              self.TMP.template_name = name
        
        self.wxExpCb.SetValue(self.TMP.template_name)
        self.TMP.template_update( name )
        
    def update(self,**kwargs):
        """ update  kwargs and widgets """
        self._update_from_kwargs(**kwargs)
        self.UpdateExperimentComBo()
        self.UpdateScanStageComBo( experiment = self.wxExpCb.GetValue() )
        
        #print("EXP TEMPLATE MSG:  162 "+self.GetName()+"_UPDATE")
        #pub.sendMessage(self.GetName()+"_UPDATE",data=True)
        
    def UpdateExperimentComBo(self,evt=None):
        """ update experiment combobox if selected """
        self.CtrlGrid.UpdateComBox(self.wxExpCb,self.TMP.get_experiments(issorted=True))
        self.CtrlGrid.EnableDisableCtrlsByName(self._ctrl_names[0], True, prefix=self.prefixes)  # experiment ctrl first
        self.wxExpCb.SetToolTip(wx.ToolTip("Template path: {}".format(self.TMP.template_path) ))
        if self.verbose:
           wx.LogMessage( self.TMP.pp_list2str(self.TMP.template_name_list,head="Template path: "+self.TMP.template_path))
        
    def UpdateScanComBo( self,scan_list=None ):
        """
        :param scan_list:
        :return:
        """
        if not self.wxExpScanCb: return
        if not scan_list:
           scan_list = self.TMP.get_sorted_scans()
        self.CtrlGrid.UpdateComBox( self.wxExpScanCb,scan_list )
        
        if not self.wxExpStageCb: return
        state = bool(len(scan_list))
        if state:
           self.wxExpStageCb.SetValue( scan_list[0]  )

        self.CtrlGrid.EnableDisableCtrlsByName(self._ctrl_names[1],state,prefix=self.prefixes)

    def UpdateStageComBo(self,stage_list=None):
        """
        :param stage_list:
        :return:
        """
        if not self.wxExpStageCb:return
        if not stage_list:
           stage_list = self.TMP.stages
        
        stage_list = self.IOUtils.expandvars(stage_list)
        self.CtrlGrid.UpdateComBox(self.wxExpStageCb, stage_list)
        
        state = bool(len(stage_list))
        if state:
           self.wxExpStageCb.SetValue(self.wxExpStageCb.GetItems()[0])
        self.CtrlGrid.EnableDisableCtrlsByName(self._ctrl_names[2:],state, prefix=self.prefixes)

    def UpdateScanStageComBo( self,experiment=None ):
        """
        fill scan

        Parameter
        ---------
         experiment name
        """
        if experiment:
           if not self.TMP.template_update( experiment ):
              self.TMP.template_data_reset()
               
           if self.wxExpScanCb:
              self.UpdateScanComBo()
           if self.wxExpStageCb:
              self.UpdateStageComBo()
        else:
            if self.wxExpScanCb:  self.CtrlGrid.UpdateComBox(self.wxExpScanCb, [])
            if self.wxExpStageCb: self.CtrlGrid.UpdateComBox(self.wxExpStageCb,[])
            self.EnableDisableCtrlsByName(self._ctrl_names[1:],status=False,prefix=self.prefixes)
   #---
    def show_stage_dlg(self):
        dlg = wx.DirDialog(None,"Choose Stage directory","",wx.DD_DEFAULT_STYLE | wx.DD_DIR_MUST_EXIST)
        dlg.SetPath(self.wxExpStageCb.GetValue())
        if (dlg.ShowModal() == wx.ID_OK):
            if self.TMP.template_data:
                l = [dlg.GetPath()]
                l.extend(self.TMP.stages)
                self.UpdateComBo(self.wxExpStageCb,l)
        dlg.Destroy()
   #---
    def ClickOnCtrl(self,evt):
        """ click on button or combobox send event
        """
        obj = evt.GetEventObject()
        if not obj: return
        #print("gui wxlib  exptemp ClickOnCtrl: "+self.GetName())
        #print( obj.GetName() )
       #--- ExpComBo
        if obj.GetName() ==  self.wxExpCb.GetName():
           self.UpdateScanStageComBo( obj.GetValue() )
           evt.Skip()
       #--- ExpBt
        elif obj.GetName() == self.wxExpBt.GetName():
           self.update()
           #evt.Skip()
       #--- ExpStageBt start change Dir DLG
        elif obj.GetName() == self.wxExpStageBt.GetName():
             self.show_stage_dlg()
      #--- ExpBt 
        elif obj.GetName() == self.wxExpUpdateBt.GetName():
             self.send_message("UPDATE",evt)
            #evt.Skip()
        else:
            evt.Skip()
  #---      
    def _ApplyLayout(self):
        """ Apply Layout via wx.GridSizers"""
       #--- Label + line
        vbox = wx.BoxSizer(wx.VERTICAL)
        vbox.Add( self.CtrlGrid,1, wx.ALIGN_LEFT|wx.EXPAND|wx.ALL,2)
        self.SetSizer(vbox)
        self.Fit()
        self.SetAutoLayout(1)
        self.GetParent().Layout()

