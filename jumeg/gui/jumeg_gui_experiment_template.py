#!/usr/bin/env python3
# -+-coding: utf-8 -+-

"""
JuMEG GUI to setup an experiment template
"""

#--------------------------------------------
# Authors: Frank Boers <f.boers@fz-juelich.de>
#
#--------------------------------------------
# Date: 21.11.18
#--------------------------------------------
# License: BSD (3-clause)
#--------------------------------------------
# Updates
#--------------------------------------------

import os
import json
import wx
from   pubsub import pub

import wx.propgrid as wxpg
#---
from jumeg.base.jumeg_base                                import jumeg_base as jb
#--- jumeg wx main stuff
from jumeg.gui.wxlib.jumeg_gui_wxlib_main_frame           import JuMEG_wxMainFrame
from jumeg.gui.wxlib.jumeg_gui_wxlib_main_panel           import JuMEG_wxMainPanel
#--- experiment template panel and exp template
from jumeg.gui.wxlib.jumeg_gui_wxlib_experiment_template  import JuMEG_wxExpTemplate
#--- parameter/properties
from jumeg.gui.wxlib.utils.jumeg_gui_wxlib_utils_controls      import JuMEG_wxMultiChoiceDialog,JuMEG_wxControlButtonPanel,JuMEG_wxControls
from jumeg.gui.wxlib.utils.jumeg_gui_wxlib_utils_property_grid import JuMEG_wxPropertyGridPageBase,JuMEG_wxPropertyGridPageNotebookBase,JuMEG_wxPropertyGridSubProperty

__version__='2019.05.14.001'

class JuMEG_wxTMPMakeDirDialog(JuMEG_wxMultiChoiceDialog):
    """
    shows a dialog to select start paths [stages]
    and check buttons to create sub directory structure
    :param: parent
    :param: message
    :param: caption
    :param: choices=[]
   
    Example
    -------
      dlg = JuMEG_wxTMPMakeSubDirDialog(self,"Make Dir Tree for Stages and Paths",self.GetLabel()+" Experiment Template",choices=stages)
      if (dlg.ShowModal() == wx.ID_OK):
         selections = dlg.GetSelections()

    """
    
    def __init__(self,parent,message,caption,choices=[],**kwargs):
        super().__init__(parent,message,caption,choices=choices,**kwargs)
      
    @property
    def MakeExperimnetPaths(self):    return self.chbox_exp_path.GetValue()
    @property
    def MakeSegmentationPaths(self):  return self.chbox_seg_path.GetValue()

    def update(self,**kwargs):
        self.chbox_exp_path = wx.CheckBox(self,-1,'make experiment paths')
        self.chbox_exp_path.SetValue(True)
        self.chbox_seg_path = wx.CheckBox(self,-1,'make segmentation paths')
        self.chbox_seg_path.SetValue(True)
   
        self.CtrlBox.Add(self.chbox_exp_path,0,wx.ALL | wx.EXPAND,5)
        self.CtrlBox.Add(self.chbox_seg_path,0,wx.ALL | wx.EXPAND,5)


class JuMEG_wxTmpPGM_BTIExport(JuMEG_wxPropertyGridPageBase):
    """
    JuMEG_wxTmpPGM_Experiment(label="Experiment",data=data["experiment"])

    bti_export={
              "bti_path"  : ["/data/MEG/meg_store2/megdaw_data21"],
              "bti_suffix": ",rfDC",
              "fif_path"  : [],
              "fif_suffix":"-raw.fif",
              "scan"      : null,
              "id"        : [],
              "emptyroom" : true,
              "overwrite" : false,
              "fakesHS"   : false
             }
    """
    
    def __init__(self,parent,**kwargs):
        super().__init__(parent,**kwargs)
    
    def update(self,**kwargs):
       #--- 1.1 global props
        self.PGMPage.Append(wxpg.PropertyCategory("3.1 - 4D/BTI-Export"))
        for k in self._data:
            if k.endswith("_path"):
               self.PGMPage.Append(wxpg.ArrayStringProperty(k,value=self._data.get(k,[])))
            #--- ToDo Multi selection Dirs wxdemo MultiDirProperty
            # self.PGMPage.Append( wxpg.DirProperty(k,value=self._data.get(k,[])))
            else:
                self._set_property_ctrl(k,self._data.get(k))


class JuMEG_wxTmpPGM_Experiment(JuMEG_wxPropertyGridPageBase):
   """
    JuMEG_wxTmpPGM_Experiment(label="Experiment",data=data["experiment"])
    
    defaults stored in
    jmeg.gui.wxlib.jumeg_gui_wxlib_experiment_template.__DEFAULT_EXPERIMENT_TEMPLATE__
    
   """
   def __init__(self,parent,**kwargs):
       super().__init__(parent,**kwargs)
    
   def update(self,**kwargs):
      #--- 1.1 global props
       self.PGMPage.Append(wxpg.PropertyCategory("1.1 - Global Properties") )
       self.PGMPage.Append(wxpg.StringProperty(label="Name",name="name",value=self._data.get('name',"TEST") ))
       
       for k in ["stages","scans","bads_list"]:
           if k in self._data.keys():
              self.PGMPage.Append(wxpg.ArrayStringProperty(label=k.capitalize().replace("_", " "),name=k,value=self._data.get(k,[])))
      
      #--- 1.2 path props
       self.PGMPage.Append( wxpg.PropertyCategory("1.2 - Stage Properties") )
       if self._data.get("path"):
          self.PGMPage.Append(JuMEG_wxPropertyGridSubProperty("Path",name="stage_path",value=self._data["path"]))
          
      #--- 1.3 Freesurfer Segmentation
       pg_seg      = self.PGMPage.Append( wxpg.PropertyCategory("1.3 - Freesurfer Segmentation") )
      
       if self._data.get("segmentation"):
          for k in self._data["segmentation"].keys():
              if isinstance(self._data["segmentation"][k],dict):
                 self.PGMPage.Append(JuMEG_wxPropertyGridSubProperty(k.capitalize(),name="segmentation_"+k,value=self._data["segmentation"][k]))
              else:
                 self._set_property_ctrl(k,self._data["segmentation"][k])
      
class JuMEG_wxTmpPGNB_Experiment(JuMEG_wxPropertyGridPageNotebookBase):
    """
    show template properties within a wx.PropertyGridManager
    experment parameter
    
    
    :param property_grid_manager:  <wx.PropertyGridManager>
    :param pgm                  :  <wx.PropertyGridManager> short cut
    :param title                : title
    :param data                 : dict; experiment-template <experiment> key,values
    :return
     wx.Propertygrid.Page
    
    :Example:
    ---------
     self.PropertyGridNoteBoook = JuMEG_wxTmpPGNB_Experiment(self.PanelA.Panel,name=self.title.replace(" ","_").upper() + "_TMP_PROP_GRID_NB")
     self.PropertyGridNoteBoook.update(data=data)
    """
    
    def __init__(self,parent,name="PGNB",**kwargs):
        super().__init__(parent,name=name,**kwargs)
        
    def update(self,**kwargs):
        self._init(**kwargs)
        self._types=["experiment","bti_export"]
        
        type="experiment"
        if self._data.get(type):
           self._pgmp[type] = JuMEG_wxTmpPGM_Experiment(self,label="Experiment Parameter",prefix="PGM_EXPTMP_"+type.upper(),data=self._data[type])
           self.AddPage(self._pgmp[type],type.capitalize() )
        
        type="bti_export"
        if self._data.get(type):
           self._pgmp[type] = JuMEG_wxTmpPGM_BTIExport(self,label="4D/BTI-Export",prefix="PGM_EXPTMP_"+type.upper(),data=self._data[type])
           self.AddPage(self._pgmp[type],"4D/BTI-Export" )
   
    def GetData(self):
        """
        overwrite wx.Propertygrid.GetData
        change PG data dict :
        "stage_path"        => experiment[path]
        "segmentation_path" => experiment[segmentation][ṕath]
        to DEFAULT_EXPERIMENT_TEMPLATE dict structure
        """
        data = JuMEG_wxPropertyGridPageNotebookBase.GetData(self)
        data["experiment"]["path"] = dict()
        data["experiment"]["path"] = data["experiment"].pop("stage_path")
        data["experiment"]["segmentation"]         = dict()
        data["experiment"]["segmentation"]["path"] = data["experiment"].pop("segmentation_path")
        return data
        
        
class JuMEG_wxTemplatePanel(JuMEG_wxMainPanel):
   """
   :param: wx.Panel.name <JUMEG_TEMPLATE_PANEL>
   :param: stage start dir  <os.getenv("JUMEG_PATH", os.getcwd()) + "/jumeg/"
   """
   def __init__(self, parent,name="JUMEG_WX_TEMPLATE_PANEL",**kwargs):
       super().__init__(parent,name=name)
       self._data=None
       self._template_panel_name = "EXPERIMENT_TEMPLATE"
       self._init(**kwargs)
     
   def GetDataName(self,data,key="experiment",name="name"):
       """
       :param data: template dict
       :param key: key who stores the template name e.g  to build filename prefix
       :param name:  template name  e.g. the filename prefix
       :return: template name or <None>
       """
       if data.get(key):
          return data[key].get(name)
       return None
   
   def GetExperimentPanelName(self):
       if self.Template:
          return self.GetName()+"."+self.Template.GetName()
       return None
       
   def update_template_panels(self):
       self.Template = JuMEG_wxExpTemplate(self.TopPanel,name=self.GetName()+"."+self._template_panel_name,ShowScan=False,ShowStage=False)
       self.PropertyGridNoteBoook = JuMEG_wxTmpPGNB_Experiment(self.PanelA.Panel,name=self.title.replace(" ","_").upper() + "_TMP_PROP_GRID_NB")
       
   def update(self,**kwargs):
       """
       setup ctrls
       :param: name of the panel
       :param: stage tepmlate start path e.g  env(JUMEG_PATH) or ./jumeg/
       :param: title of the property grid
       
       """
       self.SetName( kwargs.get("name",self.GetName()) )
       self.stage  = kwargs.get("stage", os.getenv("JUMEG_PATH", os.getcwd()) + "/jumeg/" )
       self.title  = kwargs.get("title","Experiment Template Parameter")
       self.update_template_panels()

       ds = 1
       LEA= wx.ALIGN_LEFT | wx.EXPAND | wx.ALL
      #-- Top
       self.TopPanel.GetSizer().Add(self.Template, 1,LEA,ds)
      #--- A
       self.PanelA.SetTitle(v=self.title)
       self.PanelA.Panel.GetSizer().Add(self.PropertyGridNoteBoook,1,LEA,ds)
       
       self.SplitterAB.Unsplit()  # no PanelB
      #--- ctrl buttons will be packed in <AutoLayout>
       bt_ctrls= (["Close","CLOSE",wx.ALIGN_LEFT,"close program",None],
                  ["Make DirTree","MAKE_DIR_TREE",wx.ALIGN_LEFT,"make directory tree for stages and paths",None],
                  ["Save As","SAVE",wx.ALIGN_RIGHT,"SAVE TEMPLATE",None])
 
       self._pnl_cmd_buttons = JuMEG_wxControlButtonPanel(self.MainPanel,label=None,control_list=bt_ctrls)
       self.ShowCmdButtons   = True
       
       self.Bind(wx.EVT_BUTTON,  self.ClickOnCtrl)
       self.Bind(wx.EVT_COMBOBOX,self.ClickOnCtrl)
      #---
       self.UpdatePropertyGrid() #name="default")
   
   def CallUpdatePropertyGrid(self,data=False):
       """ helper function for pussub call from ExperimnetTemplate CLS"""
       
       if data:
          self.UpdatePropertyGrid()
          
   def UpdatePropertyGrid(self,name=None,TMP=None):
       """
        update Template Parameter within a wx.PropertyGrid

        Parameter
        ---------
         name: name of experiment
         TMP : the template structure as dict
       """
       # print(self.Template.TMP._template_data)
       
       if name:
          self.Template.update_template(name=name)
       if TMP:
          data = TMP.template_data
       else:
          data = self.Template.TMP.template_data
      
       if self.verbose:
          wx.LogMessage("Update "+ self.title +" : " + self.Template.GetExperiment() ) #data["experiment"].get("name"))
          wx.LogMessage(jb.pp_list2str(data))
       
       self.PropertyGridNoteBoook.update(data=data)
       
   def init_pubsub(self, **kwargs):
       """ init pubsub call overwrite """
       #pub.subscribe(self.ClickOnApply,self.GetName().upper()+".BT_APPLY")
       pub.subscribe(self.CallUpdatePropertyGrid, self.Template.GetName()+"_UPDATE")
       
  #---
   def Cancel(self):
       print("CLICK ON CANCEL")
       
   def MakeDirTreeDLG(self):
       """
       list satges  as ck boxes
       ck path
       
        ToDo
            new thread set busy + speed bar gauge + cancel bt
            make dirtree stages and paths if path is true
            make segmentation paths
       
       :return:
       """
     #--- get experiment data
       data   = self.PropertyGridNoteBoook.GetData()["experiment"]
       stages = data["stages"]
       dlg    = JuMEG_wxTMPMakeDirDialog(self,"Make Dir Tree for Stages and Paths",self.GetLabel()+" Experiment Template",choices=stages)
       if (dlg.ShowModal() == wx.ID_OK):
           selections = dlg.GetSelections()
           for idx in selections:
               stage=stages[idx] +"/"+data["name"]
               if dlg.MakeExperimnetPaths:
                  l = self.Template.IOUtils.make_dirs_from_list(stage=stage,path_list=data["path"].values())
                  wx.LogMessage("Make Experiment Paths from list:\n {}".format(l))
               if dlg.MakeSegmentationPaths:
                  l=self.Template.IOUtils.make_dirs_from_list(stage=stage,path_list=data["segmentation"]["path"].values())
                  wx.LogMessage("Make Segmentation Paths from list:\n {}\n".format(l))
       dlg.Destroy()
       
   def ClickOnSave(self):
       """
       show File Save DLG
       save template data in json format

       """
       data = self.PropertyGridNoteBoook.GetData()
     #--- make file name
       self.Template.TMP.template_name = self.GetDataName(data)
       #fjson  = self.Template.TMP.template_full_filename
       SaveDLG= wx.FileDialog(self, message='Save Template data.',
                              wildcard='template (*.'+self.Template.TMP.template_extention+')|*.json|All Files|*',
                              style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT|wx.FD_PREVIEW)

       SaveDLG.SetDirectory(self.Template.TMP.template_path)
       SaveDLG.SetFilename(self.Template.TMP.template_filename)

       if SaveDLG.ShowModal() == wx.ID_OK:
         #--- update templatev info
          data["info"] = self.Template.TMP.update_template_info(gui_version=__version__)
          
          fout = os.path.join( SaveDLG.GetDirectory(),SaveDLG.GetFilename() )
          if self.verbose:
             wx.LogMessage(" ---> experiment template             : " + data["experiment"]["name"])
             wx.LogMessage("  --> saving experiment template file : " + fout)
             if self.debug:
                wx.LogDebug("   -> experiment template data: \n" + json.dumps(data,indent=4))
          try:
              with open(fout, "w") as f:
                   f.write(json.dumps(data,indent=4))
                   os.fsync(f.fileno()) # make to disk
              f.close()
              if self.verbose:
                 wx.LogMessage(" ---> done saving experiment template file: " + fout)
            #--- update Experiment Template lists and update ExpTmp ComboBox with new template name
            #---  TEST_jumeg_experiment_template.json  => TEST
              name=SaveDLG.GetFilename().split("_" + self.Template.TMP.template_postfix)[0]
              self.Template.update_template(name=name)
         
          except Exception as e:
              wx.LogError("Save failed!\n" + jb.pp_list2str(e,head="Error writing experiment template file: "+fout) )
              pub.sendMessage("MAIN_FRAME.MSG.ERROR",data="Error writing template file: "+fout)
              raise
          finally:
              SaveDLG.Destroy()
           
              
   def ClickOnCtrl(self, evt):
       """ click on button or combobox send event """
       obj = evt.GetEventObject()
      
       if not obj: return
       
       #--- ExpComBo
       if obj.GetName() == self.Template.wxExpCb.GetName():  #"COMBO.EXPERIMENT":
          self.UpdatePropertyGrid()
       elif obj.GetName()=="MAKE_DIR_TREE":
            self.MakeDirTreeDLG()
       elif obj.GetName() == "CLOSE":
            pub.sendMessage('MAIN_FRAME.CLICK_ON_CLOSE',evt=evt)
       elif obj.GetName() == "CANCEL":
            pub.sendMessage('MAIN_FRAME.CLICK_ON_CANCEL',evt=evt)
       elif obj.GetName() == "SAVE":
            self.ClickOnSave()
       else:
            evt.Skip()

class JuMEG_GUI_ExperimentTemplateFrame(JuMEG_wxMainFrame):
    def __init__(self,parent,id,title,pos=wx.DefaultPosition,size=[1024,768],name='JuMEG Experiment Template',*kargs,**kwargs):
        style = wx.DEFAULT_FRAME_STYLE | wx.NO_FULL_REPAINT_ON_RESIZE
        super().__init__(parent,id, title, pos, size, style, name,**kwargs)
        self.template_path = None
        self.verbose       = False
   #---
    def _update_kwargs(self,**kwargs):
        self.verbose = kwargs.get("verbose",self.verbose)
   #---
    def update(self,**kwargs):
       #---
        self.UpdateAboutBox()
       #---
        self._init_MenuDataList()
       #---
        return JuMEG_wxTemplatePanel(self,name="JuMEG_EXPERIMENT_TEMPLATE_PANEL",**kwargs)
   #---
    def wxInitStatusbar(self):
        self.STB = self.CreateStatusBar(4)
        #self.STB.SetStatusStyles([wx.SB_RAISED,wx.SB_SUNKEN,wx.SB_RAISED,wx.SB_SUNKEN])
        self.STB.SetStatusWidths([-1,1,-1,4])
        self.STB.SetStatusText('Experiment',0)
        self.STB.SetStatusText('Path',2)
   #--- 
    def UpdateAboutBox(self):
        self.AboutBox.name        = self.GetName() #"JuMEG MEEG Merger INM4-MEG-FZJ"
        self.AboutBox.description = self.GetName()#"JuMEG MEEG Merger"
        self.AboutBox.version     = __version__
        self.AboutBox.copyright   = '(C) 2018 Frank Boers'
        self.AboutBox.developer   = 'Frank Boers'
        self.AboutBox.docwriter   = 'Frank Boers'

if __name__ == '__main__':
   app = wx.App()
   frame = JuMEG_GUI_ExperimentTemplateFrame(None,-1,'JuMEG Experiment Template FZJ-INM4',ShowLogger=True,debug=True,verbose=True)
   app.MainLoop()
