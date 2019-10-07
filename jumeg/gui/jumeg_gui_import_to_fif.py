#!/usr/bin/env python3
# -+-coding: utf-8 -+-

"""
JuMEG GUI to merge MEG (FIF) and EEG data (BrainVision)
call <jumeg_merge_meeeg> with meg and eeg file and parameters
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
import wx
from   pubsub import pub

#--- jumeg wx stuff
from jumeg.gui.wxlib.jumeg_gui_wxlib_main_frame           import JuMEG_wxMainFrame
from jumeg.gui.wxlib.jumeg_gui_wxlib_main_panel           import JuMEG_wxMainPanel
from jumeg.gui.wxlib.utils.jumeg_gui_wxlib_utils_controls import JuMEG_wxControlGrid
#--- Merger CTRLs
from jumeg.gui.wxlib.jumeg_gui_wxlib_experiment_template  import JuMEG_wxExpTemplate
from jumeg.gui.wxlib.jumeg_gui_wxlib_psel_bti             import JuMEG_wxPselBTi
#---
from jumeg.gui.wxlib.jumeg_gui_wxlib_pbshost              import JuMEG_wxPBSHosts
from jumeg.gui.jumeg_gui_wx_argparser                     import JuMEG_GUI_wxArgvParser
#---
from jumeg.base.jumeg_base                                import jumeg_base as jb
from jumeg.base.ioutils.jumeg_ioutils_subprocess          import JuMEG_IoUtils_SubProcess


import logging
from jumeg.base import jumeg_logger
jumeg_logger.setup_script_logging()
logger = logging.getLogger('jumeg') # init a logger

__version__="2019.09.27-001"

class JuMEG_wxImport_BTiCtrlGrid(wx.Panel):
    def __init__(self,parent,*kargs,**kwargs):
        super().__init__(parent,*kargs)
        self._names={"stage_ck":"CK_STAGE","stage_txt":"TXT_STAGE","emptyroom_ck":"CK_EMPTYROOM","emptyroom_txt":"TXT_EMPTYROOM",
                    "path_flbt":"FLBT_PATH","path_combo":"COMBO_PATH"}
        self._ctrl_gird          = None
        self._init(**kwargs)
    
    @property
    def CtrlGrid(self): return self._ctrl_grid

    #--- fif stage txt
    @property
    def StagePostfix(self):
        return self._get_ctrl("stage_txt").GetValue()

    @StagePostfix.setter
    def StagePostfix(self,v):
        self._get_ctrl("stage_txt").SetValue(v)

    #--- fif stage ck
    @property
    def UseStagePostfix(self):
        return self._get_ctrl("stage_ck").GetValue()

    @UseStagePostfix.setter
    def UseStagePostfix(self,v):
        self._get_ctrl("stage_ck").SetValue(v)

    #--- emptyroom txt
    @property
    def EmptyroomPostfix(self):
        return self._get_ctrl("emptyroom_txt").GetValue()

    @EmptyroomPostfix.setter
    def EmptyroomPostfix(self,v):
        self._get_ctrl("emptyroom_txt").SetValue(v)

    #--- emptyroom ck
    @property
    def UseEmptyroom(self):
        return self._get_ctrl("emptyroom_ck").GetValue()

    @UseEmptyroom.setter
    def UseEmptyroom(self,v):
        self._get_ctrl("emptyroom_ck").SetValue(v)

    #--- Bti Path Combobox
    @property
    def BtiPath(self):
        return self._get_ctrl("path_combo").GetValue()

    @BtiPath.setter
    def BtiPath(self,v):
        if isinstance(v,(list)):
            self.CtrlGrid.UpdateComBox(self._get_ctrl("path_combo"),v)
        else:
            self._get_ctrl("path_combo").SetValue(v)

    def _get_name(self,k): return self._names[k]
    
    def _get_ctrl(self,k):
        return self.FindWindowByName( self.GetName().upper() +"."+ self._names[k])
    
    def EmptyroomCheckbox(self): return self._get_ctrl("emptyroom_ck")
    
    def _init(self,**kwargs):
        self._update_from_kwargs(**kwargs)
        self._wx_init(**kwargs)
        self._ApplyLayout()
        
    def _update_from_kwargs(self,**kwargs):
        self.SetName( kwargs.get("name","BTI_CTRL") )
        self._bti_path = kwargs.get("bti_path",[])
        
    def _wx_init(self,**kwargs):
       #---  export parameter -> CLS
        ctrls = [
                ["CK",   self._get_name("stage_ck"),"Stage Postfix",True,"add postfix to fif stage <experiment name>/mne", None],
                ["TXT",  self._get_name("stage_txt"),"","postfix append to fif stage <experiment name>/mne",None],
                ["CK",   self._get_name("emptyroom_ck"),"Empty Room Files",True,"looking for <subject id> and <scan> the last run in the last session and define it as <emptyroom> file",None],
                ["TXT",  self._get_name("emptyroom_txt"),"","emptyroom postfix append to fif file if this is an empty-romm measurement",None],
                ["FLBT", self._get_name("path_flbt"),"BTi Export Path", "select path to search and export BTi PDFs",None],
                ["COMBO",self._get_name("path_combo"),"BTI_PATH",self._bti_path,"select bti path to export data from",None]]
     
        for i in range(len(ctrls)):
            if ctrls[i][1].startswith(ctrls[i][0]): #TXT CK
               ctrls[i][1] = self.GetName().upper() + "." + ctrls[i][1]
            else:
               ctrls[i][1] = self.GetName().upper() + "." + ctrls[i][0] + "_" + ctrls[i][1]
               
        self._ctrl_grid = JuMEG_wxControlGrid(self,label="FIF Export",drawline=True,control_list=ctrls,cols=2,AddGrowableCol=[1],set_ctrl_prefix=False)
        self._ctrl_grid.SetBackgroundColour("grey90")
       
        self.Bind(wx.EVT_COMBOBOX,self.ClickOnCtrls)
        self.Bind(wx.EVT_CHECKBOX,self.ClickOnCtrls)
 
   #---
    def ClickOnCtrls(self,evt):
        """ pass to parent event handlers """
        evt.Skip()

    def _ApplyLayout(self):
        """ default Layout Framework """
        self.Sizer = wx.BoxSizer(wx.VERTICAL)
        self.Sizer.Add(self.CtrlGrid,1,wx.ALIGN_LEFT | wx.EXPAND | wx.ALL,1)
        self.Fit()
        self.SetAutoLayout(1)
        self.GetParent().Layout()

class JuMEG_wxImportToFIFPanel(JuMEG_wxMainPanel):
      """
      """
      def __init__(self, parent,**kwargs):
          super().__init__(parent,name="JUMEG_IMPORT_TO_FIF_PANEL",ShowTitleB=True)
        
          self.ShowCmdButtons= True
          self.ShowTopPanel  = True
          self.ShowTitleA    = True
          self.ShowTitleB    = True
          self.ShowMinMaxBt  = True
          
          self._template_panel_name = "EXPERIMENT_TEMPLATE"
          
          self.module_path      = os.getenv("JUMEG_PATH_JUMEG", os.getcwd()) + "/tools"
          self.module_name      = "jumeg_io_import_to_fif"
          self.module_extention = ".py"
          self.SubProcess       = JuMEG_IoUtils_SubProcess()
          self._init(**kwargs)

      @property
      def fullfile(self):
          return self.module_path + "/" + self.module_name + self.module_extention

      def GetFIFStage(self):
          """
          fif stage from experimnet -stage and
          if  FIF stage postfic is checked add postfix : <experiment name> /<mne>
          to stage
          
          :return: fif stage
          
          experiment name   :  "M100"
          fif_stage         :  /data/exp
          fif_stage_post_fix: <M100> + "/mne"
          
          fif_stage = /data/exp/M100/mne
          """
          fif_stage = os.path.expandvars( os.path.expanduser(self.ExpTemplate.GetStage() ))
         
          if self.BtiCtrl.UseStagePostfix:
             fif_stage_postfix = self.BtiCtrl.StagePostfix
          if not fif_stage.endswith(fif_stage_postfix):
             fif_stage += "/"+fif_stage_postfix
          # os.path.expandvars(os.path.expanduser(fif_stage))
          
          return fif_stage
          
      def update(self,**kwargs):
          self.stage  = kwargs.get("stage", self.module_path)
          self.module = kwargs.get("module",self.module_name)
         #-- update wx CTRLs
          ds=1
          LEA = wx.ALIGN_LEFT | wx.EXPAND | wx.ALL
         #-- Top
          self.ExpTemplate = JuMEG_wxExpTemplate(self.TopPanel,name=self.GetName() + "." + self._template_panel_name)
          self.HostCtrl    = JuMEG_wxPBSHosts(self.TopPanel, prefix=self.GetName())
          self.TopPanel.GetSizer().Add(self.ExpTemplate,3,LEA,ds)
          self.TopPanel.GetSizer().Add(self.HostCtrl,1, wx.ALIGN_RIGHT | wx.EXPAND | wx.ALL,ds)
         #--- A IDs;PDFs
          self.PDFBox = JuMEG_wxPselBTi(self.PanelA.Panel,name=self.GetName() + ".PDFBOX_BTI",**kwargs)
          self.PanelA.Panel.GetSizer().Add(self.PDFBox,1,LEA,ds)
          self.PanelA.SetTitle("BTi PDFs")
         #--- B right
         #--- PanelB -> combobox for <template.experimnet.bti_data.bti_path> selection
          
          bti_pnl = wx.Panel(self.PanelB.Panel,-1)
         #--- bti ctrl parameter , stage,emptyroom
          self.BtiCtrl = JuMEG_wxImport_BTiCtrlGrid(bti_pnl,name=self.GetName()+".BTI_CTRL")
         #--- import parameter
          self.AP = JuMEG_GUI_wxArgvParser(bti_pnl,name=self.GetName()+".AP",use_pubsub=self.use_pubsub,fullfile=self.fullfile,
                                           module=self.module_name,ShowParameter=True)
             
          vbox = wx.BoxSizer(wx.VERTICAL)
          vbox.Add(self.BtiCtrl,0,LEA,ds)
          vbox.Add(self.AP,1,LEA,ds)
          bti_pnl.SetAutoLayout(True)
          bti_pnl.SetSizer(vbox)
          bti_pnl.Fit()
          
          self.PanelB.Panel.GetSizer().Add(bti_pnl,1,LEA,ds)
          self.PanelB.SetTitle("BTi Export & MNE Parameter")
         #---
          self.Bind(wx.EVT_BUTTON,  self.ClickOnCtrls)
          self.Bind(wx.EVT_COMBOBOX,self.ClickOnCtrls)
          self.Bind(wx.EVT_CHECKBOX, self.ClickOnCtrls)
          self.update_parameter()
          
      def update_on_display(self):
          self.SplitterAB.SetSashPosition(self.GetSize()[0] / 2.0,redraw=True)

      def update_parameter(self):
          """
          update argparser default parameter from template
          set choices in BTiCTLs, set value to first item
          """
          for k in self.ExpTemplate.TMP.bti_data.keys():
              #print(k)
              #print(self.ExpTemplate.TMP.bti_data.get(k))
              self.AP.update_parameter(k,self.ExpTemplate.TMP.bti_data.get(k))
        
         #--- search box pattern matching
          self.PDFBox.SearchBox.SetValue("PDF",self.ExpTemplate.TMP.bti_data.get("pdf_name","c,rfDC"))
          
          self.BtiCtrl.BtiPath          = self.ExpTemplate.IOUtils.expandvars(self.ExpTemplate.TMP.bti_data.get("bti_path"))
          self.BtiCtrl.BtiPath          = self.ExpTemplate.TMP.bti_data.get("bti_path")[0]
          self.BtiCtrl.StagePostfix     = self.ExpTemplate.TMP.name+"/mne"
          self.BtiCtrl.EmptyroomPostfix = self.ExpTemplate.TMP.bti_data.get("emptyroom","empty")
          
      def init_pubsub(self,**kwargs):
          """ init pubsub call overwrite """
          pub.subscribe(self.ClickOnExperimentTemplateUpdate,self.ExpTemplate.GetMessage("UPDATE"))
         
      def ClickOnExperimentTemplateSelectExperiment(self,stage=None,scan=None,data_type=None):
          """
          
          :param stage:
          :param scan:
          :param data_type:
          :return:
          """
          self.update_parameter()
      
      def ClickOnExperimentTemplateUpdate(self,stage=None,scan=None,data_type=None):
          """
          
          :param stage:
          :param scan:
          :param data_type:
          :return:
          """
          self.AP.SetParameter(pdf_stage=self.BtiCtrl.BtiPath)
          self.PDFBox.update(stage=self.BtiCtrl.BtiPath,scan=scan,pdf_name=self.AP.GetParameter("pdf_fname"),
                             emptyroom=self.BtiCtrl.UseEmptyroom,verbose=self.verbose,debug=self.debug)
   
      def ClickOnApply(self):
          """
          apply to subprocess
          """
          self.PDFBox.verbose = self.verbose
          pdfs = self.PDFBox.GetSelectedPDFs()  # returns list of list [[pdf,extention] ..]
          if not pdfs :
             wx.CallAfter( pub.sendMessage,"MAIN_FRAME.MSG.ERROR",data="\nPlease select PDFs first\n in: "+self.GetName())
             return

          cmd_command = self.AP.get_fullfile_command(ShowFileIO=False)
          joblist     = []

         #--- del  "--fif_extention="
          cmd_list = []
          for s in cmd_command.split():
              if s.startswith("--fif"):
                 continue #del cmd_list[idx]
              cmd_list.append(s)
            
          cmd_command = " ".join(cmd_list)
          fif_stage   = self.GetFIFStage()
          logger.info("FIF stage 2: {}".format(fif_stage))
          
          for pdf in pdfs:
              cmd  = cmd_command
              #cmd += " --pdf_stage=" + os.path.expandvars( os.path.expanduser(os.path.dirname( self.PDFBox.GetStage() +"/"+ pdf )))
              cmd += " --pdf_stage="    + self.PDFBox.GetStage() + "/" + os.path.dirname(pdf[0])
              cmd += " --fif_stage="    + fif_stage
              cmd += " --fif_extention="+ pdf[1]
              joblist.append( cmd )
          
         # logger.info(joblist)
          
          if self.verbose:
             wx.LogMessage(jb.pp_list2str(joblist, head="MEEG Merger Job list: "))
             wx.LogMessage(jb.pp_list2str(self.HostCtrl.HOST.GetParameter(),head="HOST Parameter"))
          if joblist:
             # wx.CallAfter(pub.sendMessage,"SUBPROCESS.RUN.START",jobs=joblist,host_parameter=self.HostCtrl.HOST.GetParameter(),verbose=self.verbose)
             wx.CallAfter(self.SubProcess.run,jobs=joblist,host_parameter=self.HostCtrl.HOST.GetParameter(),verbose=self.verbose)
             
      def ClickOnCancel(self,evt):
          wx.LogMessage( "Click <Cancel> button" )
          #wx.CallAfter(pub.sendMessage,"MAIN_FRAME.MSG.INFO",data="<Cancel> button is no in use")
          self.SubProcess.Cancel()
          
      def ClickOnCtrls(self, evt):
          obj = evt.GetEventObject()
          #print("\n ---> ClickOnCTRL:".format( self.GetName() ))
          #print("OBJ Name => "+ obj.GetName() )
          
          if obj.GetName() == self.ExpTemplate.wxExpCb.GetName():
             self.update_parameter()
          elif obj.GetName() == self.GetName()+".BT.APPLY":
             self.ClickOnApply()
          elif obj.GetName() == self.GetName()+".BT.CLOSE":
             wx.CallAfter( pub.sendMessage,"MAIN_FRAME.CLICK_ON_CLOSE",evt=evt)
          elif obj.GetName() == self.GetName()+".BT.CANCEL":
               self.ClickOnCancel(evt)
          elif obj.GetName() == self.BtiCtrl.EmptyroomCheckbox().GetName():
               self.PDFBox.UpdateEmptyroom(self.BtiCtrl.UseEmptyroom)
          else:
             evt.Skip()
             
    
class JuMEG_GUI_ImportToFIFFrame(JuMEG_wxMainFrame):
    def __init__(self,parent,id,title,pos=wx.DefaultPosition,size=[1024,768],name='JuMEG MEEG Merger INM4-MEG-FZJ',*kargs,**kwargs):
        style = wx.DEFAULT_FRAME_STYLE | wx.NO_FULL_REPAINT_ON_RESIZE
        super().__init__(parent,id, title, pos, size, style, name,**kwargs)
        self.template_path = None

    def update(self,**kwargs):
        return JuMEG_wxImportToFIFPanel(self,**kwargs)

    def UpdateAboutBox(self):
        self.AboutBox.description = "importing 4D/BTI data to FIF format"
        self.AboutBox.version     = __version__
        self.AboutBox.copyright   = '(C) 2019 Frank Boers <f.boers@fz-juelich.de>'
        self.AboutBox.developer   = 'Frank Boers'
        self.AboutBox.docwriter   = 'Frank Boers'

if __name__ == '__main__':
   app = wx.App()
   frame = JuMEG_GUI_ImportToFIFFrame(None,-1,'JuMEG Import To FIF',ShowLogger=True,ShowCmdButtons=True,ShowParameter=True,debug=False,verbose=True)
   app.MainLoop()
