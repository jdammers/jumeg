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
from pubsub import pub

#--- jumeg wx stuff
from jumeg.gui.wxlib.jumeg_gui_wxlib_main_frame           import JuMEG_wxMainFrame
from jumeg.gui.wxlib.jumeg_gui_wxlib_main_panel           import JuMEG_wxMainPanel
from jumeg.gui.wxlib.utils.jumeg_gui_wxlib_utils_controls import JuMEG_wxSplitterWindow,JuMEG_wxCMDButtons,JuMEG_wxControlGrid,JuMEG_wxControlButtonPanel

#--- Experiment Template
from jumeg.gui.wxlib.jumeg_gui_wxlib_experiment_template  import JuMEG_wxExpTemplate
#---Merger CTRLs
from jumeg.gui.wxlib.jumeg_gui_wxlib_psel_meeg            import JuMEG_wxPselMEEG
#---
from jumeg.gui.wxlib.jumeg_gui_wxlib_pbshost              import JuMEG_wxPBSHosts
from jumeg.gui.jumeg_gui_wx_argparser                     import JuMEG_GUI_wxArgvParser
#---
from jumeg.base.jumeg_base                                import jumeg_base as jb
from jumeg.base.ioutils.jumeg_ioutils_subprocess          import JuMEG_IoUtils_SubProcess

__version__="2019.05.14.001"

class JuMEG_wxMEEGMergerPanel(JuMEG_wxMainPanel):
      """
      GUI Panel to merge EEG and MEG data into MNE fif format
      """

      def __init__(self, parent,name="JUMEG_MEEG_MERGER",**kwargs):
          super().__init__(parent,name=name,ShowTitleB=False)
      
          self.ShowCmdButtons   = True
          self.ShowTopPanel     = True
          self.ShowTitleA       = True
          self.ShowTitleB       = True
          self.ShowMinMaxBt     = True
          self.module_path      = os.getenv("JUMEG_PATH") + "/jumeg/tools/"
          self.module_name      = "jumeg_io_merge_meeg"
          self.module_extention = ".py"
          self.SubProcess       = JuMEG_IoUtils_SubProcess()
          self._template_panel_name = "EXPERIMENT_TEMPLATE"
          self._init(**kwargs)

      @property
      def fullfile(self): return self.module_path+"/"+self.module_name+ self.module_extention

      def update(self,**kwargs):
          self.stage  = kwargs.get("stage", os.getenv("JUMEG_PATH_JUMEG", os.getcwd()) + "/preproc" )
        #-- update wx CTRLs
          self.PanelA.SetTitle(v="PDF`s")
        #---
          ds=1
          LEA = wx.ALIGN_LEFT | wx.EXPAND | wx.ALL
         #-- Top
          self.ExpTemplate = JuMEG_wxExpTemplate(self.TopPanel,name=self.GetName()+"."+self._template_panel_name)
          self.HostCtrl    = JuMEG_wxPBSHosts(self.TopPanel, prefix=self.GetName())
          self.TopPanel.GetSizer().Add(self.ExpTemplate,3,LEA,ds)
          self.TopPanel.GetSizer().Add(self.HostCtrl,1, wx.ALIGN_RIGHT | wx.EXPAND | wx.ALL,ds)
         #--- A IDs;PDFs
          self.PDFBox = JuMEG_wxPselMEEG(self.PanelA.Panel,name=self.GetName()+".PDFBOX_MEEG",**kwargs)
          self.PanelA.Panel.GetSizer().Add(self.PDFBox, 1, LEA,ds)
         # --- B  right
          self.AP = JuMEG_GUI_wxArgvParser(self.PanelB.Panel,name=self.GetName()+".AP",use_pubsub=self.use_pubsub, fullfile=self.fullfile,
                                           module=self.module_name, ShowParameter=True)
          self.PanelB.Panel.GetSizer().Add(self.AP, 1, LEA,ds)
          self.PanelB.SetTitle("")
         #---
          self.Bind(wx.EVT_BUTTON, self.ClickOnCtrls)
          self.update_argparser_parameter()
     
      def update_on_display(self):
          self.SplitterAB.SetSashPosition(self.GetSize()[0] / 2.0,redraw=True)
   
      def update_argparser_parameter(self):
          """ update parameter BADS_LIST from template"""
          self.AP.update_parameter("BADS_LIST",self.ExpTemplate.TMP.bads)
           
      def _update_hosts(self):
          pass

      def ClickOnExperimentTemplateUpdate(self,stage=None,scan=None,data_type=None):
          """
          call PDFSelectionBox.update_ids and update PDF.ID.listbox
          reset PDFs for new selection

          Parameter
          ---------
           stage:     stage / path to data
           scan:      name of scan
           data_type: mne / eeg
          """
          self.PDFBox.update(stage=stage,scan=scan,reset=True,verbose=self.verbose,debug=self.debug)
          self.update_argparser_parameter()
          
      def init_pubsub(self, **kwargs):
          """ init pubsub call overwrite """
          # pub.subscribe(self.ClickOnApply,self.GetName().upper()+".BT_APPLY")
          pub.subscribe(self.ClickOnExperimentTemplateUpdate, self.ExpTemplate.GetMessage("UPDATE"))
   
      def ClickOnApply(self):
          """
          get selected pdfs structure
          make commands with argparser parameter
          apply cmds to subprocess

          """
         
          self.PDFBox.verbose = self.verbose
          pdfs = self.PDFBox.GetSelectedPDFs()
          if not pdfs:
             wx.CallAfter(pub.sendMessage,"MAIN_FRAME.MSG.ERROR",data="\nPlease select PDFs first\n in: " + self.GetName())
             return
     
          #cmd_parameter       = self.AP.GetParameter()
          cmd_command = self.AP.get_fullfile_command(ShowFileIO=True)
          joblist     = []

         #--- del  "stage"
          cmd_list = cmd_command.split()
          for k in ["--meg_stage","--eeg_stage","-smeg","-seeg","--list_path"]:
              for idx in range(len(cmd_list)):
                  if cmd_list[idx].startswith(k):
                     del cmd_list[idx]
                     break
          
          cmd_command = " ".join(cmd_list)
         # print(cmd_command)
          
          for subject_id in pdfs.get('mne'):
              for idx in range( len( pdfs['mne'][subject_id] ) ):
                  if not pdfs['mne'][subject_id][idx]["selected"]: continue
                  cmd  = cmd_command
                  eeg_idx = pdfs["eeg_index"][subject_id][idx]
                  cmd += " --meg_stage=" + pdfs["stage"]
                  cmd += " -fmeg " + pdfs["mne"][subject_id][idx]["pdf"]
                  cmd += " --eeg_stage=" + pdfs["stage"]
                  cmd += " -feeg " + pdfs["eeg"][subject_id][eeg_idx]["pdf"]
                  #cmd += " "+ cmd_parameter
                  joblist.append( cmd )
           
          if self.verbose:
             wx.LogMessage(jb.pp_list2str(joblist, head="MEEG Merger Job list: "))
             wx.LogMessage(jb.pp_list2str(self.HostCtrl.HOST.GetParameter(),head="HOST Parameter"))
          if joblist:
            # wx.CallAfter(pub.sendMessage,"SUBPROCESS.RUN.START",jobs=joblist,host_parameter=self.HostCtrl.HOST.GetParameter(),verbose=self.verbose)
             wx.CallAfter(self.SubProcess.run,jobs=joblist,host_parameter=self.HostCtrl.HOST.GetParameter(),verbose=self.verbose)
             
      def ClickOnCancel(self,evt):
          wx.LogMessage( "<Cancel> button is no in use" )
          wx.CallAfter( pub.sendMessage,"MAIN_FRAME.MSG.INFO",data="<Cancel> button is no in use")

      def ClickOnCtrls(self, evt):
          obj = evt.GetEventObject()
          #print(obj.GetName())
          if obj.GetName() == self.GetName()+".BT.APPLY":
             self.ClickOnApply()
          elif obj.GetName() == self.GetName()+".BT.CLOSE":
             wx.CallAfter( pub.sendMessage, "MAIN_FRAME.CLICK_ON_CLOSE",evt=evt)
          #else:
          #   evt.Skip()

class JuMEG_GUI_MEEGMergeFrame(JuMEG_wxMainFrame):
    def __init__(self,parent,id,title,pos=wx.DefaultPosition,size=[1024,768],**kwargs):
        style = wx.DEFAULT_FRAME_STYLE | wx.NO_FULL_REPAINT_ON_RESIZE
        super().__init__(parent,id, title, pos, size, style,**kwargs)
        self.template_path = None

    def update(self,**kwargs):
        return JuMEG_wxMEEGMergerPanel(self,**kwargs)
       
    def UpdateAboutBox(self):
        self.AboutBox.description = "merging EEG data <*.vhdr> into MEG data <*.fif>"
        self.AboutBox.version     = __version__
        self.AboutBox.copyright   = '(C) 2018 Frank Boers <f.boers@fz-juelich.de>'
        self.AboutBox.developer   = 'Frank Boers'
        self.AboutBox.docwriter   = 'Frank Boers'

if __name__ == '__main__':
   app = wx.App()
   frame = JuMEG_GUI_MEEGMergeFrame(None,-1,'JuMEG MEEG MERGER',module="jumeg_preproc_merge_meeg",function="get_args",ShowLogger=True,ShowCmdButtons=True,ShowParameter=True,debug=True,verbose=True)
   app.MainLoop()
