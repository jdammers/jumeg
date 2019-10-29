#!/usr/bin/env python3
# -+-coding: utf-8 -+-

""" JuMEG preprocessing GUI based on mne-python
This is a container widged for pre processing GUIs

- I/O                : exporting 4D data to FIF
- Experiment Template: editing or setting up a new experiment template
- MEEG Merger        :  merging MEG and EEG data
- Noisy Channel detection
- Noise Reducer
- ICA : artifact cleaning ICA
- Filter: filtering
- Epocher:
-Tools
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

import wx
from   pubsub import pub
import wx.lib.scrolledpanel as scrolled

#--- jumeg cls
#from jumeg.jumeg_base                                     import jumeg_base as jb

#--- jumeg wx stuff
from jumeg.gui.wxlib.jumeg_gui_wxlib_main_frame           import JuMEG_wxMainFrame
from jumeg.gui.wxlib.jumeg_gui_wxlib_main_panel           import JuMEG_wxMainPanel
#---
from jumeg.gui.jumeg_gui_experiment_template              import JuMEG_wxTemplatePanel
from jumeg.gui.jumeg_gui_import_to_fif                    import JuMEG_wxImportToFIFPanel
from jumeg.gui.jumeg_gui_meeg_merger                      import JuMEG_wxMEEGMergerPanel

__version__="2019.05.14.001"

class JuMEG_wxPreProcPanel(JuMEG_wxMainPanel):
    """
    Parameters:
    ----------
    Results:
    --------
    wx.Panel

    """
    def __init__(self,parent,**kwargs):
        super().__init__(parent)
     #--- wxPanel CLS
        self._proc_list    = ["Experiment Template","Import To FIF","MEEG Merger"]
        self._NB           = None
        self._init(**kwargs)

    @property
    def NB(self): return self._NB
    
    @property
    def process_list(self): return self._proc_list

    def _get_param(self,k1,k2):
        return self._param[k1][k2]
    def _set_param(self,k1,k2,v):
        self._param[k1][k2]=v

    def ShowHelp(self):
        print(self.__doc__)

    def update( self, **kwargs ):
        self.PanelA.SetTitle(v="PreProc Tools")
       #--- Notebook
        if self._NB:
           self._NB.DeleteAllPages()
        else:
           self._NB = wx.Notebook(self.PanelA.Panel,-1,style=wx.BK_DEFAULT)
           
        i=1
        self._proc_pnl=dict()
        self._proc_pnl["Experiment Template"] = JuMEG_wxTemplatePanel(self.NB,ShowLogger=True)
        self._proc_pnl["Import To FIF"]       = JuMEG_wxImportToFIFPanel(self.NB,ShowLogger=True)
        self._proc_pnl["MEEG Merger"]         = JuMEG_wxMEEGMergerPanel(self.NB,ShowLogger=True,ShowCmdButtons=True,ShowParameter=True)

        for p in self.process_list:
            self.NB.AddPage(self._proc_pnl[p],p)

        self.PanelA.Panel.GetSizer().Add(self.NB,1,wx.ALIGN_CENTER|wx.EXPAND|wx.ALL,3)
        self.SplitterAB.Unsplit()
        
    def init_pubsub( self, **kwargs ):
        """ init pubsub call overwrite """
        pass
        #--- verbose
        # pub.subscribe(self.SetVerbose,'MAIN_FRAME.VERBOSE')
        #---
        # pub.subscribe(self.ShowHelp,"MAIN_FRAME.CLICK_ON_HELP")

    #def update_from_kwargs(self,**kwargs):
    #    self.SetBackgroundColour(kwargs.get("bg","grey88"))

#----
class JuMEG_GUI_PreProcFrame(JuMEG_wxMainFrame):
    def __init__(self,parent,id,title,pos=wx.DefaultPosition,size=[1024,768],name='JuMEG_PreProc',*kargs,**kwargs):
        style = wx.DEFAULT_FRAME_STYLE | wx.NO_FULL_REPAINT_ON_RESIZE
        super().__init__(parent,id, title, pos, size, style, name,**kwargs)
        self.template_path = None
        self.verbose       = False
   #---
    def _update_kwargs(self,**kwargs):
        self.verbose = kwargs.get("verbose",self.verbose)
   #---
    def wxInitMainMenu(self):
        """
        overwrite
        add change of LoggerWindow position horizontal/vertical
        """
        self.MenuBar.DestroyChildren()
        self._init_MenuDataList()
        self._update_menubar()
        self.AddLoggerMenu(pos=1,label="Logger")
   #---
    def update(self,**kwargs):
       #---
        self.UpdateAboutBox()
       #---
        #self._init_MenuDataList()
       #---
        return JuMEG_wxPreProcPanel(self,name="JuMEG_PREPROC_PANEL",**kwargs)
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
   app    = wx.App()
   frame  = JuMEG_GUI_PreProcFrame(None,-1,'JUMEG_PREPROC_UTILITY',ShowLogger=False,ShowCmdButtons=False,debug=True,verbose=True)
   app.MainLoop()
