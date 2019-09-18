import sys,os
import wx

from jumeg.gui.tsv.wxutils.jumeg_tsv_wx_dlg_subplot import TSVSubPlotDialog
from jumeg.gui.tsv.wxutils.jumeg_tsv_wx_dlg_group   import TSVGroupDialog

__version__="2019-09-13-001"
#try:
#    from agw import floatspin as FS
#except ImportError: # if it's not there locally, try the wxPython lib.
#    import wx.lib.agw.floatspin as FS

"""
 self.mvp[:,0] = dborder
          if (self.opt.plot.cols >1):      
             mat = np.zeros( (self.opt.plot.rows,self.opt.plot.cols) )
             mat += np.arange(self.opt.plot.cols)
             self.mvp[:,0] +=  mat.T.flatten() * ( dw + 2 *dborder)          
"""           

def jumeg_tsv_wxutils_dlg_group_settings(opt=None):
    # dlg = TSVGroupDialog(title="JuMEG TSV Channel Group Options",**kwargs)
    dlg = TSVGroupDialog(title="JuMEG TSV Groups Option",opt=opt)
   # opt.info()
    if dlg.ShowModal() == wx.ID_OK:
       dlg.Destroy()
       return opt
    else:
       opt.info()
       return None

def jumeg_tsv_wxutils_dlg_plot_settings(type="subplot",**kwargs):
    """
    :param opt:
    :param type: <subplot>
    :return:
    """
    dlg = TSVSubPlotDialog(title="JuMEG TSV SubPlot Options",**kwargs)
    if dlg.ShowModal() == wx.ID_APPLY:
       param = dlg.GetParameter()
       dlg.Destroy()
       # return dlg.Destroy()
       return param
    else:
       return None
       
def jumeg_tsv_wxutils_openfile(w,path=None):

    fout = None
    path = None
    wildcard = "Sugested Bads RAW (*bcc-raw.fif)|*bcc-raw.fif|Sugested Bads Empty (*bcc-empty.fif)|*bcc-empty.fif|FIF files (*.fif)|*.fif|All files (*.*)|*.*"

    if path is None:
       path = os.getcwd()

   # dlg = wx.FileDialog(w, "Choose a file", path, "","FIF (*.fif)|*.fif", wx.FD_OPEN| wx.FD_FILE_MUST_EXIST|wx.CHANGE_DIR)
    dlg = wx.FileDialog(w, "Choose a file", path,wildcard=wildcard,style=wx.FD_OPEN| wx.FD_FILE_MUST_EXIST|wx.FD_CHANGE_DIR)

    if dlg.ShowModal() == wx.ID_OK:
       fout = dlg.GetPath()
    dlg.Destroy()
    return fout

def jumeg_tsv_wxutils_opendir(w):
         dlg = wx.DirDialog(w, "Choose a directory:", style=wx.DD_DEFAULT_STYLE | wx.DD_NEW_DIR_BUTTON)
         if dlg.ShowModal() == wx.ID_OK:
             dlg_path = dlg.GetPath()
         dlg.Destroy()
         return dlg_path

def MsgDlg(w, string, caption = "JuMEG TSV", style=wx.YES_NO|wx.CANCEL):
    """Common MessageDialog."""
    dlg = wx.MessageDialog(w, string, caption, style)
    result = dlg.ShowModal()
    dlg.Destroy()
    return result

