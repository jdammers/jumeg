import sys,os
import wx

from jumeg.tsvgl.wxutils.jumeg_tsv_wx_dlg_subplot import TSVSubPlotDialog
from jumeg.tsvgl.wxutils.jumeg_tsv_wx_dlg_group   import TSVGroupDialog

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

def jumeg_wx_dlg_group(opt=None):
    dlg = TSVGroupDialog(title="JuMEG TSV Groups Option",opt=opt)
   # opt.info()
    if dlg.ShowModal() == wx.ID_OK:
       dlg.Destroy()
       #print"TEST GRP DLG"
       #print opt
       #import pprint
       # pprint.pprint(opt)
       opt.info()
       return opt
    else:
       opt.info()
       return None
         
def jumeg_wx_dlg_subplot(opt=None):
        
    dlg = TSVSubPlotDialog(title="JuMEG TSV Plot Option",opt=opt)
    if dlg.ShowModal() == wx.ID_OK:
       dlg.Destroy()
       # return dlg.Destroy()
       return opt
    else:
       return None
       
    
def jumeg_wx_openfile(w,path=None):

    fout     = None
    wildcard = "FIF files (*.fif)|*.fif|All files (*.*)|*.*"

    if path is None:
       path = os.getcwd()

   # dlg = wx.FileDialog(w, "Choose a file", path, "","FIF (*.fif)|*.fif", wx.FD_OPEN| wx.FD_FILE_MUST_EXIST|wx.CHANGE_DIR)
    dlg = wx.FileDialog(w, "Choose a file", path,wildcard=wildcard,style=wx.FD_OPEN| wx.FD_FILE_MUST_EXIST|wx.CHANGE_DIR)

    if dlg.ShowModal() == wx.ID_OK:
       fout = dlg.GetPath()
       print"DLG: " + fout
    dlg.Destroy()
    return fout

def jumeg_wx_opendir(w):
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


def jumeg_wx_utils_about_box():
    """
    modified from : http://http://zetcode.com/wxpython/dialogs/
    :return:
    """

    description="Data Browser/Editor\nMEG Data Analysis at INM4-MEG-FZJ"

    licence="""Copyright 2014-2015, authors of Jumeg
    All rights reserved.
    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the names of Jumeg authors nor the names of any
      contributors may be used to endorse or promote products derived from
      this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS AS IS AND
    ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
    WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY
    DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
    (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
    LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
    ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
    SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE."""

    info = wx.AboutDialogInfo()

    #info.SetIcon(wx.ICON_INFORMATION)
    info.SetName('JuMEG Time Series Viewer (TSV)')
    info.SetVersion('0.007')
    info.SetDescription(description)
    info.SetCopyright('(C) 2014 - 2015 Frank Boers')
    info.SetWebSite('https://github.com/jdammers/jumeg')
    info.SetLicence(licence)
    info.AddDeveloper('Frank Boers')
    info.AddDocWriter('Frank Boers')
    info.AddArtist('JuMEGs')

    wx.AboutBox(info)


"""
import wx
class MainWindow(wx.Frame):
    def __init__(self, parent, title):
        wx.Frame.__init__(self, parent)

        #add position panel
        posPnl = wx.Panel(self)
        lbl1 = wx.StaticText(posPnl, label="Position")
        lbl2 = wx.StaticText(posPnl, label="Size")
        sizeCtrl = wx.TextCtrl(posPnl)

        posPnlSzr = wx.BoxSizer(wx.HORIZONTAL)
        posPnlSzr.Add(lbl1, 1, wx.GROW)
        posPnlSzr.Add(sizeCtrl, 1, wx.GROW)
        posPnlSzr.Add(lbl2, 1, wx.GROW)

        posPnl.SetSizer(posPnlSzr)

        #create a top leverl sizer to add to the frame itself
        mainSzr = wx.BoxSizer(wx.VERTICAL)
        mainSzr.Add(posPnl, 1, wx.GROW)

        self.SetSizerAndFit(mainSzr)
        self.Show()


app = wx.App(False)
frame = MainWindow(None, "Trading Client")
app.MainLoop()
"""