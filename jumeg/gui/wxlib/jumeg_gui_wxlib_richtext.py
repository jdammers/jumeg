#!/usr/bin/env python3
# -+-coding: utf-8 -+-

"""
"""

#--------------------------------------------
# Authors: Frank Boers <f.boers@fz-juelich.de> 
# copy of wxdemos RichTextCtrl
#-------------------------------------------- 
# Date: 07.03.19
#-------------------------------------------- 
# License: BSD (3-clause)
#--------------------------------------------
# Updates
#--------------------------------------------

from six import BytesIO
import sys,os
import wx
import wx.richtext as rt
from pubsub import pub

__version__= '2019.05.14.001'

#----------------------------------------------------------------------
class JuMEG_wxRichTextFrame(wx.Frame):
    def __init__(self, *args, size=(700,500),style=wx.DEFAULT_FRAME_STYLE|wx.STAY_ON_TOP,**kwargs):
        wx.Frame.__init__(self, *args,style=style)
        self._size=size
        
        self.MakeMenuBar()
        self.MakeToolBar()
        self.CreateStatusBar()

        self.SetStatusText("JuMEG Text Editor; select a text file")
        self.FDlgPath = "."
        self.FDlgWildcard = "text files (*.txt)|*.txt|all files (*.*)|*.*"
        self._wx_init(**kwargs)
    
    @property
    def CtrlPanel(self): return self._pnl_ctrl
    
    def _init_CtrlPanel(self,**kwargs):
        pass
        
    def _wx_init(self,**kwargs):
        self._update_from_kwargs(**kwargs)
        self._init_rtc()
        self._pnl_ctrl= wx.Panel(self)
        self._init_CtrlPanel(**kwargs)
        pub.subscribe(self.OnFileExit,"MAIN_FRAME.CLICK_ON_CLOSE")
        self._ApplyLayout()

    def _init_rtc(self):
        #self._pnl_rtc = wx.Panel(self)
        self.rtc      = rt.RichTextCtrl(self,style=wx.VSCROLL|wx.HSCROLL|wx.NO_BORDER);
        
        #vbox = wx.BoxSizer(wx.VERTICAL)
        #vbox.Add(self.rtc,1,wx.ALIGN_LEFT|wx.ALL | wx.EXPAND,5)
        #self._pnl_rtc.SetSizer(vbox)
        #self._pnl_rtc.SetAutoLayout(True)
        #self._pnl_rtc.Layout()
        #self._pnl_rtc.Fit()
        
        self.rtc.Freeze()
        self.rtc.BeginSuppressUndo()
        self.rtc.BeginParagraphSpacing(0, 20)
        self.rtc.BeginAlignment(wx.TEXT_ALIGNMENT_CENTRE)
        # Create and initialize text attributes
        self.textAttr = rt.RichTextAttr()
        self.SetFontStyle(fontColor=wx.Colour(0, 0, 0), fontBgColor=wx.Colour(255, 255, 255), fontFace='Times New Roman', fontSize=10, fontBold=False, fontItalic=False, fontUnderline=False)
        self.rtc.EndSuppressUndo()
        self.rtc.Thaw()
        
    def _ApplyLayout(self):
        self.Sizer = wx.BoxSizer(wx.VERTICAL)
        self.Sizer.Add(self.rtc,1, wx.ALIGN_LEFT|wx.ALL|wx.EXPAND,5)
        self.Sizer.Add(self._pnl_ctrl,0,wx.ALIGN_LEFT|wx.ALL|wx.EXPAND,1)
        self.SetSizer(self.Sizer)
        self.SetAutoLayout(True)
        self.Layout()
        self.Fit()
        self.SetSize(self._size)
        wx.CallAfter(self.rtc.SetFocus)

    def _update_from_kwargs(self,**kwargs):
        self.SetStatusText( kwargs.get("status_text", self.GetStatusBar().GetStatusText()) )
        self.FDlgPath     = kwargs.get("path", self.FDlgPath)
        self.FDlgWildcard = kwargs.get("wildcard", self.FDlgWildcard)

    def LoadFile(self,fin):
        if not fin: return
        self.FDlgPath=os.path.dirname(fin)
        self.rtc.LoadFile(fin,1)
        self.SetStatusText(fin)
        self.Center()
        self.Show(True)
        
    def SetFontStyle(self, fontColor = None, fontBgColor = None, fontFace = None, fontSize = None,
                     fontBold = None, fontItalic = None, fontUnderline = None):
        if fontColor:
            self.textAttr.SetTextColour(fontColor)
        if fontBgColor:
            self.textAttr.SetBackgroundColour(fontBgColor)
        if fontFace:
            self.textAttr.SetFontFaceName(fontFace)
        if fontSize:
            self.textAttr.SetFontSize(fontSize)
        if fontBold != None:
            if fontBold:
                self.textAttr.SetFontWeight(wx.FONTWEIGHT_BOLD)
            else:
                self.textAttr.SetFontWeight(wx.FONTWEIGHT_NORMAL)
        if fontItalic != None:
            if fontItalic:
                self.textAttr.SetFontStyle(wx.FONTSTYLE_ITALIC)
            else:
                self.textAttr.SetFontStyle(wx.FONTSTYLE_NORMAL)
        if fontUnderline != None:
            if fontUnderline:
                self.textAttr.SetFontUnderlined(True)
            else:
                self.textAttr.SetFontUnderlined(False)
        self.rtc.SetDefaultStyle(self.textAttr)

    def OnURL(self, evt):
        wx.MessageBox(evt.GetString(), "URL Clicked")


    def OnFileOpen(self, evt):
        # This gives us a string suitable for the file dialog based on
        # the file handlers that are loaded
        dlg = wx.FileDialog(self, "Choose a filename",
                            wildcard=self.FDlgWildcard,defaultDir=self.FDlgPath,
                            style=wx.FD_OPEN)
        
        if dlg.ShowModal() == wx.ID_OK:
            path = dlg.GetPath()
            if path:
               fileType = wx.richtext.RICHTEXT_TYPE_TEXT #types[dlg.GetFilterIndex()]
               self.rtc.LoadFile(path,fileType)
               self.FDlgPath = path
        dlg.Destroy()


    def OnFileSave(self, evt):
        if not self.rtc.GetFilename():
            self.OnFileSaveAs(evt)
            return
        self.rtc.SaveFile(self.rtc.GetFilename(),wx.richtext.RICHTEXT_TYPE_TEXT)


    def OnFileSaveAs(self, evt):
        """
        save in plaintext format
        :param evt:
        :return:
        """
        # wildcard,types = rt.RichTextBuffer.GetExtWildcard(save=True)
        
        dlg = wx.FileDialog(self, "Choose a filename",
                            wildcard=self.FDlgWildcard,defaultDir=self.FDlgPath,
                            style=wx.FD_SAVE|wx.FD_OVERWRITE_PROMPT)
        dlg.SetFilename( os.path.basename( self.rtc.GetFilename()))
        if dlg.ShowModal() == wx.ID_OK:
           path = dlg.GetPath()
           if path:
              self.rtc.SaveFile(path,wx.richtext.RICHTEXT_TYPE_TEXT)
              self.FDlgPath = os.path.dirname(path)
        dlg.Destroy()

    def OnFileExit(self, evt):
        self.Close(True)


    def OnFont(self, evt):
        if not self.rtc.HasSelection():
            return

        r = self.rtc.GetSelectionRange()
        fontData = wx.FontData()
        fontData.EnableEffects(False)
        attr = wx.TextAttr()
        attr.SetFlags(wx.TEXT_ATTR_FONT)
        if self.rtc.GetStyle(self.rtc.GetInsertionPoint(), attr):
            fontData.SetInitialFont(attr.GetFont())

        dlg = wx.FontDialog(self, fontData)
        if dlg.ShowModal() == wx.ID_OK:
            fontData = dlg.GetFontData()
            font = fontData.GetChosenFont()
            if font:
                attr.SetFlags(wx.TEXT_ATTR_FONT)
                attr.SetFont(font)
                self.rtc.SetStyle(r, attr)
        dlg.Destroy()


    def OnColour(self, evt):
        colourData = wx.ColourData()
        attr = wx.TextAttr()
        attr.SetFlags(wx.TEXT_ATTR_TEXT_COLOUR)
        if self.rtc.GetStyle(self.rtc.GetInsertionPoint(), attr):
            colourData.SetColour(attr.GetTextColour())

        dlg = wx.ColourDialog(self, colourData)
        if dlg.ShowModal() == wx.ID_OK:
            colourData = dlg.GetColourData()
            colour = colourData.GetColour()
            if colour:
                if not self.rtc.HasSelection():
                    self.rtc.BeginTextColour(colour)
                else:
                    r = self.rtc.GetSelectionRange()
                    attr.SetFlags(wx.TEXT_ATTR_TEXT_COLOUR)
                    attr.SetTextColour(colour)
                    self.rtc.SetStyle(r, attr)
        dlg.Destroy()



    def OnUpdateBold(self, evt):
        evt.Check(self.rtc.IsSelectionBold())

    def OnUpdateItalic(self, evt):
        evt.Check(self.rtc.IsSelectionItalics())

    def OnUpdateUnderline(self, evt):
        evt.Check(self.rtc.IsSelectionUnderlined())

    def OnUpdateAlignLeft(self, evt):
        evt.Check(self.rtc.IsSelectionAligned(wx.TEXT_ALIGNMENT_LEFT))

    def OnUpdateAlignCenter(self, evt):
        evt.Check(self.rtc.IsSelectionAligned(wx.TEXT_ALIGNMENT_CENTRE))

    def OnUpdateAlignRight(self, evt):
        evt.Check(self.rtc.IsSelectionAligned(wx.TEXT_ALIGNMENT_RIGHT))


    def ForwardEvent(self, evt):
        # The RichTextCtrl can handle menu and update events for undo,
        # redo, cut, copy, paste, delete, and select all, so just
        # forward the event to it.
        self.rtc.ProcessEvent(evt)


    def MakeMenuBar(self):
        def doBind(item, handler, updateUI=None):
            self.Bind(wx.EVT_MENU, handler, item)
            if updateUI is not None:
                self.Bind(wx.EVT_UPDATE_UI, updateUI, item)

        fileMenu = wx.Menu()
        doBind( fileMenu.Append(-1, "&Open\tCtrl+O", "Open a file"),
                self.OnFileOpen )
        doBind( fileMenu.Append(-1, "&Save\tCtrl+S", "Save a file"),
                self.OnFileSave )
        doBind( fileMenu.Append(-1, "&Save As...\tF12", "Save to a new file"),
                self.OnFileSaveAs )
        fileMenu.AppendSeparator()
        doBind( fileMenu.Append(-1, "E&xit\tCtrl+Q", "Quit this program"),
                self.OnFileExit )

        editMenu = wx.Menu()
        doBind( editMenu.Append(wx.ID_UNDO, "&Undo\tCtrl+Z"),
                self.ForwardEvent, self.ForwardEvent)
        doBind( editMenu.Append(wx.ID_REDO, "&Redo\tCtrl+Y"),
                self.ForwardEvent, self.ForwardEvent )
        editMenu.AppendSeparator()
        doBind( editMenu.Append(wx.ID_CUT, "Cu&t\tCtrl+X"),
                self.ForwardEvent, self.ForwardEvent )
        doBind( editMenu.Append(wx.ID_COPY, "&Copy\tCtrl+C"),
                self.ForwardEvent, self.ForwardEvent)
        doBind( editMenu.Append(wx.ID_PASTE, "&Paste\tCtrl+V"),
                self.ForwardEvent, self.ForwardEvent)
        doBind( editMenu.Append(wx.ID_CLEAR, "&Delete\tDel"),
                self.ForwardEvent, self.ForwardEvent)
        editMenu.AppendSeparator()
        doBind( editMenu.Append(wx.ID_SELECTALL, "Select A&ll\tCtrl+A"),
                self.ForwardEvent, self.ForwardEvent )

        #doBind( editMenu.AppendSeparator(),  )
        #doBind( editMenu.Append(-1, "&Find...\tCtrl+F"),  )
        #doBind( editMenu.Append(-1, "&Replace...\tCtrl+R"),  )

        formatMenu = wx.Menu()
        doBind( formatMenu.Append(-1, "&Font..."), self.OnFont)

        mb = wx.MenuBar()
        mb.Append(fileMenu, "&File")
        mb.Append(editMenu, "&Edit")
        mb.Append(formatMenu, "F&ormat")
        self.SetMenuBar(mb)


    def MakeToolBar(self):
        pass
        '''
        def doBind(item, handler, updateUI=None):
            self.Bind(wx.EVT_TOOL, handler, item)
            if updateUI is not None:
                self.Bind(wx.EVT_UPDATE_UI, updateUI, item)

        tbar = self.CreateToolBar()

        tbar.Realize()
        '''

#----------------------------------------------------------------------

if __name__ == '__main__':
   app = wx.App()
   win = JuMEG_wxRichTextFrame(None, -1, "JuMEG Text Editor",
                                size=(700, 500),style = wx.DEFAULT_FRAME_STYLE)
   
   #f="208548_INTEXT01_002.vhdr"
   win.FDlgWildcard='HDR files (*.vhdr)|*.vhdr|all files (*.*)|*.*)'
   
   # win.LoadFile()
   
   win.Show(True)
   app.MainLoop()

