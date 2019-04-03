#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 15:28:58 2018

@author: fboers
"""
import numpy as np
import wx,os,sys
import wx.lib.masked as masked
import wx.lib.scrolledpanel as scrolled

from jumeg.gui.wxlib.jumeg_gui_wxlib_richtext import JuMEG_wxRichTextFrame
from pubsub  import pub

from jumeg.gui.utils.jumeg_gui_utils_pdfs  import JuMEG_Utils_PDFsMEEG,JuMEG_Utils_PDFsBTi,JuMEG_Utils_PDFsMNE

__version__="2019-02-11-001"


class JuMEG_wxPDFMEEGTextEditor(JuMEG_wxRichTextFrame):
    """
    Txt Editor to show vhdr,vmrk file from eeg recordings
    change <DataFile> and <MarkerFile> names for correct matching
    """
    def __init__(self,name="JuMEG PDFs MEEG Editor",wildcard="HDR files (*.vhdr)|*.vhdr|marker files (*.vmrk)|*.vmrk|all files (*.*)|*.*",
                 status_text="JuMEG Text Editor; select a text file",path="."):
        super().__init__(None,-1,name,size=(700,500),style=wx.DEFAULT_FRAME_STYLE | wx.STAY_ON_TOP,
                         wildcard=wildcard,
                         status_text=status_text,
                         path=path
                         )
        
    '''
    ToDo later  change <DataFile> and <MarkerFile> names for correct matching if error
    def _init_CtrlPanel(self,**kwargs):
        self.CtrlPanel.SetBackgroundColour("grey70")
        self._ck_update_vhdr = wx.CheckBox(self.CtrlPanel,-1,'update <DataFile name> [.eeg] in  vhdr- and vmrk-file')
        self._ck_update_vmrk = wx.CheckBox(self.CtrlPanel,-1,'update <MarkerFile name> [.vmrk]')
        self._bt_update      = wx.Button(self.CtrlPanel,-1,'update')
        self._bt_update.Bind(wx.EVT_BUTTON,self.update)
        
        hbox = wx.BoxSizer(wx.HORIZONTAL)
        hbox.Add(self._ck_update_vhdr,0,wx.ALIGN_LEFT|wx.ALL | wx.EXPAND,5)
        hbox.Add(self._ck_update_vmrk,0,wx.ALIGN_LEFT|wx.ALL | wx.EXPAND,5)
        hbox.Add(self._bt_update,0,wx.ALIGN_LEFT|wx.ALL,5)

        self.CtrlPanel.SetSizer(hbox)
        self.CtrlPanel.SetAutoLayout(True)
        self.CtrlPanel.Layout()
        self.CtrlPanel.Fit()
    
    def update(self,evt):
        if not self.rtc.GetFilename():
           return
        fname = self.rtc.GetFilename()
        print(fname)

       #--- update MarkerFile
       # load MarkerFile
       # rename MarkerFile
       # change MarkerFile in vhdr
        if self._ck_update_vmrk.GetValue():


        #---
        # change DataFile in vhdr
        # change dataFile  in vmrk
        # and save
    '''

class JuMEG_wxIdSelectionBoxBase(wx.Panel):
    """
     JuMEG_wxSelectionBox
      input
         parent widged
           title  : ID
           stage  : stage path to experiment e.g. /data/XYZ/exp/M100
           pubsub : True; use wx.pubsub msg systen sends like:
                    pub.sendMessage('EXPERIMENT_TEMPLATE',stage=stage,experiment=experiment,TMP=template_data)
           or button event <ID_SELECTION_BOX_UPDATE> from Update button
           id_mask: #######; mask for id selection
           verbose: False
           bg     : grey90 backgroundcolor
    """
    
    def __init__(self,parent,name="ID_SELECTION_BOX",*kargs,**kwargs):
        super().__init__(parent,name=name)
        self.PDFs = None
        # self._init()
    
    @property
    def stage(self):
        return self.PDFs.stage
    
    @stage.setter
    def stage(self,v):
        self.PDFs.stage = v
    
    @property
    def ItemList(self):
        return self._ItemList
    
    @ItemList.setter
    def ItemList(self,v):
        self._item_list = v
        self.wx_update_lb()
    
    @property
    def ItemSelectedList(self):
        self._item_selected_list = []
        for i in self._LB.GetSelections():
            self._item_selected_list.append(self._LB.GetString(i))
        return self._item_selected_list
    
    def GetMessage(self,msg):
        return self.GetName() + "." + msg.upper()
    
    def GetSelectedPDFs(self):
        return self.PDFs.GetPDFsFromIDs(ids=self.ItemSelectedList)
    
    def _update_from_kwargs(self,**kwargs):
        self.pubsub = kwargs.get("pubsub",True)
        self.id_mask = kwargs.get("id_mask",'#######')
        self.stage = kwargs.get("stage",None)
        self._title = kwargs.get("title","I D s")
        self.verbose = kwargs.get("verbose",False)
    
    def _initID(self,**kwargs):
        pass
    
    def _init(self,**kwargs):
        self._initPDFs(**kwargs)
        self._update_from_kwargs(**kwargs)
        self._item_list = []
        self._item_selected_list = []
        self._wx_init(**kwargs)
    
    def _wx_init(self,**kwargs):
        bg = kwargs.get("bg",'grey80')
        self.SetBackgroundColour(bg)
        
        self._wxTxtIdInfo = wx.StaticText(self,-1,label="000 / 000")
        #--- de/select toggle bt
        self._wxBtSelect = wx.Button(self,-1,name=self.GetName().upper() + ".ID_SELECT",label='SELECT')
        self._wxBtSelect.Bind(wx.EVT_BUTTON,self.ClickOnButton)
        #--- id mask
        self._wxMaskId = masked.TextCtrl(self,-1,'',mask=self.id_mask)
        #--- id listbox
        self._LB = wx.ListBox(self,-1,style=wx.LB_MULTIPLE | wx.BORDER_SUNKEN)
        self._LB.Bind(wx.EVT_LISTBOX,self.ClickOnDeSelect)
        #--- Update Bt
        self._wxBtUpdate = wx.Button(self,-1,name=self.GetName().upper() + ".UPDATE",label="Update")
        self._wxBtUpdate.Bind(wx.EVT_BUTTON,self.ClickOnButton)
        self._ApplyLayout()
    
    def ClickOnButton(self,evt):
        obj = evt.GetEventObject()
        if not obj.GetName().startswith(self.GetName()):
            return
        #--- de/select bt
        if obj.GetName().endswith('ID_SELECT'):
            #--- deselect all first
            for i in range(self._LB.GetCount()):
                self._LB.Deselect(i)
            if obj.GetLabel().startswith('DE'):
                obj.SetLabel('SELECT')
            else:
                obj.SetLabel('DESELECT')
                mask = self._wxMaskId.GetValue().strip()
                for i in range(self._LB.GetCount()):
                    if mask:
                        if (self._LB.GetString(i).startswith(mask)):
                            self._LB.SetSelection(i)
                    else:
                        self._LB.SetSelection(i)
            self._update_selection()
            #--- update button
        elif obj.GetName().endswith('UPDATE'):
            self.ClickOnUpdate()
        else:
            evt.Skip()
    
    def ClickOnUpdate(self):
        pub.sendMessage(self.GetMessage("UPDATE"),stage=self.stage,pdfs=self.GetSelectedPDFs())
    
    def listener(self,path=None,experiment=None,TMP=None):
        #print("\n--->call to listener ---from =>" + self.GetName())
        self.PDFs.path = path
        self._item_list = []
        self._item_list = self.ID.update_meg_ids()
        self.PDFs.update()
        #if self._item_list:
        self.wx_update_lb()
    
    def ClickOnUpdate(self):
        pass
        #if self.pubsub:
        #  print("PUBSUB: "+self.GetName().upper()+".UPDATE")
        # pub.sendMessage(self.GetName().upper()+".UPDATE",id_list=self.ItemSelectedList)
    
    def _update_selection(self):
        self._item_selected_list = self._LB.GetSelections()
        self._wxTxtIdInfo.SetLabel("%4d / %d" % (len(self._item_selected_list),self._LB.GetCount()))
        return self._item_selected_list
    
    def ClickOnDeSelect(self,evt):
        self._update_selection()
    
    def wx_update_lb(self):
        self._LB.Clear()
        if self._item_list:
            self._LB.InsertItems(sorted(list(set(self._item_list))),0)
        elif not self.stage:
            wx.MessageBox("ERROR no directory found\n <stage> is not defiend",'Error',wx.OK | wx.ICON_ERROR)
        self._update_selection()
    
    def update(self,stage=None,scan=None,data_type='mne'):  #,experiment=None,TMP=None):
        """
        :param stage:
        :param scan:
        :param data_type:
        :return:
        """
        pass
    
    def _ApplyLayout(self):
        ds = 4
        ds1 = 1
        LEA = wx.LEFT | wx.EXPAND | wx.ALL
        REA = wx.RIGHT | wx.EXPAND | wx.ALL
        
        vbox = wx.BoxSizer(wx.VERTICAL)
        #--- Label + De/Selected IDs
        hbox1 = wx.BoxSizer(wx.HORIZONTAL)
        hbox1.Add(wx.StaticText(self,0,label=self._title),0,wx.LEFT,ds1)
        hbox1.Add((0,0),1,LEA,1)
        hbox1.Add(self._wxTxtIdInfo,0,LEA,ds1)
        vbox.Add(hbox1,0,LEA,ds)
        vbox.Add(wx.StaticLine(self),0,LEA,ds)
        #---  mask field
        sbox = wx.StaticBox(self,-1,'MASK')
        sz = wx.StaticBoxSizer(sbox,wx.VERTICAL)
        sz.Add(self._wxMaskId,0,REA,ds)
        #---
        vbox.Add(self._wxBtSelect,0,LEA,ds)
        vbox.Add(sz,0,LEA,ds)
        vbox.Add(self._LB,1,LEA,ds)
        vbox.Add(self._wxBtUpdate,0,LEA,ds)
        
        self.SetSizer(vbox)


class JuMEG_wxIdSelectionBoxMEEG(JuMEG_wxIdSelectionBoxBase):
    """
     JuMEG_wxSelectionBoxMEEG
      input
         parent widged
           stage  : stage path to experiment e.g. /data/XYZ/exp/M100
           pubsub : True; use wx.pubsub msg systen sends like:
                    pub.sendMessage('EXPERIMENT_TEMPLATE',stage=stage,experiment=experiment,TMP=template_data)
           or button event <ID_SELECTION_BOX_UPDATE> from Update button
           id_mask: #######; mask for id selection
           verbose: False
           bg     : grey90 backgroundcolor
    """
    
    def __init__(self,parent,name="ID_SELECTION_BOX_MEEG",*kargs,**kwargs):
        super().__init__(parent,name=name,*kargs,**kwargs)
        self._init()
    
    def _initPDFs(self,**kwargs):
        self.PDFs = JuMEG_Utils_PDFsMEEG(**kwargs)
    
    def GetSelectedEEGIndex(self):
        return self.PDFs.GetEEGIndexFromIDs(ids=self.ItemSelectedList)
    
    def GetSelectedEEGs(self):
        return self.EEG.GetPDFsFromIDs(ids=self.ItemSelectedList)
    
    def GetPDFsFromIDs(self,ids=None):
        """
         call to MEG/MNE obj function
        """
        if not ids:
            ids = self.ItemSelectedList
        if not ids:
            pub.sendMessage("MAIN_FRAME.MSG.ERROR",data="\n ID Selection Box: Please select IDs first")
        return self.PDFs.GetPDFsFromIDs(ids=ids),self.PDFs.GetEEGIndexFromIDs(ids=ids)
    
    def GetSelections(self,ids=None):
        if not ids:
            ids = self.ItemSelectedList
        return self.PDFs.GetPDFsFromIDs(ids=ids),self.PDFs.GetEEGsFromIDs(ids=ids),self.PDFs.GetEEGIndexFromIDs(ids=ids)
    
    def update(self,stage=None,scan=None,data_type='mne'):
        """

        :param stage:
        :param scan:
        :param data_type:
        :return:
        """
        if stage:
            self.stage = stage
        self._item_list = []
        self.PDFs.update(scan=scan)
        self._item_list = self.PDFs.GetIDs()
        self.wx_update_lb()
        #pub.sendMessage(self.GetName().upper() + ".UPDATE")
    
    def ClickOnUpdate(self):
        cnts = 0
        
        #aa= self.GetSelections()
        #for ai in aa:
        #    print("\n--------")
        #    print(ai)
        
        pdfs = { "stage":self.stage,"mne":None,"eeg":None,"eeg_index":None }
        pdf_selection,eeg_selection,pdfs["eeg_index"] = self.GetSelections()
        if pdf_selection:
            pdfs["mne"] = pdf_selection[0]
            n_pdfs = pdf_selection[1] or 0
        if eeg_selection:
            pdfs["eeg"] = eeg_selection[0]
        
        pub.sendMessage(self.GetMessage("UPDATE"),pdfs=pdfs,n_pdfs=pdf_selection[1])


class JuMEG_wxIdSelectionBoxBTi(JuMEG_wxIdSelectionBoxBase):
    """
     JuMEG_wxSelectionBoxBTi
      input
         parent widged
           stage  : stage path to experiment e.g. /data/XYZ/exp/M100
           pubsub : True; use wx.pubsub msg systen sends like:
                    pub.sendMessage('EXPERIMENT_TEMPLATE',stage=stage,experiment=experiment,TMP=template_data)
           or button event <ID_SELECTION_BOX_UPDATE> from Update button
           id_mask: #######; mask for id selection
           verbose: False
           bg     : grey90 backgroundcolor
    """
    
    def __init__(self,parent,name="ID",*kargs,**kwargs):
        super().__init__(parent,name=name,*kargs,**kwargs)
        self._init()
    
    def _initPDFs(self,**kwargs):
        self.PDFs = JuMEG_Utils_PDFsBTi(**kwargs)
    
    def GetSelections(self,ids=None):
        if not ids:
            ids = self.ItemSelectedList
        if not ids:
            pub.sendMessage("MAIN_FRAME.MSG.ERROR",data="\n ID Selection Box: Please select IDs first")
            return
        return self.PDFs.GetPDFsFromIDs(ids=ids)
    
    def update(self,**kwargs):
        """

        :param stage:
        :param scan:
        :param pdf_name:
        :return:
        """
        self._update_from_kwargs(**kwargs)
        self.PDFs.update(**kwargs)
        self.PDFs.info()
        
        self._item_list = self.PDFs.GetIDs()
        
        self.wx_update_lb()
        pub.sendMessage(self.GetName().upper() + ".UPDATE")
    
    def ClickOnUpdate(self):
        n_pdfs = 0
        
        pdfs = { "stage":self.stage,"bti":None,"n_pdfs":None }
        pdf_selection = self.GetSelections()
        if pdf_selection:
            pdfs["bti"] = pdf_selection[0] or None
            n_pdfs = pdf_selection[1] or 0
        
        pub.sendMessage(self.GetMessage("UPDATE"),pdfs=pdfs,n_pdfs=n_pdfs)


class JuMEG_wxPDFBase(wx.Panel):
    """
    JuMEG PDFBox Base CLS for <4D/BTi> and  <MEEG Merger> Posted Data File (PDF) Box
    pdfs=None, title='PDFs', stage=None, bg="grey90", pubsub=True,list=None,
    """
    def __init__(self, parent,name="PDFBOX",*kargs,**kwargs):
        super().__init__(parent,name=name,*kargs)
        self._fmt_ckbox = "{:7} {:10} {:8} {:14} {:10} {:10}"
        self._font      = wx.Font(10,wx.FONTFAMILY_TELETYPE,wx.FONTSTYLE_NORMAL,wx.FONTWEIGHT_NORMAL)
        self._separator = "_"
        self._fout_extention = "-raw.fif"
        
    def _init(self,**kwargs):
        self._ckbox = {}
        self._cbbox = {}
        self._pdfs  = None
        self._isSelected     = True
        self._CB_BACK_COLOR  = 'white'
        self._CB_ERROR_COLOR = 'red'
        
        self._init_defaults()
        self._wx_init(**kwargs)
        self._ApplyLayout()
        
    @property
    def PDFs(self): return self._pdfs

    @property
    def fout_extention(self): return self._fout_extention
    @fout_extention.setter
    def fout_extention(self,v): self._fout_extention=v
   
    def GetStage(self):
        return self.PDFs.get("stage")
    
    def UpdateSelectedPDFs(self):
        pass
    
    def _wx_init(self,**kwargs):
        """  """
        self._pnl_info = wx.Panel(self,style=wx.SUNKEN_BORDER)
        self._pnl_info.SetBackgroundColour(kwargs.get("bginfo","grey80"))
        self._pnl_pdf = scrolled.ScrolledPanel(self,-1,style=wx.TAB_TRAVERSAL | wx.SUNKEN_BORDER,
                                               name=self.GetName() + ".PDFBOX.SCROLLPANEL")
        self._pnl_pdf.SetBackgroundColour(kwargs.get("bg","grey90"))

    # --- Label + line
        ds = 4
        hbox = wx.BoxSizer(wx.HORIZONTAL)
        headline = wx.StaticText(self._pnl_info,-1,label=self._headline)
        headline.SetFont(self._font)
        hbox.Add(headline,1,wx.LEFT | wx.EXPAND | wx.ALL,ds)
        self._pdfinfo_txt = wx.StaticText(self._pnl_info,-1,label="  0 /  0  ")

        stl = wx.BU_EXACTFIT | wx.BU_NOTEXT  # | wx.BORDER_NONE
        self._bt_deselect = wx.Button(self._pnl_info,-1,name=self.GetName() + ".PDFBOX.BT.CLEAR",style=stl)
        self._bt_deselect.SetBitmapLabel(wx.ArtProvider.GetBitmap(wx.ART_DELETE,wx.ART_MENU,(12,12)))
   
        self._bt_deselect.Bind(wx.EVT_BUTTON, self.DeSelectAll, self._bt_deselect)

        hbox.Add(self._pdfinfo_txt,0,wx.LEFT | wx.EXPAND | wx.ALL,ds)
        hbox.Add(self._bt_deselect,0,wx.LEFT | wx.EXPAND | wx.ALL,ds)

        self._pnl_info.SetSizer(hbox)
        self._pnl_info.SetAutoLayout(True)
        self._pnl_info.Fit()

    def _ApplyLayout(self):
        ds = 4
        self.Sizer = wx.BoxSizer(wx.HORIZONTAL)
      #--- add info & PDFbox
        vbox = wx.BoxSizer(wx.VERTICAL)
        vbox.Add(self._pnl_info,0,wx.LEFT | wx.EXPAND | wx.ALL)
        vbox.Add(wx.StaticLine(self),0,wx.LEFT | wx.EXPAND | wx.ALL,ds)
        vbox.Add((0,0),0,wx.EXPAND | wx.ALL,ds)
        vbox.Add(self._pnl_pdf,1,wx.LEFT | wx.EXPAND | wx.ALL,ds)
    
        self.Sizer.Add(vbox,1,wx.EXPAND | wx.ALL,ds)
      #---
        self.SetAutoLayout(True)
        self.SetSizer(self.Sizer)
        self.Fit()
        self.Update()
        self.Refresh()

    def _ApplyPDFLayout(self,GridSizer):
        """
        :param GridSizer: wx.Sizer e.g.: FlexGridSizer
        final stuff to layout the PDF panel
        """
        if not self._pnl_pdf: return
        
        self.Bind(wx.EVT_CHECKBOX,self._ClickOnCkBox)
        self.Bind(wx.EVT_COMBOBOX,self._ClickOnCombo)
        self.Bind(wx.EVT_BUTTON,  self._ClickOnButton)
    
        self._update_pdfinfo()
    
        self._pnl_pdf.SetSizer(GridSizer)
        GridSizer.Fit(self._pnl_pdf)
    
        self._pnl_pdf.SetAutoLayout(1)
        self._pnl_pdf.SetupScrolling()
        self._pnl_pdf.FitInside()
    
        self.Fit()
        self.Update()
        self.Refresh()
        self.GetParent().Layout()

    def reset(self):
        """
        destroy all PDFs in pdf panel
        :return:
        """
        if self._pnl_pdf:
           for child in self._pnl_pdf.GetChildren():
               child.Destroy()
        self._cbbox = {}
        self._ckbox = {}
        self._update_pdfinfo()
        self.GetParent().Layout()
    
    def DeSelectAll(self,evt):
        """
        select or deselect all checkboxes
        """
        self._isSelected = not( self._isSelected )
        for subject_id in self._ckbox:
            for ckb in self._ckbox[subject_id]:
                ckb.SetValue(self._isSelected)
        
#--- helper functions
    def _checkbox_label(self,label,sep="_"):
        """
         makes a formated string from a list
        """
        return self._fmt_ckbox.format(*label.split(sep))

    def _update_pdfinfo(self):
        """"""
        sel  = 0
        n_ck = 0
        for item in (self._ckbox):
            for ck in (self._ckbox[item]):
                sel += int(ck.GetValue())
                n_ck += 1
        spdf = "{:3d} / {:3d}".format(sel,n_ck)
        self._pdfinfo_txt.SetLabel(spdf)
       
    def _update_ckbox_file_exist(self,ckbox,f):
        if os.path.isfile(f.partition('-')[0] + ',' + self._fout_extention):
           ckbox.SetForegroundColour(wx.BLUE)
           ckbox.SetValue(False)
           
    def _find_obj_by_name(self, obj, n):
        # https://stackoverflow.com/questions/653509/breaking-out-of-nested-loops
        # ckbt = next(( x for x in self.__ckbox if x.GetName() == n), None)
        for item in obj:
            for x in obj[item]:
                if x.GetName() == n:
                    return x

    def _ClickOnCkBox(self, evt):
        """
        checks if an eeg file is selected in combobox
        if not ckbox wille unchecked
        :param evt:
        :return:
        """
        obj = evt.GetEventObject()
        n = obj.GetName()
        if obj.GetValue():
           combo = self._find_obj_by_name(self._cbbox, n)
           if combo:
              if combo.GetValue():
                 obj.SetValue(True)
              else:
                 obj.SetValue(False)

        self._update_pdfinfo()
    
    def _ClickOnCombo(self,evt):
        pass
    def _ClickOnButton(self,evt):
        pass


class JuMEG_wxPDFBTi(JuMEG_wxPDFBase):
    """
    JuMEG 4D/BTI
    show  selected files called <Posted Data File> (PDF)
    pdfs=None, titl e='PDFs', stage=None, bg="grey90", pubsub=True,list=None,
    emptyroom=True
    """
    
    def __init__(self,parent,name="PDFBOX_BTI",*kargs,**kwargs):
        super().__init__(parent,name=name,*kargs)
        self._init(**kwargs)
    
    @property
    def emptyroom(self):
        return self._emptyroom
    
    @emptyroom.setter
    def emptyroom(self,v):
        self.UpdateEmptyroom(v)
    
    def _init_defaults(self):
        self._pdfs = None
        self._n_pdfs = 0
        self._emptyroom = True
        self._emptyroom_colour = wx.BLUE
        self._emptyroom_extention = "-empty.fif"
        self._fif_extention = "-raw.fif"
        self._separator = '/'
        self._fmt_ckbox = "{:>8}  {:>12}  {:>10}  {:>8}  {:>5}     {}"
        self._headline = "{} {:^6} {:^12} {:^10} {:^12} {:^6} {:^10}".format("4D/BTi","Id","Scan","Date","Time","Run",
                                                                             "PDF Name")
    
    def _update_from_kwargs(self,**kwargs):
        self._emptyroom = kwargs.get("emptyroom",self.emptyroom)
        self._emptyroom_extention = kwargs.get("emptyroom_extention",self._emptyroom_extention)
        self._fif_extention = kwargs.get("fif_extention",self._fout_extention)
    
    def _init_wx_search_ctrl(self,parent,**kwargs):
        self._pnl_search = wx.Panel(parent,style=wx.SUNKEN_BORDER)
        self._pnl_search.SetBackgroundColour(kwargs.get("bgsearch","grey95"))
    
    def _ck_file_size(self,pdf,ckhs=False,min_size=1):
        """
        checks filesize > 1 for pdf,config

        :param pdf: dict
              {pdf': '0815/M100/18-01-01@00:01/1/c,rfDC', 'size': 528711512, 'hs_file': 118552, 'config': 193616, 'selected': False}
        :param ckhs: <False>
        :param min_size: minimum filsize
        :return:
         True/False
        """
        key_list = ["size","config"]
        if ckhs:
            key_list.append("hs_file")
        
        for fsize in key_list:
            if pdf.get(fsize) < min_size:
                return False
        return True
    
    def UpdateSelectedPDFs(self):
        """
        update pdfs <self.PDFs> withn CTRLs selections
        id->index -><{pdf":f,"size":os.stat(f).st_size,"hs_file":size_hs,"config":size_cfg,"selected" = checkbox value  True/False}
        :return
        list of list of selected pdfs and file extention (-raw.fif or -empty.fif)
        [ [ "pdf1", "-raw.fif"], .. [ "pdf10","-empty.fif"] ]

        """
        pdfs = []
        if not self._pdfs:
           print("ERROR Please select PDFs first !!!",file=sys.stderr)
           return None
        
        for subject_id in self._pdfs["bti"]:
            for idx in range(len(self._pdfs["bti"][subject_id])):
                if not self._ckbox[subject_id][idx].GetValue(): continue
                
                self._ckbox[subject_id][idx].SetValue(self._ck_file_size(self._pdfs["bti"][subject_id][idx]))
                self._pdfs["bti"][subject_id][idx]["selected"] = self._ckbox[subject_id][idx].GetValue()
                
                if self._pdfs["bti"][subject_id][idx]["selected"]:
                    pdfs.append([self._pdfs["bti"][subject_id][idx]["pdf"],self._cbbox[subject_id][idx].GetValue()])
        return pdfs
    
    def update(self,pdfs=None,n_pdfs=None,reset=True):
        """
        update PSEL box with pdf strcture
        :param pdfs  : pdf structure
        :param n_pdfs: number of pdfs files
        :param reset : will reset PSEL Box
        :return:

        pdf:
         {'stage': '/home/fboers/MEGBoers/data/megdaw_data21',
          'bti': {
                  '0815': [{'pdf': '0815/M100/18-01-01@00:01/1/c,rfDC', 'size': 528711512, 'hs_file': 118552, 'config': 193616, 'selected': False},
                           {'pdf': '0815/M100/18-01-01@00:01/2/c,rfDC', 'size': 525419312, 'hs_file': 118552, 'config': 193616, 'selected': False},
                           {'pdf': '0815/M100/18-01-01@00:01/3/c,rfDC', 'size': 524170552, 'hs_file': 118552, 'config': 193616, 'selected': False},
                           ]}}
        """
        if reset:
            self.reset()
        
        if pdfs:
            self._pdfs = pdfs
            self._n_pdfs = n_pdfs or 0
        
        if not self._n_pdfs: return
        if not self._pdfs:   return
        
        try:
            n_subjects = len([*self._pdfs["bti"]])
        except:
            return
        
        ds = 5
        LEA = wx.LEFT | wx.EXPAND | wx.ALL
        fgs1 = wx.FlexGridSizer(self._n_pdfs + n_subjects,2,ds,ds)
        
        for subject_id in self._pdfs["bti"]:
            self._ckbox[subject_id] = []
            self._cbbox[subject_id] = []
            
            for idx in range(len(self._pdfs["bti"][subject_id])):
                #--- bti ckbox
                # 0815/M100/18-01-01@00:01/1/c,rfDC'
                pdf_name = self._pdfs["bti"][subject_id][idx]["pdf"]
                fpdf = self._pdfs["stage"] + "/" + self._pdfs["bti"][subject_id][idx]["pdf"]
                # --- init  mne checkbox
                ckb = wx.CheckBox(self._pnl_pdf,wx.NewId(),
                                  label=self._checkbox_label(pdf_name.replace("/","_").replace('@',"_")),
                                  name=subject_id + '_' + str(idx))
                ckb.SetValue(True)
                ckb.SetFont(self._font)
                ckb.SetForegroundColour(wx.BLACK)
                ckb.SetToolTip(wx.ToolTip(fpdf))
                
                self._update_ckbox_file_exist(ckb,fpdf)
                self._ckbox[subject_id].append(ckb)
                fgs1.Add(ckb,0,LEA,ds)
                
                # --- init eeg file selectioncombobox
                cbb = wx.ComboBox(self._pnl_pdf,wx.NewId(),choices=[''],style=wx.CB_READONLY | wx.CB_SORT)
                cbb.SetItems([self._fif_extention,self._emptyroom_extention])
                cbb.SetName(subject_id + '_' + str(idx))
                self._cbbox[subject_id].append(cbb)
                fgs1.Add(cbb,1,LEA,ds + 1)
                
                if self._check_for_emptyroom(ckb,self._pdfs["bti"][subject_id][idx]):
                    cbb.SetValue(self._emptyroom_extention)
                else:
                    cbb.SetValue(self._fif_extention)
            
            fgs1.Add(wx.StaticLine(self._pnl_pdf),0,wx.LEFT | wx.EXPAND | wx.ALL)
            fgs1.Add(wx.StaticLine(self._pnl_pdf),0,wx.LEFT | wx.EXPAND | wx.ALL)
        
        self._ApplyPDFLayout(fgs1)
    
    def _check_for_emptyroom(self,ckb,pdf):
        """
         update colour and tooltip of wx.CheckBox
         if pdf is a emptyroom file

        :param ckb: check box
        :param pdf: dict with <emptyroom> key
        :return:
        """
        if self.emptyroom:
            if pdf.get("emptyroom"):
                ckb.SetForegroundColour(self._emptyroom_colour)
                ckb.SetToolTip(ckb.GetToolTip().GetTip() + "\n maked as <empty room> file")
                return True
        ckb.SetForegroundColour(wx.BLACK)
        return False
    
    def UpdateEmptyroom(self,v):
        self._emptyroom = v
        for subject_id in self._pdfs["bti"]:
            for idx in range(len(self._pdfs["bti"][subject_id])):
                #print("ER:{} {}".format(v,self._pdfs["bti"][subject_id][idx]))
                self._check_for_emptyroom(self._ckbox[subject_id][idx],self._pdfs["bti"][subject_id][idx])

class JuMEG_wxPDFMEEG(JuMEG_wxPDFBase):
    """
    JuMEG MEEG Merger PDF
    GUI to merge MEG and EEG data
    show  selected files called <Posted Data File> (PDF)
    pdfs=None, title='PDFs', stage=None, bg="grey90", pubsub=True,list=None,
    """
    def __init__(self, parent,name="PDFBOX_MEEG",*kargs,**kwargs):
        super().__init__(parent,name=name,*kargs)
        self._init(**kwargs)
    
    @property
    def EEG(self): return self._pdfs['eeg']
    @property
    def MNE(self): return self._pdfs['mne']
    
    def _init_defaults(self):
        self._fout_extention = "meeg-raw.fif"
        self._fmt_ckbox      = "{:>8}  {:>12}  {:>10}  {:>8}  {:>5}     {}"
        self._headline       = "{} {:^6} {:^12} {:^12} {:^9} {:^6} {:^14}".format("MNE/RAW","Id","Scan","Date","Time","Run","PDF Name")
        self.TxtEditvhdr     = None
        self.TxtEditvmrk     = None

    def UpdateSelectedPDFs(self):
        """
        update pdfs <self.PDFs> withn CTRLs selections
        id->mne ->index -><selected> = checkbox value  True/False
        id eeg_index-> <idx> = conmbox selction index
        :return updated pdfs
        
        Example:  self.PDFs
        -------
         'mne': {   '0815': [{ pdf': 'mne/0815/M100/180101_1333/1/0815_M100_180101_1333_1_c,rfDC-raw.fif',
                              'selected': True,size': 234630828 },
                             {pdf': 'mne/0815/M100/180101_1333/2/0815_M100_180101_1333_2_c,rfDC-raw.fif',
                              'selected': True,size': 234630828}]}
                              
         'eeg': {   '0815': [{'pdf': 'eeg/M100/0815_M100_01.vhdr','size': 6403800,'vhdr': 1557,'vmrk': 506},
                             {'pdf': 'eeg/M100/0815_M100_02.vhdr','size': 6403800,'vhdr': 1557,'vmrk': 506}}]
                             
                                 
         'eeg_index': {'0815': array([ 0,  1, -1, -1, -1,  3, -1, -1,  2, -1, -1])},
          matching index between MEG and EEG data
         'stage': '/data/exp/M100'
         
        """
        if not self._pdfs: return False
        for subject_id in self._pdfs.get('mne'):
            if subject_id not in (self._pdfs.get('eeg')): continue
            for idx in range(len(self._pdfs['mne'][subject_id])):
                self._pdfs['mne'][subject_id][idx]["selected"] = self._ckbox[subject_id][idx].GetValue()
                self._pdfs["eeg_index"][subject_id][idx]       = self._cbbox[subject_id][idx].GetSelection()
                
        return self._pdfs
        
    def update(self,pdfs=None,n_pdfs=None,reset=True):
        """
        update PSEL box with pdf structure
        shows for each PDF
         CheckBox with PDF name,
         Combobox with eeg-files,
         button menu to show vhdr,vmrk file

        :param pdfs  : pdf structure
        :param n_pdfs: number of pdfs files
        :param reset : will reset PSEL Box
        :return:
        
        Example:
        --------
        pdf structure :
        
        {'stage': '/data/exp/MEG0',
         'mne': {'0815': [{'pdf': 'mne/0815/MEG0/130718_1337/1/0815_MEG0_130718_1337_1_c,rfDC-raw.fif', 'size': 217975788, 'seleted': False},
                            {'pdf': 'mne/0815/MEG0/130718_1337/2/0815_MEG0_130718_1337_2_c,rfDC-raw.fif', 'size': 222899740, 'seleted': False},
                            {'pdf': 'mne/0815/MEG0/130718_1338/1/0815_MEG0_130718_1338_1_c,rfDC-raw.fif', 'size': 736558196, 'seleted': False},
                            {'pdf': 'mne/0815/MEG0/130718_1338/2/0815_MEG0_130718_1338_2_c,rfDC-raw.fif', 'size': 651060356, 'seleted': False},
         'eeg': {'0815': [{'pdf': 'eeg/MEG0/0815_MEG0_01.vhdr', 'size': 5971800, 'vhdr': 1557, 'vmrk': 506, 'seleted': False},
                            {'pdf': 'eeg/MEG0/0815_MEG0_02.vhdr', 'size': 6070200, 'vhdr': 1557, 'vmrk': 506, 'seleted': False},
                            {'pdf': 'eeg/MEG0/0815_MEG0_03.vhdr', 'size': 19783800, 'vhdr': 1557, 'vmrk': 16132, 'seleted': False},
                            {'pdf': 'eeg/MEG0/0815_MEG0_04.vhdr', 'size': 17252400, 'vhdr': 1557, 'vmrk': 96359, 'seleted': False},
         'eeg_index': {'0815': array([ 0,  1, -1, -1, -1, -1, -1, -1, -1, -1])}
         }
        """
      
        if reset:
           self._pdfs=None
           self.reset()
        if pdfs:
           self._pdfs = pdfs
       
        n_subjects = len(pdfs['mne'])
        
        ds   = 5
        LEA  = wx.LEFT | wx.EXPAND | wx.ALL
        fgs1 = wx.FlexGridSizer(n_pdfs + n_subjects, 4, ds, ds)
        fgs1.AddGrowableCol(1, proportion=2)

        for subject_id in pdfs.get('mne'):
            self._ckbox[subject_id] = []
            self._cbbox[subject_id] = []

            if subject_id not in (pdfs.get('eeg')): continue
               
            for idx in range( len(pdfs['mne'][subject_id]) ):
              #--- meg ckbox
                fpdf = pdfs["stage"] + "/" + pdfs['mne'][subject_id][idx]["pdf"]
                pdf_name = os.path.basename(fpdf)
                   
               # --- init  mne checkbox
                ckb = wx.CheckBox(self._pnl_pdf, wx.NewId(), label=self._checkbox_label(pdf_name), name="CK."+subject_id + '.' + str(idx))
                ckb.SetValue(True)
                ckb.SetForegroundColour(wx.BLACK)
                ckb.SetFont(self._font)
                ckb.SetToolTip(wx.ToolTip(fpdf))

                self._update_ckbox_file_exist(ckb,fpdf)
                self._ckbox[subject_id].append(ckb)
                fgs1.Add(ckb,0,LEA,ds)
              
              # --- init eeg file selectioncombobox
                cbb = wx.ComboBox(self._pnl_pdf, wx.NewId(), choices=[''], style=wx.CB_READONLY | wx.CB_SORT)
                   
                cbb.SetItems(  [ x["pdf"] for x in pdfs['eeg'][subject_id] ] )  # will clear cbb first
                
                cbb.SetName("CB."+subject_id + '.' + str(idx))
               # --- if eeg vhdr exist
                if pdfs["eeg_index"][subject_id][idx] > -1:
                   eeg_idx = pdfs["eeg_index"][subject_id][idx]
                   feeg    = pdfs["stage"]+"/"+pdfs['eeg'][subject_id][eeg_idx]["pdf"]
                   cbb.SetValue( pdfs["eeg"][subject_id][eeg_idx]["pdf"] )
                   cbb.SetToolTip(wx.ToolTip(feeg))
                   cbb.SetBackgroundColour(self._CB_BACK_COLOR)

               # ToDo update ToolTip path if OnSelection
               # except:
                else:
                   ckb.SetValue(False)
               
                self._cbbox[subject_id].append(cbb)
                fgs1.Add(cbb,1,LEA,ds)
              #---
                bt_vhdr = wx.Button(self._pnl_pdf, -1,name="FILE_VHDR_BT."+subject_id+".{}".format(idx),style=wx.BU_EXACTFIT | wx.BU_NOTEXT)
                bt_vhdr.SetBitmapLabel(wx.ArtProvider.GetBitmap(wx.ART_FILE_OPEN,wx.ART_MENU, (12, 12)))
                bt_vhdr.SetToolTip("Show eeg <vhdr> file")
                fgs1.Add(bt_vhdr,1,LEA,ds)
                bt_vmrk = wx.Button(self._pnl_pdf, -1,name="FILE_VMRK_BT."+subject_id+".{}".format(idx),style=wx.BU_EXACTFIT | wx.BU_NOTEXT)
                bt_vmrk.SetBitmapLabel(wx.ArtProvider.GetBitmap(wx.ART_FILE_OPEN,wx.ART_MENU, (12, 12)))
                bt_vmrk.SetToolTip("Show eeg <vmrk> file")
                fgs1.Add(bt_vmrk,1,LEA,ds)
         
           #---
            fgs1.Add(wx.StaticLine(self._pnl_pdf), 0, wx.LEFT | wx.EXPAND | wx.ALL)
            fgs1.Add(wx.StaticLine(self._pnl_pdf), 0, wx.LEFT | wx.EXPAND | wx.ALL)
            fgs1.Add(wx.StaticLine(self._pnl_pdf), 0, wx.LEFT | wx.EXPAND | wx.ALL)
            fgs1.Add(wx.StaticLine(self._pnl_pdf), 0, wx.LEFT | wx.EXPAND | wx.ALL)
          
            self._ApplyPDFLayout(fgs1)
    
    def _set_cbox_tooltip(self,subject_id,idx):
        i=int(idx)
        self._cbbox[subject_id][i].SetToolTip( self.GetStage() +"/"+self._cbbox[str(subject_id)][i].GetValue())
        #print("set TP: {}  {}".format(subject_id,i))
        #print(self.GetStage() )
        #print(self._get_tooltip( self._cbbox[subject_id][i] ) )
   
    def _get_cbox_tooltip(self,subject_id,idx):
        i=int(idx)
        if not self._cbbox[subject_id][i].GetToolTip(): return
        return self._cbbox[subject_id][i].GetToolTip().GetTip()
   
    def _get_tooltip(self,obj):
        if not obj.GetToolTip(): return
        return obj.GetToolTip().GetTip()
       
    def _ShowTextEditor(self,f,vmrk=False):
        #---ToDo in new CLS
        #--- highlight
        #DataFile=203404_INTEXT01_180412_1013.01.eeg
        #MarkerFile=203404_INTEXT01_180412_1013.01.vmrk
        #--- do correction asfile from combo
        # ck-button correct data file name as
        # ck-button correct mrk file name as
        
        print("TXT EDITOR TEST: "+f)
        print(vmrk)
        if vmrk:
           if self.TxtEditvmrk:
              return
           self.TxtEditvmrk = JuMEG_wxPDFMEEGTextEditor(name="JuMEG MEEG Editor <vmrk>",wildcard="Marker files (*.vmrk)|*.vmrk|all files (*.*)|*.*")
           self.TxtEditvmrk.LoadFile(f.replace(".vhdr",".vmrk"))
           return
        elif self.TxtEditvhdr:
             return
        self.TxtEditvhdr = JuMEG_wxPDFMEEGTextEditor(name="JuMEG MEEG Editor <vhdr>",wildcard="HDR files (*.vhdr)|*.vhdr|all files (*.*)|*.*")
        self.TxtEditvhdr.LoadFile(f)

    #--- highlight
        #DataFile=203404_INTEXT01_180412_1013.01.eeg
        #MarkerFile=203404_INTEXT01_180412_1013.01.vmrk
        #--- do correction asfile from combo
        # ck-button correct data file name as
        # ck-button correct mrk file name as
        
    def _ClickOnButton(self,evt):
        obj = evt.GetEventObject()
        vmrk=False
        fhdr=None
        if obj.GetName().startswith("FILE_VHDR_BT"):
           vmrk=False
        elif obj.GetName().startswith("FILE_VMRK_BT"):
           vmrk=True
        else: return
        
        s,subject_id,idx = obj.GetName().split(".")
        try:
            fhdr = self._get_cbox_tooltip(subject_id,idx)
        except:
            if vmrk:
               msg= "select <vmrk> file first !!!\n subject id: {}".format(subject_id)
            else:
               msg = "select <vhdr> file first !!!\n subject id: {}".format(subject_id)
            pub.sendMessage('MAIN_FRAME.MSG.ERROR',data=msg)
            return
        if fhdr:
           self._ShowTextEditor(fhdr,vmrk=vmrk)
    
    
    def _ClickOnCombo(self, evt):
        """
        check if eeg file is selected in other comboboxes
        if so draw combo text in red
        :param evt:
        :return:
        """
        obj  = evt.GetEventObject()
        n    = obj.GetName()

       #--- update cb subject_id and idx
        s,subject_id,idx = obj.GetName().split(".")
        self._set_cbox_tooltip(subject_id,idx)

        ckbt = self._find_obj_by_name(self._ckbox, n)
        if ckbt:
            if obj.GetValue():
               ckbt.SetValue(True)
            else:
               ckbt.SetValue(False)
            self._update_pdfinfo()

            # --- ck for if eeg file is multi selected
            # --- https://stackoverflow.com/questions/11528078/determining-duplicate-values-in-an-array
            a = []
            # --- fill array with selected index
            for cb in self._cbbox[subject_id]:
                cb.SetBackgroundColour(self._CB_BACK_COLOR)
                a.append(cb.GetSelection())
            # --- ck for unique
            uitem = np.unique(a)
            # --- exclude zeros == deselected files
            for i in uitem:
                # if i < 1: continue
                double_idx = np.where(a == i)[0]
                if double_idx.shape[0] < 2: continue  # no double selection
                for idx in double_idx:
                    if self._cbbox[subject_id][idx].GetValue():
                        self._cbbox[subject_id][idx].SetBackgroundColour(self._CB_ERROR_COLOR)

class JuMEG_wxPselBase(wx.Panel):
    """
    JuMEG PSEL Pation Selection Box to show subject IDs in a listbox and PDFs
    pdfs=None, title='PDFs', stage=None, bg="grey90", pubsub=True,list=None,
    """
    def __init__(self, parent,name="PDF_SELECTION_BOX",*kargs,**kwargs):
        super().__init__(parent,name=name,*kargs)
        self._id__box  = None
        self._pdf_box = None
    
    @property
    def IDSelectionBox(self): return self._id_box
    @property
    def PDFSelectionBox(self):return self._pdf_box
    
    def GetStage(self):
        return self.PDFSelectionBox.GetStage()
        
    def _init(self,**kwargs):
        """ init slots """
        self._init_defaults()
        self._wx_init(**kwargs)
        self._ApplyLayout()
        self.init_pubsub(**kwargs)
        
    def _init_defaults(self,**kwargs):
        pass
   
    def _update_from_kwargs(self,**kwargs):
        pass
    
    def _wx_init(self,**kwargs):
        pass

    def _ApplyLayout(self):
        ds = 4
        self.Sizer = wx.BoxSizer(wx.HORIZONTAL)
        if self._id_box:
           self.Sizer.Add(self.IDSelectionBox,0,wx.LEFT | wx.EXPAND | wx.ALL)
       #--- add info & PDFbox
        self.Sizer.Add(self.PDFSelectionBox,1,wx.LEFT | wx.EXPAND | wx.ALL,ds)
       # ---
        self.SetAutoLayout(True)
        self.SetSizer(self.Sizer)
        self.Fit()
        self.Update()
        self.Refresh()

    def GetSelectedPDFs(self):
        pass
    
    def GetMessage( self, msg ): return self.GetName()+ "." +msg.upper()

    def update_pdfbox(self,**kwargs):
        pass
    
    def update(self,**kwargs):
        self._update_from_kwargs(**kwargs)
        if kwargs.pop("reset",None):
           self.PDFSelectionBox.reset()
        self.IDSelectionBox.update(**kwargs)
   
    def init_pubsub(self, **kwargs):
        """ init pubsub call overwrite """
        pub.subscribe(self.update_pdfbox,self.IDSelectionBox.GetMessage("UPDATE"))
        
class JuMEG_wxPselMEEG(JuMEG_wxPselBase):
    """JuMEG MEEG Merger pation Selection Box wit ID listbox and  Posted Data File (PDF) Box
    pdfs=None, title='PDFs', stage=None, bg="grey90", pubsub=True,list=None,
    """
    def __init__(self, parent,name="PDFBOX_MEEG",*kargs,**kwargs):
        super().__init__(parent,name=name,*kargs,**kwargs)
        self._init(**kwargs)
     
    @property
    def fmeg_extention(self): return self._pdf_box.fout_extention
    @fmeg_extention.setter
    def fmeg_extention(self,v):
        self._pdf_box.fout_extention=v
        
    def _update_from_kwargs(self,**kwargs):
        pass
        #for k in kwargs:
        #    if k in self.__slots__:
        #            self.__setattr__(k,kwargs.get(k,self.__getattribute__(k)))
    def GetSelectedPDFs(self):
        """
        get selected PDFS for each subject_id
            mne fif file
            eeg file
            eeg_index
         more info <PDFSelectionBox.UpdateSelectedPDFs>
        
        :return:
         updated pdfs
        """
        return self.PDFSelectionBox.UpdateSelectedPDFs()
      
        
    def _wx_init(self,**kwargs):
       self._update_from_kwargs(**kwargs)
       self.SetBackgroundColour(kwargs.get("bg","grey90"))
       self._id_box  = JuMEG_wxIdSelectionBoxMEEG(self,**kwargs)
       self._pdf_box = JuMEG_wxPDFMEEG(self,**kwargs)

    def update_pdfbox(self,pdfs=None,n_pdfs=None):
        """
        update PDFBox with new data
        :param pdfs  : pdfs data structure
        :param n_pdfs: number of pdfs
        :return:
        
        """
        self.PDFSelectionBox.update(pdfs=pdfs,n_pdfs=n_pdfs)
        
class JuMEG_wxPselBTi(JuMEG_wxPselBase):
    """JuMEG BTi Posted Data File (PDF) Box
    pdfs=None, title='PDFs', stage=None, bg="grey90", pubsub=True,list=None,
    """
    
    def __init__(self,parent,name="PDFBOX_BTI",*kargs,**kwargs):
        super().__init__(parent,name=name,*kargs,**kwargs)
        self._init(**kwargs)
    
    @property
    def fif_extention(self):   return self._pdf_box.fout_extention
    @fif_extention.setter
    def fif_extention(self,v): self._pdf_box.fout_extention = v
   
    def UpdateEmptyroom(self,v):
        self._pdf_box.UpdateEmptyroom(v)
        
    def GetSelectedPDFs(self):
        """
        get selected PDFS for each subject_id
            mne fif file
            eeg file
            eeg_index
         more info <PDFSelectionBox.UpdateSelectedPDFs>

        :return:
         updated pdfs
        """
        return self.PDFSelectionBox.UpdateSelectedPDFs()
    
    def _wx_init(self,**kwargs):
        self._update_from_kwargs(**kwargs)
        self.SetBackgroundColour(kwargs.get("bg","grey90"))
        self._id_box  = JuMEG_wxIdSelectionBoxBTi(self,**kwargs)
        self._pdf_box = JuMEG_wxPDFBTi(self,**kwargs)
    
    def update_pdfbox(self,pdfs=None,n_pdfs=None):
        """
        :param pdfs: dict()
              {'stage':<start path>,
              'bti': {<subject_id>:[{'pdf': '0815/M100/18-01-01@00:01/1/c,rfDC', 'size': 528711512, 'hs_file': 118552, 'config': 193616, 'selected': False, 'emptyroom': False},
                                    {'pdf': '0815/M100/18-01-01@00:01/2/c,rfDC', 'size': 525419312, 'hs_file': 118552, 'config': 193616, 'selected': False, 'emptyroom': False}
                                   ]}
        :n_pdfs: number of pdfs
        
        :return:
        list of selected pdfs as list [<pdf>, <file-extention>]
       
        """
        #print("PDFBOX BTI update pdfs")
        #print(pdfs)
        #print("-" * 50)
        
        self.PDFSelectionBox.update(pdfs=pdfs,n_pdfs=n_pdfs)


class JuMEG_wxIdSelectionBoxFIF(JuMEG_wxIdSelectionBoxBase):
    """
     JuMEG_wxSelectionBoxFIF
      input
         parent widged
           stage  : stage path to experiment e.g. /data/XYZ/exp/M100
           pubsub : True; use wx.pubsub msg systen sends like:
                    pub.sendMessage('EXPERIMENT_TEMPLATE',stage=stage,experiment=experiment,TMP=template_data)
           or button event <ID_SELECTION_BOX_UPDATE> from Update button
           id_mask: #######; mask for id selection
           verbose: False
           bg     : grey90 backgroundcolor
    """
    
    def __init__(self,parent,name="ID_SELECTION_BOX_FIF",*kargs,**kwargs):
        super().__init__(parent,name=name,*kargs,**kwargs)
        self._init()
    
    def _initPDFs(self,**kwargs):
        self.PDFs = JuMEG_Utils_PDFsMNE(**kwargs)
    
    def GetPDFsFromIDs(self,ids=None):
        """
         call to MEG/MNE obj function
        """
        if not ids:
            ids = self.ItemSelectedList
        if not ids:
            pub.sendMessage("MAIN_FRAME.MSG.ERROR",data="\n ID Selection Box: Please select IDs first")
        return self.PDFs.GetPDFsFromIDs(ids=ids)
    
    def GetSelections(self,ids=None):
        if not ids:
            ids = self.ItemSelectedList
        return self.PDFs.GetPDFsFromIDs(ids=ids)
    
    def update(self,stage=None,scan=None,data_type='mne'):
        """

        :param stage:
        :param scan:
        :param data_type:
        :return:
        """
        if stage:
            self.stage = stage
        self._item_list = []
        self.PDFs.update(scan=scan)
        self._item_list = self.PDFs.GetIDs()
        self.wx_update_lb()
        #pub.sendMessage(self.GetName().upper() + ".UPDATE")
    
    def ClickOnUpdate(self):
        cnts = 0
        
        #aa= self.GetSelections()
        #for ai in aa:
        #    print("\n--------")
        #    print(ai)
        
        pdfs = { "stage":self.stage,"fif":None}
        pdf_selection = self.GetSelections()
        if pdf_selection:
            pdfs["fif"] = pdf_selection[0]
            n_pdfs = pdf_selection[1] or 0
        pub.sendMessage(self.GetMessage("UPDATE"),pdfs=pdfs,n_pdfs=pdf_selection[1])


class JuMEG_wxPDFsFIF(JuMEG_wxPDFBase):
    """
    JuMEG FIF
    show  selected files called <Posted Data File> (PDF)
    pdfs=None, titl e='PDFs', stage=None, bg="grey90", pubsub=True,list=None,
    emptyroom=True
    """
    
    def __init__(self,parent,name="PDFBOX_BTI",*kargs,**kwargs):
        super().__init__(parent,name=name,*kargs)
        self._init(**kwargs)
    
    def _init_defaults(self):
        self._pdfs = None
        self._n_pdfs = 0
        self._emptyroom_colour = wx.BLUE
        self._fif_extention = "-raw.fif"
        self._separator = '/'
        self._fmt_ckbox = "{:>8}  {:>12}  {:>10}  {:>8}  {:>5}     {}"
        self._headline = "{} {:^6} {:^12} {:^10} {:^12} {:^6} {:^10}".format("FIF","Id","Scan","Date","Time","Run",
                                                                             "PDF Name")
    
    def _update_from_kwargs(self,**kwargs):
        self._fif_extention = kwargs.get("fif_extention",self._fout_extention)
    
    def _init_wx_search_ctrl(self,parent,**kwargs):
        self._pnl_search = wx.Panel(parent,style=wx.SUNKEN_BORDER)
        self._pnl_search.SetBackgroundColour(kwargs.get("bgsearch","grey95"))
    
    def UpdateSelectedPDFs(self):
        """
        update pdfs <self.PDFs> withn CTRLs selections
        id->index -><{pdf":f,"size":os.stat(f).st_size,"hs_file":size_hs,"config":size_cfg,"selected" = checkbox value  True/False}
        :return
        list of list of selected pdfs and file extention (-raw.fif or -empty.fif)
        [ [ "pdf1", "-raw.fif"], .. [ "pdf10","-empty.fif"] ]

        """
        pdfs = []
        
        for subject_id in self._pdfs["fif"]:
            for idx in range(len(self._pdfs["fif"][subject_id])):
                if not self._ckbox[subject_id][idx].GetValue(): continue
                
                self._ckbox[subject_id][idx].SetValue(self._ck_file_size(self._pdfs["fif"][subject_id][idx]))
                self._pdfs["fif"][subject_id][idx]["selected"] = self._ckbox[subject_id][idx].GetValue()
                
                if self._pdfs["fif"][subject_id][idx]["selected"]:
                    pdfs.append([self._pdfs["fif"][subject_id][idx]["pdf"],self._cbbox[subject_id][idx].GetValue()])
        return pdfs
    
    def update(self,pdfs=None,n_pdfs=None,reset=True):
        """
        update PSEL box with pdf strcture

        :param pdfs  : pdf structure
        :param n_pdfs: number of pdfs files
        :param reset : will reset PSEL Box
        :return:

        pdf:
         {'stage': '/home/fboers/MEGBoers/data/megdaw_data21',
          'bti': {
                  '0815': [{'pdf': '0815/M100/18-01-01@00:01/1/c,rfDC', 'size': 528711512, 'hs_file': 118552, 'config': 193616, 'selected': False},
                           {'pdf': '0815/M100/18-01-01@00:01/2/c,rfDC', 'size': 525419312, 'hs_file': 118552, 'config': 193616, 'selected': False},
                           {'pdf': '0815/M100/18-01-01@00:01/3/c,rfDC', 'size': 524170552, 'hs_file': 118552, 'config': 193616, 'selected': False},
                           ]}}
        """
        if reset:
            self.reset()
        
        if pdfs:
            self._pdfs = pdfs
            self._n_pdfs = n_pdfs or 0
        
        if not self._n_pdfs: return
        if not self._pdfs:   return
        
        try:
            n_subjects = len([*self._pdfs["fif"]])
        except:
            return
        
        ds = 5
        LEA = wx.LEFT | wx.EXPAND | wx.ALL
        fgs1 = wx.FlexGridSizer(self._n_pdfs + n_subjects,1,ds,ds)
        
        for subject_id in self._pdfs["fif"]:
            self._ckbox[subject_id] = []
            self._cbbox[subject_id] = []
            
            for idx in range(len(self._pdfs["fif"][subject_id])):
                #--- bti ckbox
                # 0815/M100/18-01-01@00:01/1/c,rfDC'
                pdf_name = self._pdfs["fif"][subject_id][idx]["pdf"]
                fpdf = self._pdfs["stage"] + "/" + self._pdfs["fif"][subject_id][idx]["pdf"]
                # --- init  mne checkbox
                ckb = wx.CheckBox(self._pnl_pdf,wx.NewId(),
                                  label=self._checkbox_label(pdf_name.replace("/","_").replace('@',"_")),
                                  name=subject_id + '_' + str(idx))
                ckb.SetValue(True)
                ckb.SetFont(self._font)
                ckb.SetForegroundColour(wx.BLACK)
                ckb.SetToolTip(wx.ToolTip(fpdf))
                
                self._update_ckbox_file_exist(ckb,fpdf)
                self._ckbox[subject_id].append(ckb)
                fgs1.Add(ckb,0,LEA,ds)
            
            fgs1.Add(wx.StaticLine(self._pnl_pdf),0,wx.LEFT | wx.EXPAND | wx.ALL)
        
        self._ApplyPDFLayout(fgs1)

class JuMEG_wxPselFIF(JuMEG_wxPselBase):
    """JuMEG Posted Data File (PDF) Box
    pdfs=None, title='PDFs', stage=None, bg="grey90", pubsub=True,list=None,
    """
    def __init__(self,parent,name="PDFBOX_FIF",*kargs,**kwargs):
        super().__init__(parent,name=name,*kargs,**kwargs)
        self._init(**kwargs)
    
    @property
    def fif_extention(self):   return self._pdf_box.fout_extention
    
    @fif_extention.setter
    def fif_extention(self,v): self._pdf_box.fout_extention = v
    
    def UpdateEmptyroom(self,v):
        self._pdf_box.UpdateEmptyroom(v)
    
    def GetSelectedPDFs(self):
        """
        get selected PDFS for each subject_id
            mne fif file
            eeg file
            eeg_index
         more info <PDFSelectionBox.UpdateSelectedPDFs>

        :return:
         updated pdfs
        """
        return self.PDFSelectionBox.UpdateSelectedPDFs()
    
    def _wx_init(self,**kwargs):
        self._update_from_kwargs(**kwargs)
        self.SetBackgroundColour(kwargs.get("bg","grey90"))
        self._id_box  = JuMEG_wxIdSelectionBoxFIF(self,**kwargs)
        self._pdf_box = JuMEG_wxPDFsFIF(self,**kwargs)
    
    def update_pdfbox(self,pdfs=None,n_pdfs=None):
        """
        :param pdfs: dict()
              {'stage':<start path>,
              'bti': {<subject_id>:[{'pdf': '0815/M100/18-01-01@00:01/1/c,rfDC', 'size': 528711512, 'hs_file': 118552, 'config': 193616, 'selected': False, 'emptyroom': False},
                                    {'pdf': '0815/M100/18-01-01@00:01/2/c,rfDC', 'size': 525419312, 'hs_file': 118552, 'config': 193616, 'selected': False, 'emptyroom': False}
                                   ]}
        :n_pdfs: number of pdfs

        :return:
        list of selected pdfs as list [<pdf>, <file-extention>]

        """
        #print("PDFBOX BTI update pdfs")
        #print(pdfs)
        #print("-" * 50)
        
        self.PDFSelectionBox.update(pdfs=pdfs,n_pdfs=n_pdfs)
