#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 15:28:58 2018

@author: fboers
"""

import wx,os,sys,glob,fnmatch
import logging

from jumeg.gui.wxlib.jumeg_gui_wxlib_psel_base  import JuMEG_wxIdSelectionBoxBase,JuMEG_wxPDFBase,JuMEG_wxPselBase,JuMEG_wxSearchBoxBase
from jumeg.gui.wxlib.jumeg_gui_wxlib_richtext   import JuMEG_wxRichTextFrame
from jumeg.gui.utils.jumeg_gui_utils_pdfs       import JuMEG_Utils_PDFsMEEG ,JuMEG_Utils_PDFsEEG

from jumeg.base.ioutils.jumeg_ioutils_find      import JuMEG_IOutils_FindPDFs,JuMEG_IoUtils_FileIO

from pubsub  import pub

logger = logging.getLogger('jumeg')

__version__="2019.07.02.001"

'''
#================================================
#========#=======================================
#   Ids  #  Search Box Pnl
# ------ #=======================================
#        #       PDFs Pnl + EEG Combobox
#        #
#--------#
# UPDATE #
#  BT    #
#========#=======================================
#================================================
-> call <UPDATE IDS> from TemplateBt => cd <stage> -> update IDs listbox
-> click on IDs Update Bt
 --> for id in IDs find_files( serach_box pattern)
 --> update PDFs

'''

class JuMEG_wxPDFMEEGTextEditor(JuMEG_wxRichTextFrame):
    """
    Txt Editor to show vhdr,vmrk file from eeg recordings
    change <DataFile> and <MarkerFile> names for correct matching
    """
    
    def __init__(self,name="JuMEG PDFs MEEG Editor",
                 wildcard="HDR files (*.vhdr)|*.vhdr|marker files (*.vmrk)|*.vmrk|all files (*.*)|*.*",
                 status_text="JuMEG Text Editor; select a text file",path="."):
        super().__init__(None,-1,name,size=(700,500),style=wx.DEFAULT_FRAME_STYLE | wx.STAY_ON_TOP,
                         wildcard=wildcard,
                         status_text=status_text,
                         path=path
                         )

#==== MEEG
class JuMEG_wxSearchBoxMEEG(JuMEG_wxSearchBoxBase):
    """
    MEEG SearchBox
    wx.Search ctrls to search for pattern in filename
    
    """
    def __init__(self,parent,**kwargs):
        super().__init__(parent,**kwargs)
        self.separator = "_"
        
        self._controls = [
            ["Scan",12],
            ["Session",11],
            ["Run",3],
            ["PDF_Name",128,"*c,rfDC-raw.fif"],
            ["EEG Name",10,"*.vhdr"]
            ]
            
        self._init(**kwargs)
        
        self.isRecursive  = True
        self.isIgnoreCase = True
        
    @property
    def growable_cols(self): return len(self._controls)-2
    
    def GetPattern(self):
        """
        BTI pattern from sub folders:

        scan/session/run/c,rfDc

        :return:
        pattern, file_extention
        """
        pat = []
        
        if len(self.SearchControls) > 1:
            for idx in range(len(self.SearchControls) - 1):
                if not self.SearchControls[idx].GetValue():  #continue
                    pat.append("*")
                else:
                    pat.append(self.SearchControls[idx].GetValue())
        pat.append("")
        return self.separator.join(pat),self.SearchControls[-1].GetValue()


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
        self.datatype = kwargs.get("datatype","mne")
        self._init()
    
    @property
    def stage(self):
        return self.PDFs.stage
    
    @stage.setter
    def stage(self,v):
        self.PDFs.stage = v
     
    def _init(self,**kwargs):
        self._initPDFs(**kwargs)
        self._update_from_kwargs(**kwargs)
        self._item_list = []
        self._item_selected_list = []
        self._wx_init(**kwargs)
        
    def _initPDFs(self,**kwargs):
        self.PDFs = JuMEG_Utils_PDFsMEEG(**kwargs)
       # self.EEG  = JuMEG_Utils_PDFsEEG(**kwargs)
       
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

   # def _update_from_kwargs(self,**kwargs):
   #     super()._update_from_kwargs(**kwargs)
    #    self.datatype = kwargs.get("datatype","mne")
        
       # if self.PDFs:
       #    self.PDFs.scan    = kwargs.get("scan",None)
       #    self.PDFs.session = kwargs.get("session",None)
       #    self.PDFs.run     = kwargs.get("run",None)
       #    self.PDFs.verbose = kwargs.get("verbose",self.verbose)
       #    self.PDFs.debug   = kwargs.get("debug",self.debug)
        
    def update(self,**kwargs):
        """

        :param stage:
        
        :param scan:
        :param session:
        :param run:
        :param verbose: <False>
        :param debug:   <False>
        
        :param data_type: <mne>
        :return:
        """
        self._update_from_kwargs(**kwargs)
  
        self._item_list = []
  
        self.PDFs.update(**kwargs)
        self._item_list = self.PDFs.GetIDs()
        self.wx_update_lb()
        #pub.sendMessage(self.GetName().upper() + ".UPDATE")

    def ClickOnUpdate(self,evt):
        cnts = 0

        #aa= self.GetSelections()
        #for ai in aa:
        #    print("\n--------")
        #    print(ai)

        pdfs = { "stage":self.stage,"mne":None,"eeg":None,"eeg_index":None }
        pdf_selection,eeg_selection,pdfs["eeg_index"] = self.GetSelections()
        
        n_pdfs = None
        
        if pdf_selection:
            pdfs["mne"] = pdf_selection[0]
            n_pdfs = pdf_selection[1] or 0
        if eeg_selection:
            pdfs["eeg"] = eeg_selection[0]

        pub.sendMessage(self.GetMessage("UPDATE"),pdfs=pdfs,n_pdfs=n_pdfs)


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
        
        try:
           n_subjects = len(pdfs['mne'])
        except:
           pub.sendMessage('MAIN_FRAME.MSG.ERROR',data="No PDFs found")
           return
        
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


class JuMEG_wxPselMEEG(JuMEG_wxPselBase):
    """JuMEG MEEG Merger pation Selection Box wit ID listbox and  Posted Data File (PDF) Box
    pdfs=None, title='PDFs', stage=None, bg="grey90", pubsub=True,list=None,
    
    Example:
    --------
    from jumeg.gui.wxlib.jumeg_gui_wxlib_psel_meeg      import JuMEG_wxPselMEEG
    
    #--- init
    PDFBox = JuMEG_wxPselMEEG(wxPanel,name=self.GetName()+".PDFBOX_MEEG",**kwargs)
    PDFBox.update(stage=stage,scan=scan,reset=True,verbose=self.verbose,debug=self.debug)
    
    #--- get pdfs
    pdfs = self.PDFBox.GetSelectedPDFs()
   """
    
    def __init__(self,parent,name="PDFBOX_MEEG",*kargs,**kwargs):
        super().__init__(parent,name=name,*kargs,**kwargs)
        self._init(**kwargs)
    
    @property
    def fmeg_extention(self): return self._pdf_box.fout_extention
    
    @fmeg_extention.setter
    def fmeg_extention(self,v):
        self._pdf_box.fout_extention = v
    
    #def _update_from_kwargs(self,**kwargs):
    #    pass
    #    #for k in kwargs:
    #    #    if k in self.__slots__:
    #    #            self.__setattr__(k,kwargs.get(k,self.__getattribute__(k)))
    
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
        self._id_box     = JuMEG_wxIdSelectionBoxMEEG(self,**kwargs)
        self._pdf_box    = JuMEG_wxPDFMEEG(self,**kwargs)
        self._search_box = JuMEG_wxSearchBoxMEEG(self,name="PDF_MEEG_SEARCHBOX",recursive=True,ignorecase=True)
        self.ioutils = JuMEG_IoUtils_FileIO()
    
    def update_pdfbox(self,pdfs=None,n_pdfs=None):
        """
        update PDFBox with new data
        :param pdfs  : pdfs data structure
        :param n_pdfs: number of pdfs
        :return:

        """
        
        #pattern = self.ioutils.update_pattern(pattern=self.SearchBox.GetPattern(),ignore_case=self.SearchBox.isIgnoreCase)

        self.PDFSelectionBox.update(pdfs=pdfs,n_pdfs=n_pdfs) #,pattern=pattern)
    
    def update(self,**kwargs):
        """
        
        :param kwargs:
        :return:
        """
        #self._update_from_kwargs(**kwargs)
        if kwargs.pop("reset",None):
           self.PDFSelectionBox.reset()
        #if self.SearchBox:
        #   self.SearchBox.update(**kwargs)
        
        scan = kwargs.get("scan")
        
        #--- pass to PDFs search pattern
       #print(kwargs)
        kwargs.update( self.SearchBox.GetValues() )
        
        kwargs["eeg_scan"]=kwargs.get("scan")
        kwargs["scan"] = scan
        
        kwargs.update( {"recursive":self.SearchBox.isRecursive,"ignore_case": self.SearchBox.isIgnoreCase })
        #print(kwargs)
        self.IDSelectionBox.update(**kwargs)

class JuMEG_wxTestPSEL(wx.Panel):
     def __init__(self,parent,**kwargs):
         super().__init__(parent)
         self._psel = None
        #--- BIT export
         #self.stage = "$JUMEG_PATH_LOCAL_DATA/megdaw_data21/"
         self.stage = "$JUMEG_PATH_BTI_EXPORT"
         
         self.experiment=None#"JuMEGTest"
         self.scan      = None#"MEG94T"
         
         self._init(**kwargs)
         
     @property
     def PSEL(self): return self._psel
     
     def _init(self,**kwargs):
         self._psel = JuMEG_wxPselMEEG(self,stage=self.stage,scan=self.scan)
         self._wx_init(**kwargs)
         self._ApplyLayout()


     def _wx_init(self,**kwargs):
         self.SetBackgroundColour(kwargs.get("bg","grey90"))

         self._pnl1 = wx.Panel(self)
         # add combo/radio choose BTI/MEEG
         self._bt    = wx.Button(self._pnl1,-1,label="UPDATE")
         self.Bind(wx.EVT_BUTTON,self.ClickOnCtrls)
         
     def _ApplyLayout(self):
         LEA = wx.LEFT | wx.EXPAND | wx.ALL
         
         vbox1 = wx.BoxSizer(wx.VERTICAL)
         vbox1.Add(self._bt,0,LEA,1)
         self._pnl1.SetSizer(vbox1)
         self._pnl1.SetAutoLayout(True)
         self._pnl1.Fit()


         vbox = wx.BoxSizer(wx.VERTICAL)
         if self._psel:
            vbox.Add(self._psel,1,LEA,1)

         vbox.Add(self._pnl1,0,LEA,1)

         self.SetSizer(vbox)
         self.SetAutoLayout(True)
         self.Fit()
         self.Update()
         self.Refresh()

     def ClickOnCtrls(self,evt):
         obj = evt.GetEventObject()
         print( "ClickOnCtrls: {}".format(obj.GetName() ) )
         
         #pub.sendMessage("PDFBOX_BTI.ID_SELECTION_BOX.UPDATE",stage=self.stage,scan=self.scan,data_type='bti')
         
         self.PSEL.update(stage=self.stage,experiment=self.experiment,data_type="mne")

if __name__ == '__main__':
  app = wx.App()
  frame = wx.Frame(None, -1, "Test PSEL", size=(640, 480))
  win = JuMEG_wxTestPSEL(frame)
  frame.Show(True)
  app.MainLoop()
