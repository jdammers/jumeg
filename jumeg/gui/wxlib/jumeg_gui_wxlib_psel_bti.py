#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 15:28:58 2018

@author: fboers
"""

import wx,os,sys,glob,fnmatch
import logging

from jumeg.gui.wxlib.jumeg_gui_wxlib_psel_base   import JuMEG_wxIdSelectionBoxBase,JuMEG_wxPDFBase,JuMEG_wxPselBase,JuMEG_wxSearchBoxBase

from pubsub  import pub

logger = logging.getLogger('jumeg')

__version__="2019.09.27.001"

'''
#================================================
#========#=======================================
#   Ids  #  Search Box Pnl
# ------ #=======================================
#        #       PDFs Pnl
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
#==== BTI
class JuMEG_wxSearchBoxBTI(JuMEG_wxSearchBoxBase):
    def __init__(self,parent,**kwargs):
        super().__init__(parent,**kwargs)
        self.separator = "/"
        self._choices.append("empty room")
        
        self._controls = [
            ["Scan",12],
            ["Session",11],
            ["Run",3],
            ["PDF",256,"c,rfDC"]
            ]
        
        self._init(**kwargs)
    
    @property
    def isEmptyRoom(self):
        return self.GetValue("empty room")
    
    @isEmptyRoom.setter
    def isEmptyRoom(self,v):
        self.SetValue("empty room",v)
    
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
    
   # def _initIDs(self,**kwargs):
   #     self.IDs = JuMEG_IOutils_FindIds(**kwargs)
    
    def GetSelections(self,ids=None):
        if not ids:
           ids = self.ItemSelectedList
        if not ids:
           pub.sendMessage("MAIN_FRAME.MSG.ERROR",data="\n ID Selection Box: Please select IDs first")
           return list()
        return ids
        
    def update(self,**kwargs):
        """

        :param stage:
        :param scan:
        :param pdf_name:
        :return:
        """
        self._update_from_kwargs(**kwargs)
        self._item_list = self.IDs.find(**kwargs)
        self.wx_update_lb()
        
    def ClickOnUpdate(self,evt):
        """
        send event to parent -> PSEL or PSEL.PDFs
        """
        if self.pubsub:
           pub.sendMessage(self.GetMessage("UPDATE"),ids=self.GetSelections())
        else:
           evt.Skip()

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
        self._emptyroom_colour    = wx.BLUE
        self._emptyroom_extention = "-empty.fif"
        
        self._fif_extention = "-raw.fif"
        self.fif_colour     = wx.BLACK
        
        self._separator = '/'
        self._fmt_ckbox = "{:>8}  {:>12}  {:>10}  {:>8}  {:>5}     {}"
        self._headline = "{} {:^6} {:^12} {:^10} {:^12} {:^6} {:^10}".format("4D/BTi","Id","Scan","Date","Time","Run",
                                                                             "PDF Name")
    
    def _update_from_kwargs(self,**kwargs):
        self._emptyroom           = kwargs.get("emptyroom",self.emptyroom)
        self._emptyroom_extention = kwargs.get("emptyroom_extention",self._emptyroom_extention)
        self._fif_extention       = kwargs.get("fif_extention",self._fout_extention)
    
    def _init_wx_search_ctrl(self,parent,**kwargs):
        self._pnl_search = wx.Panel(parent,style=wx.SUNKEN_BORDER)
        self._pnl_search.SetBackgroundColour(kwargs.get("bgsearch","grey95"))
    
    def _ck_file_size(self,pdf,ckhs=False,stage=None,min_size=1):
        """
        checks filesize > 1 for pdf,config

        :param pdf: dict
              {pdf': '0815/M100/18-01-01@00:01/1/c,rfDC', 'size': 528711512, 'hs_file': 118552, 'config': 193616, 'selected': False}
        :param ckhs: <False>
        :param min_size: minimum filsize
        :return:
         True/False
        """
        #if self.debug:
        #   logger.debug("PDFs  stage: {}\n{}".format(stage,pdf))
        key_list = [os.path.basename( pdf.get("pdf") ),"config"]
        if ckhs:
           key_list.append("hs_file")
        d = os.path.dirname( pdf.get("pdf") )
        if stage:
           d = os.path.join(stage,d)
        try:
            for f in key_list:
                if os.stat( os.path.join(d,f) ).st_size < min_size:
                   return False
        except OSError:
               logger.exception("File not exists or empty:\n --> file: {}\n --> pdf:{}".format( os.path.join(d,f),pdf ) )
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
           logger.error("ERROR Please select PDFs first !!!",file=sys.stderr)
           return None
        
        # logger.info("PDFs: {}".format(self._pdfs))
        
        for subject_id in self._pdfs["bti"]:
            for idx in range(len(self._pdfs["bti"][subject_id])):
                if not self._ckbox[subject_id][idx].GetValue(): continue
                
                self._ckbox[subject_id][idx].SetValue(self._ck_file_size(self._pdfs["bti"][subject_id][idx],stage=self._pdfs["stage"]) )
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
            
            for idx in range( len(self._pdfs["bti"][subject_id] ) ):
                #--- bti ckbox
                # 0815/M100/18-01-01@00:01/1/c,rfDC'
                pdf_name = self._pdfs["bti"][subject_id][idx]["pdf"]
                fpdf     = self._pdfs["stage"] + "/" + self._pdfs["bti"][subject_id][idx]["pdf"]
                # --- init  mne checkbox
                ckb = wx.CheckBox(self._pnl_pdf,-1,
                                  label=self._checkbox_label(pdf_name.replace("/","_").replace('@',"_")),
                                  name=subject_id + '_' + str(idx))
                ckb.SetValue(True)
                ckb.SetFont(self._font)
                ckb.SetForegroundColour(wx.BLACK)
                ckb.SetToolTip(wx.ToolTip(fpdf))
                
               # self._update_ckbox_file_exist(ckb,fpdf)
                self._ckbox[subject_id].append(ckb)
                fgs1.Add(ckb,0,LEA,ds)
                
                # --- init file extention selectioncombobox
                cbb = wx.ComboBox(self._pnl_pdf,-1,choices=[''],style=wx.CB_READONLY | wx.CB_SORT)
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
        ckb.SetForegroundColour(self.fif_colour)
        return False
    
    def UpdateEmptyroom(self,v):
        self._emptyroom = v
        for subject_id in self._pdfs["bti"]:
            for idx in range(len(self._pdfs["bti"][subject_id])):
                #print("ER:{} {}".format(v,self._pdfs["bti"][subject_id][idx]))
                self._check_for_emptyroom(self._ckbox[subject_id][idx],self._pdfs["bti"][subject_id][idx])

    def _ClickOnCombo(self,evt):
        """
        checks if an eeg file is selected in combobox
        if not ckbox wille unchecked
        :param evt:
        :return:
        """
        obj    = evt.GetEventObject()
        id,idx = obj.GetName().split("_")
        idx    = int(idx)
        
        pdf    = self._pdfs["bti"][id][idx]
        if obj.GetValue() == self._fif_extention:
           pdf["emptyroom"]=False
        else:
           pdf["emptyroom"]=True
        
        self._check_for_emptyroom(self._ckbox[id][idx],pdf)
        

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
    
    @property
    def isEmptyRoom(self):  return self.SearchBox.isEmptyRoom
    def isEmptyRoom(self,v):
        self.SearchBox.isEmptyRoom = v
    
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
        self._id_box     = JuMEG_wxIdSelectionBoxBTi(self,**kwargs)
        self._pdf_box    = JuMEG_wxPDFBTi(self,**kwargs)
        self._search_box = JuMEG_wxSearchBoxBTI(self,name="PDF_BTI_SEARCHBOX",recursive=True,ignorecase=True)
        
        self.SearchBox.isRecursive  = True
        self.SearchBox.isIgnoreCase = True
        self.SearchBox.isEmptyRoom  = True
    
    #def update(self,**kwargs):
        #for l in self.SearchBox.GetSearchLabels():
    #        if kwargs.get( l.lower() ):
    #           self.SearchBox.SetValue(l,kwargs.get(l.lower()))
    #
    #    super().update(**kwargs)
        
    def update_pdfbox(self,ids=None,stage=None):
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
        
        pattern,file_extention = self.SearchBox.GetPattern()
        
        pdfs = {"stage": self.GetStage(),"bti":{}}
        npdfs= 0
        
        for id in self.IDSelectionBox.GetSelections():
            sdir = os.path.join( self.GetStage(),id)
            flist = self.IDSelectionBox.IDs.find_files(start_dir=sdir,pattern=pattern,file_extention=file_extention,
                                                       recursive=self.SearchBox.isRecursive,
                                                       ignore_case=self.SearchBox.isIgnoreCase,debug=self.debug)
            for fpdf in flist:
                if not pdfs["bti"].get(id):
                   pdfs["bti"][id] = []
                   scan_tmp,session = fpdf.split("/")[0:2]
                   date_tmp         = session.split("@")[0]
                
                scan,session = fpdf.split("/")[0:2]
                date         = session.split("@")[0]
               
               #--- works for  sorted file list
               #--- new scan last should be empty room
                if self.SearchBox.isEmptyRoom:
                   if scan != scan_tmp:
                      pdfs["bti"][id][-1]["emptyroom"] = True # set last one
                      scan_tmp  = scan
                      date_tmp  = session.split("@")[0]
                   elif date != date_tmp:
                      date_tmp = date
                      pdfs["bti"][id][-1]["emptyroom"] = True
              
                pdfs["bti"][id].append( {"pdf": os.path.join(id,fpdf),"selected":False,"emptyroom":False} )
                
                npdfs+=1
           
           #--- sets last one to "empty room", ToDo check for mulity day
            if self.SearchBox.isEmptyRoom and pdfs["bti"].get(id):
               pdfs["bti"][id][-1]["emptyroom"] = True
            
       #--- check for empty room
        self.PDFSelectionBox.update(pdfs=pdfs,n_pdfs=npdfs,reset=True)


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
         self._psel = JuMEG_wxPselBTi(self,stage=self.stage,scan=self.scan)
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
