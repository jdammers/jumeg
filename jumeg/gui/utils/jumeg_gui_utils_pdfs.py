#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 10:08:17 2018

@author: fboers
"""

import os,sys,glob,contextlib
#import  re,textwrap,ast
#from types import ModuleType
# from pathlib import Path

import numpy as np

import logging
logger = logging.getLogger('jumeg')

from jumeg.base.jumeg_base import JuMEG_Base_Basic
jb = JuMEG_Base_Basic()

__version__="2019.05.14.001"

#===========================================================================

class JuMEG_Utils_PDFsBase(object):
    """
    Base CLS to find Posted Data Files (PDFs) via pattern search on harddisk
    default for 4D/BTi format

    Parameters:
    -----------
    stage,scan,session,run,prefix,postfix,pdf_name,seperator,data_type,verbose,debug
    """
    __slots__ =["experiment","id","scan","session","run","prefix","postfix","pdf_name","eeg_name","seperator","pattern","recursive","ignore_case","data_type",
                "id_list","verbose","debug","_stage","_start_path","_number_of_pdfs","_total_filesize"]
    
    def __init__(self,**kwargs):
        self._init(**kwargs)

    def _init(self,**kwargs):
        """ init slots """
        for k in self.__slots__:
            self.__setattr__(k,None)

        self.seperator       = "/"  # for glob.glob match  subdirs
        self.verbose         = False
        self._pdfs           = dict()
        self._total_filesize = 0

        self._init_defaults()

        self._update_from_kwargs(**kwargs)
        
    def _init_defaults(self):
        pass
    def _info_special(self):
        return []
    
    @property
    def pdfs(self)    : return self._pdfs
    @property
    def ids(self)     : return [*self._pdfs] #.sorted()
    @property
    def stage(self)   : return self._stage
    @stage.setter
    def stage(self,v):
        if v:
           self._stage = os.path.expandvars( os.path.expanduser(v) )
        
    @property
    def number_of_ids(self): return len(self._pdfs.keys())
    @property
    def number_of_pdfs(self):return self._number_of_pdfs

    @property
    def start_path(self):
        self._start_path = self.stage
        if self.experiment:
            self._start_path += "/" + self.experiment
        if self.data_type:
            self._start_path += "/" + self.data_type
        return self._start_path
    
    def GetTotalFileSize(self):
        return self._total_filesize

    def GetTotalFileSizeGB(self):
        return self._total_filesize / 1024 ** 3

    def GetTotalFileSizeMB(self):
        return self._total_filesize / 1024 ** 2
    
    def GetIDs(self):
        return sorted([*self._pdfs])
    
    def GetPDFsFromIDs(self,ids=None):
        """
         https://stackoverflow.com/questions/3129322/how-do-%20i-get-monitor-resolution-in-python
        :param ids:
        :return:
        PDF dict for ids in idlist
        {"pdf":fullfile reletive to <stage>,"size":file size,"hs_file":True/Faslse,"config":Ture/False}
        """
        if not ids: return
        if isinstance(ids,(list)):
           cnts = 0
           pdfs = dict()
           for x in ids:
               if x in self.pdfs:
                  pdfs[x]= self.pdfs[x]
                  cnts+=len(self.pdfs[x])
         # return {x: self.pdfs[x] for x in ids if x in self.pdfs}
           return pdfs,cnts
        return {ids:self.pdfs[ids]},len(self.pdfs[ids])
    
    def _update_from_kwargs(self,**kwargs):
        for k in kwargs:
            if k in self.__slots__:
                self.__setattr__(k,kwargs.get(k,self.__getattribute__(k)))
    
        #for k in self.__slots__:
        #    self.__setattr__(k,kwargs.get(k,self.__getattribute__(k)))
        
        if kwargs.get("stage"):
           self.stage=kwargs.get("stage")
        
    def get_parameter(self,key=None):
        """
        get  parameter
        :param key:
        :return: parameter dict or value for parameter[key] or all keys
        """
        if key: return self.__getattribute__(key)
        return { slot:self.__getattribute__(slot) for slot in self.__slots__ }

    
    def get_stage_path_offset(self):
        self.stage_path_offset = self.start_path.replace(self.stage,"")
        if self.stage_path_offset.startswith("/"):
           self.stage_path_offset = self.stage_path_offset.replace("/","",1)
        return  self.stage_path_offset

    #def update(self,**kwargs):
    #    self._update_from_kwargs(**kwargs)
        
    def update_pattern(self,*args,**kwargs):
        """
        pattern for glob.glob
        https://docs.python.org/3.5/library/glob.html?highlight=glob#glob.glob
        
        :return
         pattern
        """
        self._update_from_kwargs(**kwargs)
        
        l = []
        if self.id:
            l.append(self.id)
        if self.scan:
            l.append(self.scan)
        if self.session:
            l.append(self.session)
        if self.run:
            l.append(self.run)
        if self.pdf_name:
           l.append(self.pdf_name)
        else:
           d=""
           if self.prefix: d+=self.prefix
           if self.postfix:d+=self.postfix
           if d: l.append(d)
        if len(l):
           pat = self.seperator.join(l)
        else:
           pat = self.pdf_name

        #if pat.find("**") < 0:
        #    d = pat.split(self.seperator)
        #    d[-1] = "**" + self.seperator + d[-1]
        #    pat = self.seperator.join(d)

        #--- glob.glob
        if pat.find("/**/") < 0:
           d = pat.split("/")
           d[-1]= "**/" + d[-1]
           pat = "/".join(d)

       
        if self.ignore_case:
           pat_ic = ""
           for c in pat:
               if c.isalpha():
                  pat_ic += '[%s%s]' % (c.lower(),c.upper())
               else:
                  pat_ic += c
           self.pattern = pat_ic
           return self.pattern
        
        self.pattern = pat
        return self.pattern
  
    def info(self,debug=False):
        """
        prints info stage,number of found ids and pdfs, total filesize
        :return:
        """
        msg=["---> Data Type      : {}".format(self.data_type),
             "PDF  Stage          : {}".format(self.stage),
             "start path          : {}".format(self.start_path),
             "scan                : {}".format(self.scan),
             "pdf name            : {}".format(self.pdf_name),
             "Number of IDs       : {}".format(self.number_of_ids),
             "Number of PDFS      : {}".format(self.number_of_pdfs),
             "total file size[GB] : {0:.9f}".format( self.GetTotalFileSizeGB() ),
             "last used pattern   : {}".format(self.pattern),"."*50
            ]
        msg+= self._info_special()
        
        logger.info( "\n".join(msg) )

        if self.debug or debug:
           for id in self.pdfs.keys():
               msg==["-> ID: "+id]
               for pdf in self.pdfs[id]:
                   msg.append(pdf)
           logger.debug("\n".join(msg))
           

    def _update_pdfs(self,id,f):
        pass
    
    def update(self,**kwargs):
        self._pdfs = dict()
        self._update_from_kwargs(**kwargs)
        
        self._number_of_pdfs = 0
        self._total_filesize = 0.0
        self.get_stage_path_offset()
        
        #print("PATTERN: " + self.pattern)
        #print("Start path: " + self.start_path)
        with jb.working_directory(self.start_path):
             for id in os.scandir(path="."):
                 if not id.is_dir(): continue    #os.path.isdir(id): continue
                 lpdfs = []
                 for f in glob.iglob(self.update_pattern(id=id.name),recursive=self.recursive):
                     lpdfs.append(f)
                     lpdfs.sort()
                 if not lpdfs: continue
                 if not self._pdfs.get(id):
                    self._pdfs[id.name] = []
                 for f in lpdfs:
                     self._update_pdfs(id.name,f)
                     
                 self._number_of_pdfs+=len(self._pdfs[id.name])
      
       # print(self._pdfs)
       # print(self.ignore_case)
       # print(self.update_pattern(id=id.name))
        
class JuMEG_Utils_PDFsMNE(JuMEG_Utils_PDFsBase):
    """
     CLS find MEG IDs in <stage>
     for mne [meg data] along directory structure:
     <stage/experimnet/id/scan/session/run/
     check for <-raw>
    
     Example:
     --------
      from jumeg import jumeg_base as jb
      from jumeg.gui.utils.jumeg_gui_utils_pdfs import JuMEG_Utils_PDFsMNE
      stage = "/data/meg1/exp"
      PDFs  = JuMEG_Utils_PDFsMNE(stage=stage,experimnet="TEST",scan="TEST01",verbose=True,debug=True)
      
      PDFs.update()
      PDFs.get_parameter()
      PDFs.info()
      print("\n---> DONE PDFs: "+PDFs.data_type)
    """
    
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
    
    def _init_defaults(self):
        self.data_type  = "mne"
        self.pdf_name   = None
        self.prefix     = '*'
        self.postfix    = "c,rfDC-raw.fif"
        
    def _info_special(self):
        return ["experiment          : {}".format(self.experiment),
                "prefix              : {}".format(self.prefix),
                "postix              : {}".format(self.postfix),"-"*50]

    def _update_pdfs(self,id,f):
        self._pdfs[id].append({ "pdf":self.stage_path_offset +"/"+f,"size":os.stat(f).st_size,"selected":False })
        self._total_filesize += self._pdfs[id][-1]["size"]
 
 
class JuMEG_Utils_PDFsEEG(JuMEG_Utils_PDFsBase):
    """
     CLS find EEG IDs in <stage>
     for eeg data along directory structure:
     <stage/experiment/eeg/**/*.vhdr
     check for <.vhdr> <.vmrk> <.eeg>
     
     Example:
     -------
      from jumeg import jumeg_base as jb
      from jumeg.gui.utils.jumeg_gui_utils_pdfs import JuMEG_Utils_PDFsEEG
      stage = "/data/meg_store1/exp"
      PDFs  = JuMEG_Utils_PDFsEEG(stage=stage,experimnet="TEST",scan="TEST01",verbose=True,debug=True)
      
      PDFs.update()
      PDFs.get_parameter()
      PDFs.info()
      print("\n---> DONE PDFs: "+PDFs.data_type)
    """
    
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
    
    def _init_defaults(self):
        self.data_type = "eeg"
        self.prefix    = '*'
        self.postfix   = '.vhdr'
        self.seperator = "_"
        
    def _info_special(self):
        return ["experiment          : {}".format(self.experiment),
                "prefix              : {}".format(self.prefix),
                "postix              : {}".format(self.postfix),"-"*50]
  
    def update_pattern(self,*args,**kwargs):
        """
        pattern for glob.glob
        
        "/mnt/meg_store2/exp/TEST/eeg/TEST01/**/*.vhdr"
        
        https://docs.python.org/3.5/library/glob.html?highlight=glob#glob.glob
        
        :return
         pattern
        """
        self._update_from_kwargs(**kwargs)
        
        l = []
        if self.id:
            l.append(self.id)
        if self.scan:
           l.append(self.scan)
        if self.session:
            l.append(self.session)
        if self.run:
            l.append(self.run)
        if self.eeg_name:
           l.append(self.eeg_name)
        else:
           d=""
           if self.prefix: d+=self.prefix
           if self.postfix:d+=self.postfix
           if d: l.append(d)
        
        if len(l):
           pat = self.seperator.join(l)
        else:
           pat = self.eeg_name
        
       #--- glob.glob
        if pat.find("**/") < 0:
           d = pat.split("/")
           d[-1]= "**/" + d[-1]
           pat = "/".join(d)
        
        if self.ignore_case:
           pat_ic = ""
           for c in pat:
               if c.isalpha():
                  pat_ic += '[%s%s]' % (c.lower(),c.upper())
               else:
                  pat_ic += c
           self.pattern = pat_ic
           return self.pattern

        self.pattern = pat
        return self.pattern
        
    def check_files(self,fhdr):
        """
        check filesize of vmrk,data
        :param full eeg filename

        :return:
        size of headshape file
        size of config file
        """
        try:
            size_data = os.stat(fhdr.replace(".vhdr",".eeg")).st_size
        except:
            size_data = 0
        try:
            size_mrk = os.stat(fhdr.replace(".vhdr",".vmrk")).st_size
        except:
            size_mrk = 0
    
        return size_data,size_mrk

    def _update_pdfs(self,id,fhdr):
        size_data,size_mrk= self.check_files(fhdr)
        self._pdfs[id].append({ "pdf":self.stage_path_offset +"/"+fhdr,"size":size_data,"vhdr":os.stat(fhdr).st_size,"vmrk":size_mrk}) #,"selected":False })
        self._total_filesize += self._pdfs[id][-1]["size"] + size_data + size_mrk

    def update(self,**kwargs):
        """
        :param: id_list: list of ids from mne existing raw files!!
        :param kwargs:
        
        :return:
        """
        self._update_from_kwargs(**kwargs)
        #self.id_list         = kwargs.get("id_list",self.id_list)
        self._pdfs           = dict()
        self._number_of_pdfs = 0
        self._total_filesize = 0.0
        self.get_stage_path_offset()
        
        with jb.working_directory(self.start_path):
             if self.id_list:
                for id in self.id_list:
                    lpdfs=[]
                    for f in glob.iglob(self.update_pattern(id=id),recursive=True):
                        lpdfs.append(f)
                        lpdfs.sort()
                    if not lpdfs: continue
                    if not self._pdfs.get(id):
                       self._pdfs[id] = []
                    for f in lpdfs:
                        self._update_pdfs(id,f)
            
                    self._number_of_pdfs += len(self._pdfs[id])
        #print(self._pdfs)
        #print("EEG path: {}".format(self.start_path))
        #print("EEG pat: {}".format(self.pattern))

#=====================================================================
class JuMEG_Utils_PDFsMEEG(object):
    """
    CLS to find MNE PDFs (Posted Data Files) in FIF format with their corresponding EEG files via pattern matching
    information to pass to merge MEG and EEG files
    
    Parameters:
    -----------
    prefix     : <'*'>
    postfix_meg: <'*c,rfDC-raw.fif'>
    postfix_eeg: <'*.vhdr'>
    stage      : <None>
    experimnet : <None>
    id         : <None>
    scan       : <None>
    session    : <None>
    run        : <None>
    postfix:
    prefix_mne:
    postfix_mne:
    prefix_eeg:
    postfix_eeg:
    :return:
    
    Example:
    -----------
    from jumeg.gui.utils.jumeg_gui_utils_pdfs import JuMEG_Utils_PDFsMEEG
    stage="/data/exp/"
    PDFs = JuMEG_Utils_PDFsMEEG(stage=stage,verbose=True,debug=False)
    PDFs.experiment = "TEST"
    PDFs.scan       = "TEST01"
    #--- update  PDfs  meg/mne and eeg data
    PDFs.update()
    #--- show infon
    PDFs.info()

    print( PDFs.GetIDs())
    >> ['109887', '110058']
    print( PDFs.GetEEGIndexFromIDs() )
    >> {'109887': array([ 0,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1]), '110058': array([ 0,  1, -1, -1, -1, -1, -1, -1, -1, -1])}
    print( PDFs.GetEEGIndexFromIDs( PDFs.GetIDs()[0]) )
    >> [0  1 -1 -1 -1 -1 -1 -1 -1 -1 -1]
    """

    def __init__(self,**kwargs):
        super().__init__()
        
        self._EEG  = JuMEG_Utils_PDFsEEG()
        self._MNE  = JuMEG_Utils_PDFsMNE()
        
        self._pdfs      = dict()
        self._eeg_index = dict()
        self.__NO_MATCH = -1
        self.pattern    = None
        #self.recursive  = True
        #self.ignorecase = False
        
        self._update_from_kwargs(**kwargs)
        
    @property
    def NoMatch(self): return self.__NO_MATCH
    
    @property
    def MNE(self):  return self._MNE
    @property
    def MEG(self):  return self._MNE
    @property
    def EEG(self):  return self._EEG
  
    @property
    def pdfs(self): return self._pdfs
    @property
    def eeg_index(self): return self._eeg_index
    @property
    def ids(self):  return [*self._pdfs] #.sorted()
    
    @property
    def stage(self):    return self.MNE.stage
    @stage.setter
    def stage(self,v):
        self.MNE.stage=v
        self.EEG.stage=v
    
    @property
    def experiment(self): return self.MNE.experiment
    @experiment.setter
    def experiment(self,v):
        self.MNE.experiment=v
        self.EEG.experiment=v
    
    @property
    def scan(self):       return self.MNE.scan
    @scan.setter
    def scan(self,v):
        self.MNE.scan = v
       # self.EEG.scan = v
    
    @property
    def id(self):        return self.MNE.id
    @id.setter
    def id(self,v):
        self.MNE.id=v
        self.EEG.id=v
    @property
    def session(self):    return self.MNE.session
    @session.setter
    def session(self,v):
        self.MNE.session=v
        self.EEG.session=v
    
    @property
    def run(self):        return self.MNE.run
    @run.setter
    def run(self,v):
        self.MNE.run=v
        self.EEG.run=v
   
    @property
    def prefix_meg(self)   : return self.MNE.prefix
    @prefix_meg.setter
    def prefix_meg(self,v):  self.MNE.prefix=v
    @property
    def postfix_meg(self):   return self.MNE.postfix
    @postfix_meg.setter
    def postfix_meg(self,v): self.MNE.postfix = v

    @property
    def prefix_eeg(self):    return self.EEG.prefix
    @prefix_eeg.setter
    def prefix_eeg(self,v):  self.EEG.prefix = v
    @property
    def postfix_eeg(self):   return self.EEG.postfix
    @postfix_eeg.setter
    def postfix_eeg(self,v): self.EEG.postfix = v

    @property
    def number_of_ids(self): return len(self._pdfs.keys())
    @property
    def number_of_pdfs(self):return self._number_of_pdfs

    @property
    def verbose(self): return self.MNE.verbose
    @verbose.setter
    def verbose(self,v):
        self.MNE.verbose=v
        self.EEG.verbose=v

    @property
    def debug(self): return self.MNE.debug
    @debug.setter
    def debug(self,v):
        self.MNE.debug=v
        self.EEG.debug=v

    def _update_from_kwargs(self,**kwargs):
        """
            :param stage:
            :param id:
            :param scan:
            :param session:
            :param run:
            :param postfix:
            :param prefix_mne:
            :param postfix_mne:
            :param prefix_eeg:
            :param postfix_eeg:
            :param pattern:
            :return:
        """
        #for k in kwargs:
        #    if k in self.__slots__:
        #        self.__setattr__(k,kwargs.get(k,self.__getattribute__(k)))
    
        if kwargs.get("stage"):
            self.stage = kwargs.get("stage")
    
        self.stage      = kwargs.get("stage",self.stage)
        self.experiment = kwargs.get("experiment",self.experiment)
    
        self.id      = kwargs.get("id",self.id)
        self.scan    = kwargs.get("scan",self.scan)
        self.session = kwargs.get("session",self.session)
        self.run     = kwargs.get("run",self.run)
        self.postfix = kwargs.get("postfix_meeg","meeg")
    
        self.prefix_meg  = kwargs.get("prefix",self.prefix_meg)
        self.postfix_meg = kwargs.get("postfix_meg",self.postfix_meg)
    
        self.prefix_eeg  = kwargs.get("prefix",self.prefix_eeg)
        self.postfix_eeg = kwargs.get("postfx_eeg",self.postfix_eeg)
    
        self.verbose     = kwargs.get("verbose",self.verbose)
        self.debug       = kwargs.get("debug",self.debug)
        self.pattern     = kwargs.get("pattern")
        #self.recursive   = kwargs.get("recursive",self.recursive)
        #self.ignorecase  = kwargs.get("ignorecase",self.ignorecase)
        
    def GetTotalFileSize(self):
        return self._total_filesize

    def GetTotalFileSizeGB(self):
        return self._total_filesize / 1024 ** 3

    def GetTotalFileSizeMB(self):
        return self._total_filesize / 1024 ** 2
    
    def GetIDs(self):
        return sorted([*self._MNE.pdfs])

    def GetPDFsFromIDs(self,ids=None):
        """
         call to MEG/MNE obj function
        """
        return self.MNE.GetPDFsFromIDs(ids=ids)
    
    def GetEEGsFromIDs(self,ids=None):
        """
         call to MEG/MNE obj function
        """
        return self.EEG.GetPDFsFromIDs(ids=ids)
 
    def GetEEGIndexFromIDs(self,ids=None):
        """
        https://stackoverflow.com/questions/3129322/how-do-%20i-get-monitor-resolution-in-python
        :param ids:
        :return:
        eeg index dict for ids in idlist, numpy array of index match mne and eeg data
        """
        if not ids:
           return { x:self._eeg_index[x] for x in self._eeg_index }
        if isinstance(ids,(list)):
           return {x: self._eeg_index[x] for x in ids if x in self._eeg_index}
        return self._eeg_index[ids]
        
    def info(self):
        """
        prints info stage,number of found ids and pdfs, total filesize
        :return:
        """
        logger.info("PDF  Stage          : {}\n".format(self.stage)+
                    "experiment          : {}\n".format(self.experiment)+
                    "scan                : {}\n".format(self.scan) )
                
        self.MNE.info()
        self.EEG.info()
        
        if self.debug:
           for id in self._eeg_index.keys():
               logger.debug(" --> EEG idx ID: {}\n".format(id) +  jb.pp_list2str( self._eeg_index[id] ) )
            
    def update(self,**kwargs):
        """
        for each id in id-list find mne/meg and eeg data
       
        Result:
        -------
        pdf dictionary

        """
        self._update_from_kwargs(**kwargs)
        self._eeg_index = dict()
    
        self.MNE.update(**kwargs) #pattern=self.pattern,recursive=self.recursive)
        if kwargs.get("eeg_scan"):
           kwargs["scan"]=kwargs.get("eeg_scan")
        self.EEG.update(id_list=self.MNE.GetIDs(),**kwargs)
      
    #--- search for matching scan and run
        for id_item in self.MNE.GetIDs():
            n_raw = len( self.MNE.pdfs[id_item] )
        #--- make lookup tab for matching later mne raw with eeg raw
            self._eeg_index[id_item] = np.zeros(n_raw,dtype=np.int64) + self.NoMatch
        
            if id_item in self.EEG.GetIDs():
               eeg_idx = self._match_meg_eeg_list(id_item=id_item)
            else:
               eeg_idx = np.zeros(n_raw,dtype=np.int64) + self.NoMatch
        
        #--- ck for double
            uitems,uidx,uinv = np.unique(eeg_idx,return_inverse=True,return_index=True)
            self._eeg_index[id_item][uidx] = uitems

    def _match_meg_eeg_list(self,id_item=None):
        """
         find common eeg & meg files from meg and eeg list
    
         Parameter:
         ----------
         id_item: <None>
         
         Result:
         --------
         numpy array: matching index for eeg list; -1 => no match
    
         Example:
         --------
         type of input data
    
          MNE/MEG
           path to data/205630/INTEXT01/181017_1006/1/205630_INTEXT01_181017_1006_1_c,rfDC-raw.fif
           path to data/205630/INTEXT01/181017_1006/2/205630_INTEXT01_181017_1006_2_c,rfDC-raw.fif
           ...
           path to data/205630/INTEXT01/181017_1112/1/205630_INTEXT01_181017_1112_1_c,rfDC-raw.fif
           path to data/205630/INTEXT01/181017_1112/2/205630_INTEXT01_181017_1112_2_c,rfDC-raw.fif
           path to data/205630/INTEXT01/181017_1135/1/205630_INTEXT01_181017_1135_1_c,rfDC-raw.fif
          EEG
           path to data/205630_INTEXT01_181017_1006.02.vhdr
           path to data/205630_INTEXT01_181017_1006.01.vhdr
           ...
           path to data/205630_INTEXT01_181017_1112.01.vhdr
           path to data/205630_INTEXT01_181017_1112.02.vhdr
    
          or older experiments
             meg: path to data/204260_MEG9_130619_1301_2_c,rfDC-raw.fif
             eeg: path to data/204260_MEG9_2.vhdr
         """
        meg_pdfs,mne_cnts = self.MNE.GetPDFsFromIDs(ids=id_item)
        # {'205382': [{'pdf': 'mne/205382/MEG94T0T/130624_1300/1/205382_MEG94T0T_130624_1300_1_c,rfDC-raw.fif', 'size': 222943156, 'seleted': False}, ..}
        
        eeg_pdfs,eeg_cnts = self.EEG.GetPDFsFromIDs(ids=id_item)
        # {'205382': [{'pdf': 'eeg/MEG94T/205382_MEG94T0T_01.vhdr', 'size': 6075000, 'vhdr': 1557, 'vmrk': 506, 'seleted': False},
        
        found_list = np.zeros(mne_cnts,dtype=np.int64) + self.NoMatch
        match_run  = True
        match_time = False
        match_date = False
      #--- should be only one id_item
        meg_list = meg_pdfs[id_item]
        eeg_list = eeg_pdfs[id_item]
    
        #---  e.g. 212048_FREEVIEW01_180411_1030.01.vhdr
        n = len(os.path.basename(eeg_list[0]['pdf']).replace('.','_').split('_'))
        eeg_idx0   = 2
        eeg_pattern={}
        
        if n > 5:
            match_date = True
            match_time = True
            eeg_idx1 = 5
        elif n > 4:
            match_date = True
            eeg_idx1 = 4
        else:
            eeg_idx1 = 3

        runs  = []
        dates = []
        times = []

        for pdf_eeg in eeg_list:
            s = os.path.basename(pdf_eeg["pdf"]).replace('.','_').split('_')
            runs.append(s[-2].lstrip("0") or 0)  # 003 => 3     0000 => 0
            if match_time:
               times.append(s[-3])
            if match_date:
               dates.append(s[-4])
        if runs:
           eeg_pattern["run"]  = np.array( runs)
        if times:
           eeg_pattern["time"]= np.array(times)
        if dates:
           eeg_pattern["date"]= np.array(dates)
     
        idx=0
        for pdf_meg in meg_list:
            idxs = None
            f = os.path.basename(pdf_meg["pdf"])
            try:
               #--- error if run=2a
                meg_pattern = f.split('_')[2:5] # 205382_MEG94T0T_130624_1300_1_c,rfDC-raw.fif'
             
               #--- match run
                if match_run:
                  # 003 => 3     0000 => 0
                   meg_pattern[-1] = meg_pattern[:][-1].lstrip("0") or 0
                   idxs = np.where( eeg_pattern["run"] == meg_pattern[-1])[0]
                   if not idxs.size: continue
               #--- match time
                if match_time:
                   found_idxs = np.where(eeg_pattern["time"][idxs] == meg_pattern[-2])[0]
                   if not found_idxs.size: continue
                   idxs = idxs[found_idxs]
               # --- match date
                if match_date:
                    found_idxs = np.where(eeg_pattern["date"][idxs] == meg_pattern[-3])[0]
                    if not found_idxs.size: continue
                    idxs = idxs[found_idxs]
                if isinstance(idxs,(np.ndarray)):
                    if idxs.size:
                        found_list[idx] = idxs[0]
              
            except:
                pass
            
            idx+=1
            
        if self.debug:
           msg= ["Match MEG/EEG: {}".format(found_list)]
           for i in range(len(meg_list)):
               msg.append("  ->MEG: {}".format(meg_list[i]["pdf"]))
               if len(found_list)>i:
                  if found_list[i]>-1:
                     msg.append(" ->EEG: {}".format(eeg_list[ found_list[i]]["pdf"] ))
               msg.append("   "+"-"*40)
           logger.debug("\n".join(msg) )
           
        return found_list

