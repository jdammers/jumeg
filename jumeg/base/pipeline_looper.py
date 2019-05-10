#!/usr/bin/env python3
# -+-coding: utf-8 -+-

"""
"""

#--------------------------------------------
# Authors: Frank Boers <f.boers@fz-juelich.de> 
#
#-------------------------------------------- 
# Date: 06.05.19
#-------------------------------------------- 
# License: BSD (3-clause)
#--------------------------------------------
# Updates
#--------------------------------------------
"""
sys args, argparser

I/O
loop for pdfs to process
file name as arg
file list
find in id subfolder

logger
setup logger + catch stout/in
setup defaults

config file
load read cfg

check file extention

save output raw
"""

import os,logging,yaml,argparse,glob

from jumeg.base.jumeg_base import jumeg_base as jb

logger = logging.getLogger('root')

__version__="2019.05.07.001"

class JuMEG_PDF_BASE(object):
   def __init__(self,**kwargs):
       self._stage  = "."
       self._pdfs   = None
       self.verbose = False
       self.debug   = False
       self._file_extention = ["meeg-raw.fif","rfDC-empty.fif"]

   @property
   def file_extention(self): return self._file_extention
   @file_extention.setter
   def file_extention(self,v):
       if not isinstance(v,(list)):
          self._file_extention = list(v)
       else:
          self._file_extention = v

   @property
   def pdfs(self): return self._pdfs
   
   @property
   def basedir(self): return self.stage
   @basedir.setter
   def basedir(self,v):
       self.stage = v

   @property
   def stage(self): return self._stage
   @stage.setter
   def stage(self,v):
       if v:
          self._stage = self.get_fullpath(v)
       else:
          self._stage = v
          
   def _update_from_kwargs(self,**kwargs):
       self.file_extention = kwargs.get("file_extention",self.file_extention)
       self.basedir        = kwargs.get("basedir",self.basedir)
       self.stage          = kwargs.get("stage",self.stage)
       self.verbose        = kwargs.get("verbose",self.verbose)
       self.debug          = kwargs.get("debug",self.debug)
   
   def get_fullpath(self,v):
       if v:
          return os.path.expandvars(os.path.expanduser(v))
       return None
   
   def clear(self):
       self._pdfs=[]
   
   def update(self,**kwargs):
       pass
       
class JuMEG_PDF_IDS(JuMEG_PDF_BASE):
   '''
   cls to find files in stage/subject-id subfolder matching one file-extention in <file extention list>
   
   Parameter
   ---------
   subject_ids": list of subject ids  ["1213456","234567"]
   stage          : "."
   file_extention : ["meeg-raw.fif","rfDC-empty.fif"]
   recursive      : False
   verbose        : False
   debug          : False
   
   Return
   ------
   list of files
  
   Example
   -------
    from jumeg.base.pipeline_looper import JuMEG_PDF_IDS()
    
    #--- init logger
    from jumeg.base import jumeg_logger
    jumeg_logger.setup_script_logging(logger=logger)
    logger.setLevel(logging.DEBUG)
  
    #--- run
    logger.info("Start test find files fom subject_ids list")
   
    stage = "$JUMEG_PATH_LOCAL_DATA/exp/JUMEGTest/mne"
    subject_ids = "211890","211747"

    PDF = JuMEG_PDF_IDS()
    PDF.update(stage=stage,subject_ids=subject_ids,separator= ",",recursive=True,verbose=True,debug=True)
    #--- print PDFs
    pint(PDF.pdfs)
    
   '''
   def __init__(self,**kwargs):
       super().__init__(**kwargs)
       self._ids      = None
       self.separator = ","
       self.recursive = False
       self._update_from_kwargs(**kwargs)
       
   @property
   def subject_ids(self): return self._ids
   @subject_ids.setter
   def subject_ids(self,v):
       self._ids = []
       if isinstance(v,(list)):
          self._ids = v
       elif isinstance(v,(str)):
          self._ids = v.split(",")
       elif v:
          self._ids = list(v)
          
   def _update_from_kwargs(self,**kwargs):
       super()._update_from_kwargs(**kwargs)
       self.subject_ids = kwargs.get("subject_ids",self.subject_ids)
       self.recursive   = kwargs.get("recursive",self.recursive)

   def update(self,**kwargs):
      """
        loop over subjects dir find files matching one of the file extentions
      
        :param subject_ids    : string or list of strings e.g: subject ids
        :param stage          : start dir,stage
        :param file_extention : string or list  <None>
                                if <file_extention> checks filename extention ends with extentio from list
        :param recursive      : recursive searching for files in subdirs of <stage/subject_id> using glob.iglob <False>
        :return:
         file list

         Example:
         --------
      
      """
      self._update_from_kwargs(**kwargs)
       
      if self.recursive:
         fpatt = '**/*'
      else:
         fpatt = '*'
    
      self._pdfs = []
      
      for subj in self.subject_ids:
          try:
         #--- check if its a dir
             recordings_dir = os.path.join(self.stage,subj)
             if not os.path.isdir( recordings_dir ):
                continue
             
             with jb.working_directory(recordings_dir):
                  for fext in self.file_extention:
                      if self.debug:
                         logging.debug( "  -> recordings dir: {}\n".format(recordings_dir)+
                                        "  -> extention     : {}\n".format(fext)+
                                        "  -> glob recursive: {}\n".format(self.recursive) +
                                        "  -> glob pattern  : {}".format(fpatt)
                                       )
                           
                      for f in glob.glob(fpatt + fext,recursive=self.recursive):
                          self._pdfs.append( os.path.join(recordings_dir,f) )
          except:
              logger.exception("---> error subject : {}\n".format(subj) +
                               "  -> recordings dir: {}\n".format(recordings_dir) )
      self._pdfs.sort()
      
      if self.debug:
         logger.debug( "---> PDFS:\n  ->"+"\n  ->".join(self.pdfs) )
        
      return self._pdfs

class JuMEG_PDF_LIST(JuMEG_PDF_BASE):
   """
   get files from list file

   Parameter
   ---------
   list_path      : path to list file
   list_file      : filename
   stage          : "."
   verbose        : False
   debug          : False
   
   Return
   ------
   list of files
   
   """
   def __init__(self,**kwargs):
       super().__init__()
       self._stage           = None
     #--- files from list
       self._list_file_name  = None
       self._list_file_path  = None
  
   @property
   def list_file_name(self): return self._list_file_name
   @list_file_name.setter
   def list_file_name(self,v):
       self._list_file_name=v
   @property
   def list_file_path(self): return  self._list_file_path
   @list_file_path.setter
   def list_file_path(self,v):
       self._list_file_path=self.get_fullpath(v)

   def GetFullListFileName(self):
       if self.list_file_path:
          return os.path.join(self.list_file_path,self.list_file_name)
       return self.list_file_name

   def _update_from_kwargs(self,**kwargs):
       super()._update_from_kwargs(**kwargs)
       self.list_file_path = kwargs.get("list_path",self.list_file_path)
       self.list_file_name = kwargs.get("list_name",self.list_file_name)
     
   def update(self,**kwargs):
       self._pdfs = self.get_files_from_list(**kwargs)
       return self._pdfs
       
   def get_files_from_list(self,**kwargs):
       """
       
       :param reset:
       :param kwargs:
       :return:
       """
      
       self._update_from_kwargs(**kwargs)
       found_list = []
       try:
           if not os.path.isfile( self.GetFullListFileName() ):
              logger.exception("list file not found: {}".format(self.GetFullListFileName()))
              return None
            
           # if self.debug:
           #    logger.info("  -> list file: {}".format( self.GetFullListFileName() ) )
            
           with open(self.GetFullListFileName(),'r') as f:
                for line in f:
                    line = line.strip()
                    fname = None
                    if line:
                        if (line[0] == '#'): continue
                        fname = line.split()[0]
                        if self.stage:
                           fname = os.path.join( self.stage+ "/" + fname )
                        if os.path.isfile(fname):
                           found_list.append(fname)
       except :
           logger.exception(" error in reading list file: {}".format( self.GetFullListFileName() ) )
        
       if self.debug:
          logger.debug(" --> PDF in list file: {}\n".format(self.GetFullListFileName() )+
                       "  -> counts : {}\n".format( len(found_list) )+
                       "  -> files  :\n    "+ "\n    ".join(found_list) )
        
       return found_list


class JuMEG_PDF_FILE(object):
    """
     Parameter
     ---------
      fname : filename
      fpath : filepath
     
     Return
     ------
     full filename

    """
    def __init__(self,**kwargs):
        super().__init__()
        self.clear()

    def clear(self):
        self._path = None
        self.name = None

    @property
    def pdf(self):
        return self.GetFullFileName()
    
    @property
    def path(self):
        return self._path
    
    @path.setter
    def path(self,v):
        if v:
            self._path = os.path.expandvars(os.path.expanduser(v))
        else:
            self._path = v
    
    def GetFullFileName(self):
        if self.path:
            return os.path.join(self.path,self.name)
        return self.name
    
    def _update_from_kwargs(self,**kwargs):
        self.path = kwargs.get("fpath",self.path)
        self.name = kwargs.get("fname",self.name)
    
    def update(self,**kwargs):
        self._update_from_kwargs(**kwargs)
        return self.GetFullFileName()

   
class JuMEG_PipelineLooper(JuMEG_PDF_BASE):
    """
     get PDF
     from file args
     file list
     id list
     
    """
    def __init__(self,**kwargs):
        super().__init__()
        self._PDFIDS  = JuMEG_PDF_IDS()
        self._PDFList = JuMEG_PDF_LIST()
        self._PDFFile = JuMEG_PDF_FILE()
        
    #--- toDo cls property
        self._config_file = None
        self._config      = None
        self._globals     = None # dict config,subjects,stage,fif_extention,recursive
    
    @property
    def config(self): return self._config

    @property
    def config_file(self):
        return self._config_file

    def _update_from_kwargs(self,**kwargs):
        self._PDFIDS._update_from_kwargs(**kwargs)
        self._PDFList._update_from_kwargs(**kwargs)
        self._PDFFile._update_from_kwargs(**kwargs)

    def clear(self):
        self._PDFIDS.clear()
        self._PDFList.clear()
        self._PDFFile.clear()

    def update(self,**kwargs):
        self.clear()
        self._IDList.update(**kwargs)
        self._FileList.update(**kwargs)
        self._FilesFromList.update(**kwargs)

    #---
    def check_file_extention(self,fname=None,file_extention=None):
        """
        
        :param fname         : <None>
        :param file_extention: string or list <None>
        :return:
         True/False
        """
        if not fname:
            return False
        if file_extention:
            if not isinstance(file_extention,(list)):
                file_extention = list(file_extention)
            for fext in file_extention:
                if fname.endswith(fext):
                    return True
    
        return False

    def file_list(self,**kwargs):
        """
        get file list
            from ids subfolder
            from files in a textfile
            from arguments filename
        
            
        like a contexmanager in a loop
        loop over files in filelist
        cd to file directory
        provide filename, working dir
        
        avoid copy/paste and boring repetitions
        place more error handling and file checking
        handels system ENVs
    
        https://stackoverflow.com/questions/29708445/how-do-i-make-a-contextmanager-with-a-loop-inside
        
        :param stage          : start dir,stage
        
        -> File form IDs
           :param subject_ids    : string or list of strings e.g: subject ids
           :param file_extention : string or list  <None>
                                   if <file_extention> checks filename extention ends with extentio from list
           :param recursive      : recursive searching for files in <stage/subject_id> using glob.iglob <False>
        
        -> Files fron List
           :param list_filename : filename of list file
           :param list_path     : path to list file
           !!! add <stage> to filename
         
        -> argumnets
           :param fname: filename
           :param fpath: path to filename
           
        :return:
        filename,subject id,working dir
    
         Example:
         --------
         for raw_fname,subject_id,workdir in file_list(subjects_ids=["123456","234567"],stage_dir="$MY_DATA_STAGE",
                                                       file_extention=["-raw.fif","-empty.fif"], list_filename="my_files.txt",list_path="."):
                                                       
             print("---> subject: {}\n  -> dir: {}\n  -> fname: {}".format(subject_id,workdir,raw_fname) )
             
        """
     
        self.update(**kwargs)
     
        self._pdfs=[]
        if self._PDFIDS.pdfs:
           self._pdfs.extend( self._PDFIDS.pdfs )
        if self._PDFList.pdfs:
            self._pdfs.extend(self._PDFList.pdfs)
        if self._PDFFile.pdfs:
           self._pdfs.append(self._PDFFile.pdf)
        
        self._pdfs = list( set(self._pdfs) ) # kick off double PDFs
        self._pdfs.sort()
        
        for fname in self._pdfs:
            try:
                dir = os.path.dirname(fname)
                with jb.working_directory(dir):
                     fname = os.path.basename(fname)
                     subj  = fname.split("_")[0]
                     yield fname,subj,dir
           
            except:
                logger.exception("---> error subject : {}\n".format(subj) +
                                 "  -> recordings dir: {}\n".format(dir) +
                                 "  -> file          : {}\n".format(fname))
            

#=========================================================================================
#==== MAIN
#=========================================================================================
def test1():
    logger.info("Start test1 find files fom subject_ids list")
   
    stage = "$JUMEG_PATH_LOCAL_DATA/exp/JUMEGTest/mne"
    subject_ids = "211890","211747"

    PDF = JuMEG_PDF_IDS()
    PDF.updatet(stage=stage,subject_ids=subject_ids,separator= ",",recursive=True,verbose=True,debug=True)
    
def test2():
    logger.info("Start test2 get full file")
    PDF = JuMEG_PDF_FILE()
    stage="$JUMEG_PATH_MNE_IMPORT2/MEG94T/mne"
    name = "test01.txt"
    path= stage+"/../doc"
    PDF.update(name=name,path=path,verbose=True,debug=True)
    logger.info("TEST 2: {}".format( PDF.pdfs ) )
 
    
def test3():
    logger.info("Start test3 get files fom list")
    PDF = JuMEG_PDF_FILES_FROM_LIST()
    stage="$JUMEG_PATH_MNE_IMPORT2/MEG94T/mne"
    list_name = "test01.txt"
    list_path= stage+"/../doc"
    
    PDF.update(stage=stage,list_name=list_name,list_path=list_path,verbose=True,debug=True)
   
def test4():
    logger.info("Start test4 PipelineLooper")
    PDF = JuMEG_PipelineLooper()
   #--- ids
    stage       = "$JUMEG_PATH_LOCAL_DATA/exp/JUMEGTest/mne"
    subject_ids = "211890","211747"
   #--- list
    stage = "$JUMEG_PATH_MNE_IMPORT2/MEG94T/mne"
    list_name   = "test01.txt"
    list_path   = "$JUMEG_PATH_MNE_IMPORT2/MEG94T/mne"+"/../doc"
   #--- file
    name = "test01.txt"
    path= stage+"/../doc"
    
    PDF.update(stage=stage,subject_ids=subject_ids,separator= ",",recursive=True,list_name=list_name,list_path=list_path,verbose=True,debug=True)
    #PDF.file_list(stage=stage,subject_ids=subject_ids,separator= ",",recursive=True,list_name=list_name,list_path=list_path,verbose=True,debug=True)
   
    for fname,subj  in PDF.file_list():
        print(" --> id: {} fname: {}".format(subj,fname))
   
   
if __name__ == "__main__":
  #--- init/update logger
   from jumeg.base import jumeg_logger
   logger=jumeg_logger.setup_script_logging(logger=logger)
   logger.setLevel(logging.DEBUG)
  
   
   #test1()
   #test2()
   #test3()
   test4()