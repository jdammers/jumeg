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


import os,sys,logging,yaml,argparse,glob
from jumeg.base            import jumeg_logger
from jumeg.base.jumeg_base import jumeg_base as jb

logger = logging.getLogger('jumeg')

__version__="2019.08.07.001"

class JuMEG_PDF_BASE(object):
   def __init__(self,**kwargs):
       self._stage  = "."
       self._pdfs   = None
       self._exit_on_error = False
       self.verbose = False
       self.debug   = False
       
       self._file_extention = ["meeg-raw.fif","rfDC-empty.fif"]

   @property
   def ExitOnError(self):
       return self._exit_on_error
   @ExitOnError.setter
   def ExitOnError(self,v):
       self._exit_on_error = v

   @property
   def file_extention(self): return self._file_extention
   @file_extention.setter
   def file_extention(self,v):
       if not v:
          self._file_extention = []
       elif not isinstance(v,(list)):
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
       self.ExitOnError    = kwargs.get("ExitOnError",self.ExitOnError)
       
   def get_fullpath(self,v):
       if v:
          return os.path.expandvars(os.path.expanduser(v))
       return None
   
   def clear(self):
       self._pdfs=[]
   
   def update(self,**kwargs):
       pass
       
   def info(self):
       logger.info("  -> PDFs:\n  -> " + "\n".join(self.pdfs) )
       if self.debug:
          logger.debug( jb.pp_list2str(self.__dict__) )
   
   def check_file_extention(self,fname=None,file_extention=None):
       """
       :param fname:          <None>
       :param file_extention: <None>
              if set overwrites the <file_extention> property
       :return:
              True/False
       """
       if not fname:
          return False
       if file_extention:
          self.file_extention = file_extention
    
       if self.file_extention:
          for fext in self.file_extention:
              if fname.endswith(fext):
                 return True
       return False
    
class JuMEG_PDF_IDS(JuMEG_PDF_BASE):
   '''
   cls to find files in stage/subject-id subfolder matching one file-extention in <file extention list>
   
   Parameter
   ---------
   subjects       : list of subject ids  ["1213456","234567"]
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
    subjects = "211890","211747"

    PDF = JuMEG_PDF_IDS()
    PDF.update(stage=stage,subjects=subjects,separator= ",",recursive=True,verbose=True,debug=True)
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
   def subjects(self): return self._ids
   @subjects.setter
   def subjects(self,v):
       self._ids = []
       if isinstance(v,(list)):
          self._ids = v
       elif isinstance(v,(str)):
          self._ids = v.split(",")
       elif v:
          self._ids = list(v)
          
   def _update_from_kwargs(self,**kwargs):
       super()._update_from_kwargs(**kwargs)
      #--- update subjects (ids)
       self.subjects  = kwargs.get("subjects",self.subjects)
       self.subjects  = kwargs.get("ids",self.subjects)
       
       self.recursive = kwargs.get("recursive",self.recursive)

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
      
      if self.debug:
         self.info()
         
      if self.recursive:
         fpatt = '**/*'
      else:
         fpatt = '*'
    
      self._pdfs = []
      
      for subj in self.subjects:
          try:
         #--- check if its a dir
             start_dir = os.path.join(self.stage,subj)
             if not os.path.isdir( start_dir ):
                continue
             # def find_files(start_dir=None,pattern=None,file_extention=None,debug=False)
             with jb.working_directory(start_dir):
                  for fext in self.file_extention:
                      if self.debug:
                         logging.debug( "  -> start dir     : {}\n".format(start_dir)+
                                        "  -> extention     : {}\n".format(fext)+
                                        "  -> glob recursive: {}\n".format(self.recursive) +
                                        "  -> glob pattern  : {}".format(fpatt)
                                       )
                           
                      for f in glob.glob(fpatt + fext,recursive=self.recursive):
                          self._pdfs.append( os.path.abspath( os.path.join(start_dir,f) ) )
          except:
              logger.exception("---> error subject : {}\n".format(subj) +
                               "  -> start dir     : {}\n".format(start_dir) )
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
       if not self.list_file_name:
          logger.warning("  -> list file name is not defined for reading PDFs from list, if needed set option <list_name> <list_path>")
          return None
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
           if not self.GetFullListFileName():
              return None
           
           if not os.path.isfile( self.GetFullListFileName() ):
              logger.exception("---> <list file> is not a file:\n  -> path: {}\n  -> file: {}".format(self.list_file_path,self.list_file_name))
              return None
            
           # if self.debug:
           #    logger.info("  -> list file: {}".format( self.GetFullListFileName() ) )
           files_to_exclude = []
           with open(self.GetFullListFileName(),'r') as f:
                for line in f:
                    line = line.strip()
                    fname = None
                    if line:
                       if (line[0] == '#'): continue
                       fname = line.split()[0]
                       try:
                           if not self.check_file_extention(fname):
                              msg= "\n---> error wrong file extention: skip file !!!\n  -> file extention list: {}\n".format(self.file_extention)
                              raise FileNotFoundError(msg)
                           if self.stage:
                              fname = os.path.abspath( os.path.join( self.stage+ "/" + fname ) )
                           if not os.path.isfile(fname):
                              raise FileNotFoundError("---> error file is not a real file: skip file !!!")
                              continue
                           
                           found_list.append(fname)
                           
                       except FileNotFoundError as e:
                           msg = "\n".join([ e.args[0],
                                            "  -> filename                  : {}".format(fname),
                                            "  -> file extention expected   : {}".format(self.file_extention),
                                            "  -> Exit On Error:            : {}\n".format(self.ExitOnError)])
    
                           if self.ExitOnError:
                              found_list.append(fname)
                              logger.exception(msg)
                           else:
                              logger.warning(msg)
                              
       except :
           msg=" --> error in file list; found files:\n  -> "+"\n  -> ".join(found_list)
           logger.error(msg)
           
           logger.exception(" --> error in reading list file:\n  -> path: {}\n  -> file: {}\n".format( self.list_file_path,self.list_file_name ) )
           if self.ExitOnError:
              sys.exit()
           return False
        
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
        if not self.name:
           return None
        if self.path:
           return os.path.abspath( os.path.join(self.path,self.name) )
        return self.name
    
    def _update_from_kwargs(self,**kwargs):
        self.path = kwargs.get("fpath",self.path)
        self.name = kwargs.get("fname",self.name)
    
    def update(self,**kwargs):
        self._update_from_kwargs(**kwargs)
        return self.GetFullFileName()

class JuMEG_PDF(object):
    def __init__(self):
        self._pdfs = []
        self._idx  = 0

    @property
    def idx(self): return self._idx
    @idx.setter
    def idx(self,v):
        if v < self.counts:
           self._idx = v
        else:
           self._idx = self.counts - 1
    
    @property
    def pdfs(self): return self._pdfs
    @pdfs.setter
    def pdfs(self,v):
        self._idx = 0
        if v:
           self._pdfs = list( set(v) ) # kick off double PDFs
           self._pdfs.sort()
        else:
           self._pdfs = []
        
    @property
    def current_number(self): return self._idx +1
    @property
    def counts(self): return len(self._pdfs)
    
    @property
    def file(self):
        if self.counts:
           return self._pdfs[self._idx]
        return None

    @property
    def name(self):
        if self.counts:
           return os.path.basename( self._pdfs[self._idx] )
        return None

    @property
    def dir(self):
        if self.counts:
            return os.path.dirname(self._pdfs[self._idx])
        return None

    @property
    def id(self):
        if self.counts:
           return self.name.split("_")[0]
        return None
   
class JuMEG_PipelineLooper(JuMEG_PDF_BASE):
    """
     get PDF
     from file args
     file list
     id list
     
    """
    def __init__(self,**kwargs):
        super().__init__()
        self._PDF     = JuMEG_PDF()
        self._PDFIDS  = JuMEG_PDF_IDS()
        self._PDFList = JuMEG_PDF_LIST()
        self._PDFFile = JuMEG_PDF_FILE()
     
     #--- logfile
        self._Hlog        = None
        self.log2file     = False
        self.logprefix    = "pipeline_looper"
        self.logoverwrite = False
        
    #--- config file
        self._config_file = None
        self._config      = None
        #self.overwrite    = True
        self._exit_on_error = False
        self.update(**kwargs)
        self.init( **kwargs )
   
    @property
    def ExitOnError(self):
        return self._exit_on_error
    @ExitOnError.setter
    def ExitOnError(self,v):
        self._exit_on_error = v
        self._PDFList.ExitOnError = v
        self._PDFIDS.ExitOnError  = v
        self._PDFFile.ExitOnError = v
        
    @property
    def Hlog(self): return self._Hlog
    
    @property
    def pdf(self): return self._PDF
    
    @property
    def config(self): return self._config_data
   
    @property
    def config_file(self):
        return self._config_file
    @config_file.setter
    def config_file(self,v):
        self._config_file = os.path.expandvars( os.path.expanduser(v))
    
    @property
    def subjects(self): return self._PDFIDS.subjects
    @subjects.setter
    def subjects(self,v):
        self._PDFIDS.subjects=v
    @property
    def recursive(self): return self._PDFIDS.recursive
    @recursive.setter
    def recursive(self,v):
        self._PDFIDS.recursive=v
    
    @property
    def stage(self): return self._PDFIDS.stage
    @stage.setter
    def stage(self,v):
        self._PDFIDS.stage  = v
        self._PDFList.stage = v

    @property
    def file_extention(self): return  self._PDFIDS.file_extention
    @file_extention.setter
    def file_extention(self,v):
        self._PDFIDS.file_extention  = v
        self._PDFList.file_extention = v
        
    def _update_from_kwargs(self,**kwargs):
        self._PDFIDS._update_from_kwargs(**kwargs)
        self._PDFList._update_from_kwargs(**kwargs)
        self._PDFFile._update_from_kwargs(**kwargs)
        self.ExitOnError = kwargs.get(exit_on_error,self.ExitOnError)
    
    def clear_all(self):
        self.clear_list()
        self.clear_file()
        
    def clear_list(self):
        self._PDFIDS.clear()
        self._PDFList.clear()
       
    def clear_file(self):
        self._PDFFile.clear() # needs to set path and name again
        
    def update(self,**kwargs):
        self.clear_list()
        self._PDFIDS.update(**kwargs)
        self._PDFList.update(**kwargs)
        self._PDFFile.update(**kwargs)
     
    def _update_pdf_list(self):
        try:
            self.pdf.pdfs = []
            if self._PDFIDS.pdfs:
               self.pdf.pdfs.extend(self._PDFIDS.pdfs)
            if self._PDFList.pdfs:
               self.pdf.pdfs.extend(self._PDFList.pdfs)
            if self._PDFFile.pdf:
               self.pdf.pdfs.append(self._PDFFile.pdf)
            self.pdf.pdfs.sort()
        except:
            raise Exception("\n" + "\n ---> error in update  PDFs list")
            return False
        return len( self.pdf.pdfs )
        
           
    def load_config(self,config=None):
        """
        :param config: <None>
        :return:
        """
        self._config_data = None
        if config:
           self.config_file = config
        if self.debug:
            logger.info("  -> loading config file: {} ...".format(self.config_file) )
        with open(self.config_file,'r') as f:
             self._config_data = yaml.full_load(f)
        if self.debug:
            logger.info("  -> DONE loading config file")

    def _update_config_values(self):
        """
        overwriting in congfig: stage
        :return:
        """
        for k in self.config.keys():
            self.config[k]["stage"] = self.stage
            
    def init(self,options=None,defaults=None):
        """
         init global parameter
          load config file parameter
          init <subject list>,<stage>,<fif_extention>  from opt or config or defaults

        :param options : arparser option obj
        :param defaults: default dict

        init:
         config_file   : cfg filename
         config        : dict
         subject       : list of subject ids
         stage         : start / base dir
         fif_extention : list of file extentions, files must match to get started
         recursive     : option to search for files in subfolder <True/False>
         verbose,debug
         
        :return:
        """
        
        def get_value(k,d):
            """
            :param k: key in dict
            :param list of dicts:
            :return: value of first match
            """
            v=None
            for obj in d:
                if isinstance(obj,(dict)):
                   if obj.get(k):
                      v= obj.get(k)
                      logger.info("---> TEST: {} => {}".format(k,v))
                      return v
            return v
    
        self.clear() # clear all pdf lists
        
        if not defaults:
           defaults = {}
        cfg_global = defaults # if no config file
        
        if options:
           opt = vars( options ) # get options as dict
        else:
           opt = {}

        vlist = [opt,defaults]
       #--- set flags
        self.verbose = get_value("verbose",vlist )
        self.debug   = get_value("debug",  vlist )
    
       #--- load cfg ToDo in CLS
        self.load_config( config=get_value("config",vlist))
      
       #--- get global parameter from cfg ToDo in CLS
        if self.config:
           cfg_global = self.config.get("global")
           vlist.append(cfg_global)
           
        logger.debug(" ---> config global parameter: {} ".format(cfg_global))
        #logger.debug(" ---> value list parameter: {} ".format(vlist))
   
       #--- logfile
        self.log2file      = get_value("log2file",    vlist)
        self.logprefix     = get_value("logprefix",   vlist)
        self.logoverwrite  = get_value("logoverwrite",vlist)
        
        self.recursive      = get_value("recursive",     vlist)
        self.subjects       = get_value("subjects",      vlist)
        self.stage          = get_value("stage",         vlist)
        self.file_extention = get_value("file_extention",vlist)
        #self.overwrite      = get_value("overwrite"    ,vlist)
        
        self._PDFList.list_file_path = get_value("list_path",vlist)
        self._PDFList.list_file_name = get_value("list_name",vlist)
        
        self._PDFFile.path  = get_value("fpath",    vlist)
        self._PDFFile.name  = get_value("fname",    vlist)
        
        self._update_config_values()
        
    def init_logfile(self,fname=None,mode="a"):
        if self.logoverwrite:
           mode="w"
        self._Hlog = jumeg_logger.update_filehandler(logger=logger,fname=fname,path="./log",prefix=self.logprefix,name=os.path.splitext(self.pdf.name)[0],mode=mode)
    
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
        
        Parameter
        ----------
        subject_ids,file_extention,stage,list_path,list_name,fname,fpath,recursive,verbose,debug
        
        :param stage          : start dir,stage
        
        -> File form IDs
           :param subject_ids    : string or list of strings e.g: subject ids
           :param file_extention : string or list  <None>
                                   if <file_extention> checks filename extention ends with extentio from list
           :param recursive      : recursive searching for files in <stage/subject_id> using glob.iglob <False>
        
        -> Files from List
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
       
        if not self._update_pdf_list():
           logger.info("\n---> No files in list, stop process")
           return
        
        logger.debug("---> PDF files to process: \n"+"\n".join(self.pdf.pdfs) )
        
        for self.pdf.idx in range( len( self.pdf.pdfs ) ):
            try:
                with jb.working_directory(self.pdf.dir):
                     msg = ["---> Start PreProc Ids: {}".format( self.pdf.counts ),
                            " --> subject id       : {} file number: {} / {}".format(self.pdf.id,self.pdf.current_number,self.pdf.counts),
                            "  -> raw file name    : {}".format(self.pdf.name),
                            "  -> stage            : {}".format(self.stage),
                            "  -> working dir      : {}".format(os.getcwd())]
                     
                     if not self.check_file_extention(self.pdf.name):
                        msg.append("  -> Exit On Error:   : {}".format( self.ExitOnError) )
                        msg.append("  -> error wrong file extention should be: {}".format( self.file_extention) )
                        logger.error("\n".join(msg))
                      
                        if self.ExitOnError:
                          raise Exception("\n"+"\n".join(msg))
                        
                     if self.log2file:
                        if self.logoverwrite:
                           self.init_logfile( fname=os.path.splitext( self.pdf.name)[0] +".log" )
                        else:
                           self.init_logfile()
                           
                        msg.append("  -> writing log to   : {}".format(self.Hlog.filename))
                     
                     logger.info( "\n".join(msg) )
                     
                     try:
                         yield self.pdf.name,self.pdf.id,self.pdf.dir
                     except:
                         logger.exception("---> error subject : {}\n".format(self.pdf.id) +
                                          "  -> recordings dir: {}\n".format(self.pdf.dir) +
                                          "  -> file          : {}\n".format(self.pdf.name))
                         return False
                    
                #--- do your stuff here
                
                if self._Hlog:
                   self._Hlog.close()
                   self._Hlog = None
            except:
                logger.exception("---> error subject : {}\n".format(self.pdf.id) )
                if self.ExitOnError:
                    logger.error("\nExit On Error")
                    break
                    

#=========================================================================================
#==== MAIN
#=========================================================================================
def test1():
    logger.info("Start test1 find files fom subject_ids list")
   
    stage = "$JUMEG_PATH_LOCAL_DATA/exp/JUMEGTest/mne"
    subject_ids = "211890","211747"

    PDF = JuMEG_PDF_IDS()
    PDF.update(stage=stage,subject_ids=subject_ids,separator= ",",recursive=True,verbose=True,debug=True)
    
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
    defaults={
          "stage"          : "$JUMEG_PATH_LOCAL_DATA/exp/JUMEGTest/mne",
          "fif_extention"  : ["meeg-raw.fif","rfDC-empty.fif"],
          "list_name"      : None,
          "list_path"      : None,
          "fname"          : None,
          "fpath"          : None,
          "config"         : "$JUMEG_PATH_JUMEG/../pipelines/config_file.yaml",
          "subjects"       : "211890,211747",
          "log2file"       : True,
          "logprefix"      : "preproc0",
          "overwrite"      : False,
          "verbose"        : False,
          "debug"          : False,
          "recursive"      : True
         }

    opt = None
    jpl = JuMEG_PipelineLooper(defaults=defaults)
    
    print("---> TEST4: config file")
    print(jpl.config_file)
   
    print("---> TEST4: config")
    print(jpl.config)
    print("\n\n")
    for raw_fname,subject_id,dir in jpl.file_list():
        print("  -> Test4 in loop PDF: {}".format(raw_fname))
    
   
if __name__ == "__main__":
  #--- init/update logger
   from jumeg.base import jumeg_logger
   logger=jumeg_logger.setup_script_logging(logger=logger)
   logger.setLevel(logging.DEBUG)
  
   
   #test1()
   #test2()
   #test3()
   test4()