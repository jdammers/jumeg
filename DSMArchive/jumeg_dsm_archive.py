#!/usr/bin/env python3
# -+-coding: utf-8 -+-

#--------------------------------------------
# JuMEG Archive Tool
# to archive original 4D/BTI and EEG data to IBM Tivoli Archive System DSM
# DSM option are passed via configfile
#--------------------------------------------
# Authors: Frank Boers <f.boers@fz-juelich.de>
#
#--------------------------------------------
# Date: 04.07.19
#--------------------------------------------
# License: BSD (3-clause)
#--------------------------------------------
# Updates  28.10.2019 => yaml.full_load(f)
#--------------------------------------------
"""
Example
----------
# get help
jumeg_dsm_archive.py -h

jumeg_dsm_archive.py --ids=123456,234561 --scans=M100,V1 --experiments=Audio,VIS -meg -eeg -r -v -d -arc

Configfile
------------
store DSM options in yaml format e.g.: <jumeg_dsm_archive.yaml>

info:
    description: "JuMEG DSM archive config file"

    time: "2019-07-04 00:00:00"

    user: "meg"

    version: "2019-07-04-0.001"

global:
     host: "$JUMEG_DSM_ARCHIVE_HOST"

     virtualnodename: "XYZ"

     server: "archive"

     password: "XYZ"

     id: "XYZ"

MEG:
    stages: ["$JUMEG_PATH_BTI_OD_ARCHIVE"]

    file_extentions: ["c,rfDC","hs_file","config","e,rfhp1.0Hz,COH","e,rfhp1.0Hz,COH1"]
 
    description: "Magnes3600"
    
    recursive: True

EEG:
    stages: ["$JUMEG_PATH_MNE_IMPORT1","$JUMEG_PATH_MNE_IMPORT2"]

    file_extentions: [".vhdr",".vmrk",".eeg"]
 
    description: "EEG_BrainVision"
    
    recursive: True
"""

import os,sys,logging,yaml,argparse,glob,subprocess
from subprocess import STDOUT,PIPE,Popen

from jumeg.base import jumeg_logger
from jumeg.base.jumeg_base import jumeg_base as jb
from jumeg.base.ioutils.jumeg_ioutils_find import JuMEG_IOutils_FindIds,JuMEG_IoUtils_FileIO
from jumeg.base.pipelines.jumeg_pipelines_utils_base import parser_update_flags

logger = logging.getLogger("jumeg")

__version__= "2019.10.30.001"


class JuMEG_DSMConfig(object):
    """
    CLS for DSM config file obj
    
    Example:
    --------
    self.CFG = JuMEG_DSMConfig(**kwargs)
    self.CFG.update(**kwargs)
   """
    __slots__ =["verbose","debug","data_type","_filename","_data"]
    
    def __init__(self,**kwargs):
        self._init(**kwargs)

    @property
    def host(self):
        if os.path.expandvars( self.Global.get("host") ):
           return os.path.expandvars( self.Global.get("host") )
        else:
           return "local"
        
    @property
    def Global(self): return self._data.get("global")
    
    @property
    def MEG(self): return self._data.get("MEG")
    
    @property
    def EEG(self): return self._data.get("EEG")

    @property
    def filename(self):
        return self._filename

    @filename.setter
    def filename(self,v):
        if v:
            self._filename = os.path.expandvars(os.path.expanduser(v))

    @property
    def data(self):
        return self._data.get(self.data_type)

    @property
    def stages(self):
        l=[]
        l.extend( self.data.get("stages"))
        return l

    @property
    def file_extentions(self):
        l = []
        l.extend(self.data.get("file_extentions"))
        return l
    
    @property
    def description(self):
        return self.data.get("description")

    @property
    def recursive(self):return self.data.get("reccursive")
    
    def _init(self,**kwargs):
        for k in self.__slots__:
            self.__setattr__(k,None)

        self._filename = "jumeg_dsm_archive.py"
        self._data = dict()
    
        self._update_from_kwargs(**kwargs)

    def update(self,**kwargs):
        
        self._update_from_kwargs(**kwargs)
       #--- load cfg ToDo in CLS
        self.load_config()
        
    def _update_from_kwargs(self,**kwargs):
        for k in kwargs:
            if k in self.__slots__:
               self.__setattr__(k,kwargs.get(k,self.__getattribute__(k)))
        self.filename = kwargs.get("config",self._filename)
        
    def load_config(self,config=None):
        """
        :param config: <None>
        :return:
        """
        #self._config_data = None
        if config:
           self.filename = config
        if self.debug:
           logger.info("  -> loading config file: {} ...".format(self.filename) )
     
       #--- ck cfg file or use default
        if not os.path.isfile(self.filename):
           fname = os.path.join( os.path.dirname(__file__) , os.path.basename(self.filename))
           if os.path.isfile(fname):
              logger.warning("---> Config file not found: {} using default config file {}".format(self.filename,fname) )
              self.filename = fname
       
        if self.debug:
           logger.debug("  -> Start loading config file: {}".format(self.filename))

        with open(self.filename,'r') as f:
             self._data = yaml.full_load(f)
        
        if self.debug:
           logger.debug("  -> config:\n {}".format(self._data))
           logger.debug("  -> DONE loading config file")

    def GetParameter(self,data_type,key=None):
        """
        :param self:
        :param data_type:
        :param key:
        :return:
        """
        if not data_type: return None
        if key:
           return self._data.get(data_type).get(key)
        return self._data.get(data_type)
        
    def GetDSMOptions(self):
        """
         DSM option in call:
         -se=archive -virtualnodename=xyz -password=xyz
         -> in config:
            virtualnodename: "xyz"
            server         : "xyz"
            password       : "xyz"
            id             : "xyz"
         """
        return ["-se="             + os.path.expandvars(self.Global.get("server")),
                "-virtualnodename="+ os.path.expandvars(self.Global.get("virtualnodename")),
                "-password="       + os.path.expandvars(self.Global.get("password")) ]

class JuMEG_DSMArchive(object):
    """
    CLS to archive MEG data in 4D/BTi format
    
    Example:
    --------
     jumeg_logger.setup_script_logging(name=name,opt=opt,logger=logger,level=level)

    if opt.meg:
       JDS_MEG = JuMEG_DSMArchive(config=opt.config,verbose=opt.verbose,debug=opt.debug,archive=opt.archive,
                                   overwrite=opt.overwrite)
       JDS_MEG.archive(ids=opt.ids,scans=opt.scans,data_type="MEG")
    """
    __slots__ =["run","verbose","debug","data_type","_pdfs","_id_list","_path_od","_FIDs","_FPDFs","_CFG","_ids","_scans",
                "do_archive","do_overwrite","_ext","_remove_tmpfile","_arc_info"]
   
    def __init__(self,**kwargs):
        
        self._init(**kwargs)
        
        self._CFG   = JuMEG_DSMConfig(**kwargs)
        self.Config.update(**kwargs)
       
        self._remove_tmpfile = False#True
        
        self._FIDs  = JuMEG_IOutils_FindIds()
        self._FPDFs = JuMEG_IoUtils_FileIO()
       #--- file extention for DSM calls
        self._ext   = {"on_arc":"files_on_archive.txt","on_hd":"files_on_harddisk.txt","to_arc":"files_to_archive.txt"}
        

    @property
    def FindPDFs(self): return self._FPDFs

    @property
    def FindIDs(self):  return self._FIDs
 
    @property
    def Config(self):  return self._CFG

    @property # IDS from argparser
    def IDs(self): return  self._ids
    @IDs.setter
    def IDs(self,v):
        self._ids=[]
        if v:
           self._ids.extend( v.split(",") )

    @property
    def PDFs(self): return  self._pdfs
    @PDFs.setter
    def PDFs(self,v):
        self._pdfs=[]
        self._pdfs.extend(v)

    @property
    def scans(self):
        return self._scans

    @scans.setter
    def scans(self,v):
        self._scans = []
        if v:
           self._scans.extend(v.split(","))
        
    def clear(self):
        self._pdfs =[]
        self._ids  =[]
    
    def GetDescription(self,v):
        return self.Config.description +"_"+ v
    
    def GetFilenameOnArchive(self,stage,id):
        """
        
        :param stage:
        :param id:
        :return:
         full filename
        """
        return os.path.join( stage, id +"_"+ self._ext.get("on_arc"))

    def GetFilenameOnHardDisk(self,stage,id):
        """
        
        :param stage:
        :param id:
        :return:
         full filename
        """
        return os.path.join(stage,id + "_" + self._ext.get("on_hd"))

    def GetFilenameToArchive(self,stage,id):
        """
        
        :param stage:
        :param id:
        :return:
         full filename
        """
        return os.path.join(stage,id + "_" + self._ext.get("to_arc"))
    
    def GetFlistName(self,id=None,stage=None):
        fname = ""
        if stage:
           fname += stage
        else:
           fname += "."
        fname += "/jumeg_dsm_archive"
        if id:
           fname += "_"+ id
        return fname +"_"+ self.data_type + ".txt"
        
    def _init(self,**kwargs):
        """ init slots """
        for k in self.__slots__:
            self.__setattr__(k,None)
        
        self.clear()  # clear all pdf lists
        self._update_from_kwargs(**kwargs)
        
    def _update_from_kwargs(self,**kwargs):
        for k in kwargs:
            if k in self.__slots__:
                    self.__setattr__(k,kwargs.get(k,self.__getattribute__(k)))
        if kwargs.get("scans"):
           self.scans = kwargs.get("scans")
        if kwargs.get("ids"):
           self.IDs = kwargs.get("ids")
        
        self.do_archive   = kwargs.get("archive",self.do_archive)
        self.do_overwrite = kwargs.get("overwrite",self.do_overwrite)
   
    def write_listfile(self,fname,flist,stage=None):
        with open(fname, mode='wt', encoding='utf-8') as f:
             if stage:
                for pdf in flist:
                    f.write( os.path.join(stage,pdf)+'\n' )
             else:
                f.write('\n'.join(flist))
             f.write('\n')
        if self.debug:
           logger.debug("---> write list file: {}".format(fname))
        return fname, flist
            
    def update_ids(self,stage):
        """
        
        :param stage:
        :return:
        """
        self._id_list.extend(self._FIDs.find(stage=stage))

    def isLocalHost(self,host):
        if ( host.lower() == "local" ) or ( host.lower() == os.uname()[1].lower() ) :
           return True
        return False
    
    def update_pdfs(self,stage,id=None):
        """
        
        :param stage:
        :return:
        """
        id_list = []
        if id:
           if isinstance(id,(list)):
              id_list = id
           else:
              id_list.append(id)
        else:
            id_list = self._id_list
        
        for id in id_list:
            if self.scans:
               pdfs = []
               for scan in self.scans:
                   pat = id + "/" + scan + "/**/"
                   pdfs.extend(self._FPDFs.find_files(start_dir=stage,pattern=pat,file_extention=self.Config.file_extentions,
                                                      recursive=True,debug=self.debug,abspath=False))
               return pdfs
         
            else:
                pat = id + "/**/"
                return(self._FPDFs.find_files(start_dir=stage,pattern=pat,file_extention=self.Config.file_extentions,
                                              recursive=True,debug=self.debug,abspath=False))

   
    def GetDSMArchiveInfo(self,id,stage):
        """
        get file info from dsm archive
        dsm call:
         dsmc q arch -filelist=<full file list path> -description=<INFO_ID> -se=archive -virtualnodename=<XYZ> -password=<xyz>
       
        :param id:
        :param stage:
        :return:
        """
        files2arch = []
        
        cmd = []

       #-- check if files on HD are archived
        fname,_ = self.write_listfile(self.GetFilenameOnHardDisk(stage,id),self._pdfs,stage = stage)

       #--- setup dsm cmd
        if not self.isLocalHost( self.Config.host ):
           cmd.extend(["ssh","-x",self.Config.host])
        cmd.extend(["dsmc","q","arch","-filelist=" + fname,"-description="+self.GetDescription(id)])
        cmd.extend( self.Config.GetDSMOptions() )
        
       #--- call dsm
        logger.info("  -> DSM call: {}".format( " ".join(cmd) ))
        
        dsm = subprocess.run(cmd,shell=False,stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=False,universal_newlines=True)
        
        try:
            result = dsm.stdout #.readlines()
            logger.debug("DSM output\n {}".format(result))
            
        except:
            result = dsm.stderr#.readlines()
            logger.exception("DSM STDERR files on archive\n {}".format(result))
            result = None
            
        logger.info("  -> DSM call done")
       
        if self._remove_tmpfile:
           self._FPDFs.remove_file(fname)
        
        return result.replace('\n\n', '\n') # without empty lines
       
    def GetPDFsToArchive(self,stage,id):
        """
        :param stage: startpath to data structure on filesystem
        :param id   : subject id
                      BTi/4D data structure: /id/scan/session/run/
                      with files: "c,rfDC","hs_file","config","e,rfhp1.0Hz,COH","e,rfhp1.0Hz,COH1"
         
        return pdf list in file to call for archive
        
        dsm call:
         dsmc q arch -filelist=<full file list path> -description=<INFO_ID> -se=archive -virtualnodename=<XYZ> -password=<xyz>
       
        Example:
        --------
        DSM info output for not archived files:
         ANS1345E Keine Objekte auf dem Server stimmen mit '/data/ ... /c,rfDC' überein
         ANS1345E Keine Objekte auf dem Server stimmen mit '/data/ .../config' überein
        
        DSM info output for archived files:
         202.784.104  B  06.03.2019 08:56:33    /data/ ... /c,rfDC Nie <INFO_ID>
        :return:
        list with pdfs to archived
        """
        
        pdfs_to_arc = []
        
        if self.do_overwrite:
           return self.write_listfile( self.GetFilenameToArchive(stage,id),self._pdfs,stage=stage)
     
        logger.info("  -> check and skipp archived PDFs: {}".format(id) )
        
        dsm_info = self.GetDSMArchiveInfo(id,stage)
           
        logger.debug("  -> DSM Archive Info: \n {}".format(dsm_info))
           
        if not dsm_info: return pdfs_to_arc
        
        lines = dsm_info.split("\n")
        for lidx in range(len(lines)):
            if lines[lidx].find( self._pdfs[0] ):
               break
            
        for pdf in self._pdfs:
            if lidx >= len(lines): break
            # logger.debug("  ---> start pdf: {} ".format(pdf ))
            for idx in range(lidx, len( lines) ):
              # logger.dbug("  ---> search pdf: {} -> {} -> {}".format(idx,lines[idx],pdf ))
                if lines[idx].find(pdf)> 0:
                   if len( lines[idx].strip().split(" ") ) < 12: # not archived
                      pdfs_to_arc.append(pdf)
                    # logger.debug("  ---> match: {} -> {} -> {} ".format(idx,lines[idx], pdf))
                   lidx = idx
                   break
            lidx += 1
        
        if self.verbose:
           logger.info( "  -> Files to archive: {} from PDFs: {} stage: {}  id: {}".format( len(pdfs_to_arc),len(self._pdfs),stage,id))
        if self.debug:
           logger.debug("  -> PDF list:\n  -> {}".format("\n  -> ".join(pdfs_to_arc) ))
        
       # if self.verbose or self.debug:
       #    self._update_archive_info(pdfs_to_arc)
           
        return pdfs_to_arc

    def _archive(self,stage,id):
        """
        archive pdf to dsm
        dsm call:
         dsmc archive -filelist=<full file list path> -description=<INFO_ID> -se=archive -virtualnodename=<XYZ> -password=<xyz>
       
        :param id:
        :param stage:
        :return:
        """
        
        if not self._pdfs:
           logger.error("  -> skipping DSM archiving:  no PDFs in list to archive")
           return
        
        cmd = []
      #-- check if files on HD are archived
        fname,_ = self.write_listfile( self.GetFilenameToArchive(stage,id),self._pdfs,stage=stage )
        
       #--- setup dsm cmd
        if not self.isLocalHost( self.Config.host ):
           cmd.extend(["ssh","-x",self.Config.host])
        cmd.extend(["dsmc","archive","-filelist=" + fname,"-description="+self.GetDescription(id) ])
        cmd.extend( self.Config.GetDSMOptions() )
       
       #--- call dsm
        logger.info("  -> DSM archive call: {}\n  -> {}".format( self.do_archive," ".join(cmd) ))
        
        if self.do_archive:
           cmd_str = " ".join(cmd)
           logger.info("  -> START DSM archive call: {}".format( cmd_str ) )
           try:
              '''http://queirozf.com/entries/python-3-subprocess-examples'''
              dsm = subprocess.run(cmd,shell=False,stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=False,universal_newlines=True)
              logger.debug("DSM archive  STDOUT output\n {}".format(dsm.stdout))
              logger.debug("DSM archive  STDERR output\n {}".format(dsm.stderr))
           except:
              logger.exception("DSM archive STDERR files on archive\n {}".format(dsm.stderr))
         
           logger.info("  -> DONE DSM archive id: {}".format(id))
        
        if self._remove_tmpfile:
           self._FPDFs.remove_file(fname)
       
    def archive(self,**kwargs):
        """
        archive BTI data
         id/scan/session/run
         c,rfDc
         config
         hs_file
        
        :param kwargs:
        :return:
        """
        self.clear()
        
        self._update_from_kwargs(**kwargs)
        self.Config.update(**kwargs)
        # self._arc_info = dict()
        
       #--- for stages
        for st in self.Config.stages:
            self._id_list = []
            self.PDFs     = []
    
            stage = self.FindPDFs.expandvars(st)
            logger.info("---> DSM Archive START on stage: {}".format(stage) )
    
            if not os.path.isdir(stage):
               logger.error( "---> DSM Archive => {} ERROR no such directory: {}".format(self.data_type,stage) )
               continue
           
           #--- ck use ids from kwargs or search for ids in stage
            if self.IDs:
               self._id_list.extend( self.IDs )
            else:
                self.update_ids(stage)
            
            if self.verbose:
               logger.info(" --> Stage: {}\n  -> Found IDs: {}".format(stage,len(self._id_list)))
            if self.debug:
               logger.debug(" --> {}".format(self._id_list))
           #--- for ids
            for id in self._id_list:
                self._pdfs = self.update_pdfs(stage,id) #  for scans
          
                if self.verbose:
                   logger.info(" --> Stage: {}\n  -> ID: {}\n  -> Found PDFs: {}".format(stage,id,len(self._pdfs) ))
                   
                if not self._pdfs: continue
                
                if self.debug:
                   logger.debug("\n  -> {}".join( self._pdfs ))
                
                self._pdfs = self.GetPDFsToArchive(stage,id)
                
                if self._pdfs:
                   self._archive(stage,id)


class JuMEG_DSMArchiveEEG(JuMEG_DSMArchive):
    """
    CLS to archive EEG data in BrainVision format
    
    Example:
    --------
     jumeg_logger.setup_script_logging(name=name,opt=opt,logger=logger,level=level)

     if opt.eeg:
       JDS_EEG = JuMEG_DSMArchiveEEG(data_type="EEG",ids=opt.ids,scans=opt.scans,experiments=opt.experiments,\
                                     config=opt.config,verbose=opt.verbose,debug=opt.debug)
       JDS_EEG.archive(ids=opt.ids,scans=opt.scans,data_type="EEG")
     
    """
    __slots__ =["run","_experiments","verbose","debug","data_type","_pdfs","_id_list","_path_od","_FIDs","_FPDFs","_CFG","_ids","_scans",
                "do_archive","do_overwrite","_ext","_remove_tmpfile"]
    
    def __init__(self,**kwargs):
        super().__init__(**kwargs)

    def _update_from_kwargs(self,**kwargs):
       super()._update_from_kwargs(**kwargs)
       self.experiments = kwargs.get("experiments",self._experiments)

    @property
    def experiments(self): return self._experiments
    @experiments.setter
    def experiments(self,v):
        self._experiments=[]
        if not v: return
        if isinstance(v,(list)):
            self._experiments.extend( v )
        else:
           self._experiments.extend( v.split(",") )
        self._experiments.sort()
    
    def _find_eeg_experiments(self,stage):
        """
        
        :param stage:
        :return:
        list of expeperiments with <eeg> directory
        [M100,V1]
         e.g.: <stage>/M1000/eeg
         
        """
        exps = []
        with self._FPDFs.working_directory(stage):
            #--- py 3.6
            # with os.scandir(path=".") as dirs:
            #      for d in dirs:
            #          if d.is_dir():
            #             if os.path.isdir( os.path.join(d.name,"eeg") ):
            #                exps.append(d.name)
             for d in os.listdir(path="."):
                 if os.path.isdir(d):
                    if os.path.isdir( os.path.join(d,"eeg") ):
                       exps.append(d)
             exps.sort()
            
        logger.debug("  -> Experiments with {} data: stage: {}\n  {}".format(self.data_type,stage,exps) )
        return exps
        
    def _update_pdfs(self,stage,id=None):
    
        pdfs = []
        if id:
           prefix = "**/" + id + "*"
           
        else:
           prefix = "**/*"

        if self.scans:
           for scan in self.scans:
               pat = prefix + scan +"**"
               pdfs.extend(
                    self._FPDFs.find_files(start_dir=stage,pattern=pat,file_extention=self.Config.file_extentions,\
                                           recursive=True,debug=self.debug,abspath=False))
           return pdfs

        else:
           pat = prefix
           return (self._FPDFs.find_files(start_dir=stage,pattern=pat,file_extention=self.Config.file_extentions,\
                                          recursive=True,debug=self.debug,abspath=False))
            
    def update_pdfs(self,stage):
        """
        
        :param stage:
        :param id:
        :return:
        """
        pdfs    = []
        
        if self._id_list:
           for id in self._id_list:
               pdfs.extend( self._update_pdfs(stage,id) )
           return pdfs
        else: # get all
          return  self._update_pdfs(stage)
    
    def _archive_eeg(self,stage,experiment):
        """
       
        :param self:
        :param stage:
        :param experiment
        :return:
        """
        self._id_list = []
        self.PDFs     = []

        logger.info("---> DSM Archive START {} on stage: {}".format(self.data_type,stage))
       
        if not os.path.isdir(stage):
           logger.error("---> DSM Archive => {} ERROR no such directory: {}".format(self.data_type,stage))
           return None
          
       #--- ck use ids from kwargs or search for ids in stage
        if self.IDs:
           self._id_list.extend(self.IDs)
           if self.verbose:
              logger.info(" --> Stage: {}\n  -> Found IDs: {}".format(stage,len(self._id_list) ))
           if self.debug:
              logger.debug("  -> {}".format(self._id_list))
         
       #--- for ids or all eeg data in eeg-dir
        self._pdfs = self.update_pdfs(stage)  #  for scans
        
        if self.verbose:
           logger.info(" --> Stage: {}\n  -> experimnet: {}\n  -> Found PDFs: {}".format(stage,experiment,len(self._pdfs)))
           logger.debug( "PDFs list: \n  -> {}".format( "\n  -> ".join(self._pdfs)))
        
        if self._pdfs:
           self._pdfs = self.GetPDFsToArchive(stage,experiment)
           
        if self._pdfs:
           self._archive(stage,experiment)

    def archive(self,**kwargs):
        """
        archive eeg data
         stage/experiment/eeg/ [scan] / id_xyz
          *.vhdr,*.vmrk,*.data
         
        :param kwargs:
        :return:
        """
        self.clear()
        self._update_from_kwargs(**kwargs)
        self.Config.update(**kwargs)
        experiments=[]
        experiments.extend( self.experiments )
        
       #--- for stages
        for st in self.Config.stages:
            stage = self.FindPDFs.expandvars(st)
            if not experiments:
               experiments = self._find_eeg_experiments(stage)
          #--- for experiments
            for exp in experiments:
                eeg_stage = os.path.join(stage,exp,"eeg")
                if os.path.isdir(eeg_stage):
                   self._archive_eeg( eeg_stage,exp )
      


#=======================================================================================================================
#=== apply
#=======================================================================================================================
def apply(name=None,opt=None,defaults=None,logprefix="preproc"):
   #--- init/update logger
    if opt.debug:
        level="DEBUG"
    else:
        level="INFO"
        
    jumeg_logger.setup_script_logging(name=name,opt=opt,logger=logger,level=level)

    if opt.meg:
       JDS_MEG = JuMEG_DSMArchive(config=opt.config,verbose=opt.verbose,debug=opt.debug,archive=opt.archive,
                                   overwrite=opt.overwrite)
       JDS_MEG.archive(ids=opt.ids,scans=opt.scans,data_type="MEG")
    
    if opt.eeg:
       JDS_EEG = JuMEG_DSMArchiveEEG(data_type="EEG",ids=opt.ids,scans=opt.scans,experiments=opt.experiments,
                                     archive=opt.archive,config=opt.config,verbose=opt.verbose,debug=opt.debug)
       JDS_EEG.archive(ids=opt.ids,scans=opt.scans,data_type="EEG")
      
#=======================================================================================================================
#=== get_args
#=======================================================================================================================
def get_args(argv,parser=None,defaults=None,version=None):
    """
    get args using argparse.ArgumentParser ArgumentParser
    e.g: argparse  https://docs.python.org/3/library/argparse.html

    :param argv:   the arguments, parameter e.g.: sys.argv
    :param parser: argparser obj, the base/default obj like --verbose. --debug
    :param version: adds version to description
    :return:

    Results:
    --------
     parser.parse_args(), parser
    """
    
    description = """
                  JuMEG DSM Archive Script
                   script version : {}
                   python version : {}
                  """.format(version,sys.version.replace("\n"," "))

    h_experiment = "experiment name to archive eeg data experiment or experiments e.g.: M100 or M100,V1"
    h_subjects   = "subject id or ids  e.g.: 123 or 234,456"
    h_scans      = "scans e.g.: M100,CAU"
    h_config     = "script config file, full filename"
    h_verbose    = "bool, str, int, or None"
    
    #--- parser
    if not parser:
        parser = argparse.ArgumentParser(description=description)
    else:
        parser.description = description
    
    if not defaults:
        defaults = { }
    
    #---  parameter settings  if opt  elif config else use defaults
    parser.add_argument("-e","--experiments",help=h_experiment)
    parser.add_argument("-i","--ids",help=h_subjects)
    parser.add_argument("-sc","--scans",help=h_scans)
    #---
    parser.add_argument("-c","--config",help=h_config,default="jumeg_dsm_archive.yaml")
    #--- flags
    parser.add_argument("-v","--verbose",action="store_true",help=h_verbose)
    parser.add_argument("-d","--debug",action="store_true",help="debug mode")
    parser.add_argument("-t","--test",action="store_true",help="test developer mode")

    parser.add_argument("-meg","--meg",action="store_true",help="archive meg data")
    parser.add_argument("-eeg","--eeg",action="store_true",help="archive eeg data")

    parser.add_argument("-r","--run",action="store_true",help="!!! EXECUTE & RUN this program !!!")
    parser.add_argument("-ov","--overwrite",action="store_true",help="overwrite/archive existing file on DSM")
    parser.add_argument("-arc","--archive",action="store_true",help="apply archiving")
    
    parser.add_argument("-log","--log2file",action="store_true",help="generate logfile")
    parser.add_argument("-logoverwrite","--logoverwrite",action="store_true",help="overwrite existing logfile")
    parser.add_argument("-logprefix","--logprefix",help="logfile prefix",default="dsmarchde")
    
    return parser_update_flags(argv=argv,parser=parser)

#=======================================================================================================================
#==== MAIN
#=======================================================================================================================
def main(argv):
   
    opt, parser = get_args(argv,version=__version__)
    if len(argv) < 2:
       parser.print_help()
       sys.exit(-1)
      
    if opt.test:
       os.environ["JUMEG_PATH_BTI_EXPORT"] = jb.expandvars("$JUMEG_PATH_LOCAL_DATA")+"/megdaw_data21"

    #if opt.archive:
    #   opt.run = True

    if opt.run: apply(name=argv[0],opt=opt)
    
if __name__ == "__main__":
   main(sys.argv)

