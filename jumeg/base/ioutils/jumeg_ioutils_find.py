#!/usr/bin/env python3
# -+-coding: utf-8 -+-

"""
#--------------------------------------------
# Authors: Frank Boers <f.boers@fz-juelich.de> 
#
#-------------------------------------------- 
# Date: 02.07.19
#-------------------------------------------- 
# License: BSD (3-clause)
#--------------------------------------------
# Updates
#--------------------------------------------
"""

import os,sys,glob,contextlib,re
import logging

# from pubsub  import pub

from jumeg.base.jumeg_base import JuMEG_Base_Basic
jb = JuMEG_Base_Basic()

logger = logging.getLogger('jumeg')

__version__="2019.07.12.001"

class JuMEG_IoUtils_FileIO(object):
    def __init__ (self):
        self.debug  = False
        self.verbose= False
        
#--- cp from jumeg base
    def expandvars(self,v):
        """
        expand env's from string works on list or string
         => expandvars and expanduser
        :param v: list of strings  or string
        :return: input with expanded env's
        """
        if not v: return None
        if isinstance(v,(list)):
            for i in range(len(v)):
                v[i] = os.path.expandvars(os.path.expanduser(str(v[i])))
            return v
        
        return os.path.expandvars(os.path.expanduser(str(v)))
       
    def isFile(self,fin,path=None,extention=None,head="check file exist",exit_on_error=False,logmsg=False):
        """
        check if file exist
    
        Parameters
        ----------
        :param <string>     : full filename to check
        :param extention    : file extention  <None>
        :param head         : log msg/error title
        :param logmsg       : log msg <False>
        :param exit_on_error: will exit pgr if not file exist <False>
    
        :return:
        full filename / False or call <exit>
    
        Result
        ------
        abs full file name/False
        """
        
        fck = ""
        try:
            if path.strip(): fck = path + "/"
        except:
            pass
        
        if fin:
            fck += fin
        if extention:
            fck += extention
        f = os.path.abspath(self.expandvars(fck))
        # f = os.path.abspath(os.path.expandvars(os.path.expanduser(fin)))
        if os.path.isfile(f):
            if logmsg:
               logger.info("---> " + head + "\n --> file exist: {}\n  -> abs file{:>18} {}".format(fck,':',f))
            return f
        #--- error no such file
        logger.error("---> " + head + "\n --> no such file or directory: {}\n  -> abs file{:>18} {}".format(fck,':',f))
        if exit_on_error:
            raise SystemError("File not exists: {}".format(f) )
        return False
        
    def isDir(self,v):
        """
        expand env's e.g. expand user from string
        check if is dir
    
        :param v: string
        :return: input with expanded env's or None
        """
        v = self.expandvars(v)
        if os.path.isdir(v): return v
        return False
        
    def isPath(self,pin,head="check path exist",exit_on_error=False,logmsg=False):
        """
        check if file exist
    
        Parameters
        ----------
        :param <string>     : full path to check
        :param head         : log msg/error title
        :param logmsg       : log msg <False>
        :param exit_on_error: will exit pgr if not file exist <False>
    
        :return:
        full path / False or call <exit>
    
        """
        p = os.path.abspath(self.expandvars(pin))
        if os.path.isdir(p):
            if logmsg:
                logger.info("--->" + head + "\n --> dir exist: {}\n  -> abs dir{:>18} {}".format(pin,':',p))
            return p
        #--- error no such file
        logger.error("---> " + head + "\n --> no such directory: {}\n  -> abs dir{:>18} {}".format(pin,':',p))
        if exit_on_error:
            raise SystemError(self.__MSG_CODE_PATH_NOT_EXIST)
        return False
    
    
    def remove_file(self,f):
        """
        
        :param f:
        :return:
        """
        f = self.expandvars(f)
        
        try:
            if os.path.isfile(f):
               os.remove(f)
               logger.debug("  -> removing file: {}".format(f))
               return True
        except OSError as e:  ## if failed, report it back to the user ##
            logger.exception("Error: no such file: %s \n  %s." % (e.filename, e.strerror))
            return False
    
    
    @contextlib.contextmanager
    def working_directory(self,path):
        """
        copied from
        https://stackoverflow.com/questions/39900734/how-to-change-directories-within-python
        >>You simply specify the relative directory from the current directory,
        >>and then run your code in the context of that directory.
    
        :param path:
        :return:
    
        Example:
        --------
        with working_directory(<my path>):
             names = {}
             for fn in glob.glob('*.py'):
                 print(f)
    
        """
        prev_cwd = os.getcwd()
        path = jb.expandvars(path)
        try:
            os.chdir( path )
        except Exception as e:
            logger.exception("Can`t change to directory: ".format(path),exc_info=True)
        yield
        
        os.chdir(prev_cwd)
    
    
    def find_file(self,start_dir=None,pattern="*",file_extention="*.fif",recursive=True,debug=False,abspath=False,
                  ignore_case=False):
        """
        generator
        search for files in <start_dir> or subdirs matching <pattern> and <file extention[s]> using <glob.glob>
    
        :param start_dir     : start dir to search for files
        :param pattern       : pattern to look for e.g.: <*>
        :param file_extention: list of filee extentions <*.fif>
        :param recursive     : find files in subdirs <True>
        :param ignore_case   : file pattern matching case insensitive <False>
        :param debug         : <False>
        :param abspath       : add absolute path to found files <False>
        :return:
    
          file name ; generator
    
        Example:
        --------
         from jumeg.base.jumeg_base import JuMEG_Base_Basic
         jb = JuMEG_Base_Basic()
    
         pattern = "**/"
         pdfs    = dict()
    
         for id in id_list:
             sdir = os.path.join(start_dir,id)
             for fpdf in jb.find_file(start_dir=sdir,pattern=pattern,file_extention=file_extention):
                 if fpdf:
                    if not pdfs.get(id):
                       pdfs[id]=[]
                    pdfs[id].append( {"pdf": fpdf,"selected":False} )
                    npdfs+=1
    
        """
        pattern = self.update_pattern(pattern,ignore_case=ignore_case)
        if not isinstance(file_extention,(list)):
            s = file_extention
            file_extention = list()
            file_extention.append(s)
        
        if debug or self.debug:
            logging.debug("  -> start dir      : {}\n".format(start_dir) +
                          "  -> glob pattern   : {}\n".format(pattern) +
                          "  -> file extention : {}\n".format(file_extention) +
                          "  -> glob recursive : {}\n".format(recursive) +
                          "  -> adding abs path: {}\n".format(abspath)
                          )
        
        with self.working_directory(start_dir):
            for fext in file_extention:
                for f in glob.glob(pattern + fext,recursive=recursive): # ToDo  fext re /\.vhdr|vmrk|eeg$/
                    if abspath:
                        yield os.path.abspath(os.path.join(start_dir,f))
                    else:
                        yield f
    
    
    def find_files(self,start_dir=None,pattern="*",file_extention="*.fif",recursive=True,debug=False,abspath=False,
                   ignore_case=False):
        """
        search for files in <start_dir> or subdirs matching <pattern> and <file extention[s]> using <glob.glob>
    
        :param start_dir     : start dir to search for files
        :param pattern       : pattern to look for e.g.: <*>
        :param file_extention: list of filee extentions <*.fif>
        :param recursive     : find files in subdirs <True>
        :param ignore_case   : file pattern matching case insensitive <False>
        :param debug         : <False>
        :param abspath       : add absolute path to found files <False>
        :return:
            file list
    
        Example:
        --------
         from jumeg.base.jumeg_base import JuMEG_Base_Basic
         jb = JuMEG_Base_Basic()
    
         pattern = "**/"
         pdfs    = dict()
    
         for id in id_list:
             sdir = os.path.join(start_dir,id)
             flist=jb.find_files(start_dir=sdir,pattern=pattern,file_extention=file_extention):
             for f in flist:
                 print(f)
    
        """
        pattern = self.update_pattern(pattern,ignore_case=ignore_case)
        
        if not isinstance(file_extention,(list)):
            s = file_extention
            file_extention = list()
            file_extention.append(s)
        
        if debug or self.debug:
           logger.debug("  -> start dir      : {}\n".format(start_dir) +
                          "  -> glob pattern   : {}\n".format(pattern) +
                          "  -> file extention : {}\n".format(file_extention) +
                          "  -> glob recursive : {}\n".format(recursive) +
                          "  -> adding abs path: {}\n".format(abspath)
                        )
        files_found = []
        with self.working_directory(start_dir):
            for fext in file_extention: # ToDo  fext re /\.vhdr|vmrk|eeg$/
                for f in glob.iglob(pattern + fext,recursive=recursive):
                    #print(f)
                    if abspath:
                        files_found.append(os.path.abspath(os.path.join(start_dir,f)))
                    else:
                        files_found.append(f)
        
        files_found.sort()
        return files_found
    
    
    def update_pattern(self,pat,ignore_case=False):
        """
        update pattern if <ignore_case>
    
        :param pat:
        :param ignore_case: <False>
        :return:
          pattern
        """
        if not pat:
           pat= "*"
        else:
           pat = re.sub('\*\*+','**',pat)
           if pat.find("**") < 0:  # search in subdir
              d = pat.split("/")
              d[-1] = "**/" + d[-1]
             #d[0] = "**/" + d[0]
              pat = "/".join(d)
        
        if ignore_case:
            pat_ic = ""
            for c in pat:
                if c.isalpha():
                    pat_ic += '[%s%s]' % (c.lower(),c.upper())
                else:
                    pat_ic += c
            return pat_ic
        return pat
    
    
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
    
    
class JuMEG_IOutils_FindIds(JuMEG_IoUtils_FileIO):
    __slots__ = ["_stage","pattern","_ids","_start_path"]
    
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        for k in self.__slots__:
            self.__setattr__(k,None)
        self.pattern = "^[0-9]+"
        self._ids = []
        
        # self._update_from_kwargs(**kwargs)
    
    def _update_from_kwargs(self,**kwargs):
        self.stage    = kwargs.get("stage",self.stage)
        self.pattern  = kwargs.get("pattern")  # !!! reset pattern
    
    @property
    def NumberOfIds(self):
        return len(self._ids)
    
    def GetIDs(self):
        return self._ids
    
    @property
    def stage(self):
        return self._stage
    
    @stage.setter
    def stage(self,v):
        if v:
            self._stage = self.expandvars(v)
    
    def find(self,**kwargs):
        """
        :param stage  : stage, start path
        :param pattern: search pattern
       
        :return:
        list of ids
        """
        self._update_from_kwargs(**kwargs)
        self._ids = []
        
        pat = None
        if self.pattern:
            pat = re.compile(self.pattern)
        
        with self.working_directory(self.stage):
           #--- works in py 3.6
           #  with os.scandir(path=".") as dirs:
           #     for id in dirs:
           #         if not id.is_dir(): continue  #os.path.isdir(id): continue
           #         if pat:
           #             if pat.search(id.name):
           #                 self._ids.append(id.name)
           #         else:
  
           #             self._ids.append(id.name)
             for id in  os.listdir(path="."):
                 if not os.path.isdir(id): continue
                 if pat:
                    if pat.search(id):
                       self._ids.append(id)
                 else:
                    self._ids.append(id)
        
        self._ids.sort()
        
        return self._ids


class JuMEG_IOutils_FindPDFs(JuMEG_IoUtils_FileIO):
    __slots__ = ["_stage","pattern","_ids","_pdfs","_start_path"]
    
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        for k in self.__slots__:
            self.__setattr__(k,None)
        self._pdfs = dict()
        self._ids  = []
        # self._update_from_kwargs(**kwargs)
    
    def _update_from_kwargs(self,**kwargs):
        self.stage   = kwargs.get("stage",self.stage)
        self.ids     = kwargs.get("ids",  self.ids)
        self.pattern = kwargs.get("pattern")  # !!! reset pattern
    
    @property
    def pdfs(self): return self._pdfs
    
    @property
    def stage(self):
        return self._stage
    @stage.setter
    def stage(self,v):
        if v:
            self._stage = expandvars(v)
    
    @property
    def NumberOfIds(self):
        return len(self._ids)
    
    @property
    def ids(self): return self._ids
    @ids.setter
    def ids(self,v):
        if isinstance(v,list):
           self._ids= v
        else:
           self._ids=list(v)
        
    def GetIDs(self):
        return self._ids
    
    @property
    def stage(self):
        return self._stage
    
    @stage.setter
    def stage(self,v):
        if v:
            self._stage = jb.expandvars(v)
    
    def find(self,**kwargs):
        self._update_from_kwargs(**kwargs)
        self._pdfs = []
        
        pat = None
        if self.pattern:
           pat = re.compile(self.pattern)
        
        with self.working_directory(self.stage):
            
            for id in self.ids:
                self._pdfs[id] = self.find_files(start_dir=self._stage,pattern="*",file_extention="*.fif",recursive=True,
                                            debug=False,abspath=False,ignore_case=False)
        
        return self._pdfs

