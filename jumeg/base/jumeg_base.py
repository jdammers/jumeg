# -*- coding: utf-8 -*-

'''
JuMEG Base Class to provide wrapper & helper functions

Authors: 
         Prank Boers     <f.boers@fz-juelich.de>
         Praveen Sripad  <praveen.sripad@rwth-aachen.de>
License: BSD 3 clause

---> update 23.06.2016 FB
---> update 20.12.2016 FB
 --> add eeg pick-cls
 --> eeg BrainVision IO support
---> update 23.12.2016 FB
 --> add opt feeg in get_filename_list_from_file
 --> to merge eeg BrainVision with meg in jumeg_processing_batch
---> update 04.01.2017 FB
 --> add neww CLS JuMEG_Base_FIF_IO
 --> to merge eeg BrainVision with meg in jumeg_processing_batch
---> update 05.09.2017 FB
 --> update get_empty_room_fif
 --> use CLS function to read <empty room fif file> instead of mne.io stuff
---> update 05.04.2018 FB
 --> update JuMEG_Base_PickChannels.picks2label
 --> get lable list from index picks
---> update 13.04.2018 FB
 --> add pprint.PrettyPrinter(indent=4) as pp
  -> prints formated dict self.pp( my-dict)
 --> add line function
  -> prints line --- self.line()
---> update 05.07.2018 FB
 --> add update_bad_channels
  -> returns only unique bads
---> update 24.08.2018 FB
 --> update print()
  -> add isEmptyString,isNumber
---> update 03.04.2019 FB
 --> user logger from logging avoid print
---> update 02.05.2019 FB
 --> JuMEG_BasePickChannels add def check_dead_channels
'''

import os,sys,six,contextlib,re,numbers
import numpy as np
import glob

# py3 obj from pathlib import Path

import warnings
with warnings.catch_warnings():
     warnings.filterwarnings("ignore",category=DeprecationWarning)
     import mne

from pprint import PrettyPrinter #,pprint,pformat

import logging
logger = logging.getLogger("jumeg")
#logger.setLevel('DEBUG')

__version__="2019.09.13.001"

'''
class AccessorType(type):
    """
    meta class example
    http://eli.thegreenplace.net/2011/08/14/python-metaclasses-by-example
    """
    def __init__(self, name, bases, d):
        type.__init__(self, name, bases, d)
        accessors = {}
        prefixs = ["__get_", "__set_", "__del_"]
        for k in d.keys():
            v = getattr(self, k)
            for i in range(3):
                if k.startswith(prefixs[i]):
                    accessors.setdefault(k[4:], [None, None, None])[i] = v
        for name, (getter, setter, deler) in accessors.items():
            # create default behaviours for the property - if we leave
            # the getter as None we won't be able to getattr, etc..

            # [...] some code that implements the above comment

            setattr(self, name, property(getter, setter, deler, ""))

'''

class JuMEG_Base_Basic(object):
    def __init__ (self):
        super(JuMEG_Base_Basic, self).__init__()

        self._verbose       = False
        self._debug         = False
        self._template_name = None
        self._do_run        = False
        self._do_save       = False
        self._do_plot       = False

        self._pp = PrettyPrinter(indent=4)
        #self.Log = JuMEG_Logger() # use jumeg_logger.py
        #self.Log = logging.getLogger() # root
        
        self.__MSG_CODE_PATH_NOT_EXIST = 1001
        self.__MSG_CODE_FILE_NOT_EXIST = 1002
        
    
    def get_function_name(self):
        return sys._getframe(2).f_code.co_name
    def get_function_fullfilename(self):
        return sys._getframe(2).f_code.co_filename

    @property
    def python_version(self):
        return sys.version_info
    
    @property    
    def version(self): return __version__
    
    @property
    def verbose(self):  return self._verbose
    @verbose.setter
    def verbose(self,v):self._verbose=v
    
    @property
    def debug(self): return self._debug
    @debug.setter
    def debug(self,v):
        self._debug = v
        if self._debug:
           self.verbose=True

    @property
    def run(self):   return self._do_run
    @run.setter
    def run(self,v): self._do_run=v
    
    @property
    def save(self): return self._do_save
    @save.setter 
    def save(self,v): self._do_save=v
    
    @property
    def plot(self): return self._do_plot
    @plot.setter
    def plot(self,v): self._do_plot=v   
   
    def line(self,n=40,char="-"):
        """ line: prints a line for nice printing  and separate
        Parameters:
        -----------
        n   : factor to print n x times character <n=50>
        char: character to print n x times        <"-">
      
        Returns:
        ----------
        print a seperator line 
        
        Example:
        ---------
        from jumeg.jumeg_base import jumeg_base as jb   
        jb.line()
        --------------------------------------------------
        
        jb.line(n=10,char="x")
        xxxxxxxxxx
        """
        print(char * n)

    def pp_list2str(self,param,head=None):
        """
        call PrettyPrinter with dict or list
        Parameter
        ---------
         dict,string

        Result
        -------
         pretty formated string
        """
        if isinstance(param,(dict,list)):
           if head:
              return head+"\n" + ''.join(map(str,self._pp.pformat(param)))
           return ''.join(map(str,self._pp.pformat(param)))
        if head:
           return head+"\n" + self._pp.pformat(param)
        return self._pp.pformat(param)
        
    def is_number(self,n):
        """ 
        check if input is a number:  isinstance of int or float
         no check for numpy ndarray
         https://stackoverflow.com/questions/40429917/in-python-how-would-you-check-if-a-number-is-one-of-the-integer-types
         
        Parameters:
        ------------            
        n: value to check
        
        Returns:
        ---------
        True / False
        
        Example
        ---------
        from jumeg.jumeg_base import jumeg_base as jb  
        
        jb.is_number(123)
         True
        """
        #print("is_number: {} type:{}".format(n, type(n)))
        try: # if n not defined
            if  isinstance(n,(numbers.Number)):
                #print("isNumber Test:")
                #print(isinstance(n,(numbers.Number)))
                return True
            return False
        except NameError as e:
            logger.exception("input not defined",exc_info=True)
            return False
            
    def isNumber(self,n):
        """
         wrapper fct. call <is_number>
        """
        return self.is_number(n)
    
    def isNotEmptyString(self,s):
        """
         check not empty string
         https://stackoverflow.com/questions/4843173/how-to-check-if-type-of-a-variable-is-string
        """
        if not s : return
        if isinstance(s,six.string_types): return bool(s.strip())
        # PY2 if isinstance(s, basestring): return bool(s.strip())
        return False
    
    def isNotEmpty(self,v):
        """
         check not empty string,int,float,list,tuple types
         not for numpy arrays
         https://stackoverflow.com/questions/4843173/how-to-check-if-type-of-a-variable-is-string
         
         no check for numpy ndarray
        """
        if isinstance( v,(int,float,list,tuple) ): return True
        if self.is_number(v):                      return True
        if isinstance(v,six.string_types):         return bool(v.strip())
        
        # PY2 if isinstance(s, basestring): return bool(s.strip())
        return False

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
                v[i] = os.path.expandvars(os.path.expanduser( str(v[i]) ))
            return v
    
        return os.path.expandvars(os.path.expanduser( str(v) ))

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
              logger.info( "---> " +head+"\n --> file exist: {}\n  -> abs file{:>18} {}".format(fck,':',f))
           return f
       #--- error no such file
        logger.error("---> " +head+"\n --> no such file or directory: {}\n  -> abs file{:>18} {}".format(fck,':',f))
        if exit_on_error:
           raise SystemError(self.__MSG_CODE_FILE_NOT_EXIST)
        return False
       
    def isDir(self,v):
        """
        expand env's e.g. expand user from string
        check if is dir
        
        :param v: string
        :return: input with expanded env's or None
        """
        v=self.expandvars(v)
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
              logger.info("--->"+head+"\n --> dir exist: {}\n  -> abs dir{:>18} {}".format(pin,':',p))
           return p
       #--- error no such file
        logger.error("---> "+head+"\n --> no such directory: {}\n  -> abs dir{:>18} {}".format(pin,':',p))
        if exit_on_error:
           raise SystemError(self.__MSG_CODE_PATH_NOT_EXIST)
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
        try:
            os.chdir(path)
        except Exception as e:
               logger.exception("Can`t change to directory: ".format(path),exc_info=True)
        yield
       
        os.chdir(prev_cwd)
    
    def find_file(self,start_dir=None,pattern="*",file_extention="*.fif",recursive=True,debug=False,abspath=False,ignore_case=False):
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
                 for f in glob.glob(pattern + fext,recursive=recursive):
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
            logging.debug("  -> start dir      : {}\n".format(start_dir) +
                          "  -> glob pattern   : {}\n".format(pattern) +
                          "  -> file extention : {}\n".format(file_extention) +
                          "  -> glob recursive : {}\n".format(recursive) +
                          "  -> adding abs path: {}\n".format(abspath)
                          )
        files_found = []
        with self.working_directory(start_dir):
            for fext in file_extention:
                for f in glob.iglob(pattern + fext,recursive=recursive):
                    if abspath:
                       files_found.append( os.path.abspath(os.path.join(start_dir,f)) )
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

"""
ToDo
in CLS __init__
self._defaults = { "meg":False,"grad":False...,  exclude_bads=False}
self._types=None

@property
def types(self): return self._types
@types.setter
def types(self,l)
    self._types = self._defaults.copy()
    for k in l
        self._types[k]=True

def meg_nobads(self,raw)
    self.types = ["meg"]
    return self._get_picks_nobads()
  
def _get_picks() # can apply update/changes to mne function
    return mne.pick_types(raw.info, **self._types)

def _get_picks_no_bads() # can apply update/changes to mne function
    self._types["exclude_bads"]=True
    return mne.pick_types(raw.info, **self._types)
 
 
 or use contextmanager YIELD

def _GP() # can apply update/changes to mne function
    Yield
    return mne.pick_types(raw.info, **self._types)
 
 def _GPNB() # can apply update/changes to mne function
 #_get_picks_no_bads
    Yield
    self._types["exclude_bads"]=True
    return mne.pick_types(raw.info, **self._types)

@ _BPNB
def meg_nobads(self,raw)
    self.types = ["meg"]
@ _BP
def meg(self,raw)
    self.types = ["meg"]
   
   
"""
class JuMEG_Base_PickChannels(object):
    """ MNE Wrapper Class for mne.pick_types
    return list of channel index from mne.raw obj e.g. for special groups
        
    Wrapper call to    
     --> mne.pick_types(raw.info, **fiobj.pick_all) 
     --> mne.pick_types(info, meg=True, eeg=False, stim=False, eog=False, ecg=False, emg=False, ref_meg='auto',
                           misc=False, resp=False, chpi=False, exci=False, ias=False, syst=False,
                           include=[], exclude='bads', selection=None)

    https://github.com/mne-tools/mne-python/blob/master/mne/io/pick.py  lines 20ff
    type : 'grad' | 'mag' | 'eeg' | 'stim' | 'eog' | 'emg' | 'ecg' |'ref_meg' | 'resp' | 'exci' | 'ias' | 'syst' | 'misc'|'seeg' | 'chpi'  
     
    Example:
    ---------   
    from jumeg.jumeg_base import jumeg_base as jb  
        
    jb.picks.meg( raw )
    return meg channel index array => 4D Magnes3600 => [0 .. 247]
         
    jb.picks.meg_nobads( raw )
    return meg channel index array without bad channels
       
    extended function:
    jb.picks.picks2labels(raw,picks)
         
    jb.picks.labels2picks(raw,labels)
         
    #---- 
    or only import this class
    from jumeg.jumeg_base import JuMEG_Base_PickChannels
         
    picks = JuMEG_Base_PickChannels()
    picks.meg_nobads( raw ) 
    """  
      
    def __init__ (self):
        """ init """
        # getting from <mne.pick_types>
        #self._pick_type_set={'meg','mag', 'grad', 'planar1','planar2','eeg','stim','eog','ecg','emg','ref_meg','misc',
        #                     'resp','chpi','exci','ias','syst','seeg','dipole','gof','bio','ecog','fnirs'}

       #--- mne version 0.17
        self._pick_type_set={'eeg', 'mag', 'grad', 'ref_meg', 'misc', 'stim', 'eog', 'ecg', 'emg', 'seeg', 'bio', 'ecog', 'hbo', 'hbr'}
        self.verbose = False
        self.debug   = False
        
    @property
    def pick_type_set(self): return self._pick_type_set

    def picks2labels(self,raw,picks):
        '''
        get channel labels from picks
        
        Parameter
        ---------
         raw obj
         
         picks <numpy array int64>
        
        Result
        -------
         return label list
       
        Example
        --------
         from jumeg.base.jumeg_base import jumeg_base as jb
         
         fraw = "test.fif"
         
         raw,fnraw = jb.get_raw_obj(fraw)
         
         picks = jb.picks.meg_and_ref_nobads(raw)
         
         labels= jb.picks.picks2labels( raw,picks )
         
         print(labels)
         
        '''
        if isinstance(picks,(int,np.int64)):
           return raw.ch_names[picks] 
        return ([raw.ch_names[i] for i in picks]) 
       
    def labels2picks(self,raw,labels):
        """
        get picks from channel labels
        call to < mne.pick_channels >
        picks = mne.pick_channels(raw.info['ch_names'], include=[labels])

        Parameter
        ---------
         raw obj
         channel label or list of labels

        Result
        -------
         picks as numpy array int64
        """
        if isinstance(labels,(list)):
           return  mne.pick_channels(raw.info['ch_names'],include=labels)
        else:
           return mne.pick_channels(raw.info['ch_names'],include=[labels])

    def bads2picks(self,raw):
        """
        mne wrapper
        get picks from bad channel labels
        call to < mne.pick_channels >
        picks = mne.pick_channels(raw.info['ch_names'], include=raw.info['bads'])

        Parameter
        ---------
         raw obj
         
        Result
        -------
         bad picks as numpy array int64 (index of bad channels)
        """
        if raw.info['bads']:
           return  mne.pick_channels(raw.info['ch_names'],include=raw.info['bads'])
        return None

    def channels(self,raw):
        """ call with meg=True,ref_meg=True,eeg=True,ecg=True,eog=True,emg=True,misc=True,stim=False,resp=False,exclude=None """
        return mne.pick_types(raw.info,meg=True,ref_meg=True,eeg=True,ecg=True,eog=True,emg=True,misc=True,stim=False,resp=False,exclude=[])
     
    def channels_nobads(self, raw):
        """ call with meg=True,ref_meg=True,eeg=True,ecg=True,eog=True,emg=True,misc=True,stim=False,resp=False,exclude='bads' """
        return mne.pick_types(raw.info, meg=True,ref_meg=True,eeg=True,ecg=True,eog=True,emg=True,misc=True,stim=False,resp=False,exclude='bads')
       
    def all(self, raw):
        """ call with meg=True,ref_meg=True,eeg=True,ecg=True,eog=True,emg=True,misc=True, stim=True,resp=True,exclude=None """
        return mne.pick_types(raw.info, meg=True,ref_meg=True,eeg=True,ecg=True,eog=True,emg=True,misc=True, stim=True,resp=True,exclude=[])       
    
    def all_nobads(self, raw):
        """ call with meg=True,ref_meg=True,eeg=True,ecg=True,eog=True,emg=True,misc=True, stim=True,resp=True,exclude='bads' """
        return mne.pick_types(raw.info, meg=True,ref_meg=True,eeg=True,ecg=True,eog=True,emg=True,misc=True, stim=True,resp=True,exclude='bads')
  #--- meg
    def meg(self,raw):
        """ call with meg=True """
        return mne.pick_types(raw.info,meg=True)      
    
    def meg_nobads(self,raw):
        ''' call with meg=True,exclude='bads' '''
        return mne.pick_types(raw.info, meg=True,exclude='bads')
    
        
    def meg_and_ref(self,raw):
        ''' call with meg=True,ref_meg=True'''
        return mne.pick_types(raw.info, meg=True,ref_meg=True, eeg=False, stim=False,eog=False)
    def meg_and_ref_nobads(self,raw):
        ''' call with meg=mag,ref_meg=True,exclude='bads' '''
        return mne.pick_types(raw.info,meg=True,ref_meg=True,eeg=False,stim=False,eog=False,exclude='bads')
    
    def meg_ecg_eog(self,raw):
        ''' call with meg=True,ref_meg=False,ecg=True,eog=True,stim=True,exclude=bads'''
        return mne.pick_types(raw.info, meg=True,ref_meg=False,eeg=False,stim=False,eog=True,ecg=True)
  
    def meg_ecg_eog_nobads(self,raw):
        ''' call with meg=True,ref_meg=False,ecg=True,eog=True,stim=True,exclude=bads'''
        return mne.pick_types(raw.info, meg=True,ref_meg=False,eeg=False,stim=False,eog=True,ecg=True,exclude='bads')
  
    def meg_ecg_eog_stim(self,raw):
        ''' call with meg=True,ref_meg=False,ecg=True,eog=Truestim=True,'''
        return mne.pick_types(raw.info, meg=True,ref_meg=False,eeg=False,stim=True,eog=True,ecg=True)
    def meg_ecg_eog_stim_nobads(self,raw):
        ''' call with meg=True,ref_meg=False,ecg=True,eog=True,stim=True,exclude=bads'''
        return mne.pick_types(raw.info, meg=True,ref_meg=False,eeg=False,stim=True,eog=True,ecg=True,exclude='bads')
  #---
    def ref(self,raw):
        ''' call with ref=True'''
        return mne.pick_types(raw.info,ref_meg=True,meg=False,eeg=False,stim=False,eog=False)

    def ref_nobads(self,raw):
        ''' call with ref=True,exclude='bads' '''
        return mne.pick_types(raw.info,ref_meg=True,meg=False,eeg=False,stim=False,eog=False,exclude='bads')
   #---
    def ecg(self,raw):
        ''' meg=False,ref_meg=False,ecg=True,eog=False '''
        return mne.pick_types(raw.info,meg=False,ref_meg=False,ecg=True,eog=False)
    def eog(self,raw):
        ''' meg=False,ref_meg=False,ecg=False,eog=True '''
        return mne.pick_types(raw.info,meg=False,ref_meg=False,ecg=False,eog=True)
   
    def ecg_eog(self,raw):
        ''' meg=False,ref_meg=False,ecg=True,eog=True '''
        return mne.pick_types(raw.info,meg=False,ref_meg=False,ecg=True,eog=True)

    def eeg(self,raw):
        ''' meg=False,ref_meg=False,ecg=False,eog=False '''
        return mne.pick_types(raw.info,meg=False,ref_meg=False,ecg=False,eog=False,eeg=True)
    def eeg_nobads(self, raw):
        ''' meg=False,ref_meg=False,ecg=False,eog=False '''
        return mne.pick_types(raw.info, meg=False, ref_meg=False, ecg=False, eog=False, eeg=True, exclude='bads')

    def eeg_ecg_eog(self, raw):
        ''' meg=False,ref_meg=False,ecg=True,eog=True,eeg=True '''
        return mne.pick_types(raw.info, meg=False, ref_meg=False, ecg=True, eog=True, eeg=True)
    def eeg_ecg_eog_nobads(self, raw):
        ''' meg=False,ref_meg=False,ecg=True,eog=True,eeg=True '''
        return mne.pick_types(raw.info, meg=False, ref_meg=False, ecg=True, eog=True, eeg=True, exclude='bads')
    
    def emg(self,raw):
        ''' meg=False,ref_meg=False,ecg=False,eog=False '''
        return mne.pick_types(raw.info,meg=False,ref_meg=False,ecg=False,eog=False,eeg=False,emg=True)
    def emg_nobads(self, raw):
        ''' meg=False,ref_meg=False,ecg=False,eog=False '''
        return mne.pick_types(raw.info, meg=False, ref_meg=False, ecg=False, eog=False, eeg=False,emg=True,exclude='bads')

    def stim(self,raw):
        ''' call with meg=False,stim=True '''
        return mne.pick_types(raw.info,meg=False,stim=True)
        
    def response(self,raw):
        ''' call with meg=False,resp=True'''
        return mne.pick_types(raw.info,meg=False,resp=True)
        
    def stim_response(self,raw):
        ''' call with meg=False,stim=True,resp=True'''
        return mne.pick_types(raw.info, meg=False,stim=True,resp=True)

    def stim_response_ecg_eog(self,raw):
        ''' call with meg=False,stim=True,resp=True'''
        return mne.pick_types(raw.info, meg=False,stim=True,resp=True,ecg=True,eog=True)

    def exclude_trigger(self, raw):
        ''' call with meg=True,ref_meg=True,eeg=True,ecg=True,eog=True,emg=True,misc=True,stim=False,resp=False,exclude=None '''
        return mne.pick_types(raw.info, meg=True,ref_meg=True,eeg=True,ecg=True,eog=True,emg=True,misc=True,stim=False,resp=False)       

    def bads(self,raw):
        """ return raw.info[bads] """
        return raw.info['bads']
 
    def update_bads(self,raw,bads=None,clear=False):
        """
        update bads in raw, delete doubles, sort
        
        :param raw  : raw obj
        :param bads : list of bads <None>
        :param clear: clear bads list in raw first <False>
        
        :return:
        
        Return
        -------
         raw.info[bads]
        """
        if clear:
           raw.info["bads"]=[]
        if bads:
           if isinstance(bads,(list)):
              raw.info["bads"].extend( bads )
           else:
              raw.info["bads"].append(bads)
        if raw.info['bads']:
           b = list(set(raw.info['bads']))
           b.sort()  # inplace !!!
           raw.info['bads'] = b
        
        return raw.info['bads']

    def check_dead_channels(self,raw,picks=None,update=True,verbose=False):
        '''
        check for dead channels ( min == max )
        e.g.
            dead channels candidates in Jülich 4D system Magnes3600 might be:
            ['MEG 007', 'MEG 010', 'MEG 142', 'MEG 156', 'RFM 011']
        :param raw    : <raw obj>
        :param picks  : channel index list
                        if None use  meg,ref,exclude bads, call <meg_and_ref_nobads(raw)>
        :param update : update bads in raw.info <True>
        :param verbose: update bads in raw.info <True>
        
        :return:
         picks without bads
        '''
        if picks is None:
           picks = self.meg_and_ref_nobads(raw)
       #--- if empty list
        if not picks.any():
           if self.verbose or verbose:
              logger.warning("  -> looking for dead channels -> no picks defined" )
           return picks
       #--- idx array: 0:dead 1:ok
        idx = np.where( raw._data[picks,:].min(axis=1) == raw._data[picks,:].max(axis=1),0,1 )
       
       #--- update bads in raw, delete doubles, sort
        if update:
           self.update_bads(raw,bads=self.picks2labels(raw,picks[np.where(idx < 1)[0]]) )
           
        if self.verbose or verbose:
          # logger.setLevel("DEBUG")
           bads_idx = np.where(idx < 1)[0]
           logger.info("  -> looking for dead channels\n"+
                       "  -> dead channels :  {}\n".format(self.picks2labels(raw,picks[bads_idx]))+
                       "  ->      index    :  {}\n".format(bads_idx)+
                       "  -> update bads in raw.info: {}".format(update)
                       )
        logger.info("  -> bads: {}\n".format(self.bads(raw) ))
        
        return picks[ np.where(idx)[0] ]

class JuMEG_Base_StringHelper(JuMEG_Base_Basic):
    """ Helper Class to work with strings """
    
    def __init__ (self):
        super(JuMEG_Base_StringHelper,self).__init__()
         
    def isString(self, s):
        """ check if is string return True/False
         http://ideone.com/uB4Kdc
        
        Example
        -------- 
        from jumeg.jumeg_base import jumeg_base as jb  
        
        jb.isString("Hell World")
         True
         
        """
        if not s: return False
        if (isinstance(s, str)):
           return True
        #--- py3 no basestring
        #try:  basestring
        #except NameError:
        #      basestring = str
        #try:
        #   if isinstance(s, (basestring)):
        #      return True
        #except NameError:
        #   return False
        
        return False    

    def isNotEmptyString(self,s):
        '''
        check if is value is a string and not empty
         e.g. s=""  
        
         Parameter
         ---------
          value to check
         
         Results
         -------
          True/False
         
          http://ideone.com/uB4Kdc
        
        Example
        -------- 
        from jumeg.jumeg_base import jumeg_base as jb  
        
        s="" 
        jb.isEmptyString(s)
        >> False
         
        '''
        if self.isString(s):
           if s.strip(): return True
        return False
         
    def str_range_to_list(self, seq_str):
        """make a list of inergers from string
        ref:
        http://stackoverflow.com/questions/6405208/how-to-convert-numeric-string-ranges-to-a-list-in-python
           
        Parameters
        ----------
        seq_str: string
        
        Returns
        --------
        list of numbers
        
        Example
        --------
        from jumeg.jumeg_base import jumeg_base as jb 
        
        jb.str_range_to_list("1,2,3-6,8,111")
        
        "1,3,5"         => [1,3,5]
        "1-5"           => [1,2,3,4,5]
        "1,2,3-6,8,111" => [1,2,3,4,5,6,8,111]
        """
        #xranges = [(lambda l: range(l[0], l[-1]+1))(map(int, r.split('-'))) for r in seq_str.split(',')]
        #--- py3 range instead of xrange
        
        xrange=[]
        for r in seq_str.split(','):
            l=r.split('-')
            if len(l)>1:
               xrange+= list(range(int(l[0]),int(l[-1]) + 1))
            else:
               xrange.append( int(l[0]) )
       #--- mk unique & list & sort
        xrange = list(set(xrange))
        xrange.sort()
        # flatten list of xranges
        #return [y for x in xranges for y in x]
        return xrange
    
    def str_range_to_numpy(self, seq_str,exclude_zero=False,unique=False): 
        """converts integer string to numpy array 
        Parameters
        ----------
        input       : integer numbers as string
        exclude_zero: exclude 0  <False>
        unique      : return only unique numbers <False>
        
        Returns
        --------
        integer numbers as numpy array dtype('int64')
        
        Example
        --------
        
        from jumeg.jumeg_base import jumeg_base as jb 
        
        s = "0,1,2,3,0,1,4,3,0"
        
        jb.str_range_to_numpy(s)
          array([0, 1, 2, 3, 0, 1, 4, 3, 0])
        
        jb.str_range_to_numpy(s,unique=True)
          array([0, 1, 2, 3, 4])
              
        jb.str_range_to_numpy(s,exclude_zero=True)
          array([1, 2, 3, 1, 4, 3])
        
        jb.str_range_to_numpy(s,unique=True,exclude_zero=True)
          array([1, 2, 3, 4]) 
           
        """
        import numpy as np

        if seq_str is None:
           return np.unique( np.asarray( [ ] ) )
        if self.isString(seq_str):
           s = re.sub(r",+",",",seq_str.replace(" ",",") )
           anr = np.asarray (self.str_range_to_list( s ) )
        else:
           anr = np.asarray( [seq_str] )
           
        if unique:
           anr = np.unique( anr ) 
        if exclude_zero:
           return anr[ np.where(anr) ] 
        return anr

class JuMEG_Base_FIF_IO(JuMEG_Base_StringHelper):
    """ handels mne fif I/O for meg and eeg [BrainVision] data
        workaround to manage different MNE versions
        
    """
    def __init__ (self):
        super(JuMEG_Base_FIF_IO, self).__init__()
        
    def set_raw_filename(self,raw,v):
        """ set filename in raw obj"
        Parameters:
        -----------
        raw     : rawobj to modify
        filename: filename
        
        Returns:
        ----------
        None
        
        Example:
        ----------
        from jumeg.jumeg_base import jumeg_base as jb 
         
        jb.set_raw_filename(raw,"/data/test-raw.fif")
        """
        if hasattr(raw,"filenames"):
           raw._filenames = []
           raw._filenames.append(v)
        else:
           raw.info['filename'] = v


    def get_raw_filename(self,raw,index=0):
        """ get filename from raw obj
        
        Parameters:
        -----------
        raw     : raw-obj to modify
        index   : index in list of filenames from raw.filenames <0>      
                  if index = list return filename list
        Returns:
        ----------
         first filename or None
        
        Example:
        ----------
         from jumeg.jumeg_base import jumeg_base as jb 
         fname = jb.get_raw_filename(raw)
        """
        if raw:
           if hasattr(raw,"filenames"): 
              if index == "list"                : return raw.filenames 
              if abs(index) < len(raw.filenames): return raw.filenames[index]
              return raw.filenames
           return raw.info.get('filename')
        return None 
    
    def __get_from_fifname(self,v=None,f=None):
        try:
           return os.path.basename(f).split('_')[v]
        except:
           return os.path.basename(f)

    def get_id(self,v=0,f=None):
        """ get id from fifname 
        
        Parameters:
        -----------
        f  : filename <None>
        
        Returns:
        ----------
        subject id in fif filename
        
        Example:
        ----------
        from jumeg.jumeg_base import jumeg_base as jb 
        f='/data/exp/FREEVIEWING/epocher/211776_FREEVIEW01_180115_1414_1_c,rfDC,meeg_bcc,tr,nr,ar-raw.fif'
        
        jb.get_id(f=f)
        "211776"
        
        """
        return self.__get_from_fifname(v=v,f=f)

    def get_scan(self,v=1,f=None):
        """ get scan from fifname 
        
        Parameters:
        -----------
        f  : filename <None>
        
        Returns:
        ----------
        scan in fif filename
        
        Example:
        ----------
        from jumeg.jumeg_base import jumeg_base as jb 
        f='/data/exp/FREEVIEWING/epocher/211776_FREEVIEW01_180115_1414_1_c,rfDC,meeg_bcc,tr,nr,ar-raw.fif'
        
        jb.get_scan(f=f)
        "FREEVIEW01"
        """
        return self.__get_from_fifname(v=v,f=f)
    
    def get_date(self,v=2,f=None):
        """ get session from fifname 
        
        Parameters:
        -----------
        f  : filename <None>
        
        Returns:
        ----------
        date in fif filename
        
        Example:
        ----------
        from jumeg.jumeg_base import jumeg_base as jb 
        f='/data/exp/FREEVIEWING/epocher/211776_FREEVIEW01_180115_1414_1_c,rfDC,meeg_bcc,tr,nr,ar-raw.fif'
        
        jb.get_session(f=f)
        "180115"
        """
        return self.__get_from_fifname(v=v,f=f)

    def get_time(self,v=3,f=None):
        """ get time from fifname 
        
        Parameters:
        -----------
        f  : filename <None>
        
        Returns:
        ----------
        time in fif filename
        
        Example:
        ----------
        from jumeg.jumeg_base import jumeg_base as jb 
        f='/data/exp/FREEVIEWING/epocher/211776_FREEVIEW01_180115_1414_1_c,rfDC,meeg_bcc,tr,nr,ar-raw.fif'
        
        jb.get_run(f=f)
        "1414"
        """
        return self.__get_from_fifname(v=v,f=f)
    
    def get_session(self,f=None):
        """ get session from fifname 
        
        Parameters:
        -----------
        f  : filename <None>
        
        Returns:
        ----------
        session in fif filename
        
        Example:
        ----------
        from jumeg.jumeg_base import jumeg_base as jb 
        f='/data/exp/FREEVIEWING/epocher/211776_FREEVIEW01_180115_1414_1_c,rfDC,meeg_bcc,tr,nr,ar-raw.fif'
        
        jb.get_session(f=f)
        "180115_1414"
        """
        return self.get_date(f=f)+"_"+self.get_time(f=f)
      
    def get_run(self,v=4,f=None):
        """ get run from fifname 
        
        Parameters:
        -----------
        f  : filename <None>
        
        Returns:
        ----------
        run in fif filename
        
        Example:
        ----------
        from jumeg.jumeg_base import jumeg_base as jb 
        f='/data/exp/FREEVIEWING/epocher/211776_FREEVIEW01_180115_1414_1_c,rfDC,meeg_bcc,tr,nr,ar-raw.fif'
        
        jb.get_run(f=f)
        "1"
        """
        return self.__get_from_fifname(v=v,f=f)

    def get_postfix(self,f=None):
        """ get postfix from fifname 
        
        Parameters:
        -----------
        f  : filename <None>
        
        Returns:
        ----------
        postfix in fif filename
        
        Example:
        ----------
        from jumeg.jumeg_base import jumeg_base as jb 
        f='/data/exp/FREEVIEWING/epocher/211776_FREEVIEW01_180115_1414_1_c,rfDC,meeg_bcc,tr,nr,ar-raw.fif'
        
        jb.get_postfix(f=f)
        "bcc,tr,nr,ar-raw"
        
        """
        return os.path.basename(f).split('_')[-1].split('.')[0]

    def get_extention(self,f=None):
        """ get extention from fifname 
        
        Parameters:
        -----------
        f  : filename <None>
        
        Returns:
        ----------
        file extention in fif filename  <fif>; without leading <.>
        
        Example:
        ----------
        from jumeg.jumeg_base import jumeg_base as jb 
        f='/data/exp/FREEVIEWING/epocher/211776_FREEVIEW01_180115_1414_1_c,rfDC,meeg_bcc,tr,nr,ar-raw.fif'
        
        jb.get_extention(f=f)
        "fif"
       
        """
        if f:
            fname = f
        else:
            fname = self.raw.info['filename']
        return os.path.basename(fname).split('_')[-1].split('.')[-1]

    def get_postfix_extention(self,f=None):
        """ get postfix with extention from fifname 
        
        Parameters:
        -----------
        f  : filename <None>
        
        Returns:
        ----------
        postfix with extention in fif filename 
        
        Example:
        ----------
        from jumeg.jumeg_base import jumeg_base as jb 
        f='/data/exp/FREEVIEWING/epocher/211776_FREEVIEW01_180115_1414_1_c,rfDC,meeg_bcc,tr,nr,ar-raw.fif'
        
        jb.get_postfix_extention(f=f)
        "bcc,tr,nr,ar-raw.fif"
        """
        if f:
            fname = f
        else:
            fname = self.raw.info['filename']
        return os.path.basename(fname).split('_')[-1]
   

class JuMEG_Base_IO(JuMEG_Base_FIF_IO):
    """I/O class to handle higher order work on raw obj
       e.g.: read in raw obj from meg, eeg or ica data
             update bad channels
    """
    def __init__ (self):
        super(JuMEG_Base_IO, self).__init__()
        
        self.picks = JuMEG_Base_PickChannels()

      #--- ToDo --- start implementig BV support may new CLS
        self.brainvision_response_shift = 1000
        self.brainvision_extention      = '.vhdr'
        self.ica_extention              = '-ica.fif'

    def get_fif_name(self, fname=None, raw=None,path=None, prefix=None,postfix=None, extention="-raw.fif", update_raw_fname=False):
        """
        changing filename with prefix postfix path and option to update filename in raw-obj
        
        Parameters:
        -----------
        fname            : base file name
        raw              : raw obj, if defined get filename from raw obj                <None>
        path             : new path, must exsist                                        <None>
        prefix           : string to add as prefix in filename                          <None>
        postfix          : string to add as postfix for applied operation               <None>
        extention        : string to add as extention                                   <-raw.fif>
        update_raw_fname : if true and raw is obj will update raw obj filename in place <False>
       
        Returns:
        ----------
        fif filename, based on input file name and applied operation
        with new path if path is not None
        
        Example:
        ----------
        from jumeg.jumeg_base import jumeg_base as jb 
        
        f ='/data/exp/FREEVIEWING/epocher/211776_FREEVIEW01_180115_1414_1_c,rfDC,meeg_bcc,tr,nr,ar-raw.fif'
        raw,fraw=jb.get_raw_obj(f)
        
        jb.get_fif_name(raw=raw)
        "/data/exp/FREEVIEWING/epocher/211776_FREEVIEW01_180115_1414_1_c,rfDC,meeg_bcc,tr,nr,ar-raw.fif"
        
        jb.get_fif_name(raw=raw,prefix="PREFIX",postfix="POSTFIX")

        """
        if raw:
           fname = self.get_raw_filename(raw)

           p, pdf = os.path.split(fname)
           fname = p + "/" + pdf[:pdf.rfind('-')]
           if prefix:
              fname = prefix +","+ fname
           if postfix:
              fname += "," + postfix
              fname = fname.replace(',-', '-')
           if extention:
              fname += extention
           if update_raw_fname:
               self.set_raw_filename(raw,fname)
           if path:
              if os.path.isdir(path):
                 fname = path +"/"+ os.path.basename( fname )
        return self.expandvars( fname )
        
    def update_bad_channels(self,fname,raw=None,bads=None,preload=True,append=False,save=False,interpolate=False,postfix=None):
        """ update bad channels in raw obj
        
        Parameters:
        -----------
        fname   : file name
        raw     : raw obj, if defined get filename from raw obj <None>
        bads    : list of bad channels                          <None> 
        postfix : add postfix to filename                       <None>
        
        Flags:
         preload : mne loar flag                                <True>
         append  : add bads to bads if true or overwrite        <False>
         postfix : add postfix to filename                      <None>
         save    : will raw with changes in bads                <False>
         interpolate: apply mne badchannel interpolation        <False>
         
        Returns:
        ----------
        return: raw, bad channel list        
        
        Example:
        ----------
        from jumeg.jumeg_base import jumeg_base as jb 
        
        f  ='/data/exp/FREEVIEWING/epocher/211776_FREEVIEW01_180115_1414_1_c,rfDC,meeg_bcc,tr,nr,ar-raw.fif'
        bads=["MEG 010","MEG 157"] or bads="MEG 010,MEG 157"
        raw,bads_list = jb.update_bad_channels(f,raw=raw,postfix="bads",save=True,bads=bads)
        
        saved bads as new fif file:
        /data/exp/FREEVIEWING/epocher/211776_FREEVIEW01_180115_1414_1_c,rfDC,meeg_bcc,tr,nr,ar,bads-raw.fif
        
        """
        #TODO: if  new bads ==  old bads in raw then  exit
   
        if save:
           preload = True
        raw,fname = self.get_raw_obj(fname,raw=raw,preload=preload)

        if not append:
           raw.info['bads']=[]
        
        if bads:
           if not isinstance(bads,list):
              bads = bads.split(',')

        if not bads:
           if not append:
              raw.info['bads']=[]
        else:
           for b in bads:
               bad_ch = None
               if (b in raw.ch_names):
                  bad_ch = b
               else:
                  if b.startswith('A'):
                     bad_ch = 'MEG '+ b.replace(" ","").strip('A').zfill(3)
                  elif b.startswith('MEG'):
                     bad_ch = 'MEG '+ b.replace(" ","").strip('MEG').zfill(3)
               if bad_ch:
                  if bad_ch not in raw.info['bads']:
                     raw.info['bads'].append(bad_ch)
        
     #--- only unique channel names sorted
        self.picks.update_bads(raw,bads=raw.info.get('bads'))
        
        fif_out = self.get_fif_name(raw=raw,postfix=postfix)

        if self.verbose:
           logger.info(" --> Update bad-channels\n"+
                       " --> FIF in  : {}\n".format(self.get_raw_filename(raw))+
                       " --> FIF out : {}\n".format(fif_out)+
                       " --> bads    : {}\n".format(raw.info['bads']))
              
        if ( interpolate and raw.info['bads'] ) :
           logger.info( self.pp_list2str(
              [" --> Update BAD channels => interpolating: {}".format(raw.info['filename']),
               " --> BADs : {}".format(raw.info['bads'])]))
           raw.interpolate_bads()
     
      #--- save raw as bads-raw.fif
        if save:
           raw.save( fif_out,overwrite=True)
        self.set_raw_filename(raw,fif_out)
             
        return raw,raw.info['bads']

#--- helper function
    def _get_ica_raw_obj(self,fname,raw=None):
        """check for <ica filename> or <ica raw obj>
        if <ica_raw> obj is None load <ica raw obj> from <ica filename>

        Parameters:
        -----------  
        fname: ica filename
        raw  : ica raw obj <None>
        
        Returns:
        --------
        <ica raw obj>,ica raw obj filename
        
        """
        if raw is None:
           if fname is None:
              assert "---> ERROR no file foumd!!\n\n"
              if self.verbose:
                 logger.info("<<<< Reading ica raw data ...")
        if fname:
           fn = self.expandvars( fname )
           if path:
              path = self.expandvars(path)
              fn   = os.path.join(path,fn)
        else:
           fn = self.get_raw_filename(raw)
   
           raw = mne.preprocessing.read_ica(fn)
         
           if raw is None:
              assert "---> ERROR in jumeg.jumeg_base.get_ica_raw_obj => could not get ica raw obj:\n ---> FIF name: " + fname
   
        return raw,self.get_raw_filename(raw)
    
            
    def get_raw_obj(self,fname,raw=None,path=None,preload=True,reload_raw=False,reset_bads=False):
        """
        load file in fif format <*.raw> or brainvision eeg data
        check for filename or raw obj
        check for meg or brainvision eeg data *.vhdr
        if filename -> load fif file
        
        Parameters
        ----------
         fname     : name of raw-file
         raw       : raw obj <None>
                     if raw: return raw and fullfilename of raw
         preload   : True
         reload_raw: reload raw-object via raw.filename <False>
         reset_bads: reset bads <False>
         
        Results
        ----------
         raw obj,fname from raw obj
        """
        
        
        if self.verbose:
           logger.info("<<<< Reading raw data ...")
           
        if self.debug:
           msg= [" --> start reading raw data:\n",
                 "  -> raw : {}\n".format(raw),
                 "  -> file: {}\n".format(fname),
                 "  -> path: {}\n".format(path)]
           if raw:
               msg.append("  -> Bads: {}".format(str(raw.info.get('bads'))))
           logger.debug("".join(msg) )

        if raw:
           fname = None
           if reset_bads:
              if "bads" in raw.info:
                 raw.info["bads"] = []
           if not reload_raw:
              return raw ,self.get_raw_filename(raw)

        if fname:
           fn = self.expandvars( fname )
           if path:
              path = self.expandvars(path)
              fn   = os.path.join(path,fn)
        else:
           fn = self.get_raw_filename(raw)
        
        if not fn:
           logger.error("ERROR no such file or raw-object:\n  -> raw obj: {}\n  -> fname: {}\n  -> path : {}".format(raw,fname,path))
           return None,None
        try:
            if not os.path.isfile(fn):
               raise FileNotFoundError("ERROR no file found: {}".format(fn))
            
            if ( fn.endswith(self.brainvision_extention) ):
               # --- changed in mne version 019.dev
               # raw = mne.io.read_raw_brainvision(fn,response_trig_shift=self.brainvision_response_shift,preload=preload)
               raw = mne.io.read_raw_brainvision(fn,preload=preload)
               #raw.info['bads'] = []
            elif (fn.endswith(self.ica_extention)):
                raw = mne.preprocessing.read_ica(fn)
            else:
               raw = mne.io.Raw(fn,preload=preload)
    
            if not raw:
               raise FileNotFoundError("ERROR could not load RAW object: {}".format(fn))
        except:
            logger.exception("---> could not get raw obj from file:\n --> FIF name: {}\n  -> file not exist".format(fn))
            return None,None
        
        if reset_bads:
           if "bads" in raw.info:
              raw.info["bads"] = []
           logger.debug("  -> resetting bads in raw")
           
        return raw,fn #self.get_raw_filename(raw)

    def get_files_from_list(self, fin):
        """ get filename or filenames from a list
        Parameters
        ----------
        filename or list of filenames
        
        Results
        -------
        files as iterables lists 
        """
        if isinstance(fin, list):
           fout = fin
        else:
           if isinstance(fin, str):
              fout = list([ fin ]) 
           else:
              fout = list( fin )
        return fout

    def get_filename_list_from_file(self, fin, start_path = None):
        """ loads filenames with options / parameter form a text file
        
        Parameters
        ----------
        fin       : text file to open
        start_path: start dir <None>
        
        Results
        --------
        list of existing files with full path and dict with bad-channels (as string e.g. A132,MEG199,MEG246)
        
        Example
        --------
        txt file format:
          
          fif-file-name  --feeg=myeeg.vhdr --bads=MEG1,MEG123 --startcode=123
          e.g.:
          0815/M100/121130_1306/1/0815_M100_121130_1306_1_c,rfDC-raw.fif --bads=A248
          0815/M100/120920_1253/1/0815_M100_120920_1253_1_c,rfDC-raw.fif
          0815/M100/130618_1347/1/0815_M100_130618_1347_1_c,rfDC-raw.fif --bads=A132,MEG199

        call with program:
        jumeg_merge_meeg -smeg /data/meg_stroe1/exp/INTEXT/eeg/INTEXT01/ -seeg /data/meg_stroe1/exp/INTEXT/mne/ -plist /data/meg_stroe1/exp/INTEXT/doc/ -flist 207006_merge_meeg.txt -sc 5 -b -v -r
         where -flist 207006_merge_meeg.txt is the text file
        """
        found_list = []
        # bads_dict  = dict()
        opt_dict = dict()
        
        assert os.path.isfile(fin),"ERROR no list-file found: %s" %( fin ) 
           
        try:
            fh = open( fin )
        except:
            assert "---> ERROR no such file list: " + fin
                
        for line in fh :
            line  = line.strip()
            fname = None
            if line :
               if ( line[0] == '#') : continue
               opt = line.split(' --')
               fname = opt[0].strip()
               if start_path :
                  if os.path.isfile( start_path + "/" + fname ):
                     fname = start_path + "/" + fname
               #print "Fname: "+fname

               if os.path.isfile( fname ):
                  found_list.append(fname)
                  #print "found Fname: "+fname
   
               opt_dict[fname]= {}
               for opi in opt[1:]:
                   opkey,opvalue=opi.split('=')
                   if opkey:
                      opt_dict[fname][ opkey.strip() ] = opvalue
                        
                        # if ('--bads' in opi):
                        #     _,opt_dict[fname]['bads'] = opi.split('--bads=')
                        # if ('--feeg' in opi):
                        #     _,opt_dict[fname]['feeg'] = opi.split('--feeg=')
        try:           
            fh.close()
        except:
            logger.exception("  -> UP`s error: can not close list-file:" +fin,exc_info=True)
        
        if self.verbose :
           logger.info(self.pp_list2str(
               [" --> INFO << get_filename_list_from_file >> Files found: %d" % ( len(found_list) ),
                found_list,"\n --> BADs: ",opt_dict,"\n"]) )

        return found_list,opt_dict

    def add_channel(self,raw,ch_name=None,ch_type=None,data=None):
        """
        Adds a channel to raw obj, works in place.

        Parameters
        ----------
         mne.io.Raw obj
         ch_name : Name of the channel to add <None>
         ch_type : channel type from mne-type
                   e.g: get channel type from eeg raw-obj.info
                        channel_type = mne.io.pick.channel_type(eeg_raw.info,channel_idx )
         data    : channel data to add as numpy array <None>

        Returns
        -------
         mne.io.Raw obj with new channel
        """

        if ch_type not in self.picks.pick_type_set:
           ch_type='misc'

        picks = self.picks.labels2picks(raw, labels=ch_name)

        if not isinstance(data,np.ndarray):
           data = np.zeros(raw.n_times)

      #--- if channel does not exist add new channel
        if picks.shape[0] == 0:
           info = mne.create_info([ch_name],raw.info['sfreq'],[ch_type])
           if len(data.shape) < 2:
              ch_raw = mne.io.RawArray(data.reshape(1, -1),info)
           else:
              ch_raw = mne.io.RawArray(data,info)
           raw.add_channels([ch_raw],force_update_info=True)

        else: #--- if channel already exists copy data keep raw-channel-dtype:
           data.dtype = raw._data[ picks[0] ].dtype
           raw._data[picks[0],:] = data

          # channel_type = mne.io.pick.channel_type(raw.info, 75)

    def update_and_save_raw(self,raw,fin=None,fout=None,save=False,overwrite=True,postfix=None,separator="-",update_raw_filenname=False):
        """
        new filename from fin or fout with postfix
        saving mne raw obj to fif format

        Parameters
        ----------
         raw       : raw obj
         fout      : full output filename if set use this
         fin       : input file name <None>
         postfix   : used to generate <output filename> from <input file name> and <postfix>
                     e.g.:
         separator : split to generate output filename with postfix  <"-">
         save      : save raw to disk
         overwrite : if overwrite  save <raw obj> to existing <raw file>  <True>
         update_raw_filenname: <False>

        Returns
        --------
         filename,raw-obj
        """
        from distutils.dir_util import mkpath
        #--- use full filname
        if fout:
           fname = os.path.expandvars(os.path.expanduser(fout))
        #--- make output file name
        else:
            fname = fin if fin else raw.filenames[0]
            fname = os.path.expandvars(os.path.expanduser(fname))  # expand envs
            if postfix:
               fpre,fext = fname.rsplit(separator)
               fname = fpre + "," + postfix + separator + fext
      
        if not raw:
           return fname,raw
        
        if save:
            try:
                if (os.path.isfile(fname) and (not overwrite)):
                    logger.info(" --> File exist => skip saving data to : " + fname)
                else:
                    logger.info(">>>> writing data to disk...\n --> saving: " + fname)
                    mkpath(os.path.dirname(fname))
                    raw.save(fname,overwrite=True)
                    logger.info(' --> Bads:' + str(raw.info['bads']) + "\n --> Done writing data to disk...")
            except:
                logger.exception("---> error in saving raw object:\n  -> file: {}".format(fname))
    
        if update_raw_filenname:
           self.set_raw_filename(raw,fname)
        return fname,raw

    def apply_save_mne_data(self,raw,fname=None,overwrite=True):
        """saving mne raw obj to fif format
        
        Parameters
        ----------
        raw      : raw obj
        fname    : file name <None>
        overwrite: <True>
        
        Returns
        ----------
        filename
        """
        from distutils.dir_util import mkpath
        if not fname:
           fname = raw.filenames[0]
        
        fname = os.path.expandvars(fname)  # expand envs
        try:
           if ( os.path.isfile(fname) and ( not overwrite) ) :
              logger.info(" --> File exist => skip saving data to : " + fname)
           else:
              logger.info(">>>> writing data to disk...\n --> saving: "+ fname)
              mkpath( os.path.dirname(fname) )
              raw.save(fname,overwrite=True)
              logger.info(' --> Bads:' + str( raw.info['bads'] ) +"\n --> Done writing data to disk...")
        except:
           logger.exception("---> error in saving raw object:\n  -> file: {}".format(fname))
           
        return fname

    def get_empty_room_fif(self,fname=None,raw=None, preload=True):
        """find empty room file for input file name or RAWobj 
        assuming <empty room file> is the last recorded file for this id scan at this specific day
        e.g.: /data/mne/007/M100/131211_1300/1/1007_M100_131211_1300_1_c,rfDC-raw.fif
        search for id:007 scan:M100 date:13:12:11 and extention: <empty.fif>
        
        Parameters
        ----------
        raw    : raw obj
        fname  : file name <None>
        preload: will load and return empty-room-raw obj instead of raw <True>
        
        Returns
        ---------
        full empty room filename, empty-room-raw obj or raw depends on preload option
        """
        import glob

        fname_empty_room = None
       
        if raw is not None:
           fname = self.get_raw_filename(raw)
         #--- first trivial check if raw obj is the empty room obj   
           if fname.endswith('epmpty.fif'):
              return(fname,raw)
               
         #--- ck if fname is the empty-room fie  
        if fname.endswith('epmpty.fif'):
           fname_empty_room = fname  
         #--- ok more difficult lets start searching ..
        else : 
            # get path and pdf (in memory of 4D filenames) from filename
           p,pdf = os.path.split(fname)
            # get session dat from file
           session_date = pdf.split('_')[2]

            # get path to scan from p and pdf
           path_scan    = p.split( session_date )[0]
         
         #--- TODO:
            # may check for the latest or earliest empty-room file
            #  make empty-room extention/pattern as  cls property
            # use jumeg.gui.util.jumeg_gui_utils_pdfs
            
           try:
               fname_empty_room = glob.glob( path_scan + session_date +'*/*/*-empty.fif' )[0]
           except:
               logger.exception("---> can not find empty room file: " + path_scan + session_date,exc_info=True)
               return

        if fname_empty_room and preload:
           if self.verbose:
              logger.info("\n --> Empty Room FIF file found: %s \n" % (fname_empty_room))
           
           return self.get_raw_obj(fname_empty_room,raw=None,preload=True) # return raw,fname

           # return( fname_empty_room, mne.io.Raw(fname_empty_room, preload=True) )
   
#---
jumeg_base       = JuMEG_Base_IO()
jumeg_base_basic = JuMEG_Base_Basic()

