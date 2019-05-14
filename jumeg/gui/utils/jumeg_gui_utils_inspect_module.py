#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 10:08:17 2018

@author: fboers

---> update 03.04.2019 FB
 --> user logger from logging avoid print
"""

import os,sys,glob,re,textwrap
import ast
import logging
from types import ModuleType

from pathlib import Path

import numpy as np

from jumeg.base.jumeg_base import JuMEG_Base_Basic
jb = JuMEG_Base_Basic()

logger = logging.getLogger('jumeg')
__version__="2019.05.14.001"

class JuMEG_Utils_Inspect_ModuleBase(object):
    """
    
     Paremeters:
     -----------    
      name     : <None>   
      prefix   : <jumeg>  
      fullfile : <None>
      extention: <.py>
      function : <get_args>
      import   : <None>
      
      verbose  : <False>
     
     Example:
     --------
      command/module: my-path/jumeg/jumeg_test01.py     
      name          : jumeg_test01
      fullfile      : my-path/jumeg/jumeg_test01.py
     
      command       : dict of {"name":None,"prefix":"jumeg","fullfile":None,
                               "extention":".py","function":"get_args","package":None}   
    """
    def __init__(self,**kwargs):
        super().__init__()
        self.__command  = {"name":None,"prefix":"jumeg","fullfile":None,"extention":".py","function":"get_args","package":None} 
        self.__isLoaded = False
        self._update_kwargs(**kwargs)
        
    @property
    def command(self): return self.__command
    
    @property
    def name(self)  : return self.__command["name"]
    @name.setter
    def name(self,v): self.__command["name"]=v
    
    @property
    def module(self)  : return self.__command["name"]
    @module.setter
    def module(self,v): self.__command["name"]=v
   
    @property
    def prefix(self)  : return self.__command["prefix"]
    @prefix.setter
    def prefix(self,v): self.__command["prefix"]=v
   
    @property
    def package(self)    : return self.__command["package"]
    @package.setter
    def pachkage(self,v)  : self.__command["package"]=v
   
    @property
    def fullfile(self)  : return self.__command["fullfile"]
    @fullfile.setter
    def fullfile(self,v): 
        self.__isLoaded = False
        self.__command["fullfile"]=v
    
    @property        
    def function(self)  : return self.__command["function"]
    @function.setter        
    def function(self,v): 
        self.__isLoaded = False
        self.__command["function"]=v
   
    @property        
    def extention(self)  : return self.__command["extention"]
    @extention.setter        
    def extention(self,v): self.__command["extention"]=v
  
    @property
    def import_name(self): return self.prefix +"."+self.name
   
   #----- 
    def update(self,**kwargs):
        self._update_kwargs(**kwargs)
           
    def _update_kwargs(self,**kwargs):
        for k in self.__command.keys():
            self.__command[k] = kwargs.get(k,self.__command[k])
   
    def info(self):
        logger.info( jb.pp_list2str(self.command,head="JuMEG Function Command"))

class JuMEG_UtilsIO_Inspect_Module_FunctionFromText(JuMEG_Utils_Inspect_ModuleBase):
    """
    special issue
    extract a function from a text file e.g. <get_arg> in <jumeg_filter.py>
    and compile it in a new module e.g. for argparser gui
    avoiding/excluding <import>  and imports of dependencies 
    
    Parameters
    -----------       
     name     : <None>   
     prefix   : <jumeg>  
     fullfile : <None>
     extention: <.py>
     function : <get_args>
     import   : <None>
      
     start_pattern: <def >
     stop_pattern : <return >
      
     verbose  : <False>
     
    Example
    -------
     from jumeg.gui.utils.jumeg_gui_utils_io import JuMEG_UtilsIO_FunctionFromText
     JFT          = JuMEG_UtilsIO_FunctionFromText()
     JFT.fullfile = os.environ["JUMEG_PATH"]+"jumeg/filter/jumeg_filter.py"
     JFT.function = "get_args"
     opt,parser   = JFT.apply() 

     parser.print_help()

    """
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.__command = {"name":None,"fullfile":None,"extention":".py","function":"get_args","start_pattern":"def","stop_pattern":"return" }
        self.__text     = None
        self.__isLoaded = False
        self._update_kwargs(**kwargs)
   
    @property
    def function_text(self)  : return self.__text
   
    @property
    def isLoaded(self)  : return self.__isLoaded
    
    @property        
    def start_pattern(self)  : return self.__command["start_pattern"]
    @start_pattern.setter        
    def start_pattern(self,v): self.__command["start_pattern"]=v
   
    @property        
    def stop_pattern(self)  : return self.__command["stop_pattern"]
    @stop_pattern.setter        
    def stop_pattern(self,v): self.__command["stop_pattern"]=v
    
    def _open_txt(self):
        """ open module-file (.py) """
        lines = None
        #print("Function File: "+self.fullfile)
        
        with open (self.fullfile,"r+") as farg:
             lines=farg.readlines() 
             farg.close()
        return lines     
     
    def _get_function_code_from_text(self,**kwargs):
        """  extract the function text from module"""
        self._update_kwargs(**kwargs)
        idx = 0
        lines        = None
        l_idx_start  = None
        l_idx_end    = None
        l_intend_pos = None
        self.__text  = None
        
       #--- read in module as txt 
        lines = self._open_txt()
        if not lines : return 
        
       #--- mk mathch pattern 
        find_start_pattern = re.compile(self.start_pattern +'\s+'+ self.function)
       #--- find module start index  
        for l in lines:
            if find_start_pattern.match(l):
               l_idx_start = idx
               break
            idx += 1                 
       #--- find intend pos first char
        idx = l_idx_start
        for l in lines[l_idx_start+1:-1]:
            idx += 1
            if not len(l.lstrip() )     :  continue
            if l.lstrip().startswith("#"): continue
            l_intend_pos = len(l) - len(l.lstrip()) # line pos of return
            break
         
       #--- find module end  
        idx = l_idx_start
        for l in lines[l_idx_start+1:-1]:
            idx += 1
            if ( l.find( self.stop_pattern) ) == l_intend_pos:
               l_idx_end=idx
               break
        self.__text = "import sys,os,argparse\n" +''.join( lines[l_idx_start:l_idx_end+1] )
        lines=[]
        return self.__text 

    def _clear(self):
        """
        clear function module space
        """
        self.__isLoaded = False
        
    def load_function(self):
        """
         load function from source code (text file)
         create a new module and execute code
         
         Result
         -------
         return function results
         
        """
        if self.isLoaded: self._clear()    
        self.__isLoaded = False
       
        if not self._get_function_code_from_text(): return
       
        self.__ModuleType = ModuleType(self.function)
        sys.modules[self.function] = self.__ModuleType
        exec( textwrap.dedent( self.function_text), self.__ModuleType.__dict__)
        self.__isLoaded = True
   
    def apply(self):
        """
         if not loaded load function from source code (text file)
         and create a new module and execupte code
         execute function
         
         Result
         -------
         return function results
        """    
        if not self.isLoaded:
           self.load_function()
        if self.isLoaded:  
           # print("OK loaded {}".format(sys.modules[self.function] )) 
           return getattr(self.__ModuleType,self.function)() #opt,parser
                  
        return None
    
       
class JuMEG_UtilsIO_Inpect_Module(object):
    """
    CLS find jumeg modules under jumeg sub folders in PYTHONOATH 
    
    Parameters:
    -----------    
    stage_prefix : jumeg stage start of jumeg directory : <jumeg> 
    prefix       : file prefix     <None>
    postfix      : file postfix    <None>
    extention    : file extention  <".py>
    permission   : file permission <os.X_OK|os.R_OK>
    function     : function name in module <get_args>
    
    Results:
    --------
    list of executable modules with fuction < get_args> in jumeg PYTHONPATH
        
    """ 
    def __init__(self,stage_prefix="jumeg",prefix="jumeg",postfix=None,extention=".py",permission=os.X_OK|os.R_OK,function="get_args",**kwargs):  
        super().__init__(**kwargs)
      
        self.stage_prefix  = stage_prefix
        self.prefix        = prefix
        self.postfix       = postfix
        self.extention     = extention
        self.permission    = permission
        self.function      = function
        self.skip_dir_list = ["old","gui","wx"]
        self._module_list  = [] 
        self._jumeg_path_list=[]   
        
    @property
    def module_list(self): return self._module_list
    @property
    def jumeg_path_list(self): return self._jumeg_path_list  
    
    def PathListItem(self,idx):
        return os.path.split( self.module_list[idx] )[0]
  
    def _is_function_in_module(self,fmodule,function=None):
        """
        https://stackoverflow.com/questions/45684307/get-source-script-details-similar-to-inspect-getmembers-without-importing-the
   
        """
        if function:
           self.function=function
        if not self.function: return  
        
        try:
            mtxt = open(fmodule).read()
            tree = ast.parse(mtxt)
            for fct in tree.body:
                if isinstance( fct,(ast.FunctionDef)):
                   if fct.name == self.function:
                      return True
        except:
            pass
        
    def get_jumeg_path_list_from_pythonpath(self):
        """ jumeg """
       
        self._jumeg_path_list=[]
        l=[]
        for d in os.environ["PYTHONPATH"].split(":"):
            if os.environ.get(d):
               d = os.environ.get(d)
            if d == ".":
               d = os.getcwd() 
            if os.path.isdir( d + "/" + self.stage_prefix):
               l.append( d + "/" + self.stage_prefix )
    
        self._jumeg_path_list = list( set(l) ) # exclude double
        self._jumeg_path_list.sort()
        return self._jumeg_path_list 
        
    
    def ModuleListItem(self,idx):
        """  jumeg.my subdirs.<jumeg_function name>"""
      
        if not len(self.module_list): return
        p,f = os.path.split( self.module_list[idx])
        m = p.split("/"+self.stage_prefix + "/")[-1] + "/" + os.path.splitext(f)[0]
        return m.replace("/",".")
    
    def ModuleFileName(self,idx):
        """ 
        Parameters:
        -----------
        index
        """
        return self._module_list[idx]
    
    def ModuleNames(self,idx=None):
        """
        get module name from file list
        Parameters:
        -----------
        idx: if defined return only this filename from list <None>
             else return list of filenames
        """   
        if jb.is_number(idx):
           return os.path.basename( self._module_list[idx] ).replace(self.extention,"")
        l=[]
        for f in self.module_list:
            l.append(os.path.basename(f).replace(self.extention,""))
        return l    
           
        #--- get_path_and_file
        # p,f=os.path.split( self.file_list[idx] )
    def FindModules(self,**kwargs):
        """
        find modules /commands under jumeg with defined permissions
        
        Parameters:
        -----------    
        stage_prefix: <jumeg">
        prefix      : <jumeg>
        postfix     : <None>
        extention   : <.py> 
        permission  : <os.X_OK|os.R_OK>
        function    : <get_args>
      
        Results:
        ---------
        list of module names 
        """
        self._walk(**kwargs)
        return self.ModuleNames()
    
    def update(self,**kwargs):
        """ """
        self.stage_prefix = kwargs.get("stage_prefix",self.stage_prefix)
        self.prefix       = kwargs.get("prefix",self.prefix)
        self.postfix      = kwargs.get("postfix",self.postfix)
        self.extention    = kwargs.get("extention",self.extention)
        self.permission   = kwargs.get("permission",self.permission)
        self.function     = kwargs.get("function",self.function)
        self._module_list = [] 
        self._jumeg_path_list = self.get_jumeg_path_list_from_pythonpath()
               
    def _walk(self,**kwargs):  
        """
        search recursive for executable modules
        
        Parameters:
        -----------
        stage_prefix : stage prefix <jumeg>
        prefix    : file prefix     <jumeg>
        postfix   : file postfix    <None>
        extention : file extention  <".py>
        permission: file permission <os.X_OK|os.R_OK>
            
        Results:
        --------     
        list of executable modules, full filename
        """
        self.update(**kwargs)
         
        skip_dir_set = set(self.skip_dir_list)
        for p in  ( self._jumeg_path_list ):
            for root, dirs, files in os.walk(p):
                if (set(root.split(os.sep)) & skip_dir_set): continue
                for f in files:
                    if self.prefix:
                       if not f.startswith(self.prefix): continue
                    if self.extention:
                       if not f.endswith(self.extention): continue
                    if os.access(root+"/"+f,self.permission): 
                       fmodule = os.path.join(root,f)
                       if self._is_function_in_module(fmodule):
                          self._module_list.append(fmodule) 
                     
        self._module_list.sort()
        return self._module_list
 