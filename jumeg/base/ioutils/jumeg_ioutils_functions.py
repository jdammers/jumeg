#!/usr/bin/env python3
# -+-coding: utf-8 -+-

"""
"""

#--------------------------------------------
# Authors: Frank Boers <f.boers@fz-juelich.de> 
#
#-------------------------------------------- 
# Date: 03.01.19
#-------------------------------------------- 
# License: BSD (3-clause)
#--------------------------------------------
# from jumeg.ioutils.jumeg_ioutils_functions import JuMEG_IOUtils
#
#--------------------------------------------
# Updates
#--------------------------------------------

import os,sys

# logging
# logger = logging.getLogger('jumeg')

__version__ = "2019.05.14.001"

class JuMEG_IOUtils(object):
    """
    helper cls for io
    
    Example:
    --------
    from jumeg.ioutils.jumeg_ioutils_functions import JuMEG_IOUtils
    IOUtils = JuMEG_IOUtils()
    path = IOUtils.expandvars('${JUMEG_PATH_TEMPLATE_EXPERIMENTS}')
    
    import os
    os.environ["TEST"]="ThisIsATest"
    path = IOUtils.expandvars('${TEST}')
    >>'ThisIsATest'
    
    IOUtils.make_dirs_from_list(stage=os.cwd(),path_list=["a","b","c"]
    
    """
    
    def __init__(self,**kwargs):
        #self._permisssions = stat.S_IRWXU|stat.S_IRWXU
        self._update_kwargs(**kwargs)
    
    def _update_kwargs(self,**kwargs):
        pass
    
    def expandvars(self,v):
        """
        expand env's from string works on list or string
        :param v: list of strings  or string
        :return: input with expanded env's
        """
        if isinstance(v,(list)):
           for i in range(len(v)):
               v[i] = os.path.expandvars( os.path.expanduser(v[i]) )
           return v
        
        return os.path.expandvars( str(v) )
        
    def isDir(self,v):
        """
        expand env's e.g. expand user from string
        check if is dir
        
        :param v: string
        :return: input with expanded env's or None
        """
        v=self.expandvars(v)
        if os.path.isdir(v): return v
        return
        
    def make_dirs_from_list(self,stage=None,path_list=None):
        """
        make a dirs from list
        can handle env`s
        add <stage> as start path if stage else use current dir as start path
        :param stage:
        :param path_list:
        :return:
         list of created directories
        """
        if stage:
           start_path = stage
        else:
           start_path = os.getcwd()
        l=[]
        for p in path_list:
            path = os.path.expandvars(start_path+"/"+p)
            #print("Make DirTree: {}".format(path))
            os.makedirs(path,mode=0o777,exist_ok=True)
            #os.chmod(path,self._permissions)
            l.append(path)
        return l