#!/usr/bin/env python3
# -+-coding: utf-8 -+-

"""
ToDo
this cls is under construction
store all GUI MSG in a CLS or in a dict
for unique comunication

CLS for mesages [info,warnings,exceptions,error]
 <name> <msg identifier offset>  text
 Info   :    MSG 100000 This is a  msg
 Warning:    MSG 200000 This is a  warning
 Exceptions: MSG 300000 This is an exception
 Error  :    MSG 40000  This is an error
"""

#--------------------------------------------
# Authors: Frank Boers <f.boers@fz-juelich.de> 
#
#-------------------------------------------- 
# Date: 20.12.18
#-------------------------------------------- 
# License: BSD (3-clause)
#--------------------------------------------
# Updates
#--------------------------------------------

'''
msg:{"info":{ "1000001": "OK",
              "1000002": "OK",
            },

"warning":{ "2000001": "OK",
            "2000002": "OK",
            },
"exception":{ "3000001": "OK",
              "3000002": "OK",
            },
"error"    :{ "4000001": "OK",
              "4000002": "OK",
            },

'''

class JuMEG_MessagesBase(object):
    """
    base cls for defining messages
    """
    #__slots__ = ("name")
    def __init__(self,name="Message",**kwargs):
        self.name = name
        self._msgs= []
        self._init(**kwargs)
    
    def msg(self,index):
        if self._msgs:
           return self._msgs[index]
        
    def _update_from_kwargs(selfself,**kwargs) :
        pass
    
    def _init(self,**kwargs):
        self._update_from_kwargs(**kwargs)
        