#!/usr/bin/env python3
# -+-coding: utf-8 -+-

"""
"""

#--------------------------------------------
# Authors: Frank Boers <f.boers@fz-juelich.de> 
#
#-------------------------------------------- 
# Date: 16.09.19
#-------------------------------------------- 
# License: BSD (3-clause)
#--------------------------------------------
# Updates
#--------------------------------------------

import os,sys,contextlib,argparse,time
import numpy as np
import ctypes

from numba import jit


# http://pyopengl.sourceforge.net/documentation/opengl_diffs.html
#--- OGL debug / logging SLOW !!!
import OpenGL
#OpenGL.ERROR_CHECKING = False
#OpenGL.ERROR_LOGGING = False
#--- OGL debug / logging SLOW !!!
#OpenGL.FULL_LOGGING = False  # True

#OpenGL.ERROR_ON_COPY = True

# coding: utf-8
from OpenGL.GL import *
#from OpenGL.GL import shaders

#import OpenGL.GL as gl
# import OpenGL.arrays.vbo as glvbo

#from OpenGL.GLU import *
from OpenGL.GLUT import *


import logging
from jumeg.base import jumeg_logger

from jumeg.gui.tsv.ogl.jumeg_tsv_ogl_glsl import JuMEG_TSV_OGL_Shader_PlotSignal

logger = logging.getLogger('jumeg')

__version__ = "2019-09-16-001"

'''
ToDo implement sensorlayout

#print "########## Refchan geo data:"
# This is just for info to locate special 4D-refs.
#for iref in refpick:
#    print raw.info['chs'][iref]['ch_name'],
#raw.info['chs'][iref]['loc'][0:3]
'''