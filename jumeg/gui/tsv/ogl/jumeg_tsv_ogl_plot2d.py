#!/usr/bin/env python3
# -+-coding: utf-8 -+-

# https://cyrille.rossant.net/shaders-opengl/

# PyQt4 imports
#from PyQt4 import QtGui, QtCore, QtOpenGL
#from PyQt4.QtOpenGL import QGLWidget

# PyOpenGL imports

import os,sys,contextlib,argparse,time
import numpy as np
import ctypes

from numba import jit

#from scipy import signal
#import matplotlib.pyplot as plt

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

__version__ = "2019-09-13-001"



#@jit(nopython=True)
def ortho(left,right,bottom,top,znear,zfar):
    assert (right != left)
    assert (bottom != top)
    assert (znear != zfar)
    
    M = np.zeros((4,4),dtype=np.float32)
    M[0,0] = +2.0 / (right - left)
    M[3,0] = -(right + left) / float(right - left)
    M[1,1] = +2.0 / (top - bottom)
    M[3,1] = -(top + bottom) / float(top - bottom)
    M[2,2] = -2.0 / (zfar - znear)
    M[3,2] = -(zfar + znear) / float(zfar - znear)
    M[3,3] = 1.0
    return M


@jit(nopython=True)
def ortho2D(left,right,bottom,top):
    assert (right != left)
    assert (bottom != top)
    
    M = np.zeros((4,4),dtype=np.float32)
    M[0,0] = +2.0 / (right - left)
    M[3,0] = -(right + left) / float(right - left)
    M[1,1] = +2.0 / (top - bottom)
    M[3,1] = -(top + bottom) / float(top - bottom)
    M[2,2] = -2.0
    M[3,2] = -1.0
    M[3,3] = 1.0
    return M

#@jit(nopython=True)
def _viewport_matrix(n_cols,n_plts,w,h,bw,bh):
    """
    up to dowm
    calc viewport matrix for all plots
    :param n_cols: number of cols
    :param n_plts: number of plts
    :param w     : width in pixel
    :param h     : height in pixel
    :param bw    : border width
    :param bh    : border height
    :return      : viewport matrix
    """

    n_rows = int(n_plts / n_cols)
    #n_rows = n_plts * n_cols
    plt_w  = (w / n_cols) - 2.0 * bw
    plt_h  = (h / n_rows) - 2.0 * bh
    
    
    if plt_h < 1.0:
       plt_h = (h / n_rows)
       bh    = 0.0
    if plt_h < 1.0:
       plt_h = 1.0

    if plt_w < 1.0:
       plt_w = (h / n_cols)
       bw = 0.0
    if plt_w < 1.0:
       plt_w = 1.0

    vpm = np.zeros((n_plts,4),dtype=np.float32)
    mat = np.zeros((n_rows,n_cols),dtype=np.float32)
    mat += np.arange(n_cols)

    #--- x0 pos Up to Down MEG001 -> MEG010
    vpm[:,0] = bw + mat.T.flatten() * (plt_w + bw)
   
   #--- y0 pos
    mat = np.zeros((n_cols,n_rows))
    mat += np.arange(n_rows)
   
   #-- reverse ypos -> plot ch0 to upper left
    vpm[:,1]= bh + mat[:,-1::-1].flatten() * (plt_h + 2 * bh)
    #vpm[:,1]= bh + mat[::-1].flatten() * (plt_h + 2 * bh)
    #vpm[:,1] += bh + mat.flatten() * (plt_h + 2 * bh)
    vpm[:,2] = plt_w
    vpm[:,3] = plt_h
    return vpm


def viewport_matrix(n_cols,n_plts,w,h,bw,bh):
    """
    calc viewport matrix for all plots
    plot form left column to right column
    
    :param n_cols: number of cols
    :param n_plts: number of plts
    :param w     : width in pixel
    :param h     : height in pixel
    :param bw    : border width
    :param bh    : border height
    :return      : viewport matrix
    """
    
    n_rows = int(n_plts / n_cols)
    #n_rows = n_plts * n_cols
    plt_w = (w / n_cols) - bw
    plt_h = (h / n_rows) - 2*bh
    
    if plt_h < 1.0:
        plt_h = (h / n_rows)
        bh = 0.0
    if plt_h < 1.0:
        plt_h = 1.0
    
    if plt_w < 1.0:
        plt_w = (h / n_cols)
        bw = 0.0
    if plt_w < 1.0:
        plt_w = 1.0
    
    matx = np.zeros((n_rows,n_cols),dtype=np.float32)
    matx += np.arange(n_cols)

    vpm = np.zeros((n_plts,4),dtype=np.float32)
    vpm[:,0] = bw + matx.flatten() * (plt_w + bw)  #xpos
    vpm[:,2] = plt_w
    vpm[:,3] = plt_h
    
    maty = np.zeros((n_cols,n_rows),dtype=np.float32)
    maty[0:,:] = np.arange(n_rows)
    matyT = maty.T.flatten()
    vpm[:,1]=bh + matyT[::-1] * (plt_h + 2 * bh) # ypos
    
    return vpm
    
@jit
def draw_text(txt,x,y,c,f):
    glColor4f(c[0],c[1],c[2],c[3])   #0.0,1.0,0.0,1.0)
    #glMatrixMode(GL_PROJECTION)
    #glLoadIdentity()
    #gluOrtho2D(0.0,1.0,0.0,1.0)
    #glMatrixMode(GL_MODELVIEW)
    #glRasterPos2f(x,y)
    glWindowPos2d(x,y)
    #glutBitmapString(GLUT_BITMAP_TIMES_ROMAN_24,txt.encode("utf-8"))
    glutBitmapString(f,txt.encode("utf-8"))
    #for idx_chr in str(txt):
    #    glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18,ord(str(idx_chr)))


class Grid(object):
    __slots__= ["colour","backgroundcolour","trafomatrix","xstart","xend","xstep","ystart","yend","ystep","data_status","_data","_xdiv","_ydiv"]
    def __init__(self,**kwargs):
        self.colour           = np.array([0.8,0.8,0.8,0.3],dtype=np.float32)
        self.backgroundcolour = np.array([0.9,0.9,0.9,0.2],dtype=np.float32)
        self.trafomatrix = ortho2D(-1.0,1.0,-1.0,1.0)
        self.xstart = -1.0
        self.xend   =  1.0
        self.xstep  =  0.1
        self.ystart = -1.0
        self.yend   =  1.0
        self.ystep  =  0.25
        
        self.update(**kwargs)
    @property
    def ydiv(self): return self._ydiv

    @property
    def xdiv(self):
        return self._xdiv

    @property
    def data(self): return self._data
    @data.setter
    def data(self,v):
        self._data=v
        self.data_status = False
        
    def _update_from_kwargs(self,**kwargs):
        for k in kwargs:
            if k in self.__slots__:
               self.__setattr__(k,kwargs.get(k))  # ,self.__getattribute__(k)))  can set 0.0 & None
  
    def _update_trafomatrix(self,matrix=None):
        if matrix:
           self.trafomatrix = matrix
        else:
           self.trafomatrix = ortho2D(-1.0,1.0,-1.0,1.0)
        
    def _update_data(self,**kwargs):
        """
        calc grid vertices in OGL coordinates
        (-1, 1) -------- ( 1,1)
               |       |
               |       |
        (-1,-1) -------- (1,-1)
        
        :param xstart:
        :param xend:
        :param xstep:
        :param ystart:
        :param yend:
        :param ystep:
        :param color:
        :param matrix:
        :return:
        """
        self._update_from_kwargs(**kwargs)
        # self._update_trafomatrix()
        
        # ystep =  self.scale_factor_mm[1] /self.size_in_pixel[1]
        
        x = np.arange(self.xstart,self.xend + self.xstep,self.xstep)
        #x = np.round(x,2)

        y = np.arange(self.ystart,self.yend + self.ystep,self.ystep)

        grid = np.zeros( [ x.shape[0] + y.shape[0] ,4 ],dtype=np.float32)
        
        vlines = grid[0:x.shape[0],:]
        hlines = grid[x.shape[0]:,:]

        vlines[:,0] =  x
        vlines[:,2] =  x
        vlines[:,1] = self.xstart
        vlines[:,3] = self.xend

        hlines[:,0] = self.ystart
        hlines[:,1] = y
        hlines[:,2] = self.yend
        hlines[:,3] = y
      
        self._xdiv  = vlines.shape[0]-1
        self._ydiv  = hlines.shape[0]-1
      
        self._data = grid.flatten()
    
        #logger.info("---> grid data to plot\n"+
        #            "  -> data   : {}\n".format(self._data.shape)+
        #            "  -> colour : {}".format(self.colour))

    def SetBackgroundColour(self):
        glColor4f(self.backgroundcolour[0],self.backgroundcolour[1],self.backgroundcolour[2],self.backgroundcolour[3] )
        
    def update(self,**kwargs):
        self._update_from_kwargs(**kwargs)
        self._update_data()
    
    def OnChangeData(self):
        pass
    
    
class Signals(object):
    __slots__=["srate","freq", "timepoints_status","data_status","time_start","picks_status",
               "_picks","_picks_length","_labels","_data","_data_length","_colours","_tp_data","_tp_length","_dcoffset",
               "_scale","_glshader"]
    # "_scale_div","_scale_min","_scale_max","_scale_min_abs","_scale_max_abs"
    
    def __init__(self,**kwargs):

        self._picks   = np.array([])
        self._picks_length = None
        self._labels       = []
        self.srate   = 1000.0
        self.freq    = 1.0
        self.time_start = 0.0
        
        self._data        = np.array([])
        self._data_length = None
        
        self._colours   = np.array([[1.0,0.0,0.0,0.8]],dtype=np.float32)
       
        
        self._tp_data  = np.array([])
        self._tp_length= None
        self._glshader = None

        self._dcoffset     = np.array([],dtype=np.float32)
        self._scale        = np.array([],dtype=np.float32)
       
        self._update_from_kwargs(**kwargs)
    
    """
      ToDo  update colors
      if self.signals.colors.size:
           self._colors=colors
        else:
           self._colors= np.ones( (self.data.shape[0],4),dtype=np.float32 )
           self._colors[:,0:2] = 0.0
       
    """

    @property
    def dcoffset(self):
        return self._dcoffset

    @dcoffset.setter
    def dcoffset(self,v):
        self._dcoffset = v

    @property
    def scale(self): return self._scale
    @scale.setter
    def scale(self,v):
        self._scale = v
    @property
    def samples(self): return self._data.shape[-1]
    @property
    def labels(self): return self._labels
    @labels.setter
    def labels(self,v):
        self._labels =v
  
    @property
    def picks(self): return self._picks
    @picks.setter
    def picks(self,v):
        self._picks=v
        if self._picks_length != self._picks.shape[-1]:
           self._picks_length = self._picks.shape[-1]
           self.picks_status = False
        
    @property
    def data(self): return self._data
    @data.setter
    def data(self,v):
        self._data=v
        if self._data_length != self._data.shape[-1]:
           self._data_length =  self.data.shape[-1]
           self.data_status = False
    
    @property
    def colours(self): return self._colours
    @colours.setter
    def colours(self,v):
        self._colours=v
        
    @property
    def timepoints(self): return self._tp_data
    @timepoints.setter
    def timepoints(self,v):
        self._tp_data=v
        if self._tp_length != self._tp_data.shape[-1]:
           self._tp_length  = self._tp_data.shape[-1]
           self.data_status = False  # set this to False signal VBOs need update
        else:
           self.timepoints_status=False
    
    def update(self,**kwargs):
        self._update_from_kwargs(**kwargs)
       
    def _update_from_kwargs(self,**kwargs):
        for k in kwargs:
            if k in self.__slots__:
               self.__setattr__(k,kwargs.get(k,self.__getattribute__(k)))
        if "data" in kwargs:
            self.data = kwargs.get("data")
        if "colours" in kwargs:
            self.colours = kwargs.get("colours")
        if "timepoints" in kwargs:
           self.timepoints = kwargs.get("timepoints")
    
    def GetColour(self,c):
        return self._colours[ c % self._colours.shape[0]]
    
    def GetLabel(self,i): return self._labels[i]
 
    def GetScale(self,i):
        return self._scale[i]
   
    def GetDCoffset(self,i):return self._dcoffset[i]
    
    def GetTimepoints(self):
        #if self.timepoints.shape[-1] !=  self.samples:
        #   self._tp_data   = np.arange(self.samples,dtype=np.float32) / self.srate + self.time_start
        #   self._tp_length = self._tp_data.shape[-1]
        return self.timepoints # Xpos
    
    def info(self):
        logger.info("---> signals\n"+
                    "  -> data   : {}  min: {}  max: {}\n".format(self.data.shape,0,0) + #,self.data_min,self.data_max)+
                    "  -> colours: {}".format(self.colours))
    

class GLPlotWidget(object):
    # default window size
    """
    https://www.khronos.org/opengl/wiki/Tutorial2:_VAOs,_VBOs,_Vertex_and_Fragment_Shaders_(C_/_SDL)
    
    Example:
    -------
    
    from tsv.plot.jumeg_tsv_plot_plot2d import JuMEG_TSV_OGLPlot2D
    
    
    self.GLPlot = GLPlotWidget()
    
    def OnPaint(self,picks,tsl_start,tsl_end):
        
        
        if OGLisInit:
           GLPlot.size_in_pixel = size_in_pixel
           GLPlot.signals.picks = picks # np array of channel index
           GLPlot.signals.data  = self.raw._data[picks,tsl_start:tsl_end )
           GLPlot.signals.timepoints = raw.times[tsl_start.tsl_end]
           GLPlot.plot()

    """
    __slots__= ["_size_in_pixel","_size_in_mm","backgroundcolour","pointsize","border_width","border_height","info_text","text_colour","text_font",
                "_n_cols","_n_cols_status","_signals","_grid","_isOnDraw","_isInit","_ReziseVBO","_isDemoMode","_vp_matrix","_vp_pics_index",
                "_vao_handle","_vbo_handle","_vbo_buffer","_vbo_data","_glshader"]
    
    def __init__(self,**kwargs): # w=600,h=600,samples=10000,srate=1000.0,freq=10.0,n_ch=10,demomode=False):
        
        self._signals = Signals()
        self._grid    = Grid()
        
       # self._vao_handle = np.zeros(2,dtype=np.bytes)
       # self._vbo_handle = np.zeros(2,dtype=np.bytes)
       # self._vbo_buffer = np.zeros(2,dtype=np.bytes)
        
        #self.GLShader=None
        
        self._init()
        
    def _update_from_kwargs(self,**kwargs):
       
        self.n_cols       = kwargs.get("n_cols",self.n_cols)
        self.text_colour  = kwargs.get("text_color",self.text_colour)
        
       # self.n_rows        = kwargs.get("n_rows",self.n_rows)
        self.size_in_pixel = kwargs.get("size_in_pixel",self.size_in_pixel)
        self.info_text     = kwargs.get("info_text","JuMEG Time Seies Viewer  Please select a file")
        self.signals.update(**kwargs)
        self.grid.update(**kwargs)
        
     #--- demomode
       # self.amp = 1.0
       # self.fscale = 3.0

    @property
    def n_plots(self): return self.signals.picks.shape[-1]
    @property
    def Grid(self): return self._grid
    
    @property
    def n_cols(self):
        return self._n_cols
    @n_cols.setter
    def n_cols(self,v):
        self._n_cols = v
        self._n_cols_status = False

    @property
    def n_rows(self): return int(self.n_plots / self.n_cols)

    @property
    def GLShader(self): return self._glshader
    
    @property
    def signals(self): return self._signals
    @property
    def grid(self): return self._grid
    
    @property
    def size_in_pixel(self): return self._size_in_pixel
    @size_in_pixel.setter
    def size_in_pixel(self,v):
        self._size_in_pixel=v
    
    @property
    def width(self): return self._size_in_pixel[0]
    @width.setter
    def width(self,v):
        self._size_in_pixel[0]=v
    @property
    def height(self): return self._size_in_pixel[1]
    @height.setter
    def height(self,v):
        self._size_in_pixel[1] = v

    @property
    def size_in_mm(self): return self._size_in_mm
    @size_in_mm.setter
    def size_in_mm(self,v):
        self._size_in_mm = v
        
    def _init(self,**kwargs):
        self.n_cols = 1
        self._n_cols_status = False
        self._vp_matrix = None
        self._vp_pics_index = None
        
        self.border_width  = 2.0 #self.margin + self.ticksize
        self.border_height = 1.0 #self.margin + self.ticksize
      #-- ToDo make ogl text & font class with draw-text()
        self.text_colour   = np.array([ 0.0,0.0,0.0,1.0 ],dtype=np.float32)
        self.text_font     = GLUT_BITMAP_HELVETICA_18
                            # GLUT_BITMAP_8_BY_13, GLUT_BITMAP_9_BY_15,GLUT_BITMAP_TIMES_ROMAN_10,GLUT_BITMAP_TIMES_ROMAN_24,GLUT_BITMAP_HELVETICA_10,GLUT_BITMAP_HELVETICA_12,GLUT_BITMAP_HELVETICA_18
        
        self.pointsize = 1.0
        
        self._size_in_pixel   = np.array([600,600],dtype=np.float32)
        self._size_in_mm      = np.zeros([2],dtype=np.float32)
        self.backgroundcolour = np.zeros(4,dtype=np.float32)
      
        self._update_from_kwargs(**kwargs)
        
        self._vao_handle = np.zeros(2,dtype=np.uint32)
        self._vbo_handle = np.zeros(2,dtype=np.uint32) #[GLfloat(x) for x in [0,0]]
        self._vbo_data   = np.array([],dtype=np.float32)
        
        self._glshader   = None
        
        self._isOnDraw   = False
        self._isInit     = False
        
       # self._isDemoMode = demomode
        
    @property
    def isInit(self): return self._isInit
    @property
    def isOnDraw(self): return self._isOnDraw
    @property
    def isDemoMode(self): return self._isDemoMode
   
   #---
    @property
    def VBOGrid(self):
        return self._vbo_handle[0]

    @property
    def VAOGrid(self):
        return self._vao_handle[0]
   #---
    @property
    def VBOData(self): return self._vbo_handle[1]

    @property
    def VAOData(self):
        return self._vao_handle[1]
    @property
    def VieportMatrix(self): return self._vp_matrix
    
    def SetClearScreenColour(self):
        glClearColor(self.backgroundcolour[0],self.backgroundcolour[1],self.backgroundcolour[2],self.backgroundcolour[3])
    
    def init(self):
        """
        Initialize OpenGL, VBOs, upload data on the GPU, etc.
          FYI:
          https://www.khronos.org/opengl/wiki/Layout_Qualifier_(GLSL)
          https://www.khronos.org/opengl/wiki/Tutorial1:_Rendering_shapes_with_glDrawRangeElements,_VAO,_VBO,_shaders_(C%2B%2B_/_freeGLUT)
      
        """
       #--- set background color
        self.SetClearScreenColour()
       #--- init shader obj
        self._glshader = JuMEG_TSV_OGL_Shader_PlotSignal(verbose=True,debug=True)
       #--- init grid data
        self.grid.update()
        
        self._isInit   = True

    def _update_vbo_timepoints(self):
        self._vbo_data[:,0] = self.signals.GetTimepoints()
        self.signals.timepoints_status = True
        
    def _update_vbo_data(self):
        """
        setup VertexBufferObject VBO for plotting data/signals
        :return:
        """
        self._vbo_data = np.zeros((self.signals.samples,2),dtype=np.float32)
        self._update_vbo_timepoints()
        
        #logger.info("---> signals to plot")
        #self.signals.info()
    
    def draw_splash_screen(self,txt=None):
        if txt:
           self.info_text = txt
           
        glColor4f(0.0,1.0,0.0,1.0)
      # div screenwidth -1  <>  1
        dx = - ( len(self.info_text)*24.0/ self.size_in_pixel[0] / 2.0)
        glRasterPos2f(dx,0.0)
        glutBitmapString(GLUT_BITMAP_TIMES_ROMAN_24,self.info_text.encode("utf-8"))
        
        #for idx_chr in str(txt):
        #    glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18,ord(str(idx_chr)))

    def init_ogl(self):
        """
        init opengl settings
        !!! glutInit() has to be called before!!!
        """
        
       #glClearDepth(1.0)
        glDepthFunc(GL_LESS)
        glDisable(GL_SCISSOR_TEST)
        glEnable(GL_LINE_SMOOTH | GL_DEPTH_TEST |GL_BLEND)
        
        self.SetClearScreenColour()
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLineWidth(1)

  
    def _update_vao(self,idx):  #0,1
        
        glBindVertexArray(0)
        glBindBuffer(GL_ARRAY_BUFFER,0)
        if idx > -1:
           glDeleteBuffers(GL_ARRAY_BUFFER, GLuint(self._vbo_handle[idx]) )  # np array
           #glDeleteBuffers(GL_ARRAY_BUFFER,[ self._vbo_handle[idx] ] )
           self._vao_handle[idx] = glGenVertexArrays(1)
           self._vbo_handle[idx] = 0
        else:
           self._vao_handle.append(glGenVertexArrays(1))
    
    def _update_vao_grid(self,update=False):
        """
        update/setup VertexBufferArrayObject for plotting grid
        :return:
        """
       #--- VAO for grid
        self._update_vao(0)
        glBindVertexArray(self._vao_handle[0])
        
       #---VBO for grid
        if len(self._vbo_handle):
           if self._vbo_handle[0]:
              self._clear_vbos(idx=0)
           self._vbo_handle[0] = glGenBuffers(1)
        else:
            self._vbo_handle.append(glGenBuffers(1))

        glBindBuffer(GL_ARRAY_BUFFER,self.VBOGrid)
        glBufferData(GL_ARRAY_BUFFER,self.grid.data.nbytes,self.grid.data,GL_DYNAMIC_DRAW)
        glVertexAttribPointer(0,2,GL_FLOAT,False,0,ctypes.c_void_p(0)) # "a_position" fixed in vert shader
        glEnableVertexAttribArray(0)
    
    def _update_vao_data(self):
        """
        update/setup VertexBufferArrayObject VAO for plotting data/signals
        :return:
        """
       #--- VAO for grid
        self._update_vao(1)
        glBindVertexArray(self._vao_handle[1])
     
       #--- VAO for grid
        glBindVertexArray(self._vao_handle[1])
       #---VBO for grid
        if len(self._vbo_handle)>1:
           self._vbo_handle[1]= glGenBuffers(1)
        else:
           self._vbo_handle.append( glGenBuffers(1) )
           
        glBindBuffer(GL_ARRAY_BUFFER,self.VBOData)
        glBufferData(GL_ARRAY_BUFFER,self._vbo_data.nbytes,None,GL_DYNAMIC_DRAW)
        glVertexAttribPointer(0,2,GL_FLOAT,False,0,ctypes.c_void_p(0)) # "a_position" fixed in vert shader
        glEnableVertexAttribArray(0)
        
        
    def _vbo_create(self,nbufs=1):
        """
        create grid and data vbo
        http://antongerdelan.net/opengl/vertexbuffers.html

        https://stackoverflow.com/questions/40954397/how-does-pyopengl-do-its-magic-with-glgenbuffers

        :param vbo_nbufs:
        :return:
        """
        #logger.debug("  -> creating VAO VBO")
    
        self._clear_vaos()
        self._clear_vbos()
       
        self._update_vao_grid()
        
        self._update_vao_data()

        glBindVertexArray(0)
        glBindBuffer(GL_ARRAY_BUFFER,0)
        
        #logger.debug("  -> DONE create VAO VBO")
        
    def _plot_grid(self):
        '''
         ToDo move to Grid cls
         plot the grid
         VA0 idx : 0
         VBO idx : 0
         
        :return:
        '''
       
        #logger.debug("  -> plot grid")
        
        #if self.signals.picks_status:
        #   self.signals.data_status
        #   self.grid.update() =>  n-plots n_cols
        #   _update_vao_grid
           
        try:
            self.Grid.SetBackgroundColour() #glColor4f(self.Grid.backgroundcolour) #0.9,0.9,0.9,0.0)
            glRectf(-1., 1., 1., -1)
            
            glLineWidth(1)
            self.GLShader.UseProgram(True)
            self.GLShader.SetTrafoMatrix(self.grid.trafomatrix)
            self.GLShader.SetColour(self.grid.colour)
         
            glBindVertexArray(self._vao_handle[0])
            glDrawArrays(GL_LINES,0,self.grid.data.shape[0] - 1)
        except:
            logger.exception("---> ERROR in plot grid:")
    
        finally:
            glDisableVertexAttribArray(self.VAOGrid)
            #glBindBuffer(GL_ARRAY_BUFFER, 0)
            self.GLShader.UseProgram(False)
   
    def calc_viewport_matrix(self):
        self._vp_matrix = viewport_matrix(self.n_cols,self.n_plots,self.width,self.height,self.border_width,self.border_height)
        #logger.info("---> VP matrix: cols: {} rows: {} picks: {}\n ---> matrix\n {}".format(self.n_cols,self.n_rows,self.n_plots,self._vp_matrix))
        return self._vp_matrix
    
    def __channel_scaling_matrix(self,mat,cidx):
        #mat[1,1] = +2.0 / (top - bottom)
        #mat[3,1] = -(top + bottom) / float(top - bottom)
        
        # min max dT
        # min max global
        # U/div
        res,offset = self.signals.GetScaleAndOffset(cidx)
        res *= self.Grid.ydiv / 2.0
        top    = res + offset
        bottom =-res + offset
        mat[1,1] = +2.0 / (top - bottom)
        mat[3,1] = -(top + bottom) / float(top - bottom)
        
        return mat

    def _channel_scaling_matrix(self,mat,cidx):
        bottom,top = self.signals.GetScale(cidx)
        mat[1,1] = +2.0 / (top - bottom)
        mat[3,1] = -(top + bottom) / float(top - bottom)
    
        return mat

    def _plot_data(self):
        """
        ToDo use picks, as selected channels and  pointer to raw._data
        set GL_Blend ,ALPHA
        
        :return:
        """
        try:
           #--- update vbo + data buffer
            if not self.signals.data_status:
               self._update_vbo_data()
               self._update_vao_data()
               self.signals.data_status      =True
               self.signals.timepoints_status=True
              # logger.info("  -> plot data update signals & tps")

           #--- update tp
            if not self.signals.timepoints_status:
               self._update_vbo_timepoints()
               # logger.info("  -> plot data update tps")
            
            vpm = self.calc_viewport_matrix()
           
           
           
            self.GLShader.UseProgram(False)
            idx=0
           # glViewport(0,0,self.width,self.height) # full widget
         
           
            for cidx in (self.signals.picks ):
                glViewport( vpm[idx,0],vpm[idx,1],vpm[idx,2],vpm[idx,3] )
                self._plot_grid()
                idx+=1
               
            self.GLShader.UseProgram(True)
           
            glBindVertexArray(self.VAOData)
            glBindBuffer(GL_ARRAY_BUFFER, self._vbo_handle[1])
            glLineWidth(2)
           
            picks  = self.signals.picks
            mat = ortho2D(self.signals.timepoints[0],self.signals.timepoints[-1],-1,1)
            
            idx=0
        
            #self._vp_pics_index = np.zeros([self.signals._picks_length],dtype=np.int32)
           
            for cidx in (self.signals.picks ):
               #--- ToDo calc VP in shader update trafo Matrix
                glViewport(vpm[idx,0],vpm[idx,1],vpm[idx,2],vpm[idx,3])
               
                #self._vp_pics_index[idx]=cidx
               
                mat = self._channel_scaling_matrix(mat,cidx)
                self.GLShader.SetTrafoMatrix(mat)

                c = self.signals.GetColour(cidx)
                self.GLShader.SetColour(c)
               
                dc = self.signals.GetDCoffset(cidx)
                self.GLShader.SetDCoffset(dc)
                
                self._vbo_data[:,1] = self.signals.data[cidx]
                glBufferSubData(GL_ARRAY_BUFFER,0,self._vbo_data.nbytes,self._vbo_data)
                
                # self.map_buffer()
                
                glDrawArrays(GL_LINE_STRIP,0,self._vbo_data.shape[0])
                
                #glEnable(GL_BLEND)
                #glBlendFunc(GL_SRC_ALPHA,ONE_MINUS_SRC_ALPHA)
                draw_text(self.signals.GetLabel(cidx),vpm[idx,0],vpm[idx,1] + vpm[idx,3] / 2.0,self.text_colour,self.text_font)
                #glDisable(GL_BLEND)
               # logger.info( " --> label: {} cidx: {} idx: {}  vpm: {}".format(self.signals.GetLabel(cidx),cidx,idx, vpm[idx]))
                idx += 1
        
           # self._vp_pics_index = self._vp_pics_index[::-1]
        
        except:
            logger.exception("---> ERROR ")
        
       
      
        finally:
            
            # glDisableClientState(GL_VERTEX_ARRAY)
            # glDisableVertexAttribArray(self.VAOData)
           # self.GLShader.UseProgram(False)
            glBindBuffer(GL_ARRAY_BUFFER, 0)
            glBindVertexArray(0)
    
    def _clear_vbos(self,idx=-1):
        glBindVertexArray(0)
        glBindBuffer(GL_ARRAY_BUFFER,0)
        if not self._vbo_handle.any(): return
        if idx >-1:
           glDeleteBuffers(1,self._vbo_handle[idx] )
           self._vbo_handle[idx] = 0
        else:
           glDeleteBuffers(self._vbo_handle.shape[-1],self._vbo_handle)
           self._vbo_handle *= 0
    
    def _clear_vaos(self,idx=-1):
        glBindVertexArray(0)
        if idx >-1:
           glDeleteVertexArrays(1,self._vao_handle[idx])
           self._vao_handle[idx] = 0
        else:
           glDeleteVertexArrays( self._vao_handle.shape[-1],self._vao_handle)
           self._vao_handle *= 0
  
    def clear(self):
        '''
        clear GLShader memory & delete
        clear VBO VAO
        
        https://www.khronos.org/opengl/wiki/Tutorial2:_VAOs,_VBOs,_Vertex_and_Fragment_Shaders_(C_/_SDL)
        '''
        glFlush() # finish running ogl process
        
        self.GLShader.UseProgram(False)
       
        self._clear_vaos()
        self._clear_vbos()
        
        del self._glshader # delete glsl pgr,free memory
        glFlush()  # finish running ogl process
        
    def __del__(self):
        if self._glshader:
           self.clear()
    
    def _init_plot(self):
    
        if not self.isInit:
           self.init()
           self._vbo_create()
    
        # glColor(self.backgroundcolour)
        self.SetClearScreenColour()
        glPointSize(self.pointsize)
        
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glViewport(0,0,self.width,self.height)
        # self.GLShader.UseProgram(True)
    
    def _reset_plot(self):
        glDisableClientState(GL_VERTEX_ARRAY)
        self.GLShader.UseProgram(False)
        glBindBuffer(GL_ARRAY_BUFFER,0)
        glBindVertexArray(0)
        glFlush()
        
        self._isOnDraw = False
      
    def plot(self):
        """plot the scene."""
        # clear the buffer
        if self.isOnDraw:
            return
        
        self._isOnDraw = True
        
        #t0 = time.time()
     
        self._init_plot()
        
       # self._plot_grid()
        
        self._plot_data()
        
        #t1 = time.time()
        #dt = t1-t0
        #logger.info(" --> plot data paintGL w: {} h: {} data: {} ---> dt: {}".format(self.size_in_pixel[0],self.size_in_pixel[1],self.signals.data.shape,dt))
        #logger.info(" --> plot data paintGL w mm: {} h mm: {} scale: {}".format(self.size_in_mm[0],self.size_in_mm[1],self.scale_factor_mm))
        #logger.info("GLUT MM: {}".format( glutGet(GLUT_SCREEN_WIDTH_MM) ))
        
        
        #t1 = time.time()
        
        self._reset_plot()
        
        #t2 = time.time()
        #logger.info(" --> DONE paintGL ---> dt: {}".format(t2 - t1))
        #logger.info("  -> total dt: {}\n".format(t2 - t0))

   #---demo
    def paintGL_and_swap(self):
        self.size = [glutGet(GLUT_WINDOW_WIDTH),glutGet(GLUT_WINDOW_HEIGHT)]
        self.paintGL()
        glutSwapBuffers()
        
    def drawLines(self):
        # glClear(GL_COLOR_BUFFER_BIT);
       
      
        glColor4f(1.0,0.0,1.0,1.0)
        glPointSize(1.0)
        
        glBegin(GL_LINES)
        glVertex2f(-1.0,0.0)
        glVertex2f(1.0,0.0)
        glEnd()
    
    def resize(self,width,height):
        """Called upon window resizing: reinitialize the viewport."""
        # update the window size
        self.width,self.height = width,height
        #glViewport(0, 0, width, height)
        
    def update_data(self):
        pass
    
    def update_test_data(self):
        
        #self.timepoints = np.array([])
        #self.data[:,0] = np.linspace(-1.,1.,self.samples)
      
       #--- generate VBO data
        self.vbo_data       = np.zeros((self.samples,2),dtype=np.float32)
        self.vbo_data[:,0] += np.arange(self.samples,dtype=np.float32) / self.srate
        self.timepoints = self.vbo_data[:,0]
        
        #self.vbo_data = np.zeros(2 * self.samples,dtype=np.float32)
        #self.timepoints = np.arange(self.samples,dtype=np.float32) / self.srate
        #self.vbo_data[0:-1:2] += self.timepoints
    
      #--- init data
        self.data = np.zeros((self.n_channels,self.samples),dtype=np.float32)
        y1 = self.amp * np.sin(2.0 * np.pi * self.freq * self.timepoints)
        
        for cidx in range(self.n_channels):
            y2 = np.cos(2.0 * np.pi * self.freq * (self.fscale + cidx) * self.timepoints)
            self.data[cidx,:] += y1 + y2
        
        self.colors = np.ones((self.data.shape[0],4),dtype=np.float32)
        self.colors[:,0:3] = np.random.random((self.data.shape[0],3))
        self.data_min = self.data.min()
        self.data_max = self.data.max()
        # plt.plot(data[0],data[1])
        # plt.show()
        #data = data.T.flatten()


def init_gl():
    os.putenv("DRI_PRIME","1")
    
    glutInit()
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
    glutInitWindowSize(600,600)
    glutInitWindowPosition(10,10)
    glutCreateWindow(b"GL TEST scratch16")
    
    GLPlot = GLPlotWidget(demomode=True)
    
    glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE,GLUT_ACTION_CONTINUE_EXECUTION)
    glutWMCloseFunc(GLPlot.clear)
    glutDisplayFunc(GLPlot.paintGL_and_swap)
    #glutIdleFunc(GLPlot.paintGL)
    #glutReshapeFunc(GLPlot.resizeGL)
    
    GLPlot.init_ogl()
    
    logger.info("\n---> OpenGL Information:\n"+
                 "  -> ".join(["Vendor        : {}\n".format(glGetString(GL_VENDOR)),
                              "Opengl version: {}\n".format(glGetString(GL_VERSION)),
                              "GLSL Version  : {}\n".format(glGetString(GL_SHADING_LANGUAGE_VERSION)),
                              "Renderer      : {}\n".format(glGetString(GL_RENDERER)),
                              "Extentions    : {}\n".format(glGetString(GL_EXTENSIONS))] )
               )


    glutMainLoop()


if __name__ == '__main__':
   parser = argparse.ArgumentParser()
   parser.add_argument("-v","--verbose",action="store_true",help="verbose mode",default=True)
   parser.add_argument("-d","--debug",action="store_true",help="debug mode",default=True)
   opt = parser.parse_args()
   
   jumeg_logger.setup_script_logging(name=sys.argv[0],opt=opt,logger=logger)
  
   init_gl()



'''
def map_buffer(self):
    glBindBuffer(GL_ARRAY_BUFFER,self.vbo_ids[0])  #[0])
    #glBufferData(GL_ARRAY_BUFFER,self.vbo_data.nbytes,None,GL_DYNAMIC_DRAW)
    
    #self.data_buffer = glMapBufferRange(GL_ARRAY_BUFFER,0,self.vbo_data.nbytes,
    #                                    GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT|GL_MAP_UNSYNCHRONIZED_BIT)
    
    self.data_buffer = glMapBuffer(GL_ARRAY_BUFFER,GL_WRITE_ONLY)
    
    #self._data_array_p = self.vbo_data.nbytes).from_address( self.data_array )
    self._data_array_p = ((ctypes.c_float * 2) * self.vbo_data.shape[0]).from_address(self.data_buffer)
    
    self._data_array_p = self.vbo_data  #timepoints
    
    # float_array = ((ctypes.c_float * 4) * 256).from_address(droplet)
    glUnmapBuffer(GL_ARRAY_BUFFER)  #GL_UNIFORM_BUFFER)
'''

'''
        glBindBufferBase(GL_UNIFORM_BUFFER, 0, rain_buffer);
        droplet = glMapBufferRange(GL_UNIFORM_BUFFER, 0, 256*4*4, GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT)
        float_array = ((ctypes.c_float * 4) * 256).from_address(droplet)
        for i in range(0, 256):
            float_array[i][0] = random.random() * 2 -1
            float_array[i][1] = random.random() * 2 -1
            float_array[i][2] = random.random() * math.pi * 2
            float_array[i][3] = 0.0

            https://www.khronos.org/opengl/wiki/Buffer_Object_Streaming
          #-- mk range 2x data size???
            glMapBufferRange with the GL_MAP_UNSYNCHRONIZED_BIT

        # glBufferData(NULL), glMapBufferRange(GL_MAP_INVALIDATE_BUFFER_BIT), or glInvalidateBufferData)
'''


'''

    def _vbo_create_ok(self,vao_nbufs=1,vbo_nbufs=1):
       #--- init clean
        glBindVertexArray(0)
        if self._vbo_ids:
           glDeleteBuffers(len(self._vbo_ids),self._vbo_ids);
           self._vbo_ids = None
           
        self._vbo_ids = glGenBuffers(vbo_nbufs)  # only one buf
        glBindBuffer(GL_ARRAY_BUFFER, self._vbo_ids )
        glBufferData(GL_ARRAY_BUFFER,self._grid_data.nbytes,None,GL_DYNAMIC_DRAW)
       
      #--- VAOs
        #self._vao_ids  = (GLint * vao_nbufs)()
        #glGenVertexArrays(vao_nbufs,self._vao_ids )
      #
        #self._vao_ids = glGenVertexArrays(vao_nbufs)
      #--- VBOs
        #self._vbo_ids = (GLint * vbo_nbufs)()
        #glGenBuffers(vbo_nbufs, self._vbo_ids)

        #self._vbo_ids = glGenBuffers(vbo_nbufs)
'''
'''
        if vbo_nbufs>1:
          #--- init grid buffer
           glBindVertexArray(self._vao_ids[0])
          # Bind a buffer before we can use it
           glBindBuffer(GL_ARRAY_BUFFER,self._vbo_ids[0])  #[0])
          # Now go ahead and fill this bound buffer with some data
           glBufferData(GL_ARRAY_BUFFER,self._grid_data.nbytes,None,GL_DYNAMIC_DRAW)
    
        else:
          #--- init grid buffer
           glBindVertexArray(self._vao_ids)
          # Bind a buffer before we can use it
           glBindBuffer(GL_ARRAY_BUFFER,self._vbo_ids)  #[0])
          # Now go ahead and fill this bound buffer with some data
           glBufferData(GL_ARRAY_BUFFER,self._grid_data.nbytes,None,GL_DYNAMIC_DRAW)
'''
'''
    #--- init data buffers
       # for idx in range(n):
       # glBindVertexArray(self._vao_ids[1])
      # Bind a buffer before we can use it
       # glBindBuffer(GL_ARRAY_BUFFER,self._vbo_ids[1])  #[0])
      # Now go ahead and fill this bound buffer with some data
       # glBufferData(GL_ARRAY_BUFFER,self._vbo_data[0].nbytes,None,GL_DYNAMIC_DRAW)
    
       
        glBindBuffer(GL_ARRAY_BUFFER,0)
        glBindVertexArray(0)

    def _vbo_create_test(self,vbo_nbufs=2,vao_buf=2):
        """
        create grid and data vbo
        http://antongerdelan.net/opengl/vertexbuffers.html
        :param vbo_nbufs:
        :return:
        """
        self._vbo_handle = []
        #---grid vbo
        self._vbo_ids.append(glGenBuffers(1))
        glBindBuffer(GL_ARRAY_BUFFER,self._vbo_ids[-1])
        glBufferData(GL_ARRAY_BUFFER,self._grid_data.nbytes,None,GL_DYNAMIC_DRAW)
        glBufferSubData(GL_ARRAY_BUFFER,0,self._grid_data.nbytes,self._grid_data)
       
        #---grid data vbo
        self._vbo_ids.append(glGenBuffers(1))
        glBindBuffer(GL_ARRAY_BUFFER,self._vbo_ids[-1])
        glBufferData(GL_ARRAY_BUFFER,self._vbo_data.nbytes,None,GL_DYNAMIC_DRAW)
    
        self._vao_ids = []
        self._vao_ids.append(glGenVertexArrays(1))
        glBindVertexArray(self._vao_ids[-1])
    
        #--- bind the vbo Grid data  into vao
        glBindBuffer(GL_ARRAY_BUFFER,self._vbo_ids[-1])
        glVertexAttribPointer(self._vbo_ids[-1],2,GL_FLOAT,GL_FALSE,0,None)

       #--- data buffer
        self._vbo_ids.append(glGenBuffers(1))
        self._vao_ids.append(glGenVertexArrays(1))
        glBindVertexArray(self._vao_ids[-1])

        #--- bind the vbo Grid data  into vao
        glBindBuffer(GL_ARRAY_BUFFER,id)
        glVertexAttribPointer(self._vao_ids[0],2,GL_FLOAT,GL_FALSE,0,None)

    #   glBindBuffer(GL_ARRAY_BUFFER,0)

    # glBindVertexArray(0)
'''

'''

    def _vbo_create(self,n_buf=1):

        #--- if VBO IsInit
        # glDisableVertexAttribArray(0)

        #--- init clean

        #map a GOL buffer to cp data
        #https: // www.oipapio.com / question - 4876247
        #glMapBufferRange

        glBindVertexArray(0)
        if self.vbo_buffer:
           glDeleteBuffers(len(self.vbo_buffer),self.vbo_buffer);
           self.vbo_buffer = None

        self.vbo_buffer = glGenBuffers(n_buf)  # only one buf
        glBindBuffer(GL_ARRAY_BUFFER,self.vbo_buffer)
        glBufferData(GL_ARRAY_BUFFER,self.vbo_data.nbytes,None,GL_DYNAMIC_DRAW)

    def _vbo_draw(self):

        #glEnableClientState(GL_VERTEX_ARRAY)
        glBindBuffer(GL_ARRAY_BUFFER,self.vbo_buffer)
        glBufferSubData(GL_ARRAY_BUFFER,0,self.vbo_data.nbytes,self.vbo_data)

        #--- enable arrays
        #glEnableVertexAttribArray(ar) #self.GLSL.position2d)

        #glVertexAttribPointer(self.GLSL.position2d,2,GL_FLOAT,GL_FALSE,0,None)
        #glVertexAttribPointer(ar,2,GL_FLOAT,GL_FALSE,0,None)
        glDrawArrays(GL_LINE_STRIP,0,self.timepoints.shape[0] - 1)

        #glVertexPointer(3, GL_FLOAT, 0, None)
        #glDrawElements(GL_LINES, len(self.timepoints), GL_FLOAT, None)
        #glDisableClientState(GL_VERTEX_ARRAY)

 def __vbo_create(self):
        self.vao_id = glGenVertexArrays(1)
        glBindVertexArray(self.vao_id)
      # Lets create our Vertex Buffer objects - these are the buffers
      # that will contain our per vertex data
        self.vbo_id = glGenBuffers(1)

      # Bind a buffer before we can use it
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_id) #[0])
      # Now go ahead and fill this bound buffer with some data
        glBufferData(GL_ARRAY_BUFFER,self.vbo_data.nbytes,None, GL_DYNAMIC_DRAW)




    # Now specify how the shader program will be receiving this data
    # In this case the data from this buffer will be available in the shader as the vin_position vertex attribute
    #glVertexAttribPointer(program.attribute_location('vin_position'), 3, GL_FLOAT, GL_FALSE, 0, None)

    # Turn on this vertex attribute in the shader
        #glEnableVertexAttribArray(0)


    # Lets unbind our vbo and vao state
    # We will bind these again in the draw loop
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)

'''


