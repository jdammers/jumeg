# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 22:27:34 2016

@author: fboers
"""
import sys
import numpy as np

from OpenGL.GL import *

from OpenGL.raw.GL.ARB.vertex_array_object import glGenVertexArrays,glBindVertexArray


class JuMEG_TSV_OGL_VBO(object):
    """ Helper class for using GLSL shader programs
    """
    def __init__(self):

        self.__vbo_data  = np.array([],dtype=np.float32)
        self.__vbo_id    = None
        # self.__vbo_id    = np.array([],dtype=np.uint8)
        self._vbo_isinit = False

 #===== PROPERTIES


 #--- vob_id
    def __get_vbo_id(self):
        return self.__vbo_id

    def __set_vbo_id(self,d):
        self.__vbo_id = d
        self.vbo_isinit = False

    vbo_id=property(__get_vbo_id,__set_vbo_id)

   #--- vob_data
    def __get_vbo_data(self):
        return self.__vbo_data
    def __set_vbo_data(self,d):
        self.__vbo_data = d
        self.vbo_isinit = False

    vbo_data=property(__get_vbo_data,__set_vbo_data)

   #--- vbo_data_signal
    def __get_vbo_data_y(self):
        return self.__vbo_data[1::2]
    def __set_vbo_data_y(self,d):
        self.__vbo_data[1::2] = d
    vbo_data_y=property(__get_vbo_data_y,__set_vbo_data_y)

   #--- vbo_data_timepoints
    def __get_vbo_data_x(self):
        return self.__vbo_data[0:-1:2]

    def __set_vbo_data_x(self,d):
        self.__vbo_data = np.zeros( 2*d.size,dtype=np.float32)
        self.__vbo_data[0:-1:2] = d
        self.vbo_isinit = False
        self.vbo_init()

    vbo_data_x=property(__get_vbo_data_x,__set_vbo_data_x)

    def __get_vbo_data_pts(self):
        return self.vbo_data_x.size
    data_points=property(__get_vbo_data_pts)

   #--- vob_isinit
    def __get_vbo_isinit(self):
        return self.__vbo_isinit
    def __set_vbo_isinit(self,d):
        self.__vbo_isinit = d
    vbo_isinit=property(__get_vbo_isinit,__set_vbo_isinit)


    def vbo_reset(self,attr_idx=0):
        glUseProgram(0)
        #--TODO enable/destroy all buffers
        glDisableVertexAttribArray(attr_idx)

        glBindVertexArray(0)
        self.vbo_isinit = False

    def vbo_init(self,data=None):
        if ( data ):
           self.vbo_data = data

        if self.vbo_id:
           self.vbo_reset()

       #--- vertices
        self.vbo_id = glGenBuffers(1)

        #print"VBO INIT "
        #print  self.vbo_id
        #print type(self.vbo_id)

        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_id)
        glBufferData(GL_ARRAY_BUFFER, 4*len(self.vbo_data),self.vbo_data,GL_DYNAMIC_DRAW)
       #--- plot border
       # glBindBuffer(GL_ARRAY_BUFFER, self.vbo_id[1] )
       # glBufferData(GL_ARRAY_BUFFER, 4*len(self.vbo_plot_border),self.vbo_plot_border, GL_STATIC_DRAW)

        self.vbo_isinit = True
        #print"done vbo init"

    def vbo_update(self,data=None):
         if any( data ):
            if (data.size != self.vbo_data.size) :
               self.vbo_isinit=False
            self.vbo_data=data

         if not self.vbo_isinit:
            self.vbo_init()

        #--- vertices
         glBindBuffer(GL_ARRAY_BUFFER, self.vbo_id)
        #--- TODO ck if only y/signal value can be copied step 2
         glBufferSubData(GL_ARRAY_BUFFER,0, 4*len(self.vbo_data), self.vbo_data)

    def vbo_update_x(self,data=None):
        if data.any():
            if (data.size != self.data_points) :
               self.vbo_isinit=False
        self.vbo_data_x=data

        if not self.vbo_isinit:
           self.vbo_init()

        self.vbo_update_sub_buffer()

        #--- vertices
        #glBindBuffer(GL_ARRAY_BUFFER, self.vbo_id)
        #--- TODO ck if only y/signal value can be copied step 2
        #glBufferSubData(GL_ARRAY_BUFFER,4*self.data_points, 4*len(self.vbo_data_xs), self.vbo_data_timepoints)

    def vbo_update_y(self,data=None):
        if data.any():
           if (data.size != self.data_points) :
               self.vbo_isinit=False
           self.vbo_data_y=data

        if not self.vbo_isinit:
           self.vbo_init()

        self.vbo_update_sub_buffer()

    def vbo_update_sub_buffer(self):
        #--- vertices
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_id)
        #--- TODO ck if only y/signal value can be copied step 2
        glBufferSubData(GL_ARRAY_BUFFER,0,4*len(self.vbo_data), self.vbo_data)


