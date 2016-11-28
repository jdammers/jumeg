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
        self.isinit = False

    vbo_id = property(__get_vbo_id,__set_vbo_id)

   #--- vob_data
    def __get_vbo_data(self):
        return self.__vbo_data
    def __set_vbo_data(self,d):
        self.__vbo_data = d
        self.isinit = False

    data=property(__get_vbo_data,__set_vbo_data)

   #--- vbo_data_signal
    def __get_vbo_data_y(self):
        return self.__vbo_data[1::2]
    def __set_vbo_data_y(self,d):
        self.__vbo_data[1::2] = d
    data_y=property(__get_vbo_data_y,__set_vbo_data_y)

   #--- vbo_data_timepoints
    def __get_vbo_data_x(self):
        return self.__vbo_data[0:-1:2]

    def __set_vbo_data_x(self,d):
        self.__vbo_data = np.zeros( 2*d.size,dtype=np.float32)
        self.__vbo_data[0:-1:2] = d
        self.isinit = False
        self.init()

    data_x=property(__get_vbo_data_x,__set_vbo_data_x)

    def __get_vbo_data_pts_x(self):
        return self.data_x.size
    data_points_x=property(__get_vbo_data_pts_x)
   
    def __get_vbo_data_pts_y(self):
        return self.data_y.size
    data_points_y=property(__get_vbo_data_pts_y)

    #def __get_vbo_data_pts(self):
    #    return self.data.size
    data_points = property(__get_vbo_data_pts_x)

   #--- vob_isinit
    def __get_vbo_isinit(self):
        return self.__vbo_isinit
    def __set_vbo_isinit(self,d):
        self.__vbo_isinit = d
    isinit=property(__get_vbo_isinit,__set_vbo_isinit)


    def reset(self,attr_idx=0):
        glUseProgram(0)
        #--TODO enable/destroy all buffers
        glDisableVertexAttribArray(attr_idx)

        glBindVertexArray(0)
        self.vbo_isinit = False
         # finally: self.vbo.unbind() glDisableClientState(GL_VERTEX_ARRAY); finally: shaders.glUseProgram( 0 )
  
    def init(self,data=None):
        if ( data ):
           self.data = data

        if self.vbo_id:
           self.reset()

       #--- vertices
        self.vbo_id = glGenBuffers(1)

        #print"VBO INIT "
        #print  self.vbo_id
        #print type(self.vbo_id)

        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_id)
        glBufferData(GL_ARRAY_BUFFER, 4*len(self.data),self.data,GL_DYNAMIC_DRAW)
       #--- plot border
       # glBindBuffer(GL_ARRAY_BUFFER, self.vbo_id[1] )
       # glBufferData(GL_ARRAY_BUFFER, 4*len(self.vbo_plot_border),self.vbo_plot_border, GL_STATIC_DRAW)

        self.isinit = True
        #print"done vbo init"

    def update_data(self,data=None):
         if any( data ):
            if (data.size != self.data.size) :
               self.isinit=False
            self.data=data

         if not self.isinit:
            self.init()

        #--- vertices
         glBindBuffer(GL_ARRAY_BUFFER, self.vbo_id)
        #--- TODO ck if only y/signal value can be copied step 2
         glBufferSubData(GL_ARRAY_BUFFER,0, 4*len(self.data), self.data)

    def update_xdata(self,data=None):
        if data.any():
            if (data.size != self.data_points_x) :
               self.isinit=False
        self.data_x=data

        if not self.isinit:
           self.init()

        self.update_sub_buffer()

        #--- vertices
        #glBindBuffer(GL_ARRAY_BUFFER, self.vbo_id)
        #--- TODO ck if only y/signal value can be copied step 2
        #glBufferSubData(GL_ARRAY_BUFFER,4*self.data_points, 4*len(self.vbo_data_xs), self.vbo_data_timepoints)

    def update_ydata(self,data=None):
        if data.any():
           if (data.size != self.data_points_y) :
               self.isinit=False
           self.data_y=data

        if not self.isinit:
           self.init()

        self.update_sub_buffer()

    def update_sub_buffer(self):
        #--- vertices
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_id)
        #--- TODO ck if only y/signal value can be copied step 2
        glBufferSubData(GL_ARRAY_BUFFER,0,4*len(self.data), self.data)




"""
index buffer
GLuint elementbuffer;
 glGenBuffers(1, &elementbuffer);
 glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, elementbuffer);
 glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), &indices[0], GL_STATIC_DRAW);



// Index buffer
 glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, elementbuffer);

 // Draw the triangles !
 glDrawElements(
     GL_TRIANGLES,      // mode
     indices.size(),    // count
     GL_UNSIGNED_INT,   // type   -> GL_UNSIGNED_SHORT
     (void*)0           // element array buffer offset
 );

#ff
#from OpenGL.arrays import ArrayDatatype as ADT

#from OpenGL.GL import shaders
#from OpenGL.Context.arrays import *

#import numpy as np

#from OpenGL.GL import *
from OpenGL.raw.GL.ARB.vertex_array_object import glGenVertexArrays,glBindVertexArray

#from ctypes import pointer,sizeof, c_float, c_void_p, c_uint

from OpenGL.GLUT import *
#from OpenGL.GL import *

#from linalg import matrix as m
#from linalg import quaternion as q



#from OpenGL.arrays import ArrayDatatype

from OpenGL.arrays import ArrayDatatype

from OpenGL.GL import (GL_ARRAY_BUFFER, GL_COLOR_BUFFER_BIT,
    GL_COMPILE_STATUS, GL_FALSE, GL_FLOAT, GL_FRAGMENT_SHADER,
    GL_LINK_STATUS, GL_RENDERER, GL_SHADING_LANGUAGE_VERSION,
    GL_STATIC_DRAW, GL_TRIANGLES, GL_TRUE, GL_VENDOR, GL_VERSION,
    GL_VERTEX_SHADER, glAttachShader, glBindBuffer,
    glBufferData, glClear, glClearColor, glCompileShader,
    glCreateProgram, glCreateShader, glDeleteProgram,
    glDeleteShader, glDrawArrays, glEnableVertexAttribArray,
    glGenBuffers,glGetAttribLocation,
    glGetProgramInfoLog, glGetProgramiv, glGetShaderInfoLog,
    glGetShaderiv, glGetString, glGetUniformLocation, glLinkProgram,
    glShaderSource, glUseProgram, glVertexAttribPointer)

"""