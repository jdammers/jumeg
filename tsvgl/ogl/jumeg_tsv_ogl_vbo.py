import sys
import numpy as np

from OpenGL.GL import *
from OpenGL.raw.GL.ARB.vertex_array_object import glGenVertexArrays,glBindVertexArray


'''
=> in init

	// Create the vertex buffer object
	glGenBuffers(3, vbo); # box, axis, signal
	glBindBuffer(GL_ARRAY_BUFFER, vbo[0]);


=> in display
	// Draw using the vertices in our vertex buffer object
	glBindBuffer(GL_ARRAY_BUFFER, vbo[0]);

	glEnableVertexAttribArray(attribute_coord2d);
	glVertexAttribPointer(attribute_coord2d, 2, GL_FLOAT, GL_FALSE, 0, 0);
	glDrawArrays(GL_LINE_STRIP, 0, 2000);

'''

class JuMEG_TSV_OGL_VBO(object):
    """
    Vertext Buffer Object Class 
    ->e.g. plot data in a box with axis using OGL VBO and GSL 

    https://en.wikibooks.org/wiki/OpenGL_Programming/Scientific_OpenGL_Tutorial_03
    """
    def __init__(self,gl_draw_type=GL_DYNAMIC_DRAW,gl_primitive_type=GL_POINTS):

         self.__vbo_data     = np.array([],dtype=np.float32)
         self.__vbo_id       = None
         self.__vbo_isinit   = False
         self.__vbo_gl_draw_type      = gl_draw_type
         self.__vbo_gl_primitive_type = gl_primitive_type
              
 #---- PROPERTIES

#--- gl_draw_type
    def __get_vbo_gl_draw_type(self):
        return self.__vbo_gl_draw_type
#--- gl_draw_type
    def __set_vbo_gl_draw_type(self,v):
        self.__vbo_gl_draw_type=v
    gl_draw_type = property( __get_vbo_gl_draw_type, __set_vbo_gl_draw_type)
#--- gl_primitive_type
    def __get_vbo_gl_primitive_type(self):
        return self.__vbo_gl_primitive_type
#--- gl_draw_type
    def __set_vbo_gl_primitive_type(self,v):
        self.__vbo_gl_primitive_type=v
    gl_primitive_type = property( __get_vbo_gl_primitive_type, __set_vbo_gl_primitive_type)
     
 #--- vob_id
    def __get_vbo_id(self):
        if self.__vbo_id is None:
           self.__vbo_id = glGenBuffers(1)

        return self.__vbo_id

 #   def __set_vbo_id(self,d):
 #        self.__vbo_id = d
 #        self.isinit = False

    vbo_id = property(__get_vbo_id) #__set_vbo_id)

   #--- vob_data
    def __get_vbo_data(self):
        return self.__vbo_data
    def __set_vbo_data(self,d):
        self.__vbo_data = d
        self.__vbo_isinit = False

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
        self.__vbo_isinit = False
        self.init()

    data_x=property(__get_vbo_data_x,__set_vbo_data_x)

    def __get_vbo_data_pts_x(self):
        return self.data_x.size
    data_points_x=property(__get_vbo_data_pts_x)
   
    def __get_vbo_data_pts_y(self):
        return self.data_y.size
    data_points_y=property(__get_vbo_data_pts_y)

    data_points = property(__get_vbo_data_pts_x)

   #--- vob_isinit
    def __get_vbo_isinit(self):
        return self.__vbo_isinit
    #def __set_vbo_isinit(self,d):
    #    self.__vbo_isinit = d        #--- vertices
       
    isinit=property(__get_vbo_isinit) #,__set_vbo_isinit)

    def reset(self,attr_idx=0):
        glUseProgram(0)
        #--TODO enable/destroy all buffers
        glDisableVertexAttribArray(attr_idx)

        glBindVertexArray(0)
        glBindBuffer()
        self.__vbo_isinit = False
        self.__vbo_id     = None
    
         # finally: self.vbo.unbind() glDisableClientState(GL_VERTEX_ARRAY); finally: shaders.glUseProgram( 0 )
  
    def init(self,data=None):
        if ( data ):
           self.data = data

       # if self.vbo_id:
       #    self.reset()

       #--- vertices
       # if self.vbo_id is None:
       #    self.__vbo_id = glGenBuffers(1)

        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_id)
        #glBufferData(GL_ARRAY_BUFFER, 4*len(self.data),None,self.gl_draw_type)
        glBufferData(GL_ARRAY_BUFFER, 4*len(self.data),self.data,self.gl_draw_type)
        #glBindBuffer(GL_ARRAY_BUFFER, None) #self.vbo_id)
      
        self.__vbo_isinit = True
        print"done vbo init"

    def update_data(self,data=None):
          if any( data ):
             #if (data.size != self.data.size) :
             self.__vbo_isinit=False
             self.data=data
             
          #print"VBO update data"   
          self.update_sub_buffer
        
    def update_xdata(self,data=None):
        if data.any():
          # if (data.size != self.data_points) :
           self.__vbo_isinit=False
        self.data_x=data
        self.update_sub_buffer()

    def update_ydata(self,data=None):
        if data.any():
           if (data.size != self.data_points) :
               self.__vbo_isinit= False
               #self.__vbo_data  = np.zeros( 2*data.size,dtype=np.float32)
               
           self.data_y = data
        self.update_sub_buffer

    def update_sub_buffer(self):
        #print"VBO update sub"
            
        if not self.isinit:
           self.init()           
        else:   
           #print"VBO update sub bind"
           glBindBuffer(GL_ARRAY_BUFFER, self.vbo_id)
           #print"VBO update sub   bind ok"
           glBufferSubData(GL_ARRAY_BUFFER,0,4*len(self.data), self.data)
           #print"VBO use gl sub buffer"
           #glBindBuffer(GL_ARRAY_BUFFER,0)
           



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