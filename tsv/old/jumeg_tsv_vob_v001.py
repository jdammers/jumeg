#
# http://ltslashgt.com/2007/08/31/vertex-buffer-object-pyopengl/
#--- create OGL verts buffer


#   glDisableClientState(GL_VERTEX_ARRAY);
#
#my $VertexObjID = glGenBuffersARB_p(1);
#   glBindBufferARB(GL_ARRAY_BUFFER_ARB,$VertexObjID);
#
#my $ogl_array = OpenGL::Array->new_scalar(GL_FLOAT, $data_vbo->get_dataref,$data_vbo->dim(0)*$float_size);
#   glBufferDataARB_p(GL_ARRAY_BUFFER_ARB,$ogl_array,GL_DYNAMIC_DRAW_ARB);
#  $ogl_array->bind($VertexObjID);
#  glVertexPointer_p(2,$ogl_array);
#   glEnableClientState(GL_VERTEX_ARRAY);


from OpenGL.GL import *
from OpenGL.raw import GL
from OpenGL.arrays import ArrayDatatype as ADT

#from OpenGL.GL import shaders
from OpenGL.arrays import vbo
#from OpenGL.Context.arrays import *

import numpy as np

#from OpenGL.GL import *
from OpenGL.raw.GL.ARB.vertex_array_object import glGenVertexArrays, \
                                                  glBindVertexArray

from ctypes import sizeof, c_float, c_void_p, c_uint

from OpenGL.GLUT import *
#from OpenGL.GL import *

#from linalg import matrix as m
#from linalg import quaternion as q

class JuMEGVertexBuffer(object):

      def __init__(self, usage=GL_DYNAMIC_DRAW):
          #self._vbo = GL.GLuint(0)
          #self.float_size = sizeof(c_float)

          #glBindBuffer( GL_ARRAY_BUFFER,glGenBuffers(1) )
          #glBufferData( GL_ARRAY_BUFFER, len( self.data ) * self.float_size, usage);
          self._vbo_data = np.array([])
          self._vbo_isbound=False

          print "INIT"




      def __get_vbo_isbound(self):
          return self._vbo_isbound

      def __set_vbo_isbound(self,v):
          self._vbo_isbound=v

      vbo_isbound=property(__get_vbo_isbound,__set_vbo_isbound)

    #---
      def __get_vbo_data(self):
          return self._vbo_data

      def __set_vbo_data(self,din):

          if  self._vbo_data.size:
             if (self._vbo_data.shape != din.shape):
                if self.vbo_isbound:
                   self.vbo.unbind()
                   glDisableClientState(GL_VERTEX_ARRAY)

            #--- try:
          self._vbo_data = din
          self.vbo = vbo.VBO( self._vbo_data )
          self.vbo.bind()
            #--- try:
            # glEnableClientState(GL_VERTEX_ARRAY)
            # glVertexPointer(2, GL_FLOAT,2 * sizeof(c_type), self.vbo )

            # glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(self.data), data)

            #glBufferSubDataARB_p(GL_ARRAY_BUFFER_ARB,0,$ogl_array);

            #glDrawArrays(GL_LINE_STRIP,0,$data_4_vbo_timepoints->dim(-1)-1 );

          self.vbo_isbound=True

             #   finally:

            # finally:


      data=property(__get_vbo_data,__set_vbo_data)


      def draw(self):
          glEnableClientState(GL_VERTEX_ARRAY)
          glVertexPointer(2, GL_FLOAT,2*sizeof(c_float), self.vbo )
          glDrawArrays(GL_LINE_STRIP, 0, self.data.shape[0])


      def plot(self):
          self.vbo = vbo.VBO(self.data)
          print"data shape "
          print self.data.shape
          try:
              self.vbo.bind()
              points=self.data.shape[0]
              print "points: "
              print points
              try:
                  glEnableClientState(GL_VERTEX_ARRAY)
                 # glVertexPointerf( self.vbo )
                  glVertexPointer(2, GL_FLOAT,2*4, self.vbo )
                  glDrawArrays(GL_LINE_STRIP, 0, points)

                  print "OK2"

              finally:
                  self.vbo.unbind()
                  glDisableClientState(GL_VERTEX_ARRAY)


          finally:
              # shaders.glUseProgram( 0 )
              print "ok plot done"



          #glEnableClientState(GL_VERTEX_ARRAY)
          #glBindBuffer( GL_ARRAY_BUFFER,glGenBuffers(1) )
          ##glBindBuffer(GL_ARRAY_BUFFER, self.vertices_VBO)
          ##glVertexPointer(2, GL_FLOAT, len( self.data ) * self.float_size0, 0) #offset in bytes

          #glBufferData(GL_ARRAY_BUFFER, len( self.data ) * self.float_size, self.data, GL_DYNAMIC_DRAW)

         # glVertexPointer(2, GL_FLOAT, len( self.data ) * self.float_size, 0) #offset in bytes

          #glVertexPointer(2, GL_FLOAT,0, 0) #offset in bytes
          #print "OK1"
          #glVertexAttribPointer(attribute_coord2d, 2, GL_FLOAT, GL_FALSE, 0, 0);
          #glDrawArrays(GL_LINES, 0, points);


          #glDrawArrays(GL_LINE_STRIP, 0,points)
          #print "OK2"





          #glBindBuffer(GL_ARRAY_BUFFER, 0)
          #glDisableClientState(GL_VERTEX_ARRAY)
          print "OK3"



      def reset(self):

          self.vbo.unbind()

          glDisableClientState(GL_VERTEX_ARRAY)

          self.vob_isbound=False
          self._vbo_data = np.array([])




class VertexBufferOrig(object):

  def __init__(self, data, usage):
      self.buffer = GL.GLuint(0)
      glGenBuffers(1)# self.buffer)
      self.buffer = self.buffer.value
      glBindBuffer(GL_ARRAY_BUFFER_ARB, self.buffer)
      glBufferData(GL_ARRAY_BUFFER_ARB, ADT.arrayByteCount(data), ADT.voidDataPointer(data), usage)

  def __del__(self):
      #  glBindBufferARB(GL_ARRAY_BUFFER_ARB, 0);
      # glDisableClientState(GL_VERTEX_ARRAY);
      glDeleteBuffers(1)

      #glDeleteBuffers(1, GL.GLuint(self.buffer))

  def bind(self):
      glBindBuffer(GL_ARRAY_BUFFER_ARB, self.buffer)

  def bind_colors(self, size, type, stride=0):
      self.bind()
      glColorPointer(size, type, stride, None)

  def bind_edgeflags(self, stride=0):
      self.bind()
      glEdgeFlagPointer(stride, None)

  def bind_indexes(self, type, stride=0):
      self.bind()
      glIndexPointer(type, stride, None)

  def bind_normals(self, type, stride=0):
      self.bind()
      glNormalPointer(type, stride, None)

  def bind_texcoords(self, size, type, stride=0):
      self.bind()
      glTexCoordPointer(size, type, stride, None)

  def bind_vertexes(self, size, type, stride=0):
      self.bind()
      glVertexPointer(size, type, stride, None)
