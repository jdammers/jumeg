# from OpenGL.GL import *

#import OpenGL
#from OpenGL import GL

import OpenGL 
OpenGL.ERROR_ON_COPY = True 
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

# PyOpenGL 3.0.1 introduces this convenience module...
from OpenGL.GL.shaders import *



#from OpenGL.GL import (GL_TRUE,GL_FRAGMENT_SHADER,GL_LINK_STATUS)
#from OpenGL.GL import (GL_VERTEX_SHADER,GL_COMPILE_STATUS)
 
# glAttachShader,glCompileShader,glCreateProgram,glCreateShader, glDeleteProgram,glGetAttribLocation,glDeleteShader,glGetProgramInfoLog,
# glGetProgramiv, glGetShaderInfoLog,  glGetShaderiv,glGetUniformLocation, glLinkProgram,glShaderSource,glUseProgram)


#-- ck ogl version 
#from OpenGL.extensions import alternate
#from OpenGL.GL import *
#from OpenGL.GL.ARB.shader_objects import *
#from OpenGL.GL.ARB.fragment_shader import *
#from OpenGL.GL.ARB.vertex_shader import *

#glCreateShader     =  alternate( 'glCreateShader',    glCreateShader,glCreateShaderObjectARB )
#glShaderSource      = alternate( 'glShaderSource',    glShaderSource,glShaderSourceARB)
#glCompileShader     = alternate( 'glCompileShader',   glCompileShader,glCompileShaderARB)
#glCreateProgram     = alternate( 'glCreateProgram',   glCreateProgram,glCreateProgramObjectARB)
#glAttachShader      = alternate( 'glAttachShader',    glAttachShader,glAttachObjectARB )
#glValidateProgram   = alternate( 'glValidateProgram', glValidateProgram,glValidateProgramARB )
#glLinkProgram       = alternate( 'glLinkProgram',     glLinkProgram,glLinkProgramARB )
#glDeleteShader      = alternate( 'glDeleteShader',    glDeleteShader,glDeleteObjectARB )
#glUseProgram        = alternate( 'glUseProgram',      glUseProgram,glUseProgramObjectARB )
#glGetProgramInfoLog = alternate( glGetProgramInfoLog, glGetInfoLogARB )


#OpenGL.error.NullFunctionError: Attempt to call an undefined function glCreateProgram, check for bool(glCreateProgram)

#from OpenGL.GL import *
from OpenGL.GL.ARB.shader_objects import *
from OpenGL.GL.ARB.fragment_shader import *
from OpenGL.GL.ARB.vertex_shader import *
from OpenGL.extensions import alternate
#glCreateShader = alternate( 'glCreateShader', glCreateShader,
#glCreateShaderObjectARB )

#  glBindBuffer,glEnableVertexAttribArray,glGetString, glVertexAttribPointer,GL_SHADING_LANGUAGE_VERSION


# from jumeg.jumeg_base import jumeg_base

class GLSLinfo(object):
      """ Helper class for using GLSL shader programs
      """
      def __init__(self, name=None,source=None,glsl_id=None,postfix='.v.glsl',prefix='plot',fname=None,path=None):
          self.name      = name
          self.id        = glsl_id
          self.prefix    = prefix
          self.postfix   = postfix
          self.source    = source
          self.fname     = fname
          self.path      = path



class JuMEG_TSV_OGL_GSL(object):
      """ Helper class for using GLSL shader programs
      """

      def __init__(self, vertex=None, fragment=None):

          self.VTX = GLSLinfo(name='vertex',  source=vertex,  postfix='.v.glsl')
          self.FRG = GLSLinfo(name='fragment',source=fragment,postfix='.f.glsl')

          self.program_id      = None
          self.glsl_source_dir = 'glsl_source'

          if (self.VTX.source and self.FRG.source):
             self.init_shaders(vtx=self.VTX.source,frg=self.FRG.source)

      def load_shaders_from_file(self,fin=None,pin=None,init=False):

          import os
          fglsl = None

          if pin:
             if os.path.isdir( pin ):
                fglsl = pin

          if not fglsl:
             fglsl = self.get_module_path() + '/' + self.glsl_source_dir+ '/'

          if fin:
             fglsl += '/' + str(fin)
          else:
             fglsl += self.VTX.prefix

         #---VTX
          try:
             fh = open( fglsl + self.VTX.postfix )
             self.VTX.source = fh.read()
             self.VTX.fname=fglsl

             fh.close()
          except:
             assert "ERROR  VTX no such file list: " + fglsl + self.VTX.postfix
             #return found_list

         #---FRG
          try:
             fh = open( fglsl + self.FRG.postfix )
             self.FRG.source = fh.read()
             self.FRG.fname  = fglsl
             fh.close()
          except:
             assert "ERROR  FRG no such file list: " + fglsl + self.FRG.postfix
             #return found_list

          if init:
             self.init_shaders()


      def init_shaders(self,vtx=None,frg=None):

          self.program_id = glCreateProgram()

          if vtx:
             self.VTX.source = vtx
             self.VTX.fname  = None

          #print self.VTX.source
          self.VTX.id = self.add_shader(self.VTX.source, GL_VERTEX_SHADER)

          if frg:
             self.FRG.source = frg
             self.FRG.fname  = None
          self.FRG.id = self.add_shader(self.FRG.source, GL_FRAGMENT_SHADER)


          glAttachShader(self.program_id, self.VTX.id)
          glAttachShader(self.program_id, self.FRG.id)
          glLinkProgram(self.program_id)

          if glGetProgramiv(self.program_id, GL_LINK_STATUS) != GL_TRUE:
             info = glGetProgramInfoLog(self.program_id)
             glDeleteProgram(self.program_id)
             glDeleteShader(self.VTX.id)
             glDeleteShader(self.FRG.id)
             raise RuntimeError('Error linking program: %s' % (info))
          glDeleteShader(self.VTX.id)
          glDeleteShader(self.FRG.id)

      def add_shader(self, source, shader_type):
          try:
              #print "TEST"
              #print source
              shader_id = glCreateShader(shader_type)
              glShaderSource(shader_id, source)
              glCompileShader(shader_id)
              if glGetShaderiv(shader_id, GL_COMPILE_STATUS) != GL_TRUE:
                 info = glGetShaderInfoLog(shader_id)
                 raise RuntimeError('Shader compilation failed: %s' % (info))
              return shader_id
          except:
              glDeleteShader(shader_id)
              raise

      def uloc(self,n):
          return glGetUniformLocation(self.program_id, n)

      def aloc(self,n):
          return glGetAttribLocation(self.program_id, n)

      def uniform_location(self, name):
          return glGetUniformLocation(self.program_id, name)

      def attribute_location(self, name):
          return glGetAttribLocation(self.program_id, name)

      def get_module_path(self):
         import os
         import inspect
         return os.path.dirname(os.path.abspath(inspect.getsourcefile(self.__class__)))
