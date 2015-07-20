import sys
import numpy as np

from OpenGL.GL import *

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


class ShaderProgram(object):
    """ Helper class for using GLSL shader programs
    """
    def __init__(self, vertex, fragment):
        """
        Parameters
        ----------
        vertex : str
            String containing shader source code for the vertex
            shader
        fragment : str
            String containing shader source code for the fragment
            shader

        """

        self.program_id = glCreateProgram()
        vs_id = self.add_shader(vertex, GL_VERTEX_SHADER)
        frag_id = self.add_shader(fragment, GL_FRAGMENT_SHADER)

        glAttachShader(self.program_id, vs_id)
        glAttachShader(self.program_id, frag_id)
        glLinkProgram(self.program_id)

        if glGetProgramiv(self.program_id, GL_LINK_STATUS) != GL_TRUE:
            info = glGetProgramInfoLog(self.program_id)
            glDeleteProgram(self.program_id)
            glDeleteShader(vs_id)
            glDeleteShader(frag_id)
            raise RuntimeError('Error linking program: %s' % (info))
        glDeleteShader(vs_id)
        glDeleteShader(frag_id)

    def add_shader(self, source, shader_type):
        """ Helper function for compiling a GLSL shader

        Parameters
        ----------
        source : str
            String containing shader source code

        shader_type : valid OpenGL shader type
            Type of shader to compile

        Returns
        -------
        value : int
            Identifier for shader if compilation is successful

        """
        try:
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

    def uniform_location(self, name):
        """ Helper function to get location of an OpenGL uniform variable

        Parameters
        ----------
        name : str
            Name of the variable for which location is to be returned

        Returns
        -------
        value : int
            Integer describing location

        """
        return glGetUniformLocation(self.program_id, name)

    def attribute_location(self, name):
        """ Helper function to get location of an OpenGL attribute variable

        Parameters
        ----------
        name : str
            Name of the variable for which location is to be returned

        Returns
        -------
        value : int
            Integer describing location

        """
        return glGetAttribLocation(self.program_id, name)




class JuMEG_GLS_Plotter(object):
    """ Helper class for using GLSL shader programs
    """
    def __init__(self, vertex=None, fragment=None):

        self.__vao_id   = None
        self.__vao_data = np.array([])

        self.__color =  np.array( [1.0,0.0,1.0,1.0],dtype=np.float32 )
        self.color   =  [0.0, 0.0, 1.0, 1.0] #np.array( [0.0,0.0,1.0,1.0],dtype=np.float32 )


        #self.vertex_pos=None


        self.gls_pgr = None

        self.init_shader(vertex=vertex,fragment=fragment)




    def __get_color(self):
        return self.__color
    def __set_color(self,v):
        self.__color = v
    plot_color = property(__get_color,__set_color)

    def __get_vao_id(self):
        return self.__vao_id
    def __set_vao_id(self,v):
        self.__vao_id = v
    vao_id = property(__get_vao_id,__set_vao_id)


    def __get_vao_data(self):
        return self.__vao_data
    def __set_vao_data(self,data):
        self.__vao_data = data
    vao_data = property(__get_vao_data,__set_vao_data)

    def __get_vbo_data_points(self):
        return int( self.vao_data.size / 2 )

    data_points=property(__get_vbo_data_points)

    print  "INIT"
    print 'Vendor: %s' % (glGetString(GL_VENDOR))
    print 'Opengl version: %s' % (glGetString(GL_VERSION))
    print 'GLSL Version: %s' % (glGetString(GL_SHADING_LANGUAGE_VERSION))
    print 'Renderer: %s' % (glGetString(GL_RENDERER))
    print"done\n"


    def init_shader(self, vertex=None, fragment=None):

        if self.gls_pgr:
           glUseProgram(0)
           if self.gls_pgr.shader_id:
              glDeleteShader(self.gls_pgr.shader_id)

        if vertex is None:

           vertex = """
                        #version 330
                        attribute vec2 xy_pos;
                        uniform vec4 xy_color;
                        uniform vec4 xy_color1;

                        varying vec4 frg_color;

                        void main(void)
                        {
                            frg_color = xy_color;
                            gl_Position = vec4(xy_pos,0, 1.0);
                        }
                    """
           print vertex

        if fragment is None:
           fragment = """
                         #version 330
                         varying vec4 frg_color;

                         void main(void)
                          {
                            gl_FragColor = frg_color;
                          }
                      """
           print fragment

        self.gls_pgr = ShaderProgram(vertex,fragment)
        glUseProgram(self.gls_pgr.program_id)



        self.vertIndex = self.gls_pgr.attribute_location('xy_pos')

        #glVertexAttribPointer(self.gls_pgr.attribute_location('xy_pos'), 2, GL_FLOAT, GL_FALSE, 0, None)

        # Turn on this vertex attribute in the shader
        #glEnableVertexAttribArray(0)


        self.gls_id_xy_color = self.gls_pgr.uniform_location("xy_color")
        print self.gls_id_xy_color

        cid=glGetUniformLocation(self.gls_pgr.program_id,"xy_color1")
        print cid
        glUniform4fv(self.gls_id_xy_color,1, self.color)

        # attributes
        #self.vertex_pos = glGetAttribLocation(self.gls_pgr, "vin_position")


    def init_vao_vbo(self,data=None):

        if data.size:
           self.vao_data=data

        # Lets create a VAO and bind it
        # Think of VAO's as object that encapsulate buffer state
        # Using a VAO enables you to cut down on calls in your draw
        # loop which generally makes things run faster

        self.vertexBuffer = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vertexBuffer)
        glBufferData(GL_ARRAY_BUFFER, 4*len(self.vao_data), self.vao_data,GL_DYNAMIC_DRAW)






        #self.vbo_id = glGenBuffers(1)
        #glBindBuffer(GL_ARRAY_BUFFER, self.vbo_id)
        #vertexData = numpy.array(quadV, numpy.float32)
        #glBufferData(GL_ARRAY_BUFFER, 4*len(self.vao_data), self.vao_data,GL_DYNAMIC_DRAW)
        # self.vao_id = \
        #vao_id=glGenVertexArrays(1,None)

        #self.vao_id=vao_id

        #print vao_id
        #glBindVertexArray(self.vao_id[0])

        # Lets create our Vertex Buffer objects - these are the buffers
        # that will contain our per vertex data
        #self.vbo_id = glGenBuffers(1)

        # Bind a buffer before we can use it
        #glBindBuffer(GL_ARRAY_BUFFER, self.vbo_id[0])

        # Now go ahead and fill this bound buffer with some data
        #glBufferData(GL_ARRAY_BUFFER, ArrayDatatype.arrayByteCount(self.data), self.data, GL_DYNAMIC_DRAW)

        # Now specify how the shader program will be receiving this data
        # In this case the data from this buffer will be available in the shader as the vin_position vertex attribute


        # Now do the same for the other vertex buffer
        #glBindBuffer(GL_ARRAY_BUFFER, vbo_id[1])
        #glBufferData(GL_ARRAY_BUFFER, ArrayDatatype.arrayByteCount(color_data), color_data, GL_STATIC_DRAW)
        #glVertexAttribPointer(program.attribute_location('vin_color'), 3, GL_FLOAT, GL_FALSE, 0, None)
        #glEnableVertexAttribArray(1)

        # Lets unbind our vbo and vao state
        # We will bind these again in the draw loop
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)




    def vao_update(self,data):
        if data.size:
           self.vao_data=data
        # Bind a buffer before we can use it
        glBindBuffer(GL_ARRAY_BUFFER, self.vertexBuffer)

        # Now go ahead and fill this bound buffer with some data
        glBufferSubData(GL_ARRAY_BUFFER,0, ArrayDatatype.arrayByteCount(self.vao_data), self.vao_data)


    def plot(self):

        # Specify shader to be used
        glUseProgram(self.gls_pgr.program_id)

        # Bind VAO - this will automatically
        # bind all the vbo's saving us a bunch
        # of calls
        #glBindVertexArray(self.vao_id)

        glUniform4fv(self.gls_id_xy_color,1, self.plot_color)

        # Modern GL makes the draw call really simple
        # All the complexity has been pushed elsewhere


        glEnableVertexAttribArray(self.vertIndex)


        # set buffers
        glBindBuffer(GL_ARRAY_BUFFER, self.vertexBuffer)
        glVertexAttribPointer(self.vertIndex,2, GL_FLOAT, GL_FALSE, 0, None)

        glDrawArrays(GL_LINE_STRIP, 0, self.data_points-1)

        # Lets unbind the shader and vertex array state
        glUseProgram(0)
        # disable arrays
        glDisableVertexAttribArray(self.vertIndex)

        glBindVertexArray(0)

        # Now lets show our master piece on the screen
        # SwapBuffers()

        # If the user has closed the window in anger
        # then terminate this program
        # running = running and GetWindowParam(OPENED)



    def reset(self):
        glUseProgram(0)
        glDisableVertexAttribArray(self.vertIndex)

        glBindVertexArray(0)


      #  glUseProgram(self.program)

        # set proj matrix
      #  glUniformMatrix4fv(self.pMatrixUniform, 1, GL_FALSE, pMatrix)

        # set modelview matrix
      #  glUniformMatrix4fv(self.mvMatrixUniform, 1, GL_FALSE, mvMatrix)

        # set color
      #  glUniform4fv(self.colorU, 1, self.col0)

        #enable arrays
      #  glEnableVertexAttribArray(self.vertIndex)

        # set buffers
      #  glBindBuffer(GL_ARRAY_BUFFER, self.vertexBuffer)
      #  glVertexAttribPointer(self.vertIndex, 3, GL_FLOAT, GL_FALSE, 0, None)

        # draw
      #  glDrawArrays(GL_TRIANGLES, 0, 6)

        # disable arrays
      #  glDisableVertexAttribArray(self.vertIndex)

          # set color

