from OpenGL.GL import (GL_TRUE,GL_FRAGMENT_SHADER,GL_LINK_STATUS,
     GL_VERTEX_SHADER, glAttachShader,glCompileShader,GL_COMPILE_STATUS,
     glCreateProgram,glCreateShader, glDeleteProgram,glGetAttribLocation,
     glDeleteShader,glGetProgramInfoLog, glGetProgramiv, glGetShaderInfoLog,
     glGetShaderiv,glGetUniformLocation, glLinkProgram,glShaderSource,glUseProgram)

#  glBindBuffer,glEnableVertexAttribArray,glGetString, glVertexAttribPointer,GL_SHADING_LANGUAGE_VERSION

class JuMEG_TSV_GLSL(object):
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
        self.id_vertex  = self.add_shader(vertex, GL_VERTEX_SHADER)
        self.id_frag    = self.add_shader(fragment, GL_FRAGMENT_SHADER)

        glAttachShader(self.program_id, self.id_vertex)
        glAttachShader(self.program_id, self.id_frag)
        glLinkProgram(self.program_id)

        if glGetProgramiv(self.program_id, GL_LINK_STATUS) != GL_TRUE:
            info = glGetProgramInfoLog(self.program_id)
            glDeleteProgram(self.program_id)
            glDeleteShader(self.id_vertex)
            glDeleteShader(self.id_frag)
            raise RuntimeError('Error linking program: %s' % (info))
        glDeleteShader(self.id_vertex)
        glDeleteShader(self.id_frag)

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

    def uloc(self,n):
        return glGetUniformLocation(self.program_id, n)

    def aloc(self,n):
        return glGetAttribLocation(self.program_id, n)

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

