#!/usr/bin/env python3
# -+-coding: utf-8 -+-

"""
"""

#--------------------------------------------
# Authors: Frank Boers <f.boers@fz-juelich.de> 
#
#-------------------------------------------- 
# Date: 08.08.19
#-------------------------------------------- 
# License: BSD (3-clause)
#--------------------------------------------
# Updates
#--------------------------------------------

import sys,logging
logger=logging.getLogger("jumeg")

from OpenGL.GL import *
from OpenGL.GLUT import *
#from OpenGL.GLU import *

from OpenGL.GL.shaders import *

__version__="2019-09-13-001"

class JuMEG_TSV_OGL_Shader_Base(object):
    """
    todo check if shader still compiled after clear
    """
    def __init__(self,verbose=False,debug=False):
        super().__init__()
        self._program_id   = None
        self._isInit       = False
        self._vrt_shader   = None
        self._frt_shader   = None
        self._isActive     = False
        self._attribut_list = None
        self._isBindAttributs = False
        
        self.verbose       = verbose
        self.debug         = debug

    @property
    def isBindAttributs(self): return self._isBindAttributs
    
    @property
    def isActive(self):
        return self._isActive
    
    @property
    def programID(self):
        return self._program_id

    @property
    def isInit(self):
        return self._isInit
    @property
    def vertex_shader(self): return self._vrt_shader
    @property
    def vertex_shader_text(self): return self._vrt_txt
    @vertex_shader_text.setter
    def vertex_shader_text(self,v):
        self._vrt_txt=v
    @property
    def fragment_shader(self): return self._frt_shader
    @property
    def fragment_shader_text(self):
        return self._frg_txt

    @fragment_shader_text.setter
    def frg_shader_text(self,v):
        self._frg_txt = v

    def GetUloc(self,n):
        return glGetUniformLocation(self._program_id,n)

    def SetUloc(self,n):
        #return glGetUniformLocation(self._program_id,n)
        pass
    
    def GetAloc(self,n):
        return glGetAttribLocation(self._program_id,n)
    def SetAloc(self,n):
        #return glGetAttribLocation(self._program_id,n)
        pass
    
    def GetUniformLocation(self,n):
        return glGetUniformLocation(self._program_id,n)

    def GetAttributeLocation(self,n):
        return glGetAttribLocation(self._program_id,n)
    
    def UseProgram(self,status):
        """
        if needed  init & compile shader prg
        activate the shader pgr
        get shader vars for modification
        
        :param status: True/False: select/deselect shader program
        :return:
        """
        if status:
           if not self.isInit:
              self.init()
           
           glUseProgram(self._program_id)
           self.init_vars()
        
        elif self.isActive:
             glUseProgram(0)
        
        self._isActive = status
        return self.isActive
    
    def init_vars(self):
        pass
    
    def clear(self):
        logger.debug(" --> GLSL shader program clear")
        self.UseProgram(False)
       #--- free shaders
        if self.isInit:
           logger.debug(" --> GLSL shader program delete shaders")
           self._vrt_shader = self.delete_shader(self._vrt_shader)
           self._frt_shader = self.delete_shader(self._frt_shader)
           
        if self.debug:
           stat = glGetProgramiv(self._program_id ,GL_DELETE_STATUS)
           logger.debug("---> OGL GLSL SHADER Delete Status\n  -> status {}\n  -> id: {}".format(stat,self._program_id))
           
        if self._program_id:
           glDeleteProgram(self._program_id)
        logger.debug("---> DONE OGL GLSL SHADER Delete program")

        self._program_id = 0
        self._isInit     = False
      
     
    def __del__(self):
        self.clear()
       
    def compile_shader(self,pgr_code,type):
        """
        compile a shader from source
        
        :param type: e.g.: GL_VERTEX_SHADER,GL_FRAGMENT_SHADER
        :return:
        """
        
        try:
           shader = glCreateShader(type)
           glShaderSource(shader,pgr_code)
           glCompileShader(shader)
    
           status = glGetShaderiv(shader,GL_COMPILE_STATUS)
           if not (status):
              raise RuntimeError( glGetShaderInfoLog(shader) )
        except:
           logger.exception("---> Error in compiling shader")
        return shader
   
    def delete_shader(self,shader):
        if shader:
           glDetachShader(self._program_id,shader)
           glDeleteShader(shader)
        return None

    def link_program(self,vert=None,frag=None,program_id=None):
        """
        
        :param vert:
        :param frag:
        :param program_id:
        :return:
        """
        
        if program_id:
           self._program_id = program_id
        
        if vert:
           self._vrt_shader = vert
        if frag:
           self._frt_shader = frag
        
        try:
           if not self._program_id:
              self._program_id = glCreateProgram()
           
           glAttachShader(self._program_id,self._vrt_shader)
           glAttachShader(self._program_id,self._frt_shader)
           
           self._bind_attribLocations()
           
           glLinkProgram(self._program_id)
           status = glGetProgramiv(self._program_id,GL_LINK_STATUS)

           self._vrt_shader = self.delete_shader(self._vrt_shader)
           self._frt_shader = self.delete_shader(self._frt_shader)
           
           if not (status):
              raise RuntimeError(glGetProgramInfoLog(self._program_id))
        except:
           logger.exception("---> Error in linking shader")
    
        return self._program_id
    
    def _bind_attribLocations(self):
        """
        https://www.khronos.org/opengl/wiki/Tutorial2:_VAOs,_VBOs,_Vertex_and_Fragment_Shaders_(C_/_SDL)
        :return:
        """
        self._isBindAttributs = False
        if isinstance(self._attribut_list,(list)):
           idx=0
           for attrb in self._attribut_list:
               glBindAttribLocation(self._program_id,idx,attrb)
               idx+=1
        self._isBindAttributs = True
    
    def init(self,vtxt=None,ftxt=None):
        """
        call this in subclass __init__()
        
        create vertex and fragment shader program
        https: // rdmilligan.wordpress.com / 2016 / 0 8 / 27 / opengl - shaders - using - python /
        """
        # ck & free shader pgr memory
        if self.isInit:
           self.clear()
           
        if vtxt:
           self.vertex_shader_text  = vtxt
        if ftxt:
           self.fragment_shader_text= ftxt
           
        self._program_id = None
        self._isInit     = False
       
        glsl_version = None
        stat         = None
       
        try:
          # GL_DELETE_STATUS, GL_LINK_STATUS, GL_VALIDATE_STATUS,GL_INFO_LOG_LENGTH,GL_ATTACHED_SHADERS,
          # GL_ACTIVE_ATTRIBUTES,GL_ACTIVE_ATTRIBUTE_MAX_LENGTH,GL_ACTIVE_UNIFORMS,GL_ACTIVE_UNIFORM_MAX_LENGTH.
    
            glsl_version = glGetString(GL_SHADING_LANGUAGE_VERSION)
      
            if not glsl_version:
               raise ValueError
            
            self._vrt_shader = self.compile_shader(self.vertex_shader_text,  GL_VERTEX_SHADER)
            self._frt_shader = self.compile_shader(self.fragment_shader_text,GL_FRAGMENT_SHADER)
            self._program_id = self.link_program()
            
            stat = glGetProgramiv(self._program_id ,GL_LINK_STATUS)
           
            if self._program_id:
               self._isInit   = True
               self._isActive = False
        except:
            logger.exception("---> No shading language version defined!!!\n --> call glutInit() first?")

        logger.debug("---> OGL init GLSL SHADER\n  -> GL_SHADING_LANGUAGE_VERSION: {}\n  -> status {}\n  -> id: {}".
                     format(glsl_version,stat,self._program_id))

        return self._isInit


class JuMEG_TSV_OGL_Shader_PlotSignal(JuMEG_TSV_OGL_Shader_Base):
    """
    
    """
    __slots__=["_u_colour","_u_trafo_matrix","_u_dcoffset"]
    
    def __init__(self,**kwargs):
        # self._shader_version= 330
       
       #--- Vertex shader
        self._vrt_txt ='''
        /* vertex shader
           https://www.khronos.org/opengl/wiki/Layout_Qualifier_(GLSL)
        */
        #version 330
        //precision mediump float;
        
        //--- attribute variable that contains coordinates of the vertices.
        layout(location = 0) in vec2 a_position; // location=>0 first VBO !!!
        //in vec2 a_position;
        
        //--- matrix to transfer vertex points into world coordinates
        uniform mat4 u_trafo_matrix;
        
        //--- vertex colour passed to fragment shader
        uniform vec4 u_colour; // lowp: low resolution
        out     vec4 frg_colour;

        uniform float u_dcoffset =  0.0;
        
        //uniform vec4 u_colour;
        //out     vec4 frg_colour;

        //-----------------------------------
        //--- MAIN
        //-----------------------------------
        void main()
        {
          gl_Position = u_trafo_matrix * vec4(a_position.x, a_position.y  - u_dcoffset, 0., 1.);
          frg_colour   = u_colour;
        }
        '''
       #--- Fragment shader
        self._frg_txt='''
        /* fragment shader
           Output variable of the fragment shader, which is a 4D vector containing the
           RGBA components of the pixel colour.+
        */
        #version 330
        
        // precision lowp float;
        //precision medium float;
        
        in vec4 frg_colour;
        out vec4 out_colour;

        void main(void) {
                         out_colour = frg_colour;
                         //out_colour = vec4(1., 1., 0., 1.);
                        }
        
        '''

        self._isActive = False
       #--- set list to bind attributes to buffer number
        self._attribut_list = None  # "a_position"

        #--- do shader stuff compile,link
        super().__init__(**kwargs)
        
        self.init()
        
    @property
    def position(self): return self._a_position
    
    def SetColour(self,v):
        """
        set vertex colour <u_colour>
        :param v:
        :return:
        """
        glUniform4fv(self._u_colour,1,v)
    
    def SetDCoffset(self,v):
        """
        set dcoffset>
        :param v:
        :return:
        """
        glUniform1f(self._u_dcoffset,v)
   
    def SetTrafoMatrix(self,v):
        """
        sets trafo matrix for vertex   <u_trafo_matrix>
        :param v:
        :return:
        """
        glUniformMatrix4fv(self._u_trafo_matrix,1,GL_FALSE,v)

    
    def init_vars(self):
        """
        get vars from shader pgr
        :return:
        """
        # if not self.isBindAttributs:
        self._a_position     = self.GetAloc('a_position')
        self._u_dcoffset     = self.GetUloc('u_dcoffset')
        self._u_colour       = self.GetUloc('u_colour')
        self._u_trafo_matrix = self.GetUloc('u_trafo_matrix')
