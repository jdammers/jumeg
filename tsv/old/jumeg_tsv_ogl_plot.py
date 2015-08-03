import sys
import numpy as np

from OpenGL.GL import *

from OpenGL.raw.GL.ARB.vertex_array_object import glGenVertexArrays,glBindVertexArray



class JuMEG_TSV_OGL_PLOT2D(object):
    """ Helper class for using GLSL shader programs
    """
    def __init__(self):

        self.is_on_draw= False

        self.__vbo_data = np.array([],dtype=np.float32)
        self.__vbo_color= np.array([1.0,0.0,0.0,1.0],dtype=np.float32)

        self.__vbo_id        = np.array([],dtype=np.uint8)

        self.__vbo_id_data   = 0
        self.__vbo_id_border = 1
        self.__vbo_id_ticks  = 2

        self.__vbo_plot_border = np.array([ [-1, -1],[1, -1],[1, 1],[-1, 1] ],dtype=np.float32 )

        self._vbo_isinit = False

 #===== PROPERTIES



"""


    def subplot(self):
        if self.is_on_draw:
           return

                  if  (self.plot_parameter['subplot'] == 'subplot' ): , # waterfall,sensorlayout,contour
                      self.plot_subplot()

                  if  (self.plot_parameter['subplot'] == 'waterfall' ): , # waterfall,sensorlayout,contour
                      print "waterfall\n\n"  # self.plot_waterfall()



                  if self.dragging:
                     print"no update on drag"
                     return


                  t0  = time.time()
                  tw0 = time.clock()

                  self.is_on_draw = True

                  #glMatrixMode(GL_PROJECTION)
                  #glLoadIdentity()




                #print xmin
                #print xmax
              #---start sub plots

                  dborder = self.magrin + self.ticksize
                  #--- start first channel at top


                #  !! transpose pixe border magrin to plot coordinates

                  dh = int( self.height / self.data.shape[0] ) - 2 * dborder

                  if dh <  dborder:
                     dh = int( self.height / self.data.shape[0] )
                     dborder= dh * 0.1
                     dh   -=  2 * dborder

                  w0 = dborder
                  wd = self.width - dborder *2



                  if (w0 < 1) or (wd < 50) :
                     self.is_on_draw=False
                     return False



                   #--- reshape
                  glClearColor(1.0,1.0,1.0,0.0)
                  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

                  glLineWidth(2)

                # build projection matrix
                #fov = math.radians(45.0)
                #f = 1.0/math.tan(fov/2.0)
                #zN, zF = (0.1, 100.0)
                #a = self.aspect

                  xmin=self.timepoints[0]
                  xmax=self.timepoints[-1]


                 # h0 = dborder

                 # h1 = dh
                 # ymin =-1.0
                 # ymax = 1.0

        #glViewport(w0,h0,w1,dh)
        #int window_width = glutGet(GLUT_WINDOW_WIDTH);
        #int window_height = glutGet(GLUT_WINDOW_HEIGHT);

        #glViewport(
        #  margin + ticksize,
        #  margin + ticksize,
        #  window_width - margin * 2 - ticksize,
        #  window_height - margin * 2 - ticksize
        #);
                  mvp = np.zeros( (self.n_channels,4),dtype=np.float32)
                  mvp[:,0]= dborder
                  mvp[:,1]= np.arange(self.n_channels) * (dh + 2*dborder)
                  mvp[0,1]= dborder
                  mvp[:,2]=wd
                  mvp[:,3]=dh

                 # glEnable(GL_SCISSOR_TEST);

                 # dpos = ymin + (ymax - ymin) / 2.0

                  #self.plot2d.trafo_matrix = jtr.ortho(xmin,xmax,ymin,ymax,0,1)
                  #print"--------------------START LLOP TP"
                  #self.plot2d.vbo_data_timepoints = self.timepoints

                  #print"--------------------START LLOP"
                  #self.plot2d.vbo_data_signal     = self.data[0,:]

                #print self.plot2d.vbo_data

                 # print self.height
                 # print self.width
                 # print dh
                 # print mvp

                  self.plot2d.vbo_data_timepoints = self.timepoints

                  idx = 0



                  #self.plot2d.vbo_isinit = False


                  #self.plot2d.vbo_data_signal[:]  = self.data[idx,:]
                  #self.plot2d.vbo_color           = self.plot_color[idx,:]

                  # self.plot2d.vbo_init()
                  # trafo_matrix        = jtr.ortho(xmin,xmax,self.data_min_max[idx,0],self.data_min_max[idx,1],0,1)

                  #glViewport(mvp[idx,0],mvp[idx,1],mvp[idx,2],mvp[idx,3])
                  #self.plot2d.render()

                  for idx in range( self.n_channels ):

                      self.plot2d.vbo_data_signal[:]  = self.data[idx,:]
                      self.plot2d.vbo_color           = self.plot_color[idx,:]
                      self.plot2d.trafo_matrix        = jtr.ortho(xmin,xmax,self.data_min_max[idx,0],self.data_min_max[idx,1],0,1)

                   #---TODO viewport to GLS TrafoMatrix  like C++ example or  Perspective Matrix  as GeometryShader split x,y into VBOs cp only y7signal value
                      glViewport(mvp[idx,0],mvp[idx,1],mvp[idx,2],mvp[idx,3])
                      self.plot2d.render()

                      #glMatrixMode(GL_PROJECTION)
                      #glLoadIdentity()


                     # print"\n\n"
                     #glScissor(
                     #  margin + ticksize,
                     #  margin + ticksize,
                     #  window_width - margin * 2 - ticksize,
                     #  window_height - margin * 2 - ticksize
                     #);
                      #self.set_viewport(w0,wd,h0,dh)
                      #self.set_window(xmin,xmax,ymin,ymax )


                      if self.do_plot_axis:
                     #--- draw zero line
                        glLineWidth(1)
                        glColor4f(0.0,0.0,0.0,0.0)

                        self.set_window(xmin,xmax,self.data_min_max[idx,0],self.data_min_max[idx,1])
                        dy = self.data_min_max[idx,0] + self.data_min_max[idx,1]/2

                      # y0 = self.data_min_max[idx,0] - self.data_min_max[idx,0]
                      # self.data_min_max[idx,0],self.data_min_max[idx,1
                        glBegin(GL_LINES)
                      #glVertex3f(-1.0,0.50,0.0)
                      #glVertex3f(1.0,0.50,0.0)
                        glVertex2f(xmin,dy)
                        glVertex2f(xmax,dy)
                        glEnd()

                      #glRasterPos2f( 1,mvp[idx,1]+mvp[idx,3]/2)
                        glRasterPos2f( xmin,dy)

                        glColor4f(1.0,0.0,0.0,1.0)

                        for idx_chr in str(idx):
                      #glutBitmapCharacter(GLUT_BITMAP_HELVETICA_10, ord(idx) )
                            glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, ord( str(idx_chr) ))

                        glViewport(0, 0, self.width, self.height)
                   # print idx

                     # h0 += dh + dborder

                      #print h0
                     # h1 += h0 + dh

                # swap buffers


                  glFlush()
                  glutSwapBuffers()
                 # glViewport(0, 0, self.width, self.height)

                  self.plot2d.vbo_reset()
                  self.is_on_draw = False

                  td  = time.time()  - t0
                  tdw = time.clock() - tw0

                  print "done draw Time: %10.3f  WallClk: %10.3f \n" % (td,tdw)





























 #--- vob_id
    def __get_vbo_id(self):
        return self.__vbo_id

    def __set_vbo_id(self,d):
        self.__vbo_id = d
        self.vbo_isinit = False

    vbo_id=property(__get_vbo_id,__set_vbo_id)

#--- vob_id_data
    def __get_vbo_id_data(self):
        return self.__vbo_id[ self.__vbo_id_data]

    vbo_id_data=property(__get_vbo_id_data)

#--- vob_id_border
    def __get_vbo_id_border(self):
        return self.__vbo_id[ self.__vbo_id_border]

    vbo_id_border=property(__get_vbo_id_border)

#--- vob_id_ticks
    def __get_vbo_id_ticks(self):
        return self.__vbo_id[ self.__vbo_id_ticks]

    vbo_id_ticks=property(__get_vbo_id_ticks)



 #--- vob_plot_border
    def __get_vbo_plot_border(self):
        return self.__vbo_plot_border
    def __set_vbo_plot_border(self,d):
        self.__vbo_plot_border = d
        self.vbo_isinit = False

    vbo_plot_border=property(__get_vbo_plot_border,__set_vbo_plot_border)

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
    vbo_data_signal=property(__get_vbo_data_y,__set_vbo_data_y)

   #--- vbo_data_timepoints
    def __get_vbo_data_x(self):
        return self.__vbo_data[0:-1:2]

    def __set_vbo_data_x(self,d):
        self.__vbo_data = np.zeros( 2*d.size,dtype=np.float32)
        self.__vbo_data[0:-1:2] = d
        self.vbo_isinit = False
        self.vbo_init()

    vbo_data_timepoints=property(__get_vbo_data_x,__set_vbo_data_x)

    def __get_vbo_data_points(self):
        return self.vbo_data_timepoints.size
    data_points=property(__get_vbo_data_points)

   #--- vob_isinit
    def __get_vbo_isinit(self):
        return self.__vbo_isinit
    def __set_vbo_isinit(self,d):
        self.__vbo_isinit = d
    vbo_isinit=property(__get_vbo_isinit,__set_vbo_isinit)

   #---vbo_color
    def __get_vbo_color(self):
        return self.__vbo_color
    def __set_vbo_color(self,d):
        self.__vbo_color = d
    vbo_color=property(__get_vbo_color,__set_vbo_color)


    def vbo_reset(self):
        glUseProgram(0)
        #--TODO enable/destroy all buffers
        glDisableVertexAttribArray(self.vertIndex)

        glBindVertexArray(0)
        self.vbo_isinit = False

    def vbo_init(self,data=None):
        if data:
           self.vbo_data = data

        if self.vbo_id.size:
           self.vbo_reset()

       #--- vertices
        self.vbo_id = glGenBuffers(3)

        #print"VBO INIT "
        #print  self.vbo_id
        #print type(self.vbo_id)

        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_id[0])
        glBufferData(GL_ARRAY_BUFFER, 4*len(self.vbo_data),self.vbo_data,GL_DYNAMIC_DRAW)
       #--- plot border
       # glBindBuffer(GL_ARRAY_BUFFER, self.vbo_id[1] )
       # glBufferData(GL_ARRAY_BUFFER, 4*len(self.vbo_plot_border),self.vbo_plot_border, GL_STATIC_DRAW)

        self.vbo_isinit = True
        #print"done vbo init"

    def vbo_update(self,data=None):
         if data:
            if (data.size != self.vbo_data.size) :
               self.vbo_isinit=False
            self.vbo_data=data

         if not self.vbo_isinit:
            self.vbo_init()

        #--- vertices
         glBindBuffer(GL_ARRAY_BUFFER, self.vbo_id_data)
        #--- TODO ck if only y/signal value can be copied step 2
         glBufferSubData(GL_ARRAY_BUFFER,0, 4*len(self.vbo_data), self.vbo_data)

    def vbo_update_tp(self,data=None):
        if data:
            if (data.size != self.data_points) :
               self.vbo_isinit=False
        self.vbo_data_timepoints=data

        # if not self.vbo_isinit:
        self.vbo_init()

        #--- vertices
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_id_data)
        #--- TODO ck if only y/signal value can be copied step 2
        glBufferSubData(GL_ARRAY_BUFFER,4*self.data_points, 4*len(self.vbo_data_timepoints), self.vbo_data_timepoints)

    def vbo_update_signal(self,data=None):
        if data:
           if (data.size != self.data_points) :
               self.vbo_isinit=False
           self.vbo_data_signal=data

        if not self.vbo_isinit:
           self.vbo_init()

        #--- vertices
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_id_data)
        #--- TODO ck if only y/signal value can be copied step 2
        glBufferSubData(GL_ARRAY_BUFFER,0,4*len(self.vbo_data_signal), self.vbo_data_signal)




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