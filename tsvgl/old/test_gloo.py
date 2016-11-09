#! /usr/bin/env python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2014, Nicolas P. Rougier. All rights reserved.
# Distributed under the terms of the new BSD License.
# -----------------------------------------------------------------------------
import sys,time
import numpy as np

import OpenGL.GL as gl
import OpenGL.GLUT as glut
from OpenGL.GLUT.freeglut import *

from vispy import gloo
from vispy.gloo import Program, VertexBuffer, IndexBuffer
#from vispy.geometry import MeshData

import jumeg.tsv.jumeg_transforms as jtr


# from vispy import gloo


vertex = """
attribute vec2 data2d;
uniform mat4 TrafoMatrix;
uniform vec4 color;

varying vec4 frg_color;

void main(void) {
	gl_Position = TrafoMatrix * vec4( data2d,0.0, 1.0);
	//gl_Position = vec4(coord2d, 0.0, 1.0);
	frg_color = color;
}"""


fragment = """
varying vec4 frg_color;

void main(void) {
	gl_FragColor = frg_color;
}"""

test_vertex = """
    uniform float scale;
    attribute vec4 color;
    attribute vec2 position;
    varying vec4 v_color;
    void main()
    {
        gl_Position = vec4(scale*position, 0.0, 1.0);
        v_color = color;
    } """

fragment = """
    varying vec4 v_color;
    void main()
    {
        gl_FragColor = v_color;
    }"""


grid_vertex = """
    uniform vec4 color;
    attribute vec2 position;
    varying vec4 v_color;

    void main()
    {
        gl_Position = vec4(position, 0.0, 1.0);
        v_color     = color;
    } """

grid_fragment = """
    varying vec4 v_color;
    void main()
    {
        gl_FragColor = v_color;
    } """


class TSV_TEST_GLOO(object):

      def __init__ (self,  channels=10,timepoints=10000,srate=1017.25):
          super(TSV_TEST_GLOO, self).__init__()


          self.n_channels   = channels
          self.n_timepoints = timepoints
          self.srate        = srate


          self.magrin     = 10
          self.ticksize   = 10
          self.height     = 0.0
          self.width      = 0.0

          self._is_init     = False
          self.is_on_draw   = False

          self._init_data()

        # Build program & data
        # ----------------------------------------
          self.data_pgr = Program(vertex, fragment, count=timepoints)
          #self.data_pgr'color']    = [ (1,0,0,1), (0,1,0,1), (0,0,1,1), (1,1,0,1) ]
          #self.data_pgr['position'] = [ (-1,-1),   (-1,+1),   (+1,-1),   (+1,+1)   ]
          #self.program['scale'] = 1.0

          self.xgrid_pgr = Program(grid_vertex, grid_fragment)
          self.xgrid_pgr['position'] = self.init_xgrid()
          self.xgrid_pgr['color']    = np.array([0.0,0.0,0.0,1.0],dtype=np.float32)

          self.ygrid_pgr = Program(grid_vertex, grid_fragment)
          self.ygrid_pgr['position'] = self.init_ygrid()
          self.ygrid_pgr['color']    = np.array([0.50,0.50,0.50,1.0],dtype=np.float32)

         # v = np.arange(10)
         # MeshData(vertices=None, faces=None, edges=None, vertex_colors=None, face_colors=None)


      def _init_data(self):
          import numpy as np
          ch = self.n_channels
          n  = self.n_timepoints

          self.timepoints = np.arange(n,dtype=np.float32) / self.srate
          self.data       = np.zeros((ch,n), dtype=np.float32)


          self.plot_data     = np.zeros((self.timepoints.size ,2), dtype=np.float32)
          self.plot_data[:,0]= self.timepoints  #x-value

          for i in range( ch ):
              self.data[i,:] = np.sin(self.timepoints * (2 * i+1) * 2* np.pi)

          self.plot_color       = np.repeat(np.random.uniform( size=(ch,4) ,low=.5, high=.9),1,axis=0).astype(np.float32)
          self.plot_color[:,-2] = 1.0

          self.plot_color[:,-1] = 0.0

          self.data_min_max = np.array( [ self.data.min(axis=0),self.data.max(axis=0) ] ).T
         #-- ck for min == max
          min_eq_max_idx = np.array( self.data_min_max.ptp( axis=1 )==0 )

          if min_eq_max_idx.size:
             self.data_min_max[ min_eq_max_idx] += [-1.0,1.0]


      def init_xgrid(self):
         # x axes grid; time
          v      = np.zeros((40,2),dtype=np.float32)

          v[0::2,0] = np.array( np.linspace(-1.0,1.0,20)    ,dtype=np.float32)
          v[1::2,0] = np.array( np.linspace(-1.0,1.0,20)    ,dtype=np.float32)
          v[0::2,1] = -1.0
          v[1::2,1] =  1.0
         #x = np.arange(-1, 1, 0.1,dtype=np.float32)
          #y = np.arange(-1, 1, 0.1,dtype=np.float32)
          #xx, yy = np.meshgrid(x, y)

          #print v
          return v

      def init_ygrid(self):
          # y axes grid; time
          v      = np.zeros((40,2),dtype=np.float32)

          v[0::2,1] = np.array( np.linspace(-1.0,1.0,20)    ,dtype=np.float32)
          v[1::2,1] = np.array( np.linspace(-1.0,1.0,20)    ,dtype=np.float32)
          v[0::2,0] = -1.0
          v[1::2,0] =  1.0
         #x = np.arange(-1, 1, 0.1,dtype=np.float32)
          #y = np.arange(-1, 1, 0.1,dtype=np.float32)
          #xx, yy = np.meshgrid(x, y)
          return v

      def display(self):

          if self.is_on_draw:
             return
          self.is_on_draw=True

          if not self._is_init:
               # self.InitGL()
               #glut.glutInit(sys.argv)
               self._is_init = True
               print"DONE GLOO INIT "

          gl.glClearColor(1,1,1,1)
          gl.glClear(gl.GL_COLOR_BUFFER_BIT)
#----------------------------------------------------------

          tw0 = time.clock()
          t0  = time.time()

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

          gl.glLineWidth(2)

          xmin = self.timepoints[0]
          xmax = self.timepoints[-1]

        #-- copy data
          mvp     = np.zeros( (self.n_channels,4),dtype=np.float32)
          mvp[:,0]= dborder
          mvp[:,1]= np.arange(self.n_channels) * (dh + 2*dborder)
          mvp[0,1]= dborder
          mvp[:,2]= wd
          mvp[:,3]= dh

          idx = 0

         # glEnable(GL_SCISSOR_TEST);

         #glViewport(mvp[idx,0],mvp[idx,1],mvp[idx,2],mvp[idx,3])

          for idx in range( self.n_channels ):

              self.plot_data[:,1]           = self.data[idx,:]  #!!!!TODO reshape on the fly  sub array

              self.data_pgr['data2d'].set_data( self.plot_data)
             # self.data_pgr['color']        = self.plot_color[idx,:]
              self.data_pgr['TrafoMatrix'].set_data( jtr.ortho(xmin,xmax,self.data_min_max[idx,0],self.data_min_max[idx,1],0,1) )

           #---TODO viewport to GLS TrafoMatrix  like C++ example or  Perspective Matrix  as GeometryShader split x,y into VBOs cp only y7signal value
           #--- set border scissor

              gl.glViewport(mvp[idx,0],mvp[idx,1],mvp[idx,2],mvp[idx,3])

              #self.data_pgr.draw(gl.GL_LINE_STRIP)
              self.data_pgr.draw('line_strip')

              self.xgrid_pgr.draw(gl.GL_LINES)

              self.ygrid_pgr.draw(gl.GL_LINES)

          gl.glViewport(0, 0, self.width, self.height)



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


             # if self.do_plot_axis:
             #--- draw zero line
             #   glLineWidth(1)
             #   glColor4f(0.0,0.0,0.0,0.0)

             #   self.set_window(xmin,xmax,self.data_min_max[idx,0],self.data_min_max[idx,1])
             #   dy = self.data_min_max[idx,0] + self.data_min_max[idx,1]/2

              # y0 = self.data_min_max[idx,0] - self.data_min_max[idx,0]
              # self.data_min_max[idx,0],self.data_min_max[idx,1
             #   glBegin(GL_LINES)
              #glVertex3f(-1.0,0.50,0.0)
              #glVertex3f(1.0,0.50,0.0)
             #   glVertex2f(xmin,dy)
             #   glVertex2f(xmax,dy)
              #  glEnd()



              #glRasterPos2f( 1,mvp[idx,1]+mvp[idx,3]/2)
              #  glRasterPos2f( xmin,dy)

              #  glColor4f(1.0,0.0,0.0,1.0)

              #  for idx_chr in str(idx):
              #glutBitmapCharacter(GLUT_BITMAP_HELVETICA_10, ord(idx) )
              #      glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, ord( str(idx_chr) ))


          self.is_on_draw=False

          td  = time.time()  - t0
          tdw = time.clock() - tw0

          print "done draw Time: %10.3f  WallClk: %10.3f \n" % (td,tdw)


      def reshape(self,width,height):
          gl.glViewport(0, 0, width, height)

          #glLineWidth(4)
          #self.width = width
          #self.height = height
          #self.aspect = width/float(height)
          #gl.glViewport(0, 0, self.width, self.height)
          #gl.glEnable(GL_DEPTH_TEST)
          #gl.glDisable(GL_CULL_FACE)
          #gl.glClearColor(0.8, 0.8, 0.8,1.0)


      def keyboard(self, key, x, y ):
          if key == '\033':
             sys.exit( )

# Glut init
# --------------------------------------
#glut.glutInit(sys.argv)
#glut.glutInitDisplayMode(glut.GLUT_DOUBLE | glut.GLUT_RGBA)
#glut.glutCreateWindow('Hello world!')
#glut.glutReshapeWindow(512,512)
#glut.glutReshapeFunc(reshape)
#glut.glutKeyboardFunc(keyboard )
#glut.glutDisplayFunc(display)


# Enter mainloop
# --------------------------------------
# glut.glutMainLoop()

tsv_test_gloo = TSV_TEST_GLOO()