import time,sys
import numpy as np
import ctypes

import mne

import OpenGL 
OpenGL.ERROR_ON_COPY = True 
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from OpenGL.GL.shaders import *



import jumeg.tsvgl.utils.jumeg_transforms as jtr

from  jumeg.tsvgl.ogl.jumeg_tsv_ogl_gsl    import JuMEG_TSV_OGL_GSL
from  jumeg.tsvgl.ogl.jumeg_tsv_ogl_vbo    import JuMEG_TSV_OGL_VBO

from  jumeg.tsvgl.plot2d.jumeg_tsv_plot2d_data_info import JuMEG_TSV_PLOT2D_DATA_INFO
from  jumeg.tsvgl.plot2d.jumeg_tsv_plot2d_options   import JuMEG_TSV_PLOT2D_OPTIONS
     
 
              
class JuMEG_TSV_PLOT2D_OGL(object):
      """
         Helper class for using GLSL shader programs
      """

      def __init__ (self,size=None,channels=10,timepoints=10000,sfreq=1017.25,demo_mode=True):
          super(JuMEG_TSV_PLOT2D_OGL, self).__init__()
          #super(JuMEG_TSV_PLOT2D_OGL, self).__init__()
#---
          self.__demo_mode     = demo_mode
          self.__size_in_pixel = size
          self.__size_in_mm    = None
                    
          self.raw             = None
          self.tsvdata         = None          
          
          self.opt             = JuMEG_TSV_PLOT2D_OPTIONS()
          self.info            = JuMEG_TSV_PLOT2D_DATA_INFO()
     
     #--- data raw    
          self.data_is_init          = False
          self.data_is_update        = False
          self.data_channel_selection_is_update = False    
          
      #--- test data
          self.opt.plots             = channels
          self.opt.channels.counts   = channels
          
          self.opt.time.timepoints   = timepoints
          self.opt.time.sfreq        = sfreq
          
          self.opt.time.start        = 0.0
          self.opt.time.window       = self.opt.time.end / 10.0    
          self.opt.time.scroll_speed = self.opt.time.window/2.0          
      
      #--- axis; plot parameter
          self.magrin     = 2 #10
          self.ticksize   = 2 #10
          self.height     = 0.0
          self.width      = 0.0

          self.data_is_init = False
          self.is_on_draw   = False

          self.clback = np.array([1.0,1.0,1.0,1.0],'f')

          
          self.initGL()
        #-----------------------------------------
        # GLSL vertex fragment
        #
        # ----------------------------------------
          self.init_glsl()

        #----------------------------------------
        #--> data vbo  sig-channel (x,y)
        #--> x ticks lines vbo
        #--> y ticks lines vbo
          self.init_vbo()

          self.init_plt_box()
          
          #self.xgrid_pgr = Program(grid_vertex, grid_fragment)
          #self.xgrid_pgr['position'] = self.init_xgrid()
          #self.xgrid_pgr['color']    = np.array([0.0,0.0,0.0,1.0],dtype=np.float32)

          #self.ygrid_pgr = Program(grid_vertex, grid_fragment)
          #self.ygrid_pgr['position'] = self.init_ygrid()
          #self.ygrid_pgr['color']    = np.array([0.50,0.50,0.50,1.0],dtype=np.float32)

         # v = np.arange(10)
         # MeshData(vertices=None, faces=None, edges=None, vertex_colors=None, face_colors=None)

      def print_info(self,s):
          print"\n --->  " + self.__class__.__name__ +" -> "+ str(s)  +"\n"
     
      def __get_size_in_pixel(self):
          return self.__size_in_pixel

      def __set_size_in_pixel(self,v):
          self.__size_in_pixel = v

      size_in_pixel= property(__get_size_in_pixel,__set_size_in_pixel)

      def __get_size_in_mm(self):
          return self.__size_in_mm

      def __set_size_in_mm(self,v):
          self.__size_in_mm = v

      size_in_mm = property(__get_size_in_mm,__set_size_in_mm)

      def init_glsl(self):
          # load vertex frag shader -> plot 2d data and x/y axis lines
          self.GLSL = JuMEG_TSV_OGL_GSL()
          self.GLSL.load_shaders_from_file()
          self.GLSL.init_shaders()

          #print"DONE init glsl"


      def init_vbo(self):
          self.vbo_sig_data = JuMEG_TSV_OGL_VBO()

          self.vbo_xaxis    = JuMEG_TSV_OGL_VBO()
          self.vbo_yaxis    = JuMEG_TSV_OGL_VBO()


          # init vbos
          # data
          # xaxis
          #yaxis
          #print "DONE init vbo"

          #vertPoints = someArray[:,:2].flatten().astype(ctypes.c_float)
          #vertices_gl = vertPoints.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
 
 #ToDO update data channels for group& channel selection !!!
 # will not be 248 MEG  ...
      def update_data(self): 
          
          self.print_info("start update data")
          self.data_is_update = False
                    
          if not self.data_is_init:
             self.init_raw_data()
             return
          #set color RGBA
          #set range min max adjust 
        
         # if self.info.do_update:
         #--- calc channel range
         # self.opt.channels.start=1          
         # self.opt.channels.channels_to_display = 10
         # self.opt.channels.counts    = self.raw._data.shape[0]
     
         #---cp index and select selected channels for plotting  
          if not self.data_channel_selection_is_update:
             self.selected_channel_index = self.info.selected_channel_index()
             self.data_selected = self.raw._data[self.selected_channel_index,: ]                  
            
             self.opt.channels.start  = 1          
             # self.opt.channels.channels_to_display = 10
             self.opt.channels.counts = self.data_selected.shape[0]
             self.data_channel_selection_is_update = True
             
        #--- calc channel range  0 ,-1
          ch_start,ch_end_range = self.opt.channels.index_range()     
         
        #--- calc tsl timepoints
          tsl0,tsl1_range = self.opt.time.index_range()
      
        #---
          if self.opt.time.do_scroll:
             self.timepoints = self.raw_timepoints[tsl0:tsl1_range]
             n_timepoints    = self.timepoints.size
             
         #--init VBO data with timepoints
             self.data_4_vbo        = np.zeros(( n_timepoints,2), dtype=np.float32).flatten()
             self.data_4_vbo_tp     = self.data_4_vbo[0:-1:2]
             self.data_4_vbo_tp[:]  = self.timepoints # self.plot_data[:,0]  # self.timepoints   
         
         #---init part min max foreach tp MEGs  => NO BADs?
         #--- ToDo  exclude deselected groups, channels via index
          if self.opt.do_scroll:
             # self.selected_data  
             #self.data = self.raw._data[ch_start:ch_end_range,tsl0:tsl1_range]  #.astype(np.float32)
             
             #self.data = self.raw._data[ch_start:ch_end_range,tsl0:tsl1_range]  #.astype(np.float32)
             self.data = self.data_selected[ch_start:ch_end_range,tsl0:tsl1_range]  #.astype(np.float32)
         #--init VBO data with data
             self.data_4_vbo_sig = self.data_4_vbo[1::2]
              
           #--- if remove dcoffset         
             self.data_mean        = np.mean(self.data, axis = -1)   
             self.data            -= self.data_mean[:, np.newaxis] 
             n_ch,n_timepoints     = self.data.shape
             self.data_min_max_sig = np.array([self.data.min(axis=0),self.data.max(axis=0)],dtype=np.float32).T.flatten()
 
          #--TODO opengl color buffer wx color palette get index
         
             self.data_min_max = np.zeros( [n_ch,2])
             self.data_min_max = np.array( [ self.data.min(axis=1),self.data.max(axis=1) ] ).T

         #-- ck for min == max
             min_eq_max_idx = np.array( self.data_min_max.ptp( axis=1 )==0 )

             if min_eq_max_idx.size:
                self.data_min_max[ min_eq_max_idx] += [-1.0,1.0]
        
             self.data_is_update=True      
          
          self.print_info("done update data")    
             
          return self.data_is_update
          
      
      def init_raw_data(self,raw=None):
          self.data_is_init  = False
          self.data_is_update= False 
          self.data_channel_selection_is_update = False
              
          if raw:
             self.raw=raw          
          if not self.raw:
             return       
             
          self.print_info("start init raw data")
          
          #self.tsvdata = tsvdata
          #print tsvdata.raw.info
          
        #--- update options  
          self.opt.time.timepoints = self.raw.n_times
          self.opt.time.sfreq      = self.raw.info['sfreq']
          self.opt.time.start      = 0.0
          self.opt.time.window     = 5.0
          self.opt.time.scroll_speed = 1.0
          self.raw_timepoints      = np.arange(self.opt.time.timepoints,dtype=np.float32) / self.opt.time.sfreq

         
          
         # TODO check for pretime !!!
         # self.opt.time.end        = self.opt.time.tsl2time( self.opt.time.timepoints ) 
        
        #--- calc channel range
          self.opt.channels.start=1          
          self.opt.channels.channels_to_display = 10
          self.opt.channels.counts    = self.raw._data.shape[0]
           
        #--- init info setting channel plot option e.g. color
          self.info.init_info(raw=self.raw)
            
        #--- RGBA color setting for group & channel via idx and lookup tab
          self.data_is_init = True
          self.update_data()
          
          self.print_info("done init raw data")
 
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

          #print v.flatten()
          return v.flatten()

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
          return v.flatten()

      def init_plt_box(self):            
          
          self.plt_box_vbo = glGenBuffers(1)

          self.plt_box_data = np.array([-1,-1,1,-1,1,1,-1,1],dtype=np.float32)  # {-1, -1}, {1, -1}, {1, 1}, {-1, 1}

          #glBindBuffer(GL_ARRAY_BUFFER, self.plt_box_vbo)
          #glBufferData(GL_ARRAY_BUFFER, 4*len(self.plt_box_data),self.plt_box_data,GL_DYNAMIC_DRAW)
          
       #--- plot border
       # glBindBuffer(GL_ARRAY_BUFFER, self.vbo_id[1] )
       # glBufferData(GL_ARRAY_BUFFER, 4*len(self.vbo_plot_border),self.vbo_plot_border, GL_STATIC_DRAW)


      def initGL(self):
          """
          :param size:
          :return:
          """
          self.print_info("OGL -> start initGL window")
          
          glutInit(sys.argv)
          self.clear_display()
            
          self.print_info("done OGL -> initGL")
          return True
          
      def clear_display(self):
          """
          :param size:
          :return:
          """
          self.print_info("OGL -> start clar display")
          
          glutInit(sys.argv)

          glMatrixMode(GL_PROJECTION)
          glLoadIdentity()
       
          glViewport(0,0,self.size_in_pixel.width,self.size_in_pixel.height)
          r,g,b,a = self.clback
          glClearColor(r,g,b,a)
          glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
          glEnable(GL_LINE_SMOOTH)
          glDisable(GL_SCISSOR_TEST)
          
          self.print_info("done OGL -> clear display")
          return True
      
      
      def draw_axis(self,idx):
          '''     
          transform = viewport_transform(
                                         margin + ticksize,
                                         margin + ticksize,
                                         window_width - margin * 2 - ticksize,
                                         window_height - margin * 2 - ticksize,
                                          )                                    
          glUniformMatrix4fv(uniform_transform, 1, GL_FALSE, glm::value_ptr(transform));
          
          GLuint box_vbo;
          glGenBuffers(1, &box_vbo);
          glBindBuffer(GL_ARRAY_BUFFER, box_vbo);

          static const point box[4] = {{-1, -1}, {1, -1}, {1, 1}, {-1, 1}};
          glBufferData(GL_ARRAY_BUFFER, sizeof box, box, GL_STATIC_DRAW);

          GLfloat black[4] = {0, 0, 0, 1};
          glUniform4fv(uniform_color, 1, black);

          glVertexAttribPointer(attribute_coord2d, 2, GL_FLOAT, GL_FALSE, 0, 0);
          glDrawArrays(GL_LINE_LOOP, 0, 4);
          
          ''' 

        # [    2.   873.  1906.    93.]


          mvp      = self.mvp[idx]
          magrin   = mvp[0]
          ticksize = 15
         
          print self.mvp[idx]
          trafo_mat = jtr.viewport_transform(margin + ticksize, margin + ticksize,
                                             mvp[2] - mnv[0] - margin * 2 - ticksize,
                                             mvp[3] - mvp[1] - margin * 2 - ticksize)
          
          
          print "Trafo Mat"
          print trafo_mat
          
          #jtr.viewport_transform(margin + ticksize, margin + ticksize,
          #                       self.size        window_width - margin * 2 - ticksize,
          #                               window_height - margin * 2 - ticksize,)




      def GLBatch(self,idx,ch_idx,data=None,tlines=GL_LINE_STRIP):
          
          self.trafo_matrix = jtr.ortho(self.xmin,self.xmax,self.data_min_max[idx,0],self.data_min_max[idx,1],0,1)
          #print "TRMAT"
          #print self.trafo_matrix
                  
          #---TODO viewport to GLS TrafoMatrix  like C++ example or  Perspective Matrix  as GeometryShader split x,y into VBOs cp only y7signal value
           #--- set border scissor
          #i=idx
          #idx=0          
         
          glViewport(self.mvp[idx,0],self.mvp[idx,1],self.mvp[idx,2],self.mvp[idx,3])
          
            #{-1, -1}, {1, -1}, {1, 1}, {-1, 1}
 
          #glUniform4fv(self.glsl_u_color, 1,[0.5,0.7,1.0,0.0] ) 
          # [-1. -1.  1. -1.  1.  1. -1.  1.]
           #4.  378.  622.   34.]
        
          
#glGenBuffers(1, &box_vbo);
#glBindBuffer(GL_ARRAY_BUFFER, box_vbo);

#static const point box[4] = {{-1, -1}, {1, -1}, {1, 1}, {-1, 1}};
#glBufferData(GL_ARRAY_BUFFER, sizeof box, box, GL_STATIC_DRAW);

#GLfloat black[4] = {0, 0, 0, 1};
#glUniform4fv(uniform_color, 1, black);

#glVertexAttribPointer(attribute_coord2d, 2, GL_FLOAT, GL_FALSE, 0, 0);
#glDrawArrays(GL_LINE_LOOP, 0, 4);          

          
          #glRasterPos2f(self.mvp[idx,0],self.mvp[idx,1])
          #glLoadIdentity()          
          
          #glRasterPos2f( -1.0,0.0)
          #glColor4f(0.0,0.0,0.0,1.0)
          
          
          #for idx_chr in str( self.info.plt_channels.label[ch_idx] ):
          #    glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, ord( idx_chr ) )
           #  # glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, ord( str(idx_chr) ))
       
      
         
         
          #glScissor(0.1,0.1,0.8,0.8)
          #glClearColor(0.5,0.5,0.5,0.0)
          #glClear(GL_COLOR_BUFFER_BIT)
         

          #idx=i 
          glUseProgram(self.GLSL.program_id)

         #--- set trafo matrix
          glUniformMatrix4fv(self.glsl_u_trafomatrix, 1, GL_FALSE, self.trafo_matrix)
         #--- set color
         
         
         # color_idx = self.info.plt_channels.color_index[ch_idx]
         # co = self.info.plt_color.index2colour(color_idx)
        
          
          # glUniform4fv(self.glsl_u_color, 1,self.plot_color[idx])
         
          glUniform4fv(self.glsl_u_color, 1,self.info.index2channel_color(ch_idx)  ) #co)

         #--- enable arrays
          glEnableVertexAttribArray(self.glsl_a_position2d)
          self.vbo_sig_data.vbo_update_y( data=data)#.ctypes.data )
          # self.vbo_sig_data.vbo_update_y( data=self.data[idx,:] )
  
        
          glVertexAttribPointer(self.glsl_a_position2d,2, GL_FLOAT, GL_FALSE, 0, None)
          glDrawArrays(tlines, 0, self.vbo_sig_data.data_points-1)

          glDisableVertexAttribArray(self.glsl_a_position2d)
 
         # print self.trafo_matrix
         # imat = np.array(np.identity(4),dtype=np.float32)

         
      #!!!    glUniformMatrix4fv(self.glsl_u_trafomatrix, 1, GL_FALSE, viewport_transform( x,y,width,height,window_width,window_height):
         
         # glUniformMatrix4fv(self.glsl_u_trafomatrix, 1, GL_FALSE, imat)
          #glEnableVertexAttribArray(self.glsl_a_position2d)
         # glUniform4fv(self.glsl_u_color, 1,(0.7,0.5,0.0,1.0) ) #co)

 
         # vbox = self.mvp[idx,:]
          
          #glColor4f(0.0,0.5,0.0,1.0)
         # self.plt_box_data = np.array([vbox[0],vbox[1], vbox[2] +vbox[0],vbox[1], vbox[2]+vbox[0], vbox[1]+vbox[3],vbox[0],vbox[1]+vbox[3]],dtype=np.float32)


         # print "BOX Test"
          #print self.plt_box_data
          #  self.plt_box_data = self.mvp[idx,:]

         # glBindBuffer(GL_ARRAY_BUFFER, self.plt_box_vbo)
         # glBufferData(GL_ARRAY_BUFFER, 4*len(self.plt_box_data),self.plt_box_data,GL_DYNAMIC_DRAW)
          
         # glDrawArrays(GL_LINE_LOOP, 0, self.plt_box_data.shape[0]-1 )

         # glDisableVertexAttribArray(self.glsl_a_position2d)
  
         #---
         

      def display(self):
          if self.is_on_draw:
             return
          self.is_on_draw = True

          tw0 = time.clock()
          t0  = time.time()

          if not self.data_is_init:
             print " RAW -> call to init raw data first!!!"
             self.init_raw_data();
             self.is_on_draw = False
             return
             
          print "---> START display -> OnDraw"
          self.is_on_draw = True

          self.clear_display()

#----------------------------------------------------------

#TODO channel order ascending descending ???  now up ch0 is bottom
         #---start sub plots
         # dborder = self.magrin + self.ticksize
           
          dborder=2.0          

          #--- start first channel at top

          ##  !! transpose pixel border magrin to plot coordinates

         
         # print "TEST size "
         # print self.size_in_pixel.height
         # print self.opt.plot_rows
         # print dborder
         # print self.opt
          
          dh = int( self.size_in_pixel.height / self.opt.plot_rows ) - 2 * dborder
          dw = int( self.size_in_pixel.width / self.opt.plot_cols  ) - 2 * dborder

          if dh <  dborder:
             dh = int( self.size_in_pixel.height / self.opt.plot_rows )
             dborder= dh * 0.1
             dh   -=  2 * dborder


          if dborder < 1.0:
             dborder = 1.0

          w0 = dborder
          
#TODO check for screen size and nr of channels

          if (w0 < 1) or (dw < 1) :
             self.is_on_draw=False
             return False

          glLineWidth(1)

          self.xmin = self.timepoints[0]
          self.xmax = self.timepoints[-1]
          
          #-- init  matrix   check sub plot  cols
          # self.mvp     = np.zeros( (r * c,4 ),dtype=np.float32)
        
        #--- xpos0
          
          mat = np.zeros( (self.opt.plot_rows,self.opt.plot_cols) )
          mat += np.arange(self.opt.plot_cols)
             
          self.mvp      = np.zeros( (self.opt.plot_rows*self.opt.plot_cols,4),dtype=np.float32)
          self.mvp[:,0] = dborder           
          self.mvp[:,0] +=  mat.T.flatten() * ( dw + 2 *dborder)  
        
        #-- ypos0     
          mat = np.zeros( (self.opt.plot_cols,self.opt.plot_rows) )
          mat += np.arange(self.opt.plot_rows)
         #-- reverse ypos -> plot ch0 to upper left
          self.mvp[:,1] += mat[:,-1::-1].flatten() * ( dh + 2 *dborder)  
             #self.mvp[-1::-1,1] +=  mat.flatten() * ( dh + 2 *dborder)  
          
         # self.mvp[:,1]= np.arange(self.opt.plot.rows) * (dh + 2*dborder)
          
          self.mvp[:,2]= dw
          self.mvp[:,3]= dh

        #--- init time point data and sig vbo
          self.data_4_vbo_tp[:]      = self.timepoints
          self.vbo_sig_data.vbo_data = self.data_4_vbo

       #TODO check init update vbo draw del 1. channel
          self.vbo_sig_data.vbo_update_y( data=self.data[0,:])
          
          self.glsl_a_position2d  = self.GLSL.aloc('pos2d')
          self.glsl_u_trafomatrix = self.GLSL.uloc('trafo_matrix')
          self.glsl_u_color       = self.GLSL.uloc('color')
         
         
         

         # glEnable(GL_SCISSOR_TEST);

         #glViewport(mvp[idx,0],mvp[idx,1],mvp[idx,2],mvp[idx,3])

          #print"TEST display: pixel mm"
          #print self.size_in_pixel
          #print self.size_in_mm
          #print dborder
          #print dh


         # w_dmm = (self.size_in_pixel.width - dborder *2 ) *0.1
         # h_dmm = (self.size_in_pixel.height- dborder *2 ) *0.1


          idx = 0
          for selected_ch_idx in range( self.opt.channels.idx_start,self.opt.channels.idx_end_range): #self.n_channels ):

              ch_idx = self.selected_channel_index[selected_ch_idx] 
             # print "--- ch idx------------"              
             # print idx
             # print ch_idx
             # print self.opt.channels.idx_start
             # print self.opt.channels.idx_end
             # print self.opt.channels.idx_end_range

              self.draw-axis(idx)
              
              self.GLBatch(idx,ch_idx,data=self.data[idx,:])
              print idx
              print self.mvp[idx]
              idx +=1
              
              
              #glViewport(0, 0, self.size_in_pixel.width, self.size_in_pixel.height)
                   
              #self.trafo_matrix = jtr.ortho(xmin,xmax,self.data_min_max[idx,0],self.data_min_max[idx,1],0,1)

           #---TODO viewport to GLS TrafoMatrix  like C++ example or  Perspective Matrix  as GeometryShader split x,y into VBOs cp only y7signal value
           #--- set border scissor

              #glViewport(mvp[idx,0],mvp[idx,1],mvp[idx,2],mvp[idx,3])

              #glUseProgram(self.GLSL.program_id)
              #-- set trafo matrix
              #glUniformMatrix4fv(self.glsl_u_trafomatrix, 1, GL_FALSE, self.trafo_matrix)
              #--- set color
              #glUniform4fv(self.glsl_u_color, 1, self.plot_color[idx])

              #enable arrays
              #glEnableVertexAttribArray(self.glsl_a_position2d)
              #self.vbo_sig_data.vbo_update_y( data=self.data[idx,:] )


              #glVertexAttribPointer(self.glsl_a_position2d,2, GL_FLOAT, GL_FALSE, 0, None)
              #glDrawArrays(GL_LINE_STRIP, 0, self.vbo_sig_data.data_points-1)

            # disable arrays

              #glDisableVertexAttribArray(self.glsl_a_position2d)
            #--- set border scissor
              #glViewport(mvp[idx,0],mvp[idx,1],mvp[idx,2],mvp[idx,3])

              #self.data_pgr.draw(gl.GL_LINE_STRIP)
              #self.data_pgr.draw('line_strip')

              #self.xgrid_pgr.draw(gl.GL_LINES)

              #self.ygrid_pgr.draw(gl.GL_LINES)

             # glViewport(0, 0, size.width, size.height)



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



             # glRasterPos2f( 1,mvp[idx,1]+mvp[idx,3]/2)
              #  glRasterPos2f( xmin,dy)

             # glColor4f(1.0,0.0,0.0,1.0)

             # for idx_chr in str(idx):
             #     glutBitmapCharacter(GLUT_BITMAP_HELVETICA_10, ord(idx) )
              #      glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, ord( str(idx_chr) ))
         
          glFlush()



         # self.update_global_signal_plot()

          
          glViewport(0, 0, self.size_in_pixel.width, self.size_in_pixel.height)
          # time.sleep(2000)
          
          
          td  = time.time()  - t0
          tdw = time.clock() - tw0
          
          if self.opt.verbose:
             print"\n---PLOT Channel"
             print "Ch start idx %d" %( self.opt.channels.idx_start)
             print "Ch end   idx %d" %(self.opt.channels.idx_end)
             print "Data Shape %d,%d" %(self.data.shape)
          
             print"---TIME"    
             print "Time Range Start/End     : %7.3f -> %7.3f" %(self.opt.time.start,self.opt.time.window_end)
             print "Time Range Start/End calc: %7.3f -> %7.3f" %(self.timepoints[0],self.timepoints[-1])      
             print "Time end                 : %7.3f"          %(self.opt.time.end)
          
             print "Window        :    %d" %(self.opt.time.window)
             print "Window end    :    %d" %(self.opt.time.window_end)
             print "Window end tsl:    %d" %(self.opt.time.window_end_idx)

             print "---> DONE display -> OnDraw Time: %10.3f  WallClk: %10.3f \n" % (td,tdw)
           
          self.is_on_draw=False

      def update_global_signal_plot(self):

         #---TODO Framebuffer overview data set MEGs
          glViewport(0, 0, self.size_in_pixel.width, self.size_in_pixel.height)



          w_dmm = int(self.size_in_pixel.width  * 0.3 )
          h_dmm = int(self.size_in_pixel.height * 0.3)

          si_x0 = int(self.size_in_pixel.width - w_dmm -1)
          si_x1 = w_dmm
          si_y0 = int(self.size_in_pixel.height -h_dmm -1)
          si_y1 = h_dmm

          idx          = 1 #self.n_channels
          self.mvp[idx]= np.array( [-1.0,-1.0,1.0,1.0])

         # self.xmin = self.data_min_max_tp[0,0]
         # self.xmax = self.data_min_max_tp[-1,0]

         # print self.xmin
         # print self.xmax

         # self.vbo_sig_data.vbo_data_x = self.data_min_max_tp.flatten()


          glClearColor(0.90, 0.90, 0.90, 0.0)

         # glEnable(GL_BLEND)
         # glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA)

          glScissor(si_x0,si_y0,si_x1,si_y1)
          glEnable(GL_SCISSOR_TEST)
          glClear(GL_COLOR_BUFFER_BIT)


          glDisable(GL_SCISSOR_TEST)

         # glEnable(GL_BLEND)


         # print  self.data_min_max_sig

         # print  self.data_min_max[idx]

          self.mvp[idx]=np.array([si_x0,si_y0,si_x1,si_y1],dtype=np.float32)

          # self.GLBatch(idx,tlines=GL_LINES,data=self.data_min_max_sig)

          self.GLBatch(idx,data=self.data[idx,:])

          # glDisable(GL_SCISSOR_TEST)

          glDisable(GL_BLEND)




"""
     def _init_demo_data(self):
          print "---> START init demo data"
          tw0 = time.clock()
          t0  = time.time()

          ch = self.opt.plot.rows
          n = self.opt.time.timepoints 

          self.timepoints = np.arange(n,dtype=np.float32) / self.opt.time.sfreq
          self.data       = np.zeros((ch,n), dtype=np.float32)


          self.plot_data     = np.zeros((self.timepoints.size ,2), dtype=np.float32)
          self.plot_data[:,0]= self.timepoints  #x-value

          for i in range( ch ):
              self.data[i,:] = (1+i) *np.sin(self.timepoints * (2 * i+1) * 2* np.pi)

          self.plot_color       = np.repeat(np.random.uniform( size=(ch,4) ,low=.5, high=.9),1,axis=0).astype(np.float32)
          self.plot_color[:,-2] = 1.0
          self.plot_color[:,-1] = 0.0

          self.plot_color[:,-1] = 0.0


          self.data_min_max = np.array( [ self.data.min(axis=1),self.data.max(axis=1) ] ).T
         #-- ck for min == max
          min_eq_max_idx = np.array( self.data_min_max.ptp( axis=1 )==0 )

          if min_eq_max_idx.size:
             self.data_min_max[ min_eq_max_idx] += [-1.0,1.0]

         # self.data_min_max *= 1.2

          self.data_4_vbo = np.zeros((n,2), dtype=np.float32).flatten()
          self.data_4_vbo_tp  = self.data_4_vbo[0:-1:2]
          self.data_4_vbo_sig = self.data_4_vbo[1::2]

          self.data_4_vbo_sig[:] = self.data[0,:]
          self.data_4_vbo_tp[:]  = self.timepoints

          td  = time.time()  -t0
          tdw = time.clock() -tw0
          print "---> DONE init demo data Time: %10.3f  WallClk: %10.3f \n" % (td,tdw)


"""

"""
 self.vertexBuffer = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vertexBuffer)
        vertexData = numpy.array(quadV, numpy.float32)
        glBufferData(GL_ARRAY_BUFFER, 4*len(vertexData), vertexData, 
                     GL_STATIC_DRAW)

    # render 
    def render(self, pMatrix, mvMatrix):        
        # use shader
        glUseProgram(self.program)

        # set proj matrix
        glUniformMatrix4fv(self.pMatrixUniform, 1, GL_FALSE, pMatrix)

        # set modelview matrix
        glUniformMatrix4fv(self.mvMatrixUniform, 1, GL_FALSE, mvMatrix)

        # set color
        glUniform4fv(self.colorU, 1, self.col0)

        #enable arrays
        glEnableVertexAttribArray(self.vertIndex)

        # set buffers 
        glBindBuffer(GL_ARRAY_BUFFER, self.vertexBuffer)
        glVertexAttribPointer(self.vertIndex, 3, GL_FLOAT, GL_FALSE, 0, None)

        # draw
        glDrawArrays(GL_TRIANGLES, 0, 6)

        # disable arrays
        glDisableVertexAttribArray(self.vertIndex)            
"""




