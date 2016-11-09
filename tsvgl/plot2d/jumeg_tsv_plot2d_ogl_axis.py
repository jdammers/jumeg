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

from  jumeg.tsvgl.ogl.jumeg_tsv_ogl_gsl      import JuMEG_TSV_OGL_GSL
from  jumeg.tsvgl.ogl.jumeg_tsv_ogl_vbo_test import JuMEG_TSV_OGL_VBO

from  jumeg.tsvgl.plot2d.jumeg_tsv_plot2d_data_info import JuMEG_TSV_PLOT2D_DATA_INFO
from  jumeg.tsvgl.plot2d.jumeg_tsv_plot2d_options   import JuMEG_TSV_PLOT2D_OPTIONS
     
 
              
class JuMEG_TSV_PLOT2D_OGL(object):
      """
         Helper class for using GLSL shader programs
      """

      def __init__ (self,size=None,channels2plot=10,timepoints=10000,sfreq=1017.25,demo_mode=True):
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
          self.opt.plots             = channels2plot
          self.opt.channels.counts   = channels2plot
          
          self.opt.time.timepoints   = timepoints
          self.opt.time.sfreq        = sfreq
          
          self.opt.time.start        = 0.0
          self.opt.time.window       = self.opt.time.end / 10.0    
          self.opt.time.scroll_speed = self.opt.time.window/2.0          
      
     
          self.data_is_init = False
          self.is_on_draw   = False

          self.clback = np.array([1.0,1.0,1.0,1.0],'f')
      
      #--- sHADER & VERTEX BUFFER        
          self.initGL()
          self.init_glsl()
          
          self.vbo_sig_data = JuMEG_TSV_OGL_VBO() #gl_draw_type = GL_DYNAMIC_DRAW,gl_primitive_type = GL_LINE_STRIP)
          self.axis_init()
      
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

      def update_data(self): 
          
          self.print_info("start update data")
          self.data_is_update = False
                    
          if not self.data_is_init:
             self.init_raw_data()
             return
     
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
             self.data_4_vbo_tp[:]  = self.timepoints   
         
         #---init part min max foreach tp MEGs  => NO BADs?
         #--- ToDo  exclude deselected groups, channels via index
          if self.opt.do_scroll:
             self.data = self.data_selected[ch_start:ch_end_range,tsl0:tsl1_range]  #.astype(np.float32)
         #--init VBO data with data
             self.data_4_vbo_sig = self.data_4_vbo[1::2]
              
         #---TODO check  if remove dcoffset         
             self.data_mean        = np.mean(self.data, axis = -1)   
             self.data            -= self.data_mean[:, np.newaxis] 
             n_ch,n_timepoints     = self.data.shape
          
         #--- TODO  Gimick get min/max time point  ??? 
             self.data_min_max_sig = np.array([self.data.min(axis=0),self.data.max(axis=0)],dtype=np.float32).T.flatten()
 
         #--- TODO opengl color buffer wx color palette get index
         
             self.data_min_max = np.zeros( [n_ch,2])
             self.data_min_max = np.array( [ self.data.min(axis=1),self.data.max(axis=1) ] ).T

         #--- ck for min == max
             min_eq_max_idx = np.array( self.data_min_max.ptp( axis=1 )==0 )

             if min_eq_max_idx.size:
                self.data_min_max[ min_eq_max_idx] += [-1.0,1.0]
        
             self.data_is_update=True      
          
          self.print_info("done update data")    
             
          return self.data_is_update
          
      
      def init_raw_data(self,raw=None,channels2plot=10,cols=1):
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
          self.opt.channels.start   = 1          
          self.opt.channels.counts  = self.raw._data.shape[0]
          self.opt.plots            = channels2plot
          self.opt.plot_cols        = cols 
          
        #--- init info setting channel plot option e.g. color
          self.info.init_info(raw=self.raw)
            
        #--- RGBA color setting for group & channel via idx and lookup tab
          self.data_is_init = True
          self.update_data()
          
          self.print_info("done init raw data")
 
      def axis_init(self):            
          
          self.vbo_box      = JuMEG_TSV_OGL_VBO()#gl_draw_type = GL_DYNAMIC_DRAW,gl_primitive_type = GL_LINE_LOOP)
          #self.vbo_yaxis    = JuMEG_TSV_OGL_VBO()
       

 #--- axis; plot parameter
          self.margin     = 2.0 #10
          self.ticksize   = 6.0
          
          #self.height     = 0.0
          #self.width      = 0.0
          self.axis_box_color = np.array([0.0,0.0,0.0,1.0],dtype=np.float32)
     
        #--- box[4] = {{-1, -1}, {1, -1}, {1, 1}, {-1, 1}};
          self.axis_box_data = np.array([ -1.0, -1.0, 1.0, -1.0, 1.0, 1.0,-1.0, 1.0 ],dtype=np.float32)
          self.vbo_box.data  = self.axis_box_data   
        
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
          self.print_info("OGL -> start clear display")
          
          
  #-- ToDo ck for use display list
          glMatrixMode(GL_PROJECTION)
          glLoadIdentity()
       
          glViewport(0,0,self.size_in_pixel.width,self.size_in_pixel.height)
          r,g,b,a = self.clback
          glClearColor(r,g,b,a)
          glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
          glEnable(GL_LINE_SMOOTH)
          #glHint(GL_LINE_SMOOTH,GL_FASTEST)
          glDisable(GL_SCISSOR_TEST)
          glLineWidth(1)
          
          self.print_info("done OGL -> clear display")
          return True
      
      
      def draw_channel_label(self,ch_idx):
          #glUseProgram(self.GLSL.program_id)
          glColor4f(0.0,0.0,0.0,1.0)
          glRasterPos2f(0.0,0.0) 
          for idx_chr in str( self.info.index2channel_label(ch_idx) ):
              glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, ord( str(idx_chr) ) )
                    
          
        

      def GLBatch(self,idx,ch_idx,data=None,tlines=GL_LINE_STRIP):
          '''
            # Draw using the vertices in our vertex buffer object
	      glBindBuffer(GL_ARRAY_BUFFER, vbo[0]);

	      glEnableVertexAttribArray(attribute_coord2d);
	      glVertexAttribPointer(attribute_coord2d, 2, GL_FLOAT, GL_FALSE, 0, 0);
	      glDrawArrays(GL_LINE_STRIP, 0, 2000);

          '''
          
          draw_box = True
          draw_sig = True
          #glUseProgram(self.GLSL.program_id)

          if  draw_box:
              print"---> draw box: %d" %(idx)
              imat = np.array( np.identity(4),dtype=np.float32 )
              #print imat
              glUniformMatrix4fv(self.glsl_u_trafomatrix, 1, GL_FALSE, imat)
              glUniform4fv(self.glsl_u_color, 1, self.info.index2channel_color(ch_idx) ) #self.axis_box_color ) 
              glEnableVertexAttribArray(self.glsl_a_position2d)
              glBindBuffer(GL_ARRAY_BUFFER,self.vbo_box.vbo_id)
              #self.vbo_box.vbo_update(data=self.axis_box_data)
              
              glVertexAttribPointer(self.glsl_a_position2d,2, GL_FLOAT, GL_FALSE, 0, None)
              glDrawArrays(GL_LINE_LOOP, 0,self.vbo_box.data_points)
              glDisableVertexAttribArray(self.glsl_a_position2d)
              print"---> done draw box"
           
        
          if draw_sig:
            
             #glUseProgram(self.GLSL.program_id)

             #print"---> draw sig: %d" %(idx)
             self.trafo_matrix = jtr.ortho(self.xmin,self.xmax,self.data_min_max[idx,0],self.data_min_max[idx,1],0,1)
             #print self.trafo_matrix
             glUniformMatrix4fv(self.glsl_u_trafomatrix, 1, GL_FALSE,self.trafo_matrix)  
             glUniform4fv(self.glsl_u_color, 1,self.info.index2channel_color(ch_idx)  )   
             glEnableVertexAttribArray(self.glsl_a_position2d)            
             self.vbo_sig_data.vbo_update_y( data=data) #self.axis_box_data)   
             glVertexAttribPointer(self.glsl_a_position2d,2, GL_FLOAT, GL_FALSE, 0, None)
             glDrawArrays(tlines, 0, self.vbo_sig_data.data_points-1)
             #glDrawArrays(self.vbo_sig_data.gl_primitive_type, 0, self.vbo_sig_data.data_points-1)
             glDisableVertexAttribArray(self.glsl_a_position2d)        
             #print"---> done draw sig"
          #glDisableVertexAttribArray(self.glsl_a_position2d)        
       
         # self.vbo_sig_data.reset(attr_idx=self.glsl_a_position2d)
          
          # finally: self.vbo.unbind() glDisableClientState(GL_VERTEX_ARRAY); finally: shaders.glUseProgram( 0 )
     
         #glDisableVertexAttribArray(self.glsl_a_position2d)
         #self.vbo_box.reset(attr_idx=self.glsl_a_position2d)

        
        
       
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
          self.xmin = self.timepoints[0]
          self.xmax = self.timepoints[-1]
     
         #--- init time point data and sig vbo
          self.data_4_vbo_tp[:]  = self.timepoints
          #self.vbo_sig_data.update_data(data = self.data_4_vbo)
          self.vbo_sig_data.vbo_update(data = self.data_4_vbo)
          
          #self.data_4_vbo_tp[:]  = self.timepoints
          #self.vbo_sig_data.update_xdata(data = self.timepoints)
          
          self.vbo_box.vbo_update(data = self.axis_box_data )  
        
      
          self.glsl_a_position2d  = self.GLSL.aloc('pos2d')
          self.glsl_u_trafomatrix = self.GLSL.uloc('trafo_matrix')
          self.glsl_u_color           = self.GLSL.uloc('color')             
         
          plt_border_w = self.margin + self.ticksize
          plt_border_h = self.margin + self.ticksize           
          
          plt_h = ( self.size_in_pixel.height / self.opt.plot_rows ) - 2.0 * plt_border_h
          plt_w = ( self.size_in_pixel.width / self.opt.plot_cols  ) - 2.0 * plt_border_w
        
          #plt_h = ( self.size_in_pixel.height / self.opt.plot_rows ) -  plt_border_h
          #plt_w = ( self.size_in_pixel.width / self.opt.plot_cols  ) -  plt_border_w
          '''
          if plt_h <= plt_border:
             plt_h      = int( self.size_in_pixel.height / self.opt.plot_rows )
             plt_border = plt_h * 0.1
             plt_h     -=  2 * plt_border
          if plt_w <= plt_border:
             plt_w      = int( self.size_in_pixel.width / self.opt.plot_cols )
             plt_border = plt_w * 0.1
             plt_w     -=  2 * plt_border
          '''
          if plt_w < 0.0:
             plt_border_w = 0.0
             plt_w = ( self.size_in_pixel.width / self.opt.plot_cols  )

          
          if plt_h < 0.0:
             plt_border_h = 0.0
             plt_h = ( self.size_in_pixel.height / self.opt.plot_rows )
             
          
          print"Plot Border  W : %f  H: %f " % ( plt_border_w, plt_border_h)
       
         # plt_border = 0
          
        
        #--- pos mat
        
          '''
          use with shader trafo matrix
          glm::mat4 viewport_transform(float x, float y, float width, float height) {
          float offset_x = (2.0 * x + (width - window_width)) / window_width;
          float offset_y = (2.0 * y + (height - window_height)) / window_height;

             // Calculate how to rescale the x and y coordinates:
          float scale_x = width / window_width;
          float scale_y = height / window_height;          
          '''
          
          vpm =  np.zeros( (self.opt.plot_rows*self.opt.plot_cols,4),dtype=np.float32)
        
          mat = np.zeros( (self.opt.plot_rows,self.opt.plot_cols) )
          mat += np.arange(self.opt.plot_cols)  
        #--- x0 pos           
          vpm[:,0] =  plt_border_w + mat.T.flatten() * ( plt_w  + plt_border_w)
        #--- y0 pos
          #mat[:] = 0
          
          mat = np.zeros( (self.opt.plot_cols,self.opt.plot_rows) )
          mat += np.arange(self.opt.plot_rows)
         #-- reverse ypos -> plot ch0 to upper left
          vpm[:,1] += plt_border_h + mat[:,-1::-1].flatten() * ( plt_h + 2 *plt_border_h)  
    
          #mat = np.zeros( (self.opt.plot_rows,self.opt.plot_cols) )
   
          #mat = mat.reshape( self.opt.plot_cols,self.opt.plot_rows )          
                 
          #mat += np.arange(self.opt.plot_rows)
          #vtrafo_mat[:,1] = mat.flatten() *  plt_h  + plt_border  
          vpm[:,2] = plt_w   
          vpm[:,3] = plt_h   
  
         # print"TEST plt border: %f " % ( plt_border_w)
         # print"TEST W  %f  " % ( self.size_in_pixel.width)
         # print plt_w #"GL %d " % ( glutGet(GLUT_WINDOW_WIDTH) )

         # print"TEST H  %f " % ( self.size_in_pixel.height)
         # print plt_h #glutGet(GLUT_WINDOW_HEIGHT) 
     
          #print"TEST vTrafo mat:"
          #print vpm
          #print"\n"
        
          '''
        #--- init vieport matrix    
          self.mvp       = np.zeros( (self.opt.plot_rows*self.opt.plot_cols,4),dtype=np.float32)
        
        #--- x0 pos           
          self.mvp[:,0] =  mat.T.flatten() *  plt_w # pos_border #dw +( 2 *dborder)  
       
        #-- ypos0     
          #mat = np.zeros( (self.opt.plot_cols,self.opt.plot_rows) )
          #mat += np.arange(self.opt.plot_rows)
         #-- reverse ypos -> plot ch0 to upper left
          self.mvp[:,1] += mat[:,-1::-1].flatten() * plot_border          
          
          self.mvp[:,2]= dw
          self.mvp[:,3]= dh

          '''
          
          
          glUseProgram(self.GLSL.program_id)    
           
          idx = 0
        
          for selected_ch_idx in range( self.opt.channels.idx_start,self.opt.channels.idx_end_range): #self.n_channels ):

              ch_idx = self.selected_channel_index[selected_ch_idx] 
       
              # self.draw_axis(idx,vtrafo_mat[idx], )
              glViewport(vpm[idx,0],vpm[idx,1],vpm[idx,2],vpm[idx,3])
              print"VPM: %d rows: %d cols: %d" %(idx,self.opt.plot_rows ,self.opt.plot_cols)
              print vpm[idx]
             
              self.draw_channel_label( ch_idx )
              
              self.GLBatch(idx,ch_idx,data=self.data[idx,:] )
          
              #print idx
             
              idx +=1
              
              #glViewport(0,0,self.size_in_pixel.width,self.size_in_pixel.height)
        
          glUseProgram(0)          
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




