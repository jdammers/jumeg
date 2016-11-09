#-----------------------------------------------------------
# Perl-Module MEG::MEG13::PLOT::TSV::TimeSeriesViewerUtility
# Autor: F.B.
# Date 25.01.2013
# last change 28.02.2013
#
#-----------------------------------------------------------
package MEG::MEG13::TSV::TimeSeriesViewerIUP;
       

use strict;
use OpenGL qw(:all);
use IUP ':all';

use Carp;
use PDL;
#use Fcntl;
#use PDL::NetCDF;

# use PDL::Constants qw(PI);
    $PDL::BIGPDL = 1;

use base qw( MEG::MEG13::TSV::TimeSeriesViewerBase );

my $VERSION = "2013";

#-----------------------------------------------------------------------# 
#   new CLASS
#-----------------------------------------------------------------------# 
sub new{
 my $class = shift;
 my $self  = $class->SUPER::new( 
     			       );
    bless ( $self,$class );

    $self->Init(@_);  

    return $self;

} # new
1;

#---------------------------------------------------------# 
#---- plot_range                 -------------------------#
#---------------------------------------------------------# 
sub plot_range{
my $self = shift;
   $self->{ymin}  = $self->data->min(); 
   $self->{ymin} -= $self->{ymin} * 0.12;
   $self->{ymax}  = $self->data->max() * 1.12;
 
   $self->{xmin}  = $self->xdata->at(0);
   $self->{xmax}  = $self->xdata->at(-1);
  
} # end of plot_range


#---------------------------------------------------------#
#---- tsv_iup_update_plot        -------------------------#
#---------------------------------------------------------#
sub tsv_iup_update_plot{
my $self = shift;
   
   return if ( $self->{gl_pending} );
   
   if (  $self->{do_sensor_layout} )
    {
    $self->tsv_iup_do_update_sensor_layout();
    } 
   
   else { $self->tsv_iup_do_update_plot() }

} # end of tsv_iup_update_plot 


#---------------------------------------------------------#
#---- tsv_iup_do_update_sensor_layout --------------------#
#---------------------------------------------------------#
sub tsv_iup_do_update_sensor_layout{
my $self = shift;
my $dpos;

   return if ( $self->{gl_pending} );
   $self->{gl_pending} = 1;

# my $t00 = time;

   $self->{WGL}->GLMakeCurrent();

my $chidx   = $self->{ch_idx0};
                             
my $ch_idx0 = $self->{ch_idx0};
my $ch_idx1 = $self->{ch_idx1};
my $tsl0    = $self->{tsl0} || 0;
my $tsl1    = $self->{tsl1} || $self->data->dim(-1)-1;


my $data    = $self->data->slice("$ch_idx0:$ch_idx1,$tsl0:$tsl1"); 
my $datax   = $self->xdata->slice("$tsl0:$tsl1"); 

#--- init data for vertex buffer obj
# my $data_4_vbo             = pdl( zeroes(2,$self->data->dim(-1) ) )->float();

my $data_4_vbo             = pdl( zeroes(2,$data->dim(-1) ) )->float();
my $data_4_vbo_timepoints  = $data_4_vbo->slice("(0),:");
my $data_4_vbo_signal      = $data_4_vbo->slice("(1),:");
   $data_4_vbo_timepoints .= $datax; #$self->xdata();
my $data_vbo               = $data_4_vbo->flat;

my $float_size = 4;
  

# $self->{txspos}->at($ix),$self->{tyspos}->at($ix)   
#---start sub plots
my ($w, $h) = split /x/,$self->{WGL}->RASTERSIZE;

# print"TEST : $w $h\n";
   $self->tsv_iup_setViewport(0,$w,0,$h);


#--- reshape
   glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
   glClearColor(1.0,1.0,1.0,0.0);
   glLineWidth(2);
   glMatrixMode(GL_MODELVIEW);
   glLoadIdentity();
   glColor3f(0.0,0.0,1.0);

#--- create OGL verts buffer
   glDisableClientState(GL_VERTEX_ARRAY);

my $VertexObjID = glGenBuffersARB_p(1);
   glBindBufferARB(GL_ARRAY_BUFFER_ARB,$VertexObjID);

my $ogl_array = OpenGL::Array->new_scalar(GL_FLOAT, $data_vbo->get_dataref,$data_vbo->dim(0)*$float_size);
   glBufferDataARB_p(GL_ARRAY_BUFFER_ARB,$ogl_array,GL_DYNAMIC_DRAW_ARB);
   $ogl_array->bind($VertexObjID);
   glVertexPointer_p(2,$ogl_array);
   glEnableClientState(GL_VERTEX_ARRAY);

#---start sub plots

my $dh   = 50;
my $dw   = 50;

my $w0 = 10; # pixel border
my $w1 = $w-10;
 
my $h0 = 0;
# my $dh = int( $h / $data->dim(0) );
my $h1 = $dh;


my $xmin = $datax->at(0);
my $xmax = $datax->at(-1);
my ($ymin,$ymax); 

my $xpos  = $self->{sxpos} * ($w-$dw);
 #  $xpos +=
my $ypos = $self->{sypos} * ($h-$dh);


#--- copy data to VBO   
   for ( my $i=0; $i < $data->dim(0); $i++ )
    { 

#--- sub plot window
      $w0 = $xpos->at($i);
      $w1 = $w0 + $dw;
      $h0 = $ypos->at($i);
      $h1 = $h0 + $dh;

      $self->tsv_iup_setViewport($w0,$w1,$h0,$h1);

      $ymin  = $data->slice("($i),:")->min || -1;
      $ymin -= $ymin * 0.2;
      $ymax = $data->slice("($i),:")->max  ||  1; 
      $ymax += $ymax * 0.2; 
      $self->tsv_iup_setWindow($xmin,$xmax,$ymin,$ymax );
      $dpos = $ymin + ($ymax - $ymin) / 2;

#--- draw zero line
      glLineWidth(1);
      glColor3f(0,0,0);

      glBegin(GL_LINES);
        glVertex2f($xmin,0.0);
        glVertex2f($xmax,0.0);
      glEnd();
   
     glColor3f(0.4,0.4,0.4);

  #    glBegin(GL_LINES);
  #      glVertex2f($xmin,$dpos);
  #      glVertex2f($xmax,$dpos);
  #    glEnd();
       
   
     glRasterPos2f( $xmin, $dpos );


 # my  $n               = $self->channel_info($chidx,"name")  || $chidx;
     # $self->{channel_list}[$chidx];
  
  #my ($cr,$cg,$cb,$ca) = $self->channel_info{$chidx,"color_r","color_g""color_b","color_a"} || 0;

# print "TEST $chidx ".$self->{channel_list}[$chidx]."\n";

  $self->print_string( ${$self->{channel_list}}[$chidx],$self->{sensor_layout}{gl_font} );

 
my $str = ${$self->{label_list_ptr}}[$chidx + $i] || "CH-".++$chidx;
#   $self->print_string( $str,$self->{gl_font} );


#--- start drawing signal
      glLineWidth(1);
  #    glColor3f(rand(1), rand(1),1.0);# mix color for each signal 
     
    #  glColor3f($cr,$cg,$cb,$ca);# mix color for each signal 
    
    # glColor4f($cr,$cg,$cb,$ca);# mix color for each signal 

 glColor3f(0,0,1);

#--- copy pdl data to VBO thank's vividsnow !!!
      $data_4_vbo_signal .= $data->slice("($i),:");
      $ogl_array = OpenGL::Array->new_scalar(GL_FLOAT,$data_vbo->get_dataref,$data_vbo->dim(0)*$float_size);
 
      glBufferSubDataARB_p(GL_ARRAY_BUFFER_ARB,0,$ogl_array);

      glDrawArrays(GL_LINE_STRIP,0,$data_4_vbo_timepoints->dim(-1)-1 );
 
      $h0 += $dh;
      $h1 += $dh + 1;

   #   $chidx++;

     } # for
  
  glBindBufferARB(GL_ARRAY_BUFFER_ARB, 0);
  glDisableClientState(GL_VERTEX_ARRAY);

  glFlush();
#  glutSwapBuffers();
 
  $self->{WGL}->GLSwapBuffers();

  $self->{gl_pending} = undef;

#  $t00 = time() - $t00;

#  print" done  <update_plot_pdl_to_vbo> Time to update: $t00\n";
return IUP_DEFAULT;

} # end of update_plot_pdl_to_vbo


#---------------------------------------------------------#
#---- tsv_iup_do_update_plot     -------------------------#
#---  copy pdl data to opengl vertex buffer
#---  generates a sub plot for each channel 
#---------------------------------------------------------#
sub tsv_iup_do_update_plot{
my $self = shift;
my $dpos;

   return if ( $self->{gl_pending} );
   $self->{gl_pending} = 1;

# my $t00 = time;

   $self->{WGL}->GLMakeCurrent();

my $chidx   = $self->{ch_idx0};
                             
my $ch_idx0 = $self->{ch_idx0};
my $ch_idx1 = $self->{ch_idx1};
my $tsl0    = $self->{tsl0} || 0;
my $tsl1    = $self->{tsl1} || $self->data->dim(-1)-1;


my $data    = $self->data->slice("$ch_idx0:$ch_idx1,$tsl0:$tsl1"); 
my $datax   = $self->xdata->slice("$tsl0:$tsl1"); 

#--- init data for vertex buffer obj
# my $data_4_vbo             = pdl( zeroes(2,$self->data->dim(-1) ) )->float();

my $data_4_vbo             = pdl( zeroes(2,$data->dim(-1) ) )->float();
my $data_4_vbo_timepoints  = $data_4_vbo->slice("(0),:");
my $data_4_vbo_signal      = $data_4_vbo->slice("(1),:");
   $data_4_vbo_timepoints .= $datax; #$self->xdata();
my $data_vbo               = $data_4_vbo->flat;

my $float_size = 4;
  
#---start sub plots
my ($w, $h) = split /x/,$self->{WGL}->RASTERSIZE;

# print"TEST : $w $h\n";
   $self->tsv_iup_setViewport(0,$w,0,$h);


#--- reshape
   glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
   glClearColor(1.0,1.0,1.0,0.0);
   glLineWidth(2);
   glMatrixMode(GL_MODELVIEW);
   glLoadIdentity();
   glColor3f(0.0,0.0,1.0);

#--- create OGL verts buffer
   glDisableClientState(GL_VERTEX_ARRAY);

my $VertexObjID = glGenBuffersARB_p(1);
   glBindBufferARB(GL_ARRAY_BUFFER_ARB,$VertexObjID);

my $ogl_array = OpenGL::Array->new_scalar(GL_FLOAT, $data_vbo->get_dataref,$data_vbo->dim(0)*$float_size);
   glBufferDataARB_p(GL_ARRAY_BUFFER_ARB,$ogl_array,GL_DYNAMIC_DRAW_ARB);
   $ogl_array->bind($VertexObjID);
   glVertexPointer_p(2,$ogl_array);
   glEnableClientState(GL_VERTEX_ARRAY);

#---start sub plots
my $w0 = 10;
my $w1 = $w-10;
 
my $h0 = 0;
my $dh = int( $h / $data->dim(0) );
my $h1 = $dh;


#my $xmin = $time_points->min;
#my $xmax = $time_points->max;
#my $ymin = $data->min * 1.2;
#my $ymax = $data->max * 1.2; 

my $xmin = $datax->at(0);
my $xmax = $datax->at(-1);
#my $ymin = $data->min * 1.2;
#my $ymax = $data->max * 1.2; 

my ($ymin,$ymax); 

#--- plot superimpose
unless ( $self->{do_sub_plot} )
 {
  $dh = 0;
  $ymin  = $data->min || -1;
  $ymin -= $ymin * 0.1;
  $ymax  = $data->max ||  1; 
  $ymax += $ymax * 0.1; 
  $ymin  = -600;
  $ymax  = 600;
  $self->tsv_iup_setWindow($xmin,$xmax,$ymin,$ymax );
  $dpos = $ymin + ($ymax - $ymin) / 2;
 }


#--- copy data to VBO   
   for ( my $i=0; $i < $data->dim(0); $i++ )
    { 

if ( $self->{do_sub_plot} )
 {
#--- sub plot window
      $self->tsv_iup_setViewport($w0,$w1,$h0,$h1);
      $ymin  = $data->slice("($i),:")->min || -1;
      $ymin -= $ymin * 0.2;
      $ymax = $data->slice("($i),:")->max  ||  1; 
      $ymax += $ymax * 0.2; 
      $self->tsv_iup_setWindow($xmin,$xmax,$ymin,$ymax );
      $dpos = $ymin + ($ymax - $ymin) / 2;
} # if sub_plot

#--- draw zero line
      glLineWidth(1);
      glColor3f(0,0,0);

      glBegin(GL_LINES);
        glVertex2f($xmin,0.0);
        glVertex2f($xmax,0.0);
      glEnd();
   
     glColor3f(0.4,0.4,0.4);

      glBegin(GL_LINES);
        glVertex2f($xmin,$dpos);
        glVertex2f($xmax,$dpos);
      glEnd();
       
   
     glRasterPos2f( $xmin, $dpos );


 # my  $n               = $self->channel_info($chidx,"name")  || $chidx;
     # $self->{channel_list}[$chidx];
  
  #my ($cr,$cg,$cb,$ca) = $self->channel_info{$chidx,"color_r","color_g""color_b","color_a"} || 0;

# print "TEST $chidx ".$self->{channel_list}[$chidx]."\n";

 #     $self->print_string( "CH".${$self->{channel_list}}[$chidx],$self->{gl_font} );

 
my $str = ${$self->{label_list_ptr}}[$chidx + $i] || "CH-".++$chidx;
   $self->print_string( $str,$self->{gl_font} );


#--- start drawing signal
      glLineWidth(2);
  #    glColor3f(rand(1), rand(1),1.0);# mix color for each signal 
     
    #  glColor3f($cr,$cg,$cb,$ca);# mix color for each signal 
    
    # glColor4f($cr,$cg,$cb,$ca);# mix color for each signal 

 glColor3f(0,0,1);

#--- copy pdl data to VBO thank's vividsnow !!!
      $data_4_vbo_signal .= $data->slice("($i),:");
      $ogl_array = OpenGL::Array->new_scalar(GL_FLOAT,$data_vbo->get_dataref,$data_vbo->dim(0)*$float_size);
 
      glBufferSubDataARB_p(GL_ARRAY_BUFFER_ARB,0,$ogl_array);

      glDrawArrays(GL_LINE_STRIP,0,$data_4_vbo_timepoints->dim(-1)-1 );
 
      $h0 += $dh;
      $h1 += $dh + 1;

   #   $chidx++;

     } # for
  
  glBindBufferARB(GL_ARRAY_BUFFER_ARB, 0);
  glDisableClientState(GL_VERTEX_ARRAY);

  glFlush();
#  glutSwapBuffers();
 
  $self->{WGL}->GLSwapBuffers();

  $self->{gl_pending} = undef;

#  $t00 = time() - $t00;

#  print" done  <update_plot_pdl_to_vbo> Time to update: $t00\n";
return IUP_DEFAULT;

} # end of update_plot_pdl_to_vbo


#---------------------------------------------------------#
#---- setWindow                  -------------------------#
#---------------------------------------------------------#
sub tsv_iup_setWindow{
my ($self,$l,$r,$b,$t) = @_;
 glMatrixMode(GL_PROJECTION);
 glLoadIdentity();
 gluOrtho2D($l,$r,$b,$t);
} # end of setWindow

#---------------------------------------------------------#
#---- setViewport                -------------------------#
#---------------------------------------------------------#
sub tsv_iup_setViewport{
my ($self,$l,$r,$b,$t) = @_;
 glViewport($l,$b,$r-$l,$t-$b);
}# end of setViewport


#---------------------------------------------------------# 
#--- tsv_iup_plot_options_dialog    ----------------------#
# "<text>%<x><extra>{<tip>}\n"
#my $format1 = "Real 2: %r[-1.5,1.5,0.05]\n";       # [min,max,step] example
# my $format2 = "Numeric string: %s/d+\n";           # mask example
# my $format3 = "Boolean: %b[No,Yes]{tip text}\n";   # [false,true] + tip text example
# my $format4 = "List: %l|item0|item1|item2|item3|item4|item5|item6|\n"; # list example
# my $format5 = "File: %f[OPEN|*.bmp;*.jpg|CURRENT|NO|NO]\n" # [dialogtype|filter|directory|nochangedir|nooverwriteprompt] example
#---------------------------------------------------------# 
sub tsv_iup_plot_options_dialog{
my $self = shift;

   $self->{time_max_sec} = $self->{time_min_sec} + $self->data->dim(-1) / $self->{sample_frequency};

my ($state,$ch0,$chd,$time0,$timed,$subplot,$sensor_layout) = IUP->GetParam(
    "TSV Plot Options",undef,
   "CHANNELS %t\n".
   "Start Channel: %i[1,".$self->data->dim(0)."]{Integer Tip 4}\n".
   "Channels to display: %i[1,".$self->data->dim(0)."]{Integer Tip 2}\n".
   "TIME [s] %t\n".
   "Time Start: %i[$self->{time_min_sec},$self->{time_max_sec},0.1]\n".
   "Time Delta: %i[$self->{time_min_sec},$self->{time_max_sec},0.1]\n".
 #  "Time Start: %r[$self->{time_min_sec},$self->{time_max_sec},0.01]{Integer Tip 4}\n".
 #  "Time Delta: %r[$self->{time_min_sec},$self->{time_max_sec},0.01]{Integer Tip 2}\n".
    "Sub Plot %b[$self->{do_sub_plot}]\n".
    "Sensor layout %b[$self->{do_sensor_layout}]\n",
    $self->{ch_idx0}+1,$self->{ch_idx_delta}+1,$self->{time0},$self->{time_delta},$self->{do_sub_plot},$self->{do_sensor_layout}
      );
 
# IUP->Message("Results",
#   "Delta Channels:\t$ch\n"
#   ) if $ch;

 return unless $state;
 
 $self->{ch_idx_delta} = $chd -1;
 $self->{ch_idx0}      = $ch0; 
 $self->{ch_idx1}      = 1; 
 $self->{time0}        = $time0;
 $self->{time_delta}   = $timed;
 $self->{do_sub_plot}  = $subplot;
 $self->{do_sensor_layout} = $sensor_layout;


 $self->{tsl0}         = int( $time0 * $self->{sample_frequency} );
 $self->{tsl_delta}    = int( $timed * $self->{sample_frequency} );

 $self->tsv_iup_update_all();

} # end of tsv_iup_plot_options_dialog

#my ($ret, $b, $i, $a, $s, $l, $f, $c) = IUP->GetParam(+
#   "Simple Dialog Title", undef,
#   #define dialog controls
#   "Boolean: %b[No,Yes]{Boolean Tip}\n".
#   "Integer: %i[0,255]{Integer Tip 2}\n".
#   "Angle: %a[0,360]{Angle Tip}\n".
#   "String: %s{String Tip}\n".
#   "List: %l|item1|item2|item3|{List Tip}\n".
#   "File: %f[OPEN|*.bmp;*.jpg|CURRENT|NO|NO]{File Tip}\n".
#   "Color: %c{Color Tip}\n",
#   #set default values
#   1, 100, 45, 'test string', 2, 'test.jpg', '255 0 128'
# );


#---------------------------------------------------------# 
#---- tsv_iup_init_main_menu     -------------------------#
#---------------------------------------------------------# 
sub tsv_iup_init_main_menu{
my $self= shift;

   $self->{MAIN_MENU} = IUP::Menu->new( child=>[
               IUP::Submenu->new( TITLE=>"OPTION", child=>IUP::Menu->new( child=>[
                 IUP::Item->new( TITLE=>"PLOT", ACTION=>sub{ $self->tsv_iup_plot_options_dialog }), #, VALUE=>"NO" ),
                IUP::Separator->new(),
               #  IUP::Item->new( TITLE=>"IupItem 2 Disabled", ACTION=>$self->tsv_iup_dialog_channels, ACTIVE=>"NO" ),
               ])),
             #  IUP::Item->new(TITLE=>"IupItem 3", ACTIVE=>"NO", ACTION=>$self->tsv_iup_dialog_channels ),
            #   IUP::Item->new(TITLE=>"IupItem 4", ACTIVE=>"NO", ACTION=>sub{ print"TEST\n"} ),
             ]);

} # end of tsv_iup_init_main_menu

#---------------------------------------------------------# 
#---- show                       -------------------------#
#---------------------------------------------------------# 
sub show{
my $self= shift;

   $self->{time_max_sec} = $self->{time_min_sec} + $self->data->dim(-1) / $self->{sample_frequency};
   $self->{time0}        = $self->{time_min_sec} + $self->{tsl0} / $self->{sample_frequency};
   $self->{time_delta}   = $self->{time_min_sec} + $self->{tsl_delta} / $self->{sample_frequency};

 #  $self->{label_list_ptr} = $self->{channel_list_ptr};

#  $self->plot_range();
   $self->tsv_base_check_channel_range();
   $self->tsv_base_check_time_range();  


   glutInit();

 #  $self->{WGL} = IUP::CanvasGL->new( BUFFER=>"DOUBLE", RASTERSIZE=>$self->{wSizeX}."x".$self->{wSizeY}

#---   
# Creates items, sets its shortcut keys and deactivates edit item;

#my $item_channels  = IUP::Item->new( TITLE=>"Channels", ACTION=>$self->tsv_iup_show_channels() );
#my $item_exit      = IUP::Item->new( TITLE=>"Exit\tCtrl+E" );

# Creates two menus;
#my $menu_file     = IUP::Menu->new( child=>[$item_exit] );
#my $menu_channels = IUP::Menu->new( child=>[$item_channels] );

# Creates two submenus;
#my $submenu_file     = IUP::Submenu->new( child=>$menu_file,     TITLE=>"File" );
#my $submenu_channels = IUP::Submenu->new( child=>$menu_channels, TITLE=>"Channesl" );

# Creates main menu with two submenus;
  # $self->{MAIN_MENU} = IUP::Menu->new( child=>[$submenu_file, $submenu_channels] );

my $bt_test = IUP::Button->new( TITLE=>"TEST");

#---
my $w1=  IUP::Vbox->new([ 
               $self->{WGL} = IUP::CanvasGL->new( BUFFER=>"DOUBLE", RASTERSIZE=>$self->wSizeX."x".$self->wSizeY ),
               $bt_test, 
	     #  IUP::Button->new( TITLE=>"This is testing demo Button1"),
             #  IUP::Button->new( TITLE=>"This is testing demo Button2"),
             ]);
   
   $self->{WGL}->K_ANY( sub{ $self->tsv_iup_key_functions(@_) } );
   $self->{WGL}->ACTION( sub{$self->tsv_iup_update_plot() } );

   $bt_test->ACTION( sub{print"TEST\n"} );
  
   $self->tsv_iup_init_main_menu();

my $dlg = IUP::Dialog->new( MENU=>$self->{MAIN_MENU}, child=>$w1, TITLE=>$self->{tittle},MINSIZE=>$self->{minsize} );


   $dlg->ShowXY( IUP_CENTER, IUP_CENTER );
   IUP->MainLoop();

} # show_iup

#---------------------------------------------------------# 
#---- tsv_iup_key_functions      -------------------------#
#---------------------------------------------------------# 
sub tsv_iup_key_functions{
my ($self,$w,$key) = @_;

# $w->GLMakeCurrent();

#--- scroll in time
 if ( $key == K_LEFT){ $self->{tsl0} -= $self->{tsl_delta} }
  elsif($key == K_RIGHT) { $self->{tsl0} += $self->{tsl_delta} }

#--- scroll channels  
  elsif( $key == K_UP     ) { $self->{ch_idx0}++ }
  elsif( $key == K_DOWN   ) { $self->{ch_idx0}-- }
  elsif( $key == K_PGUP   ) { $self->{ch_idx0}+= $self->{ch_idx_delta}-1 }
  elsif( $key == K_PGDN   ) { $self->{ch_idx0}-= $self->{ch_idx_delta}-1 }
#--- scroll time  
  elsif( $key == K_HOME   ) { $self->{tsl0} = 0 }
  elsif( $key == K_END    ) { $self->{tsl0} = $self->data->dim(-1)  } 
  elsif( $key == K_LEFT   ) { $self->{tsl0}--  }
  elsif( $key == K_RIGHT  ) { $self->{tsl0}++  } 
  else{ return }

  if ( $self->debug )
   {
    print"done KEY pressed ch   $key => $self->{ch_idx0}  == $self->{ch_idx1}\n";
    print"done KEY pressed time $key => $self->{tsl0}  == $self->{tsl1}\n";
   } # debug

  $self->tsv_iup_update_all();

} # end of tsv_key_functions

#---------------------------------------------------------# 
#---- tsv_iup_update_all         -------------------------#
#---------------------------------------------------------# 
sub tsv_iup_update_all{
my $self = shift;
  
   $self->tsv_base_check_time_range();
   $self->tsv_base_check_channel_range(); 
   
   if ( $self->{do_sensor_layout} )
    {
     $self->load_sensor_pos    unless( $self->{sensor_pos_loaded}   );
     $self->project_sensor_pos unless( $self->{sensor_pos_projected});
    }

   $self->tsv_iup_update_plot(); 

} # end of tsv_iup_update_all 


__DATA__




 use IUP ':all';

 my ($ret, $b, $i, $a, $s, $l, $f, $c) = IUP->GetParam(
   "Simple Dialog Title", undef,
   #define dialog controls
   "Boolean: %b[No,Yes]{Boolean Tip}\n".
   "Integer: %i[0,255]{Integer Tip 2}\n".
   "Angle: %a[0,360]{Angle Tip}\n".
   "String: %s{String Tip}\n".
   "List: %l|item1|item2|item3|{List Tip}\n".
   "File: %f[OPEN|*.bmp;*.jpg|CURRENT|NO|NO]{File Tip}\n".
   "Color: %c{Color Tip}\n",
   #set default values
   1, 100, 45, 'test string', 2, 'test.jpg', '255 0 128'
 );
 
 IUP->Message("Results",
   "Boolean:\t$b\n".
   "Integer:\t$i\n".
   "Angle:\t$a\n".
   "String:\t$s\n".
   "List Index:\t$l\n".
   "File:\t$f\n".
   "Color:\t$c\n"
 ) if $ret;



#---------------------------------------------------------#
#---- myReshape                  -------------------------#
#---------------------------------------------------------#
sub myReshape{

my($self,$w,$h) = @_;

return if (  $self->{gl_pending} );

glViewport(0,0,$w,$h);
glMatrixMode(GL_PROJECTION);
glLoadIdentity();
gluOrtho2D(0.0,$w,0.0,$h);

} # end of reshape


#---------------------------------------------------------# 
#---- myInit                     -------------------------#
#---------------------------------------------------------# 
sub myInit{
my $self = shift;
 
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glClearColor(1.0,1.0,1.0,0.0);
    glColor3f(0.0,0.0,1.0);
    glLineWidth(2);
    $self->check_channel_range();  
    $self->check_time_range(); 

}# myInit


#---------------------------------------------------------# 
#---- show                       -------------------------#
#---------------------------------------------------------# 
sub show_gl{
my $self= shift;

#$self->plot_range();

$self->{WGL}->GLMakeCurrent();

glClearColor(1.0,1.0,1.0,0.0);
glColor3f(0.0,0.0,1.0);
glLineWidth(2);

#glutInit();

#glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_ALPHA);

#glutInitWindowSize($self->{wSizeX},$self->{wSizeY});
#glutInitWindowPosition($self->{wPosX},$self->{wPosY});


#$self->{IDwindow} = glutCreateWindow("MEG TSV");
#glutDisplayFunc( sub{ $self->update_plot_pdl_to_vbo(@_) } );

glutReshapeFunc( sub{ $self->myReshape(@_) } );

# glutIdleFunc( sub{   while (Glib::MainContext->default->iteration(FALSE)) { } } );

glutKeyboardFunc( sub { $self->keyboard_functions(@_) } );
glutSpecialFunc(  sub { $self->special_functions(@_)  } );
glutSpaceballMotionFunc(  sub { $self->spaceball_motionspecial_functions(@_)  } );

$self->myInit();

$self->setWindow( $self->{xmin},$self->{xmax},$self->{ymin},$self->{ymax} );
$self->setViewport(0,$self->{wSizeX},0,$self->{wSizeY});

#$self->gtk_show_gui();

# $self->wx_show_gui();

# glutPostRedisplay();

# $self->init_menu();

glutMainLoop();

} # end of show

#---------------------------------------------------------# 
#---- calc_scroll_channel_start    -------------------------#
#---------------------------------------------------------# 
sub calc_scroll_channel_start{
my $self = shift;
   $self->_init(@_);
   $self->{scroll_channel_max}  =  $self->{max_channels} - 1;
 ${$self->{scroll_channel_start}}+= $self->{scroll_channel_delta} * ${ $self->{scroll_channel_direction} };
   $self->{scroll_channel_end}  = ${$self->{scroll_channel_start}} + $self->{data_panel_list}->nelem() -1;

   if ( $self->{scroll_channel_end} > $self->{scroll_channel_max} )
    {
      $self->{scroll_channel_end}    = $self->{scroll_channel_max};
      ${$self->{scroll_channel_start}} = $self->{scroll_channel_max} - $self->{data_panel_list}->nelem() +1;
    } # if

   if ( ( ${$self->{scroll_channel_start}} < 0 ) ||  ( $self->{scroll_channel_end} <= 0 ) )
    {
      ${$self->{scroll_channel_start}} = 0;
      $self->{scroll_channel_end}    = $self->{data_panel_list}->nelem() -1;
    } # if
} # end of calc_scroll_pos
1;



__DATA__

 #---- PLOT option channel
   scroll => {
               channel_start      =>  0,
               channel_direction  =>  1,
               channel_delta      =>  20, # xpanels * ypanels
            #   vertical_spacing => 2000;
             #---- PLOT option time
               time_start         =>  0,
               time_direction     =>  1,
               time_delta         => 10, # sec
               time_goto_latency  =>  0,
               time_old_pos       => -1,
             #--- time start end display info in de scroll time
               time_start         =>  0,
               time_end           =>  0,
              }, # scroll






#---------------------------------------------------------# 
#---- dpw_scroll_panels          -------------------------#
#---------------------------------------------------------# 
sub dpw_scroll_panels{
my $self = shift;
my $move = shift;
my $reset= shift;
  
   $self->_init(@_);

 return unless ( $self->{data_loaded} );
 $self->{scroll}{panel_delta} = $self->{xpanels} * $self->{ypanels};

 switch($move){
#my $xscroll = $fpl->Scrollbar(-orient => 'horizontal')->pack(-side => 'bottom', -fill => 'x');
#my $yscroll = $fpl->Scrollbar(-orient => 'vertical'  )->pack(-side => 'right',  -fill => 'y');


#--- jump to start
 case "start"{
              $self->{scroll}{panel_start}     = -9999;
              $self->{scroll}{panel_direction} =  1;
             }; # start
#--- move up one page
 case "pgup" {
               return if ($self->{scroll}{panel_start} < 1);
               $self->{scroll}{panel_direction} = -1;
             };
      
#--- move up one channel
 case "oneup"{
              $self->{scroll}{panel_direction} = 0;
              $self->{scroll}{panel_start}--;#    += 1;#$self->{scroll}{channel_delta} -1;
             # $self->{scroll}{channel_start}    += $self->{scroll}{channel_delta} +1;
             }; # oneup

#--- move down one channel
 case "onedown"{
                $self->{scroll}{panel_direction} = 0;
                $self->{scroll}{panel_start}++;#    -= 1; #$self->{scroll}{channel_delta} -1;
               }; # onedown
#--- move down to last channel
 case "pgdown"{
                return if ($self->{scroll}{panel_end} >= $self->{scroll}{panel_max}-1 );
                $self->{scroll}{panel_direction} = 1;
              }

#--- jump to last channel
 case "end"   {
               $self->{scroll}{panel_start}     = ${$self->{ydata}}->dim(0)+1;
               $self->{scroll}{panel_direction} = 1;
              } # end 

 case "update"{$self->{scroll}{panel_direction} = 0;
              # $self->{scroll}{channel_start}   -= $self->{scroll}{channel_delta};
              } # update     

} # end SWITCH

 return if ($reset); # set flag for start with epoch 0  

 $self->_scroll_panels;


} # end of dpw_scroll_panels
1;

#---------------------------------------------------------# 
#---- dpw_init_plot_bindings     -------------------------#
#---------------------------------------------------------# 
sub dpw_init_plot_bindings{
my ( $self,$w,$release) = @_;

  if (defined ($release) )
   {
     $$w->bind('<Home>' );
     $$w->bind('<Left>' );
     $$w->bind('<Right>');
     $$w->bind('<End>'  ); 
     $$w->bind('<Prior>');
     $$w->bind('<Down>' );
     $$w->bind('<Up>'   );
     $$w->bind('<Next>' ); 
     $$w->bind('<Configure>' ); 
     $$w->bind('<Button-3>'  );
     $$w->bind('<Motion>'    );
     return;
   } 

 $$w->bind('<Configure>'  => [sub {$self->dpw_plot() }] );

 return unless ( $self->{inter_active} );


# $$w->bind('<Configure>'=> sub {$self->dpw_rezise_plot( $w ) } );

# $$w->bind('<Button-3>' => sub {$self->dpw_clickon_panel() });

#$$w->bind('<Motion>'   => sub {$self->dpw_report_motion() });
#$$w->bind('<Motion>'   => sub {print"MOTION\n" });


 $$w->bind('<Home>'  => [sub{ 
                                           $self->wmain->Busy();
                                         #  $self->scroll_time("start");
                                           $self->wmain->Unbusy();
                                           }] );
 $$w->bind('<Left>'  => [sub{ 
                                           $self->wmain->Busy();
                                         #  $self->scroll_time("left");
                                           $self->wmain->Unbusy();
                                          }] );
 $$w->bind('<Right>' => [sub{
                                           $self->wmain->Busy();
                                          # $self->scroll_time("right");
                                           $self->wmain->Unbusy();
                                          } ]);
 $$w->bind('<End>'   => [sub{ 
                                           $self->wmain->Busy();
                                          # $self->scroll_time("end");
                                           $self->wmain->Unbusy();
                                          }] ); 

 $$w->bind('<Prior>' => [sub{ $self->dpw_scroll_panels("start")}]);
 $$w->bind('<Down>'  => [sub{ $self->dpw_scroll_panels("down") }]); 
 $$w->bind('<Up>'    => [sub{ $self->dpw_scroll_panels("up")   }]);  
 $$w->bind('<Next>'  => [sub{ $self->dpw_scroll_panels("end")  }]); 

 $$w->bind('<Button-3>'  => [sub{ $self->dpw_update_panel() }] );

# $$w->bind('<Button-2>'  => [sub{ print"TEST cursor\n";
#                                  $self->{plot}->set_cursor();
#                               
#
#                                }]); 



} # end of wde_init_plot_bindings



















# SPECIAL KEYBOARD
sub special_keyboard {
    if (($options{time}!=0) and (!$time_real)) {
        $time_real=$options{time};
        $options{time}=0;
    }
    $time{key} = [gettimeofday];

    my ($key) = @_; 

    # Special keys
    if   ($key==GLUT_KEY_LEFT){ reset_all(4); }
    elsif($key==GLUT_KEY_RIGHT){ reset_all(31); }
    elsif($key==GLUT_KEY_UP){ }
    elsif($key==GLUT_KEY_DOWN){ }
    elsif($key==GLUT_KEY_F1){ }
    elsif($key==GLUT_KEY_F2){ }
    elsif($key==GLUT_KEY_F3){ }
    elsif($key==GLUT_KEY_F4){ }
    elsif($key==GLUT_KEY_F5){ }
    elsif($key==GLUT_KEY_F6){ }
    elsif($key==GLUT_KEY_F7){ }
    elsif($key==GLUT_KEY_F8){ }
    elsif($key==GLUT_KEY_F9){ }
    elsif($key==GLUT_KEY_F10){ }
    elsif($key==GLUT_KEY_F11){ }
    elsif($key==GLUT_KEY_F12){ }
    elsif($key==GLUT_KEY_PAGE_UP){ }
    elsif($key==GLUT_KEY_PAGE_DOWN){ }
    elsif($key==GLUT_KEY_HOME){ }
    elsif($key==GLUT_KEY_END){ }
    elsif($key==GLUT_KEY_INSERT){ }
}

#---------------------------------------------------------# 
#---- key_pressed                -------------------------#
#---------------------------------------------------------# 
sub key_pressed{
my $self= shift;
print"KEY pressed \n";

} # end of key_pressed(

#---------------------------------------------------------# 
#---- gtk_show_gui               -------------------------#
#---------------------------------------------------------# 
sub gtk_show_gui{
my $self= shift;

 $self->{gtk_main_ptr} = $self->gtk_init_main();
 #$self->wx_main->Show; #Modal;
 
} # end of gtk_show_gui

#-----------------------------------------------------------------------# 
#  gtk_init_main
#-----------------------------------------------------------------------# 
sub gtk_init_main{
my $self = shift;

# Initialize GTK

my $window = Gtk2::Window->new("toplevel");
$window->set_title("Button Boxes");
$window->signal_connect( "destroy" => sub {Gtk2->main_quit;});
$window->set_border_width(10);

my $main_vbox = Gtk2::VBox->new("false", 0);
$window->add($main_vbox);

my $frame_horz = Gtk2::Frame->new("Horizontal Button Boxes");
$main_vbox->pack_start($frame_horz, TRUE, TRUE, 10);

my $vbox = Gtk2::VBox->new(FALSE, 0);
$vbox->set_border_width(10);
$frame_horz->add($vbox);

     my $button = Gtk2::Button->new_from_stock('gtk-ok');
	$button->signal_connect( 'clicked' => sub {
                                                   $self->{ch_idx1} = $self->{ch_idx0} + $self->{ch_idx_delta};
						   $self->{ch_idx0}++;
						   if ( $self->{ch_idx0}  > $self->data->dim(0)-1 )
						    {
                                                     $self->{ch_idx1} = $self->data->dim(0)-1;
						     $self->{ch_idx0} = $self->{ch_idx1} - $self->{ch_idx_delta};
                                                     $self->{ch_idx0} = 0 if ($self->{ch_idx0} < 0); 
						    }

                                                 print " click + $self->{ch_idx0}  $self->{ch_idx1}\n";
                                               #  glutPostRedisplay(); 
                    } );

     my $button1 = Gtk2::Button->new_from_stock('gtk-ok');
	$button1->signal_connect( 'clicked' => sub {
                                                   $self->{ch_idx0}--;
						   
						   if ( $self->{ch_idx0}  < 0 )
						    {
                                                    $self->{ch_idx0} = 0;
						    $self->{ch_idx0} + $self->{ch_idx_delta};
						     }
                                                   $self->{ch_idx1} = $self->{ch_idx0} + $self->{ch_idx_delta};
                                                  
                                                  print " click - $self->{ch_idx0}  $self->{ch_idx1}\n";                                                                                               #  glutPostRedisplay(); 
                    } );


	$vbox->add($button);
	$vbox->add($button1);


  $window->show_all;
  Gtk2->main;

} # gtk_init_main




__DATA__
#---------------------------------------------------------# 
#---- wx_show_gui                -------------------------#
#---------------------------------------------------------# 
sub wx_show_gui{
my $self= shift;

 $self->{wx_app}      = Wx::SimpleApp->new;
 $self->{wx_main_ptr} = $self->wx_init_main();
 $self->wx_main->Show; #Modal;
 $self->{wx_app}->MainLoop;

} # end of wx_show_gui

#-----------------------------------------------------------------------# 
#  wx_init_main
#-----------------------------------------------------------------------# 
sub wx_init_main{
my $self = shift;

$self->{title} ="TEST";
my $w = Wx::Frame->new(undef, -1, $self->{title},
                       Wx::Point->new( $self->{wx_xpos}, $self->{wx_ypos} ),
                       Wx::Size->new( $self->{wx_width}, $self->{wx_height}),
		      );
   $w->CreateStatusBar( 2 );
   $w->SetIcon( Wx::GetWxPerlIcon() );

#   Wx::EVT_CLOSE($w, sub{ $self->wx_OnClose } );

#--- some grid
my $bzv0 = Wx::BoxSizer->new( wxVERTICAL );

my %sz=();

my %wx_panel= ();

#--- process panels
   foreach  my $p ("channels","tsls")
     {
       $sz{ $p } = Wx::StaticBoxSizer->new( Wx::StaticBox->new( $w, -1,$_ ),wxHORIZONTAL );
       $bzv0->Add( $sz{$p}, 0,  &Wx::wxGROW|&Wx::wxALL, 2 );
     }
 
#--- init channels
   $self->wx_init_channels(\$w,\$sz{ channels },\$bzv0);
#--- init dicom 2 nii & idl preproc
#   $self->wx_init_tsls(\$w,\$sz{ $self->{wx_panel_list}[1] },\$bzv0);

#---
  $w->SetAutoLayout( 1 );
  $w->SetSizer( $bzv0 );
  # size the window optimally and set its minimal size
  $bzv0->Fit($w);
  $bzv0->SetSizeHints( $w );

  return \$w;

} # end of wx_init_main

#-----------------------------------------------------------------------# 
#  wx_init_channels
#-----------------------------------------------------------------------# 
sub wx_init_channels{
my ($self,$wptr,$sz_ptr,$bzv_ptr) =  @_;

my $sh = Wx::BoxSizer->new( wxHORIZONTAL );


 $self->{wx_ch0} = Wx::SpinCtrl->new($$wptr,-1);
 $self->{wx_ch0}->SetRange( 0,$self->data->dim(-1)-1 );
 $self->{wx_ch0}->SetValue( $self->{ch_idx0} );
 Wx::Event::EVT_BUTTON( $$wptr,$self->{wx_ch0},
                      sub{ 
		           $self->{ch_idx0} = $self->{wx_ch0}->GetValue();
			 
			 } );

 $sh->Add($self->{wx_ch0}, 0, wxALL, 10 );

 $self->{wx_ch1} = Wx::SpinCtrl->new($$wptr,-1);
 $self->{wx_ch1}->SetRange( 0,$self->data->dim(-1)-1 );
 $self->{wx_ch1}->SetValue( $self->{ch_idx1} );
 Wx::Event::EVT_BUTTON( $$wptr,$self->{wx_ch1},
                      sub{ 
		           $self->{ch_idx1} = $self->{wx_ch1}->GetValue();
			 
			 });
 $sh->Add($self->{wx_ch1}, 0, wxALL, 10 );

 $$sz_ptr->Add( $sh,0, &Wx::wxEXPAND | &Wx::wxGROW|&Wx::wxALL,2);




} # end wx_init_channels


#--- init dicom 2 nii & idl preproc
#   $self->wx_init_tsls(\$w,\$sz{ $self->{wx_panel_list}[1] },\$bzv0);



