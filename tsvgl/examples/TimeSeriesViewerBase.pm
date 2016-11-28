#-----------------------------------------------------------
# Perl-Module MEG::MEG13::TSV::TimeSeriesViewerBase
# Autor: F.B.
# Date 29.01.2013
# last change 29.01.2013
#
#-----------------------------------------------------------
package MEG::MEG13::TSV::TimeSeriesViewerBase;

use strict;
use OpenGL qw(:all);

use Carp;
use PDL;
use PDL::Constants qw(PI);
    $PDL::BIGPDL = 1;


use base qw(MEG::MEG13::GLOBAL::MEGGlobalFct);

#-----------------------------------------------------------------------# 
#   CLASS
#-----------------------------------------------------------------------# 
sub new {
my $class = shift;
my $self  = $class->SUPER::new( 
                                tittle=> "FZJ INM4 MEG Time Series Viewer",
                                wmain_ptr => undef,

		   	        wSizeX    => 680,
                                wSizeY    => 480,
                                wPosX     => 10,
                                wPosY     => 10,
                                wMinsize  =>"300x300",
			       #---
			         WGL      => undef,

			       #--- channels scroll
                                 ch_idx0      =>  0,
                                 ch_idx_delta => 20,
                                 ch_idx1      => 19,
                                #--- tsl / time scroll
                                 tsl0      =>     0,
                                 tsl_delta => 20000,
                                 tsl1      => undef,
                                 time0     => undef,
                                 time1     => undef,
				 time_min_sec => 0,

                                #---
				 gl_pending => undef,
                                 gl_font     => GLUT_BITMAP_HELVETICA_18,
#GLUT_BITMAP_HELVETICA_10,
#GLUT_BITMAP_HELVETICA_12
#GLUT_BITMAP_HELVETICA_18

#GLUT_BITMAP_TIMES_ROMAN_10
#GLUT_BITMAP_TIMES_ROMAN_24

                                 sample_frequency => 1000,

  sensor_layout=>{
                  sensor_pos_path => $ENV{MEG_HOME}."/templates/sensor_pos",
                  sensor_pos_file => "Juelich3600.spos",
		  gl_font         => GLUT_BITMAP_HELVETICA_10,

                 },
	       
  

	                      );
    bless ( $self,$class );

    $self->Init(@_);  

    return $self;

} # new
1;


sub wmain { $_[0]->{wmain} }
sub wmain_ptr { \$_[0]->{wmain} }

sub WGL { $_[0]->{WGL} }

sub data{ ${$_[0]->{data_ptr}}  }
sub data_ptr{ $_[0]->{data_ptr} }

sub label_list{ 
 if ( ref $_[0]->{label_list} )
  {  
  ${ $_[0]->{label_list} }[$_[1]] = $_[2] if defined( ( $_[2] ) ); 
  return ${ $_[0]->{label_list}} [$_[1]] if defined( ( $_[1] ) ); 
  return @${ $_[0]->{label_list} };
  }

   $_[0]->{label_list}[$_[1]] = $_[2] if defined( ( $_[2] ) ); 
  return $_[0]->{label_list}[$_[1]] if defined( ( $_[1] ) ); 
  return @$_[0]->{label_list};
} # end of label_list



sub xdata{ ${$_[0]->{xdata_ptr}}  }
sub xdata_ptr{ $_[0]->{xdata_ptr} }

sub wSizeX{ $_[0]->{wSizeX} } 
sub wSizeY{ $_[0]->{wSizeY} } 


# "color_r","color_g""color_b","color_a"} || 0;

sub channel_info{
my $self = shift;
my $idx  = shift;  
my @a    = (); 
 
   foreach (@_)
    {
      push( @a, $self->{channel_info}{$idx}{$_} ) if ( defined( $self->{channel_info}{$idx}{$_} ) );
    }

 return @_;

} # end of channel_info


#---------------------------------------------------------# 
#---- tsv_base_plot_range        -------------------------#
#---------------------------------------------------------# 
sub tsv_base_plot_range{
my $self = shift;
   $self->{ymin}  = $self->data->min(); 
   $self->{ymin} -= $self->{ymin} * 0.12;
   $self->{ymax}  = $self->data->max() * 1.12;
 
   $self->{xmin}  = $self->xdata->at(0);
   $self->{xmax}  = $self->xdata->at(-1);
  
} # end of tsv_base_plot_range




#---------------------------------------------------------# 
#---- load_sensor_pos               ----------------------#
#---------------------------------------------------------# 
sub load_sensor_pos{ 
my $self = shift;
  
   $self->{channel_list} = undef; 

   $self->{sensor_pos_loaded} = undef;

   $self->{s_xpos}         = pdl(null)->float; 
   $self->{s_ypos}         = pdl(null)->float;
   $self->{s_zpos}         = pdl(null)->float;

# ( $self->{channel_list},$self->{xpos},$self->{ypos},$self->{zpos} ) =
#   rcols( "$self->{path_sensor_pos}/$self->{sensor_pos_file}",0,1,2,3, { COLSEP => ',', PERLCOLS => [ 0 ]} );


#--- !! strange format
#--- Channel    X/m      Y/m      Z/m        Nx       Ny       Nz  
#---   A1       -0.000,  -0.001,  +0.002,  -0.0016, -0.0197, +0.9998
my $aptr;
my ($ch,$x);
   @{$self->{channel_list}} = (); 
 
 
 ( $aptr,$self->{s_ypos},$self->{s_zpos} ) =
   rcols( "$self->{sensor_layout}{sensor_pos_path}/$self->{sensor_layout}{sensor_pos_file}",0,1,2,3, { COLSEP => ',', PERLCOLS => [ 0 ]} );
  
   $self->{s_xpos} = pdl( zeroes( $self->{s_ypos}->dim(-1) ) )->float; 

   for (0 .. $#$aptr)
    { 
        $$aptr[$_]=~s/\s+//;
      ( $ch,$x ) = split( /\s+/,$$aptr[$_] );
        $self->{channel_list}[$_] =  $ch;
        $self->{s_xpos}->set($_,$x);
   } # for

#  if ($self->DEBUG)
#   {
#     $self->pdlinfo("POS X",\$self->{xpos},1,1);
#     $self->pdlinfo("POS Y",\$self->{ypos},1,1);
#     $self->pdlinfo("POS Z",\$self->{zpos},1,1);
#   }

   $self->{s_xpos}  = ( $self->{s_xpos} * 100)->float; 
   $self->{s_ypos}  = ( $self->{s_ypos} * 100)->float;
   $self->{s_zpos}  = ( $self->{s_zpos} * 100)->float;
 
   $self->{s_xpos} -= $self->{s_xpos}->at(0); 
   $self->{s_ypos} -= $self->{s_ypos}->at(0);
   $self->{s_zpos} -= $self->{s_zpos}->at(0);

   $self->{tyspos} =  $self->{s_xpos};
   $self->{txspos} = -$self->{s_ypos};
   $self->{tzspos} =  $self->{s_zpos};


my $pos_data = pdl( zeroes(3,$self->{s_xpos}->dim(-1) ) )->float();
   $pos_data->slice("(0),:").= $self->{s_xpos}; 
   $pos_data->slice("(1),:").= $self->{s_ypos}; 
   $pos_data->slice("(2),:").= $self->{s_zpos}; 

   $self->{pos_data_ptr} = \$pos_data;

   $self->{sensor_pos_loaded} = 1;

} # end of load_sensor_pos

#---------------------------------------------------------# 
#---- sl_project_sensor_pos         ----------------------#
#--- idl transform sensor data into layout format
#---     xy[0,*] = xy[0,*] + (z^pow * xy[0,*])
#--      xy[1,*] = xy[1,*] + (z^pow * xy[1,*])
#-- pow = 1.6
#---------------------------------------------------------# 
sub project_sensor_pos{ 
my $self = shift;
   $self->{sensor_pos_projected} = undef;

my $zz  = pdl( $self->{tzspos} )->float() ;# ->pow(1.6);
   $zz /= $self->{tzspos}->max(); #norm
   $zz  = -$zz + 1;
   $zz  = $zz->pow(1.6);

 #  $zz += 1;
#   $zz->inplace->pow(1.6);
 #  $zz -= 1;
#  $zz->badvalue(0.0);

#  $self->pdlinfo("POS Z",\$self->{tzspos},1,1);

#  $self->pdlinfo("POS TZZ",\$zz,1,1);


#   $self->pdlinfo("POS TX",\$self->{txspos},1,1);

#   $self->pdlinfo("POS TX",\$self->{txspos},1,1);
 
   $self->{txspos} += $self->{txspos} * $zz;
   $self->{tyspos} += $self->{tyspos} * $zz;

 #  $self->pdlinfo("POS TX",\$self->{txspos},1,1);

 #  $self->pdlinfo("POS TX",\$self->{txspos},1,1);
   #$self->pdlinfo("POS TZZ",\$zz,1,1);

# norm 0 - 1
   $self->{sxpos}  = pdl( $self->{txspos} );
   $self->{sxpos} -= $self->{sxpos}->min();
   $self->{sxpos} /= $self->{sxpos}->max();
 
   $self->{sypos}  = pdl( $self->{tyspos} );
   $self->{sypos} -= $self->{sypos}->min();
   $self->{sypos} /= $self->{sypos}->max();

   $self->{sensor_pos_projected} = 1;

#   $self->pdlinfo("POS",\$pos,1,1);
} # end of project_sensor_pos





#-----------------------------------------------------------------------# 
#--- print_string                 --------------------------------------#
#-----------------------------------------------------------------------# 
sub print_string{
my ($self,$str,$font) = @_;
my @c = split '', $str;
   for(@c) { glutBitmapCharacter($font, ord $_) }
} # end of print_string


#---------------------------------------------------------# 
#---- tsv_base_check_channel_range  ----------------------#
#---------------------------------------------------------# 
sub tsv_base_check_channel_range{
my $self = shift;
   
   if( $self->{ch_idx0} >= $self->data->dim(0) - $self->{ch_idx_delta} -1 )
     { $self->{ch_idx0} = $self->data->dim(0) - $self->{ch_idx_delta} -1  }
   elsif ($self->{ch_idx0} < 0 ) { $self->{ch_idx0} = 0 }
  
   $self->{ch_idx1} = $self->{ch_idx0}  +  $self->{ch_idx_delta};
   $self->{ch_idx1} = $self->data->dim(0) -1 if ( $self->{ch_idx1} > $self->data->dim(0) -1 );

   print"done check_channel_range  => $self->{ch_idx0}  == $self->{ch_idx1}\n" if ( $self->debug );

} # end of tsv_base_check_channel_range  

#---------------------------------------------------------# 
#---- tsv_base_check_time_range   ------------------------#
#---------------------------------------------------------# 
sub tsv_base_check_time_range{
my $self = shift;
  
   if( $self->{tsl0} >= $self->data->dim(-1) - $self->{tsl_delta} -1 )
     { $self->{tsl0} = $self->data->dim(-1) - $self->{tsl_delta} -1 }
    elsif ( $self->{tsl0} < 0 ){ $self->{tsl0} = 0 }
  
   $self->{tsl1} = $self->{tsl0}  +  $self->{tsl_delta};

   $self->{tsl1} = $self->data->dim(-1) -1 if ( $self->{tsl1} > $self->data->dim(-1) -1 );

 
   $self->{time0}        = $self->{time_min_sec} + $self->{tsl0} / $self->{sample_frequency};
   $self->{time_delta}   = $self->{time_min_sec} + $self->{tsl_delta} / $self->{sample_frequency};

 
  print"done check_time_range  => $self->{tsl0}  == $self->{tsl1}\n" if ( $self->debug );

} # end of tsv_base_check_channel_range  









__DATA__
   wmain             => undef,
   wframe            => undef,
   DPW_FRAME         => undef,
   window_data_name  => undef,
   window_xaxis_name => undef,
  # ID                => undef;                                        
   data_loaded       => undef,
   on_rezise         => undef,   
                                    
   wSizeX            => 1000,
   wSizeY            => 3000,
 
   channel_info_ptr => undef,
   group_ptr        => undef,

#--- init colour 
   color_list => [qw/black white red green blue cyan magenta yellow orange darkgray lightgray/],
   group_list => [qw/DUMMY MEG MEGB  REFERENCE EEG  EXTERNAL TRIGGER UTILITY DERIVED SHORTED/],
#--- init group color  
   group=>{
              MEG       => { color => "red",    yscaling_minmax => 2000 },
              REFERENCE => { color => "green",  yscaling_minmax => 4000 },
              EEG       => { color => "cyan",   yscaling_minmax => 0.5  },
              EXTERNAL  => { color => "magenta",yscaling_minmax => 1    },
              TRIGGER   => { color => "blue",   yscaling_minmax => 16   },
              UTILITY   => { color => "white",  yscaling_minmax => 2000 },
              DERIVED   => { color => "white",  yscaling_minmax => 2000 },
              SHORTED   => { color => "white",  yscaling_minmax => 2000 },
           }, # group 

#--- plot device names
   wDevXAXIS => undef, # 'ICAAxisWindow/ptk'
   wDevDATA  => undef, # 'ICAPlotWindow/ptk'


#---- SUB PLOT  X Y 
   xpanels               => 1,
   ypanels               => 10,

#---- PLOT option Y Scaling
   yscaling_type         => "adjust", # fixed

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

     active_channel => "", # cursor motion
     error          => 0,
     verbose        => 0,  
 
 ); # $self

  
   bless ( $self,$class );
    
   $self->_init(@_); 
 
  
   return $self;
  
} # new
1;

#--- helper FKts
sub version{$_[0]->{version}}

sub wmain{ ${ $_[0]->{wmain} } }
sub mframe { ${ $_[0]->{mframe} } }
sub dpw_frame { $_[0]->{DPW_FRAME} }

sub id { $_[0]->{DE_DATA}->id() } 
sub DE_DATA{ $_[0]->{DE_DATA} }

#=======================================================================
#
# Function List
#
#=======================================================================
#

#---------------------------------------------------------# 
#---- dpw_check_window           -------------------------#
#---------------------------------------------------------# 
sub dpw_check_window{
my ( $self, $w ) = @_;

#--- window
   return -1 unless ( defined ( $$w ) );
   if ( Exists( $$w ) ) { $$w->deiconify(); $$w->raise(); $$w->focus(); return 1; } 
    else { return -1 };

} # end of dpw_check_window
1;


#---------------------------------------------------------# 
#---- dpw_clickon_panel          -------------------------#
#---------------------------------------------------------# 
sub dpw_clickon_panel{
my $self = shift;
my $panel =  $self->dpw_get_active_panel();
my $idx   =  $self->{scroll}{channel_start} + $panel -1 ;
   return($panel,${$self->{data_idxptr}}->at($idx),$idx);

} # end of dpw_clickon_panel

#---------------------------------------------------------# 
#---- wde_get_active_panel       -------------------------#
#---------------------------------------------------------# 
sub dpw_get_active_panel{
my $self = shift;
my $e   = $self->{DE_DATA}->XEvent;
my $pfx = int( $self->{DE_DATA}->width  / $self->{plot}->xpanels() );
my $pfy = int( $self->{DE_DATA}->height / $self->{plot}->ypanels() );

#-- from left to right
my $pix   = int( $e->x / $pfx ) + 1;
my $piy   = int( $e->y / $pfy );

return ($self->{plot}->xpanels() * $piy + $pix); 
   
} # end of wde_get_active_panel

#-----------------------------------------------------------------------# 
#--- show                         --------------------------------------#
#-----------------------------------------------------------------------# 
sub show{
my $self = shift;
   $self->{DPW_FRAME} = $self->mframe->Frame(-borderwidth=>1,-relief=>'ridge' )
                                              ->pack(qw/-side left -fill both -expand 1/);

   $self->dpw_frame->configure(-cursor => 'top_left_arrow');
   $self->wmain->idletasks; # !! call for pgplot
   $self->dpw_init_plot(\$self->{DPW_FRAME});

} # end of show
1;

#---------------------------------------------------------# 
#---- dpw_init_plot              -------------------------#
#---------------------------------------------------------# 
sub dpw_init_plot{
my ($self, $wptr ) = @_;
my $dev;

my $fpl = $$wptr->Frame(-borderwidth=>1,-relief=>'sunken')->pack(qw/-side left -fill both -expand 1/);

#my $xscroll = $fpl->Scrollbar(-orient => 'horizontal')->pack(-side => 'bottom', -fill => 'x');
#my $yscroll = $fpl->Scrollbar(-orient => 'vertical'  )->pack(-side => 'right',  -fill => 'y');

 ( $self->{dev_data} = $self->{window_data_name} )=~s/\/ptk$//;

   $self->{DE_DATA} = $fpl->Pgplot(
                          -name      => $self->{dev_data},
   		          -share     => 1,
		          -mincolors => 16,
			  -maxcolors => 32, 
                          -relief    =>"flat", #'sunken',
			  -bd        =>1,# 2,
			  -bg        => 'black',
			  -fg        => 'white',
                          -height    => '10c',
                          -width     => '13c',


		       )->pack(qw/-side top -fill both -expand 1/);

 $self->{DE_DATA}->bind('<Configure>'  => [sub {$self->dpw_rezise_plot() }] );
# $self->{DE_DATA}->bind('<Button-3>' => sub {$self->dpw_clickon_panel() });

 return unless( $self->{window_xaxis_name} );

 #$self->{DE_DATA}->configure(-yscrollcommand => ['set', $yscroll]);
 #$yscroll->configure(-command => ['yview',$self->{DE_DATA}]);

 #$xscroll->configure(-command => ['xview',$self->{DE_DATA}]); 
 #$self->{DE_DATA}->configure(-xscrollcommand => ['set', $xscroll]);

#  $self->{DE_DATA}->bind('<Motion>' => sub{ $self->wde_report_motion } );
#  $xscroll->bind('<Motion' => sub{ print"XSC TEST:\n"; foreach ($xscroll->get()){print"$_\n"} } );

 ( $self->{dev_xaxis} = $self->{window_xaxis_name} )=~s/\/ptk$//;

  $self->{DE_XAXIS}  = $fpl->Pgplot(
                          -name      => $self->{dev_xaxis},
			  -share     => 1,
		          -mincolors => 16,
			  -maxcolors => 32, 
                          -height    => '1.5c',
                          -relief    => "flat", #'ridge',
			  -bd        => 1,
			  -bg        => 'black',
			  -fg        => 'white',
		       )->pack(qw/-side bottom -fill x/);

# $self->{DE_XAXIS}->bind('<Configure>'  => sub {$self->dpw_rezise_plot( \$self->{DE_DATA} )} );


#--- redraw plot
# $self->{DE_DATA}->bind('<Configure>'  => sub {$self->dpw_rezise_plot( \$self->{DE_DATA} )} );
## $self->{DE_DATA}->bind('<Button-1>' => sub {$self->dpw_report_motion() });
## $self->{DE_DATA}->bind('<Motion>' => sub {$self->wde_report_motion() });

# $self->dpw_rezise_plot( \$self->{DE_DATA} );

# $self->{DE_DATA}->bind('<Button-3>' => sub {$self->dpw_clickon_panel() });

# $self->dpw_init_plot_bindings(\$self->{DE_DATA});
#$self->dpw_init_plot_bindings(\$fpl);
#$self->dpw_init_plot_bindings($wptr);




} # end of dpw_init_plot
1;
 
#---------------------------------------------------------# 
#---- dpw_rezise_plot            -------------------------#
#---------------------------------------------------------# 
sub dpw_rezise_plot{
my ( $self, $wptr ) = @_;
   return if (defined( $self->{on_rezise} ) );

my $m2p = $self->{DE_DATA}->fpixels('1m');
my $wx  = int($self->{DE_DATA}->width / $m2p ) -5;
my $wy  = int($self->{DE_DATA}->height/ $m2p ) -5;
 
   if ( ( $self->{ wSizeX } != $wx ) or ( $self->{ wSizeY } != $wy ) )
    {
      $self->{ wSizeX } = $wx;
      $self->{ wSizeY } = $wy;
      $self->dpw_rezise() if $self->{data_loaded};
    } # if

# $self->{on_rezise} = undef;

} # end of dpw_rezise_plot
1;

#---------------------------------------------------------# 
#---- dpw_rezise                 -------------------------#
#---------------------------------------------------------# 
sub dpw_rezise{
my $self = shift; 
   return if $self->{on_rezise};

  return -1 if $self->error();
  $self->{on_rezise} = 1;

 # $self->dpw_init_plot_bindings($self->{wmain},1);
  $self->wmain->idletasks;
  $self->wmain->Busy();
  

 
  if ($self->{xpanels} > 1 || $self->{ypanels} >1 )
   {
     $self->dpw_init_sub_plot();
   }
  $self->dpw_scroll_channels(1);
#  $self->dpw_init_plot_bindings($self->{wmain});

  $self->wmain->Unbusy();
  $self->wmain->idletasks;

  $self->{on_rezise} = undef;

} # end of dpw_rezise
1;

#---------------------------------------------------------# 
#---- dpw_int_sub_plot           -------------------------#
#---------------------------------------------------------# 
sub dpw_init_sub_plot{
 my $self = shift; 

#--- plot

 $self->{scroll}{channel_direction} = 0;
 $self->{scroll}{channel_start}     = 0;

my $group_ptr = \$self->{group};
   $group_ptr =  $self->{group_ptr} if ( defined( $self->{group_ptr} ) );

 unless ( defined( $self->{plot} ) )
  {  

    $self->{plot}= MEG::Magnes3600::ICA::ICAPGPlot_SubPlotUtil->new(  
                                                    wDevXAXIS  => $self->{window_xaxis_name},
                                                    wDevDATA   => $self->{window_data_name},
                                                    wNX                      =>  $self->{xpanels},
                                                    wNY                      =>  $self->{ypanels},
                                                    wSizeY                   =>  $self->{wSizeY},
                                                    wSizeX                   =>  $self->{wSizeX},
                                                    yscaling                 =>  $self->{'yscaling_type'}, 
                                                  # yscaling_min             =>  $self->{'yscaling_min'},  
                                                  # yscaling_max             =>  $self->{'yscaling_max'}, 
                                                    scroll_channel_start     => \$self->{scroll}{channel_start},
                                                    scroll_channel_direction => \$self->{scroll}{channel_direction},
                                                    ydata                    =>  $self->{data_yptr},
                                                    xdata                    =>  $self->{data_xptr}, 
                                                    data_idxptr              =>  $self->{data_idxptr},  
                                                    channel_info             =>  $self->{channel_info_ptr},
                                                    group_info               =>  $group_ptr,
                                                    max_channels             =>  $self->{max_channels},
                                                    verbose                  =>  $self->{verbose} 
                                                   );

 } else{ $self->{plot}->init_plot(
                                                    wNX                      =>  $self->{xpanels},
                                                    wNY                      =>  $self->{ypanels},
                                                    wSizeY                   =>  $self->{wSizeY},
                                                    wSizeX                   =>  $self->{wSizeX},
                                                    yscaling                 =>  $self->{'yscaling_type'}, 
                                                  # yscaling_min             =>  $self->{'yscaling_min'},  
                                                  # yscaling_max             =>  $self->{'yscaling_max'}, 
                                                    scroll_channel_start     => \$self->{scroll}{channel_start},
                                                    scroll_channel_direction => \$self->{scroll}{channel_direction},
                                                    ydata                    =>  $self->{data_yptr},
                                                    xdata                    =>  $self->{data_xptr}, 
                                                    overlay_data_ptr         =>  $self->{overlay_data_ptr},
                                                    channel_info             =>  $self->{channel_info_ptr},
                                                    group_info               =>  $group_ptr,
                                                    max_channels             =>  $self->{max_channels},
                                                    verbose                  =>  $self->{verbose} 
                                                  );
} # else



 } # end of dpw_init_sub_plot
1;


#---------------------------------------------------------# 
#---- dpw_start                  -------------------------#
#---------------------------------------------------------# 
sub dpw_start{
my $self = shift; 
   $self->_init(@_); 

   $self->{ wSizeX } = -1 if $self->{data_loaded};

   $self->dpw_rezise_plot();
 
} # end of dpw_start
1;
 
#---------------------------------------------------------# 
#---- dpw_update_overlay_channels ------------------------#
#---------------------------------------------------------# 
sub dpw_update_overlay_channels{
my $self = shift; 
 $self->_init(@_); 

 $self->{plot}->update_overlay(
       overlay_data_ptr    => $self->{overlay_data_ptr},
       ydata               => $self->{data_yptr},
       xdata               => $self->{data_xptr},
       data_idxptr         => $self->{data_idxptr},  
     );  

} # end of dpw_update_overlay_channels 

#---------------------------------------------------------# 
#---- dpw_scroll_channel_pos     -------------------------#
#---------------------------------------------------------# 
sub dpw_scroll_channel_pos{
my $self = shift; 
   $self->_init(@_); 

   $self->{plot}->scroll_channels(
                                scroll_channel_start     => \$self->{scroll}{channel_start}, 
                                scroll_channel_direction => \$self->{scroll}{channel_direction},
                                ydata                    =>  $self->{data_yptr},
                                xdata                    =>  $self->{data_xptr},
                                data_idxptr              =>  $self->{data_idxptr},  
                                overlay_data_ptr         =>  $self->{overlay_data_ptr},
                              );

} # end of dpw_scroll_channel_pos
1;

#---------------------------------------------------------# 
#---- dpw_scroll_channels        -------------------------#
#---------------------------------------------------------# 
sub dpw_scroll_channels{
my $self = shift;
my $move = shift;
   $self->_init(@_);

 return unless ( $self->{data_loaded} );
 $self->{scroll}{channel_delta} = $self->{xpanels} * $self->{ypanels};

 switch($move){
#--- jump to start
 case "start"{
              $self->{scroll}{channel_start}     = -9999;
              $self->{scroll}{channel_direction} =  1;
             }; # start
#--- move up one page
 case "pgup" { $self->{scroll}{channel_direction} = -1 };
      
#--- move up one channel
 case "oneup"{
              $self->{scroll}{channel_direction} = 0;
              $self->{scroll}{channel_start}--;#    += 1;#$self->{scroll}{channel_delta} -1;
             # $self->{scroll}{channel_start}    += $self->{scroll}{channel_delta} +1;
              }; # oneup
#--- move down one channel
 case "onedown"{
                $self->{scroll}{channel_direction} = 0;
                $self->{scroll}{channel_start}++;#    -= 1; #$self->{scroll}{channel_delta} -1;
               }; # onedown
#--- move down to last channel
 case "pgdown"{ $self->{scroll}{channel_direction} = 1}

#--- jump to last channel
 case "end"   {
               $self->{scroll}{channel_start} = ${$self->{data_idxptr}}->dim(0) + 1;
               $self->{scroll}{channel_direction} = 1;
              } # end 

 case "update"{$self->{scroll}{channel_direction} = 0;
              # $self->{scroll}{channel_start}   -= $self->{scroll}{channel_delta};
              } # update     

} # end SWITCH

 $self->dpw_scroll_channel_pos(@_);


} # end of dpw_scroll_channels
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

 $$w->bind('<Prior>' => [sub{ $self->dpw_scroll_channels("start")}]);
 $$w->bind('<Down>'  => [sub{ $self->dpw_scroll_channels("down") }]); 
 $$w->bind('<Up>'    => [sub{ $self->dpw_scroll_channels("up")   }]);  
 $$w->bind('<Next>'  => [sub{ $self->dpw_scroll_channels("end")  }]); 

# $$w->bind('<Button-2>'  => [sub{ print"TEST cursor\n";
#                                  $self->{plot}->set_cursor();
#                               
#
#                                }]); 



} # end of wde_init_plot_bindings


#---------------------------------------------------------# 
#---- dpw_set_cursor            -------------------------#
#---------------------------------------------------------# 
sub dpw_set_cursor{
my $self = shift;

   $self->{plot}->set_cursor();

} # end of dpw_set_cursor




#---------------------------------------------------------# 
#---- dpw_update_plot_selection   -------------------------#
#---------------------------------------------------------# 
sub dpw_update_plot_selection{
my($self,$panel,$idx) = @_;

   $self->{plot}->update_plot_selection($panel,$idx);

} # end of dpw_update_plot_selection


__DATA__



#---------------------------------------------------------# 
#---- de_get_cursor_info         -------------------------#
#---------------------------------------------------------# 
sub de_get_cursor_info{
my $self = shift; #,$panel,$idx,$label) = @_;

return unless $self->{plot};

   #  $self->{plot}->get_cursor_info(@_);

my ( $x,$y) = $self->{plot}->set_cursor(@_);
     $self->{cursor_xpos} = sprintf "%.4f", $x;
     $self->{cursor_ypos} = sprintf "%.4f", $y; 



} # end of de_get_cursor_info


