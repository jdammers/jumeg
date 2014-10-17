#!/usr/bin/env perl
#=======================================================================
# jumeg_export_4D_to_fif
#
# Autor: F B 12.09.2014
#
# last change 25.09.2014
#
#=======================================================================
# perl script to export 4D data into fif format
# uses jumeg file and directory convention
# call to mne bti2fiff function
#
#=======================================================================

# BEGIN{ unshift( @INC,$ENV{MEG_PERL_DEVL_LIB} ) if ( defined( $ENV{MEG_PERL_DEVL_LIB} )) };

use strict;
use warnings;

use Getopt::Long;
 
use File::List;
use File::Path;
use File::Basename;
use File::Copy;
use File::Slurp;
use Carp;

my $VERSION ="0.0013";

my @IDs = ();

my $path_bti = "/data/meg_store2/megdaw_data21";

   $path_bti = $ENV{MEG_BTI_DATA_DISK} if ( defined( $ENV{MEG_BTI_DATA_DISK} ) );

my $path_fif = $ENV{PWD};

my ($id,$id_str,$scan,$f,$fsuffix,$p,$fbti,$pbti,$path,$pout,$verbose,$debug,$do_run); 

my $bti_suffix = ",rfDC";
my $fif_suffix = "-raw.fif";

my ($help,$keep_existing_mne_files);


#---------------------------------------------------------# 
#--- help  ----------------------------------------------#
#---------------------------------------------------------# 
sub help{
my $u = "undef";
 
   print"\n\n----> HELP  jumeg_export_4D_to_fif  ----\n";
   print"      perl script to export 4D data into fif format\n";
   print"      uses JuMEG file and directory convention\n";
   print"      call to <mne bti2fiff function> for all 4D files matching the search options\n";
   print"-"x60 ."\n"; 
   print"----> Version           : $VERSION\n";
   print"-"x60 ."\n"; 
   print "      Options          : Example              : Value\n";   
   print"-"x60 ."\n"; 
   print "----> -path_4d|p4d     : path to 4D data disk : ".($path_bti or $u )."\n"; 
   print "----> -path_fif|pfif   : path to fif/mne path : ".($path_fif or $u )."\n";
   print "----> -scan            : TEST01               : ".($scan     or $u )."\n";
   print "----> -id              : 0815,007             : ".($id       or $u )."\n";
   print "----> -4d_suffix       : ,rfDC                : ".($bti_suffix or $u) ."\n"; 
   print "----> -fif_suffix      : ,-raw.fif            : ".($fif_suffix or $u) ."\n";
   print"-"x60 ."\n"; 
   print"----> Flags\n";
   print "----> -run|-r          : if defined     : ".($do_run  or $u )."\n";
   print "---->   -r => will run the script\n";  
   print "----> -keep|-k         : if defined     : ".($keep_existing_mne_files  or $u )."\n";
   print "---->   -k => will keep existing files, no new export\n";  
   print "----> -verbose|-v      : if defined     : ".($verbose or $u )."\n";
   print "---->   -v => get more info\n";
   print"-"x60 ."\n"; 
   print "----> -help|-h         : if defined     : ".($help    or $u )."\n";
   print "---->   -h will show this \n";
   print"-"x60 ."\n";
   print"Examples:
          export all 4D data for scan TEST01
           jumeg_import_4D_to_fif -scan TEST01 -v -run
  
          export only 4D data for id 0815 and 007 and scan TEST01 form 4D data disk to mne path 
           jumeg_import_4D_to_fif -scan TEST01T -id 0815,007 -p4d <path_in> -pmne <path_out> -v -run

         \n";
   exit;
 
} # end of help
1;


#---
   GetOptions( 
              "path_4d|p4d=s"     => \$path_bti,
              "path_fif|pfif=s"   => \$path_fif,
              "id=s"              => \$id_str, 
              "scan|s=s"          => \$scan,
              "4d_suffix=s"       => \$bti_suffix,
              "bti_suffix=s"      => \$bti_suffix,
              "fif_suffix=s"      => \$fif_suffix, 
             #--- flags
              "run"               => \$do_run,
	      "verbose|v"         => \$verbose,
              "debug|d"           => \$debug,
              "keep|k"            => \$keep_existing_mne_files,
              "help|h"            => \$help,
            );

#--- some error checks
   unless(-d $path_bti)
    {
      $help = 1;
      carp"\n\n!!!ERROR!!! no directory found for 4D/Bti data: $path_bti\n---> use -p4d <start path to 4d data>\n\n" 
     }      
   
   unless( defined($scan) )
    {
     $help=1;
     carp "\n\n!!!ERROR!!! no option <scan> defined\n ---> use  <jumeg_import_4D_to_fiff -scan ... > or <jumeg_import_4D_to_fiff -h>\n\n" 
    }

   &help if( $help );

#---

   if ( $id_str )
     {
       @IDs=();
       push( @IDs, split(",",$id_str ) );
     }

   unless(@IDs)
    { 
    print"start search on 4D data disk: $path_bti\n";

     opendir(my $dh, $path_bti) || croak "can not open dir $path_bti: $!";
  my @dirs = readdir $dh;
     foreach my $d ( @dirs )
      {
       print"---> check for $scan =>DIR: $d ";
        
       if (-d $path_bti."/".$d."/".$scan)
        {
         print"\t ---> OK\n";
         push( @IDs,$d);
        }
       else { print"\n" }

      } # foreach
     
     closedir $dh; 
    }
#---
   print"\n\nScan        : $scan\n";
   print"IDs         : ".scalar(@IDs)       ." ==> @IDs\n";


my $idx     = 0;
my $pattern = $scan.".*.".$bti_suffix;  

my $i  = 0;
my $ii = 0;
my $iii= 0;

   foreach $id ( @IDs )
    {  
    $i++;
    $ii = 0;

my  $path_id_scan = $path_bti."/".$id."/".$scan;
    unless(-d $path_id_scan ) { warn "\n ERROR $id $scan=> no such directory: $path_id_scan\n"; exit; }

my  $search  = new File::List("$path_id_scan");
    print"search pattern: $pattern\n" if ($verbose);

my  @sel     = ();
    push( @sel, @{ $search->find("$pattern\$") } ); 
   
    foreach $f ( @sel )
     {  
      ( $fbti,$p, $fsuffix ) = fileparse( $f, $bti_suffix);
        $fbti .= $fsuffix;
     my $size =  ( -s $f ) / (1024 * 1024);

      # next if ( -s $f < 600000000);
    
        $p =~s/$path_bti//;
        $p =~s/@/_/g;
        $p =~s/-//g;
        $p =~s/://g;

        #$p = join("/",$id,$scan,$p);
        $p =~s/\/+/\//g;

     my $pfif  = $path_fif."/".$p;
        $pfif  =~s/\/+/\//g;

     my $fif  = $p;
        $fif  =~s/^\/+//;
        $fif  =~s/\/+/_/g;
        $fif .= $fbti.$fif_suffix;
        #$fname =~s/_//;     
     
        next if ( (-e $pfif."/".$fif) and $keep_existing_mne_files);
        
        $ii++;
        $iii++;
     my $str  = "ID count   : ".sprintf('%6d',$i)."/".scalar(@IDs) ." ===> ";
        $str .= "Fcount ==> ".sprintf('%6d',$ii)." ==> ".sprintf('%6d',$iii);

        print "===> 4D/Bti $str : $f\n"; 
        print "---> FIF / MNE file        : $fif  size [Mb]:  ".sprintf('%.3f', $size)."\n";
        print "---> FIF / MNE path        : $pfif\n";
       
        mkpath($pfif); 
        
     my $cmd ="mne bti2fiff -p $f -o $pfif/$fif";
        print "---> CMD: $cmd\n" if ($verbose);
        system $cmd if($do_run);
        print "---> CMD: DONE export 4D/Bti data to FIF/MNE: $fif\n\n";

     } # foreach f

} # foreach id


