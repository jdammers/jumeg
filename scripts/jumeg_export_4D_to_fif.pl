#!/usr/bin/env perl
#=======================================================================
# jumeg_export_4D_to_fifjumeg_export_4D_to_fif.pl
# Autor: F B 12.09.2014
#=======================================================================
# update 08.06.2016
# update 12.12.2016  filesize option to search & count total size to export
# update 12.01.2017  fix bug in filesize option to search & count ...
# update 02.06.2017  --hs_file => use export headshape or fakeHS
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

my ($id,$id_str,$scan,$f,$fsuffix,$p,$fbti,$pbti,$path,$pout,$verbose,$debug,$do_run,$fempty); 

my $bti_suffix       = ",rfDC";
my $fif_suffix       = "-raw.fif";
my $fif_suffix_empty = "-empty.fif";
my $auto_empty       = 0;
my $fakehs;

my $data_size        = 0.0;
my $search_size_str  = "";
my $search_size      = 0.0;  # byte
my $serach_size_unit ="byte";
my $search_size_operation = "gt";

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
   print "----> -size|s -/+ <size> <[k|K] [M|m] [G|g]>  : ".($search_size_str or $u) ."\n";
   print "  -->  examples:
                 search for files > 100kb:   -s  100k or -s +100k
                 search for files < 100kb:   -s -100k 
                 search for files > 100Mb:   -s 100M or -s +100M 
                 search for files > 100Gb:   -s 100G or -s +100G\n";

   print"-"x60 ."\n"; 
   print"----> Flags\n";
   print "----> -run|-r          : if defined     : ".($do_run  or $u )."\n";
   print "---->   -r => run the script\n";  
   print "----> -empty|-e         : if defined     : ".($auto_empty or $u )."\n";
   print "---->   -e => automatic search for empty room measurement\n";
   print "              looks for the last run in last session of the day\n";  
   print "----> -keep|-k         : if defined     : ".($keep_existing_mne_files  or $u )."\n";
   print "---->   -k => keep existing files, no new export\n";
   print "----> -fakeHS|fhs => use fake/dumy headshape file  : ".($fakehs or $u)."\n";
   print "----> -verbose|-v      : if defined     : ".($verbose or $u )."\n";
   print "---->   -v => more info\n";
   print"-"x60 ."\n"; 
   print "----> -help|-h         : if defined     : ".($help    or $u )."\n";
   print "---->   -h => show this \n";
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


#---------------------------------------------------------# 
#--- calc_search_file_size       -------------------------#
#---------------------------------------------------------# 
sub calc_search_file_size{
my $s      = shift;
my $sz     = 0.0;
my $op     = "ge";
my $factor = 1.0;

my $fk = 1024;
my @a  = ();
   @a  = ($s=~/^([+,-]?)(\d*)(\w?)/g);
   
   $op= "le" if (shift(@a) eq '-' );
   $sz = shift(@a);

my $unit = ( lc( shift(@a) ) or "b" );

   if ( $unit eq 'k')     {$sz *= $fk }
    elsif( $unit eq 'm')  {$sz *= $fk ** 2 }
     elsif( $unit eq 'g'){$sz *= $fk ** 3 }

   return ($sz,$op,$unit); 

}# end ofcalc_ search_file_size

#---
   GetOptions( 
              "path_4d|p4d=s"     => \$path_bti,
              "path_fif|pfif=s"   => \$path_fif,
              "id=s"              => \$id_str, 
              "scan|s=s"          => \$scan,
              "4d_suffix=s"       => \$bti_suffix,
              "bti_suffix=s"      => \$bti_suffix,
              "fif_suffix=s"      => \$fif_suffix,
              "size|s=s"          => \$search_size_str,  
             #--- flags
              "run"               => \$do_run,
              "empty|e"           => \$auto_empty,
              "verbose|v"         => \$verbose,
              "debug|d"           => \$debug,
              "keep|k"            => \$keep_existing_mne_files,
              "fakeHS|fHS"        => \$fakehs,
              "help|h"            => \$help,
            );

#--- some error checks
   unless(-d $path_bti)
    {
      $help = 1;
      carp"\n\n!!!ERROR!!! no directory found for 4D/Bti data: $path_bti\n---> use -p4d <start path to 4d data>\n" 
     }      
   
   unless( defined($scan) )
    {
     $help=1;
     carp "\n\n!!!ERROR!!! no option <scan> defined\n ---> use  <jumeg_import_4D_to_fiff -scan ... > or <jumeg_import_4D_to_fiff -h>\n" 
    }

   &help if( $help );

#--- ck size option
   ($search_size,$search_size_operation,$serach_size_unit) = calc_search_file_size($search_size_str);
#---

   if ( $id_str )
     {
       @IDs=();
       push( @IDs, split(",",$id_str ) );
     }

   unless(@IDs)
    { 
     print"start search on 4D data disk: $path_bti\n";

     opendir(my $dh,$path_bti) || croak "can not open dir $path_bti: $!";
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

my  $path_id_scan = $path_bti."/".$id."/".$scan."/";
    unless(-d $path_id_scan ) { carp "\n ERROR $id $scan=> no such directory: $path_id_scan\n";}

my  $search  = new File::List("$path_id_scan");
    print"search pattern: $pattern\n" if ($verbose);

my  @sel     = ();
    push( @sel, @{ $search->find("$pattern\$") } ); 

my @check_for_empty_room_list = ();

    foreach $f ( @sel )
     {  
      ( $fbti,$p, $fsuffix ) = fileparse( $f, $bti_suffix);
        $fbti .= $fsuffix;
     my $size =  ( -s $f );
  
        if($search_size_operation eq "ge"){ next if ( $size <= $search_size); }
         elsif( $search_size_operation eq "le"){ next if ( $size => $search_size); }
        $pbti = $p;
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
       
        push( @check_for_empty_room_list, $pfif ."/". $fif ) if ($auto_empty); 
       
        next if ( (-e $pfif."/".$fif) and $keep_existing_mne_files );
        unlink( $pfif."/".$fif ) if (-e $pfif."/".$fif);
        
        $ii++;
        $iii++;
     my $str  = "ID count   : ".sprintf('%6d',$i)."/".scalar(@IDs) ." ===> ";
        $str .= "Fcount ==> ".sprintf('%6d',$ii)." ==> ".sprintf('%6d',$iii);

        print "===> 4D/Bti $str : $f\n"; 
        print "---> FIF / MNE file        : $fif  size [Mb]:  ".sprintf('%.3f', $size/1024**2)."\n";
        print "---> FIF / MNE path        : $pfif\n";
        $data_size += $size;

        mkpath($pfif."/plots");  # include plots directory 
        
    
     my $cmd  = "mne bti2fiff -overwrite=True -p $f -o $pfif/$fif";
        $cmd .= " -c ".$pbti."config";
        if (-e $pbti.'/hs_file')
          { $cmd.= " --head_shape=$pbti\/hs_file" }
        elsif($fakehs){$cmd.= " --head_shape=.\/hs_file"}
     
        print "---> CMD: $cmd\n" if ($verbose);
        system $cmd if($do_run);
        print "---> CMD: DONE export 4D/Bti data to FIF/MNE: $fif\n\n";
      } # foreach f
  
 
  my @unsorted_time   = ();
 #--- sort for empty room file as number date.time.run
     foreach $f ( @check_for_empty_room_list )
      {
       my @a = split("_",$f);
          push( @unsorted_time, $a[-4].$a[-3].sprintf('%03d',$a[-2]) );
      }
  my @sorted_indexes   = sort { $unsorted_time[$b] <=> $unsorted_time[$a] } 0..$#unsorted_time;
  my @sorted_file_list = @check_for_empty_room_list[ @sorted_indexes ];
  
  my $date_of_empty;
  my @empty_room_list = ();
     foreach $f ( @sorted_file_list )
      {
      #--- check for sessions at differnt days
      my $d = (split("_",$f))[-4];
         unless ( scalar(@empty_room_list) )
          {
           push(@empty_room_list,$f);
           $date_of_empty = $d; 
          }
         if ($d != $date_of_empty)
          {  
            push (@empty_room_list,$f);
            $date_of_empty = $d; 
            next;
           }       
       } # foreach f
     
     foreach $f ( @empty_room_list )
      {  
       print "empty Room FIF: $f\n";
     ( $fempty = $f )=~s/$fif_suffix$/$fif_suffix_empty/;
       print "empty Room : $fempty\n";
       rename($f,$fempty) if ($do_run);
      }  # foreach emty room
     
} # foreach id

print "===> 4D/Bti Total file size [Gb]:  ".sprintf('%.3f', $data_size /1024**3)."\n";
   
