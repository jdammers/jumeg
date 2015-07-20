#!/usr/bin/env perl
#=======================================================================
# jumeg_freesurfer.pl
#
# Autor: F B 20.02.2015
#
# last change 02.03.2015
#
#=======================================================================
# perl script run freesurfer cmd for MNE
#=======================================================================

# BEGIN{ unshift( @INC,$ENV{MEG_PERL_DEVL_LIB} ) if ( defined( $ENV{MEG_PERL_DEVL_LIB} )) };

use strict;
use warnings;

use Getopt::Long;
 
#use File::List;
#use File::Basename;
#use File::Copy;
use File::Slurp;

use Carp;
use File::Path;
use Cwd qw(cwd chdir);

#--- install JSON from cpan 
#---> perl -MCPAN -e shell
#---> install JSON

use JSON qw();


my $VERSION ="0.0013";

my @IDs = ();

my $mri_iso_extention  = "_1x1x1mm.nii";
my $mri_orig_extention = "_1x1x1mm_orig.nii";

my $template_extention = "jumeg_experiment_template.json";

my ($experiment,$template,$template_path,$ftemplate);
my ($do_run,$create_pbs,$verbose,$debug,$help);

my ($stage,$id,$id_str,$fin,$fout,$fsuffix,$p,$path,$path_id,$pout,$cmd); 

my $use_iso = 0; # => use mri-orig nifti
my $clean   = 0;



#---------------------------------------------------------# 
#--- help  ----------------------------------------------#
#---------------------------------------------------------# 
sub help{
my $u = "undef";
 
   print"\n\n----> HELP jumeg_appply_freesurfer.pl  ----\n";
   print"      perl script to apply fresurfer cmds for MNE data analysis\n";
   print"      uses JuMEG file and directory convention\n";
   print"      set up enviroment variable JUMEG_PATH_TEMPLATE\n";
   print"      call to freesurfer cmds & programs\n";
   print"-"x60 ."\n"; 
   print"----> Version           : $VERSION\n";
   print"-"x60 ."\n"; 
   print "      Options          : Example              : Value\n";   
   print"-"x60 ."\n"; 
   print "----> -id              : list of one or more ids => 007,008,1224 : ".($id_str or $u )."\n"; 
   print "                         if not set all ids in template file will be processed\n"; 
   print "----> -experiment|exp  : experiment name                         : ".($experiment or $u )."\n";
   print"-"x60 ."\n"; 
   print"----> Optinal will use template parameter instead of <experiment>\n";
   print "----> -template|tmp     : <my template name> in json format      : ".($template or $u) ."\n"; 
   print "----> -template_path|tp : <my template path>                     : ".($template_path or $u) ."\n";
   print"-"x60 ."\n"; 
   print"----> Flags\n";
   print "----> -run|-r           : if defined     : ".($do_run  or $u )."\n";
   print "---->   -r => run the script\n";  
   print "----> -iso|i            : if defined     : ".($use_iso or $u )."\n";
   print "---->   -i => use iso mri niftis instead of original\n";
   print "----> -clean|-c         : if defined     : ".($clean or $u )."\n";
   print "---->   -c => clean id-fresurfer directory\n";  
   print "----> -verbose|-v       : if defined     : ".($verbose or $u )."\n";
   print "---->   -v => more info\n";
   print "----> -pbs              : if defined     : ".($create_pbs or $u )."\n";
   print "---->      => creates a pbs script for processing on a cluster\n";
   print"-"x60 ."\n"; 
   print "----> -help|-h          : if defined     : ".($help    or $u )."\n";
   print "---->   -h => show this \n";
   print"-"x60 ."\n";
   print"Examples:
          apply freesurfer for experiment template <MEG94T> for subjects 0815 and 007 and clean before -> remove id-fresurfer-dir
          
            jumeg_appply_freesurfer.pl -exp MEG94T -id 0815,007 -c -run -v
  
          apply freesurfer for experiment template <MEG94T> for all subjects and clean before -> remove id-fresurfer-dir

            jumeg_appply_freesurfer.pl -exp MEG94T -c -run -v
  
         \n";

     exit;
 
} # end of help
1;



#---
   GetOptions( 
              # "stage=s"           => \$stage,
              "id=s"              => \$id_str, 
              "experiment|exp=s"  => \$experiment,
              "template|tmp=s"    => \$template,
              "template_path|tp=s"=> \$template_path,
              "run|r"             => \$do_run,
              "iso|i"             => \$use_iso,
              "clean|c"           => \$clean,
              "pbs"               => \$create_pbs,
              "verbose|v"         => \$verbose,
              "debug|d"           => \$debug,
              "help|h"            => \$help,
            );


   &help if( $help );

#--- init template file name
   $ftemplate = "";
   
   if ($template)
    { 
      $ftemplate  = $template_path."/" if ($template_path);
      $ftemplate .= $template;
    }
    elsif ($experiment)
     {
       if ($template_path){ $ftemplate = $template_path }
        else{ $ftemplate = $ENV{JUMEG_PATH_TEMPLATE_EXPERIMENTS} }
       $ftemplate.= "/".$experiment;
     }

    $ftemplate.= "_".$template_extention if ( $ftemplate !~/$template_extention$/ );

    croak "===> !!! ERROR No such experiment template file: ". $ftemplate unless(-e $ftemplate);


#--- read experiment template (json)
#-> http://stackoverflow.com/questions/15653419/parsing-json-file-using-perl

my $json_text = do {
                    open(my $json_fh, "<:encoding(UTF-8)", $ftemplate)
                    or die("Can't open \$filename\": $!\n");
                    local $/;
                    <$json_fh>
                   };

my $JS   = JSON->new;
my $TMP = $JS->decode($json_text);

#--- read IDs
   push(@IDs, @{ $TMP->{experiment}{ids} } );

#--- init mri data path
   $stage               = $TMP->{experiment}{path}{stage};
my $path_mri_iso        = $stage."/".$TMP->{experiment}{mri}{path}{mri}; # mrdata/iso/mri
my $path_mri_orig       = $stage."/".$TMP->{experiment}{mri}{path}{mri_orig};
my $path_mri_freesurfer = $stage."/".$TMP->{experiment}{mri}{path}{freesurfer};

#---
   if ( $id_str )
     {
       @IDs=();
       push( @IDs, split(",",$id_str ) );
     }

#my $path_mri_iso        = $stage."/mrdata/iso/mri";
#my $path_mri_orig       = $stage."/mrdata/mri_orig";
#my $path_mri_freesurfer = $stage."/mrdata/freesurfer";


#---
   print"\n===> Freesurfer IDs  : ".scalar(@IDs)       ." ==> @IDs\n";
   print" stage           : $stage\n";
   print" path mri iso    : $path_mri_iso\n"; 
   print" path mri orig   : $path_mri_orig\n";
   print" path freesurfer : $path_mri_freesurfer\n";
   print" template file   : $ftemplate\n";
 
   if ($create_pbs)
    {
  my $pbs = 0;
  my $opt = " -tmp $ftemplate";
     $opt.= " -r";
     $opt.= " -i"  if ($use_iso);
     $opt.= " -cl" if ($clean);
     $opt.= " -v"  if ($verbose);
     
  my $MY_WORKDIR = $path_mri_freesurfer."/tmp_mrcluster"; 
     mkpath($MY_WORKDIR);

  my $fpbs_id = $MY_WORKDIR."/".$experiment."_id_list.tmp";
     write_file($fpbs_id,join("\n",@IDs) );
  
  my $fpbs_out = $MY_WORKDIR."/".$experiment."_apply_freesurfer_mrcluster.pbs";

  my @pbs =();
     push(@pbs,'#!/bin/tcsh','#PBS -W umask=002','#PBS -N JuMEGJob');
     push(@pbs,'#PBS -l pmem=2gb');
     push(@pbs,'#PBS -d '.$MY_WORKDIR);
     push(@pbs,'#PBS -e '.$MY_WORKDIR);
     push(@pbs,'#PBS -o '.$MY_WORKDIR);
     push(@pbs,'#PBS -j oe');
     push(@pbs,'#PBS -q cpu');
     push(@pbs,'#PBS -l nodes=1:ppn=1');
     push(@pbs,'#PBS -r n');
     push(@pbs,"\n\n");
     push(@pbs,'set SUBJECT_ID=`awk "NR==${PBS_ARRAYID}" '.$fpbs_id);
     push(@pbs,$0.$opt." -id ". '${SUBJECT_ID}');

     write_file($fpbs_out,join("\n",@pbs) );
     system "chmod 770 $fpbs_out";

    print"\n---> INFO for mrcluster & where to find the tmp files:";
    print "--> Id list file: $fpbs_id\n";
    print" --> PBS script  : $fpbs_out\n\n";
    print" --> call for mrcluster jobs:\n";
 # cmd ---> send it to the cluster
  my $cmd="ssh -X mrcluster qsub -t 0-$#IDs $fpbs_out";
     print "$cmd\n\n";

     exit;
} # if create_pbs

my $i  = 0;
my $ii = 0;

   foreach $id ( @IDs )
    {  
     $i++;
     $ii = 0;
     
     $path_id = $stage."/mne/".$id;
 # my $path_id_fs = $path_id."/freesurfer/".$id;
 
  my $path_fs    = $path_mri_freesurfer;
  my $path_fs_id = $path_mri_freesurfer."/".$id;

     mkpath($path_fs);

    #--- mk subjects dir
     chdir($path_fs);

     if( (-d $id) and $clean and $do_run )
      { 
       print"ERROR id-dir already exists => ". cwd() ." removing dir: $id\n";
       system "rm -rf $id";
      }
     
     $cmd="mksubjdirs $id";
     print" --> $cmd\n" if $verbose;
     system $cmd if $do_run;

    #---  
     if ($use_iso) { $fin = $path_mri_iso."/".$id.$mri_iso_extention }
      else{ $fin = $path_mri_orig."/".$id.$mri_orig_extention }
 

     $fin.=".gz" unless (-e $fin);
     croak "ERROR NO such mri input file: ". $fin unless(-e $fin);

     $fout= $path_fs_id."/mri/orig/001.mgz";
     $cmd ="mri_convert $fin $fout";
     print" --> $cmd\n" if $verbose;
     system $cmd if $do_run;
 
   
  my @fs_cmd=("recon-all -autorecon-all -sd $path_fs -subjid",
              "mne_setup_mri --overwrite --subject",
              "mne_setup_source_space --overwrite --subject", 
              "mne_watershed_bem --overwrite --subject",
             );

    #---
     foreach my $c (@fs_cmd)
      {
    
       $cmd = $c." ".$id;
       print"\n===> Freesurfer ID  : $id\n";
       print" --> $cmd\n\n";
       system $cmd if $do_run;
       print" -->DONE \n\n";

      } # foreach cmd

  my $pw = $path_fs_id."/bem/watershed";
  my $pb = $path_fs_id."/bem";
    
     symlink($pw."/".$id."_brain_surface",      $pb."/".$id."-brain.surf");
     symlink($pw."/".$id."_inner_skull_surface",$pb."/".$id."_inner_skull.surf");
     symlink($pw."/".$id."_outer_skin_surface", $pb."/".$id."_outer_skin.surf");
     symlink($pw."/".$id."_outer_skull_surface",$pb."/".$id."_outer_skull.surf");
   
   #---  
     $cmd = "mne_make_scalp_surfaces -s $id";
     print" --> $cmd\n\n" if $verbose;
     system $cmd if $do_run;
 
   #---  
     $cmd ="mne_setup_forward_model --subject $id --ico 4 --surf";
     print" --> $cmd\n\n" if $verbose;
     system $cmd if $do_run;

   } # foreach id





__DATA__



FYI: 

---> run on mrcluster (PBS & torque)

--->generate id list: -> write_file("id_list.txt",join("\n",@IDs) );

--->cat id_list.txt
001
002
003
...
00n

---> generate pbs-script for cluster
---> !!! ck path and permission for cluster and beos !!!

#!/bin/tcsh
#PBS -S /bin/tcsh
#PBS -W umask=002
#PBS -N JuMEGJob
#PBS -l pmem=2gb
#PBS -d MY_WORKDIR
#PBS -e MY_WORKDIR
#PBS -o MY_WORKDIR
#PBS -j oe
#PBS -q cpu
#PBS -l nodes=1:ppn=1
#PBS -r n

set SUBJECT_ID=`awk "NR==${PBS_ARRAYID}" ./id_list.txt`

jumeg_apply_freesurfer.pl -r -cl -id ${SUBJECT_ID};


---> send it to the cluster
qsub -t 1-15 /data/meg_store2/exp/MEG94T/source/mne/fs_id_array_run_on_cluster.csh 



---> ck if jobs are running
qstat -t -u MY_USER_NAME

#-- kill all jobs in array
qdel MY_PID[].qserv




# FB:  this schould be run in another script


    #  Align the coordinate frames - results in trans file.
      echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
      echo ">>>> before going on the coordinate frames have to be align, i.e.           <<<<"   
      echo ">>>> coregistration must be done, resulting in a trans file.                <<<<"
      echo ">>>> --> to apply coregistration in IPython following commands have to be   <<<<"
      echo ">>>>     used:                                                              <<<<"
      echo ">>>>         import mne                                                     <<<<"
      echo ">>>>         mne.gui.coregistration()                                       <<<<"
      echo ">>>>     (after coregistration the 'Average Point Error' should be < 3.0mm) <<<<"
      echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
  
    elif [ $1 = "2" ]
      then
        # loop over all experiments
        for exp in basename_cleaned_files; do
          echo "Running for experiment file $SUBJECTS_DIR/$SUBJECT/${basename_cleaned_files}.fif"
        
     #$SUBJECTS_DIR/sensor_connectivity_script.py $SUBJECTS_DIR/$SUBJECT/cleaned_${SUBJECT}_${exp}.fif $SUBJECTS_DIR/$SUBJECT/${SUBJECT}_${exp}-eve.fif ${SUBJECT} ${exp}
     
          # Calculation of forward solution (MEG only)
          echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
          echo ">>>>                       calculate forward solution                       <<<<"
          echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
          mne_do_forward_solution --mindist 5 --subject $SUBJECT --src $SUBJECTS_DIR/$SUBJECT/bem/${SUBJECT}-7-src.fif --bem $SUBJECTS_DIR/$SUBJECT/bem/${SUBJECT}-5120-5120-5120-bem-sol.fif --meas $SUBJECTS_DIR/$SUBJECT/${basename_cleaned_files}.fif --fwd $SUBJECTS_DIR/$SUBJECT/${basename_cleaned_files}-7-src-fwd.fif --megonly --overwrite
#   
#     # Make a sensitivity map (did not work)
#     mne_sensitivity_map --fwd $SUBJECTS_DIR/$SUBJECT/${SUBJECT}_audi_cued-fwd.fif --map 1 --w ${SUBJECT}_audi_cued-fwd-sensmap

          # Calculation of the inverse operator
          echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
          echo ">>>>                       calculate inverse operator                       <<<<"
          echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
          mne_do_inverse_operator --fwd $SUBJECTS_DIR/$SUBJECT/${basename_cleaned_files}-7-src-fwd.fif --depth --meg --noisecov $SUBJECTS_DIR/$SUBJECT/${SUBJECT}${pattern_cov_matrix}.fif

        done # for experiment

        # Finally, analyzing data
        echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
        echo ">>>> All preprocessing steps are done! Finally the data have to be analyzed <<<<"
        echo ">>>> by using for example the mne_analyze or mne_make_movie script...       <<<<"
        echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
  
    else
      echo "ERROR: You have to select which steps should be performed (1 or 2)! Now the script hasn't done anything...."
  fi
  
done # for subject
