import argparse

def msg_on_attribute_error(info):
    print "\n ===> FYI: NO parameters for <" +info+ "> defined\n\n"


def jumeg_tsv_utils_get_args():
    
    info_global="""
     JuMEG Time Series Viewer [TSV] Start Parameter 
    
     ---> view time series data FIF file   
      jumeg_tsv.py --fname=110058_MEG94T_121001_1331_1_c,rfDC-raw.fif --path=/localdata/frank/data/MEG94T/mne/110058/MEG94T/121001_1331/1 -v

      jumeg_tsv.py --fname=110058_MEG94T_121001_1331_1_c,rfDC_30sec-raw.fif --path=/localdata/frank/data/MEG94T/mne/110058/MEG94T/121001_1331/1 -v
      
    """
    
   # parser = argparse.ArgumentParser(description='JuMEG TSV Start Parameter')
  
    parser = argparse.ArgumentParser(info_global)
  
    parser.add_argument("-p","--path", help="start path to data files")
    parser.add_argument("-f","--fname",help="Input raw FIF file")
 
    parser.add_argument("-exp","--experiment", help="experiment name")
    parser.add_argument("-bads","--bads",help="list of channels to mark as bad --bads=A1,A2")
    parser.add_argument("-v","--verbose", action="store_true",help="verbose mode")
    parser.add_argument("-d","--debug",  action="store_true",help="debug mode")
    parser.add_argument("-dd","--dd",    action="store_true",help="dd-debug mode")
    parser.add_argument("-ddd","--ddd",  action="store_true",help="ddd-debug mode")
   #--- mne opt
   
    parser.add_argument("-td","--duration",   type=float,help="Time window for plotting (sec)",default=10.0)
    parser.add_argument("-t0","--start",      type=float,help="Initial start time for ploting",default=0.0)
    parser.add_argument("-nch","--n_channels",type=int,  help="Number of channels to plot at a time",default=20)
    parser.add_argument("-nco","--n_cols",    type=int,  help="Number of cols to plot at a time",default=1)
   #--- plot option
   # parser.add_argument("--subplot",    type=int,  help="Subplot configuration --subplot=3,2 ")
 
    return parser.parse_args(),parser

