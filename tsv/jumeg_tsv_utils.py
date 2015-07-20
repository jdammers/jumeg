import argparse


def jumeg_tsv_utils_args():
    parser = argparse.ArgumentParser(description='JuMEG TSV Start Parameter')
    
    parser.add_argument("-p","--path", help="start path to data files")
    parser.add_argument("-f","--fname",help="Input raw FIF file")
 
    parser.add_argument("-exp","--experiment", help="experiment name")
    parser.add_argument("-bads","--bads",help="list of channels to mark as bad --bads=A1,A2")
    parser.add_argument("-v","--verbose", action="store_true",help="verbose mode")
    parser.add_argument("-d","--debug", action="store_true",help="debug mode")
   #--- mne opt
   
    parser.add_argument("--duration",   type=float,help="Time window for plotting (sec)",default=10.0)
    parser.add_argument("--start",      type=float,help="Initial start time for ploting",default=0.0)
    parser.add_argument("--n_channels", type=int,  help="Number of channels to plot at a time",default=20)
   #--- plot option
   # parser.add_argument("--subplot",    type=int,  help="Subplot configuration --subplot=3,2 ")
 
    return parser.parse_args()

