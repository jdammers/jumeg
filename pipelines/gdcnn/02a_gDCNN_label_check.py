"""
gDCNN label_check

    check ECG and EOG labelled ICA components and artifact rejection performance

"""

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#
# import libraries
#
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import os.path as op
from mne.report import Report
from distutils.dir_util import mkpath
from dcnn_base import DCNN_CONFIG
from dcnn_main import DCNN
from dcnn_utils import find_files
from dcnn_logger import setup_script_logging
logger = setup_script_logging()


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# select your config file
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# fnconfig = 'config_4D_CAU.yaml'
# fnconfig = 'config_4D_INTEXT.yaml'
fnconfig = 'config_4D_Freeviewing.yaml'
# fnconfig = 'config_MEGIN.yaml'
# fnconfig = 'config_CTF_Paris.yaml'
# fnconfig = 'config_CTF_Philly.yaml'


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# load config details and file list to process
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# read config
cfg = DCNN_CONFIG(verbose=True)
cfg.load(fname=fnconfig)
dcnn = DCNN(**cfg.config)

# get file list to process
path_in = dcnn.path.data_train
path_out = dcnn.path.report + '/'
mkpath(path_out)

# read list of files
fnames = find_files(path_in, pattern='*.npz')
msg=["ICA Check & AR performance",
     "  -> path_in : {}".format(path_in),
     "  -> path_out: {}".format(path_out),
     " --> files:\n  -> {}".format("\n  -> ".join(fnames))
     ]
logger.info("\n".join(msg))
assert fnames,"ERROR no files found"

# report
# prefix   = '_'.join(str.split(str.split( dcnn.path.data_train, dcnn.path.basedir)[-1],'/')[2:])
prefix = cfg.config['meg']['location'] + '_' + cfg.config['meg']['exp_name']
fnreport = path_out + prefix + '_label_check'
report   = Report(title='Check IC labels & AR Performance')
logger.info("START MNE report: {}".format(fnreport))

do_save_h5 = False

# ++++++++++++++++++++++++++++++++++++++++++++++
#
# loop across files to process
#
#++++++++++++++++++++++++++++++++++++++++++++++
for fname in fnames:

    # init object with config details
    dcnn.load_gdcnn(fname)
    name = op.basename(fname[:-4])
    # print('>>> working on %s' % name)

    #  check IC labels (and apply corrections)
    figs, captions = dcnn.plot_ica_traces(fname)
    report.add_figs_to_section(figs, captions=captions, section='ICA components', replace=True)

    figs, captions = dcnn.plot_artifact_performance()
    report.add_figs_to_section(figs, captions=captions, section='AR performance', replace=True)

if do_save_h5:
    report.save(fnreport + '.h5', overwrite=True)
report.save(fnreport + '.html', overwrite=True, open_browser=True)
logger.info("\nDONE MNE report: {}".format(fnreport))

