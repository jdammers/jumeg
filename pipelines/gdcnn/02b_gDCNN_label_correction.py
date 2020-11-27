"""
gDCNN label_check

    check ECG and EOG labeled ICA components and apply correction if applicable.

"""

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#
# import libraries
#
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
from dcnn_base import DCNN_CONFIG
from dcnn_main import DCNN
from dcnn_utils import find_files


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# select your config file
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# fnconfig = 'config_4D_CAU.yaml'
# fnconfig = 'config_4D_INTEXT.yaml'
# fnconfig = 'config_MEGIN.yaml'
# fnconfig = 'config_CTF_Paris.yaml'
fnconfig = 'config_CTF_Philly.yaml'


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# load config details and file list to process
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
cfg = DCNN_CONFIG(verbose=True)
cfg.load(fname=fnconfig)

# get file list to process
path_in = cfg.config['path']['data_train']
fnames = find_files(path_in, pattern='*.npz')

# ++++++++++++++++++++++++++++++++++++++++++++++
#
# loop across files to process
#
#++++++++++++++++++++++++++++++++++++++++++++++
for fname in fnames:

    # init object with config details
    dcnn = DCNN(cfg.config)
    dcnn.load_gdcnn(fname)

    # check IC labels (and apply corrections)
    dcnn.check_labels(save=True, path_out=cfg.config['path']['data_train'])

print ('>>> finished ICA label check')