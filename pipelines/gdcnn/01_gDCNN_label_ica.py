"""
gDCNN label_ica
Authors:
 - f.boers@fz-juelich.de
 - j.dammers@fz-juelich.de
 - n.kampel@fz-juelich.de

Extended version of:
[1] Ahmad Hasasneh, Nikolas Kampel, Praveen Sripad, N. Jon Shah, and Juergen Dammers
    "Deep Learning Approach for Automatic Classification of Ocular and Cardiac
    Artifacts in MEG Data"
    Journal of Engineering, vol. 2018, Article ID 1350692,10 pages, 2018.
    https://doi.org/10.1155/2018/1350692
"""

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#
# import libraries
#
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import os.path as op
from dcnn_base import DCNN_CONFIG
from dcnn_main import DCNN
from dcnn_utils import (find_files, logger)


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# select your config file
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# fnconfig = 'config_example.yaml'
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
path_in = op.join(cfg.config['path']['basedir'],cfg.config['path']['data_meg'])
fnames = find_files(path_in, pattern='*-raw.fif')

# ++++++++++++++++++++++++++++++++++++++++++++++
#
# loop across files
#
#++++++++++++++++++++++++++++++++++++++++++++++
for fnraw in fnames:

    # init DCNN obejct with config details
    dcnn = DCNN(**cfg.config)
    dcnn.verbose = True

    # read raw data and apply noise reduction and downsample data
    print ('>>> working on %s' % op.basename(fnraw[:-4]))
    dcnn.meg.update(fname=fnraw)

    # chop filtered data, apply ICA on chops, auto-label ICs and save results to disk
    fgdcnn = dcnn.label_ica(save=True)

print ('>>> finished ICA auto-labeling')
