#!/usr/bin/env python
'''
Example code to use the jumeg suggest bads functionality.
'''

from jumeg import suggest_bads

# provide the path of the filename:
raw_fname = '/Users/psripad/fzj_sciebo/noisy_channel_detection_2016/jul017/jul017_BadChTst-2_16-12-06@11:50_1_c,rfDC-raw.fif'

mybads, raw = suggest_bads(raw_fname, show_raw=False, summary_plot=False)
