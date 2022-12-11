#!/usr/bin/env python
'''Plotting vrious methods used to generate surrogates.'''

import numpy as np
import mne
from jumeg.jumeg_surrogates import Surrogates, check_power_spectrum
import matplotlib.pyplot as pl

mysurr = Surrogates.SimpleTestData()

# do shuffling
shuffled = mysurr.shuffle_time_points(mysurr.original_data)

pl.figure('shuffled')
pl.title('shuffled')
pl.plot(mysurr.original_data[0])
pl.plot(shuffled[0], color='r')

# do shifting
shifted = mysurr.shift_data(mysurr.original_data)

pl.figure('shifted')
pl.title('Shifted')
pl.plot(mysurr.original_data[0])
pl.plot(shifted[0], color='r')

# do phase randomization
phase_random = mysurr.randomize_phase(mysurr.original_data)

pl.figure('phase_randomize')
pl.title('phase_randomize')
pl.plot(mysurr.original_data[0])
pl.plot(phase_random[0], color='r')

# do phase randomize as in scot
phase_random_scot = mysurr.randomize_phase_scot(mysurr.original_data)

pl.figure('phase_randomize_scot')
pl.title('phase_randomize_scot')
pl.plot(mysurr.original_data[0])
pl.plot(phase_random_scot[0], color='r')

check_power_spectrum(mysurr.original_data, phase_random)
check_power_spectrum(mysurr.original_data, phase_random_scot)

pl.show()
