from pylab import *
from scot.datatools import randomize_phase
from sklearn.utils import check_random_state
from jumeg.jumeg_plot import plot_phases_polar
import scipy as sci

np.random.seed(1234)
s = np.sin(np.linspace(0,10*np.pi,1000)).T
x = np.vstack([s, np.sign(s)]).T

data = x[:, 0]
data = data.reshape(-1, 1)
rng = check_random_state(None)
# data = np.asarray(x)
data_freq = np.fft.rfft(data)

phases = np.angle(data_freq)
# amplitudes = np.abs(data_freq)
rng.shuffle(phases)
data_freq = np.abs(data_freq) * np.exp(1j*phases)

# data_freq = np.abs(data_freq) * np.exp(1j*rng.random_sample(data_freq.shape)*2*np.pi)
surr = np.fft.irfft(data_freq, data.shape[-1])
pl.plot(data[:, 0]); pl.plot(surr[:, 0])

# y = randomize_phase(x)
subplot(2,1,1)
title('Phase randomization of sine wave and rectangular function')
plot(x[:, 0]), axis([0,1000,-3,3])
subplot(2,1,2)
plot(surr), axis([0,1000,-3,3])
plt.show()

from scipy.signal import welch
psd_x = welch(x, fs=100., window='hann', nperseg=256)
psd_y = welch(y, fs=100., window='hann', nperseg=256)

plt.figure()
plt.plot(np.abs(np.fft.fft(data)) ** 2)
plt.plot(np.abs(np.fft.fft(surr)) ** 2)

a = np.abs(np.fft.rfft(data)) ** 2
b = np.abs(np.fft.rfft(surr)) ** 2
# assert np.allclose(np.abs(np.fft.fft(data)) ** 2, np.abs(np.fft.fft(surr)) ** 2)

# using numpy
data = x[:, 0]
data = data.reshape(-1, 1)
rng = check_random_state(None)
data_freq = np.fft.fft(data)

# phases = np.angle(data_freq)
# amplitudes = np.abs(data_freq)
# rng.shuffle(phases)
# data_freq = np.abs(data_freq) * np.exp(1j*phases)

data_freq = np.abs(data_freq) * np.exp(1j*rng.random_sample(data_freq.shape)*2*np.pi)
# surr = np.fft.ifft(data_freq, data.shape[-1])
surr = np.fft.ifft(data_freq, data.shape[-1])

# using scipy

data = x[:, 0]
# data = data.reshape(-1, 1)
rng = check_random_state(None)
data_freq = sci.fftpack.rfft(data)

# phases = np.angle(data_freq)
# amplitudes = np.abs(data_freq)
# rng.shuffle(phases)
# data_freq = np.abs(data_freq) * np.exp(1j*phases)

data_freq = np.abs(data_freq) * np.exp(1j*rng.random_sample(data_freq.shape)*2*np.pi)
# surr = sci.fftpack.irfft(data_freq, data.shape[-1])
surr = sci.fftpack.irfft(data_freq)
