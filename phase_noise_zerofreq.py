import numpy as np
import matplotlib.pyplot as plt
from phase_noise_mod import *

TIME_LEN = 1. # in seconds
# discretization
SAMP_FREQ = 131072 # in herz
NUM_SAMP = int(TIME_LEN * SAMP_FREQ)
# signal parameters
PN_SC_MIN = 1. / 20000         # high frequency part in secons
PN_SC_MAX = 1. / 5             # low frequency part in seconds # both PN_SC_* and PHASE_DEV will affect phase noise
PHASE_DEV = 2 * np.pi * 0.06   # in rad * herz
# spectrum mean
NUM_MEAN = 200

# begin script
xvals = np.linspace(0, TIME_LEN, NUM_SAMP)

# compute mean
acc_sig1 = np.zeros([NUM_SAMP,])
acc_sig2 = np.zeros([NUM_SAMP,])
for midx in range(NUM_MEAN):
    signal = make_phase_noise(TIME_LEN, SAMP_FREQ, 0., PN_SC_MIN, PN_SC_MAX, PHASE_DEV)
    acc_sig1 += np.abs(np.fft.fft(signal)) ** 2

# normalize
acc_sig1 /= np.max(acc_sig1)
# decibels
spectre1 = 10 * np.log(acc_sig1) / np.log(10)
# 1/f**3 noise which will be equal to signal at 1 Hz according to AD MT-008 paper at page 3
spectre3 = -3 * 10 * np.log(xvals[1:] * SAMP_FREQ) / np.log(10)
spectre3 = np.concatenate([np.zeros([1,]), spectre3])
# 1/f**2 noise
spectre4 = -2 * 10 * np.log(xvals[1:] * SAMP_FREQ) / np.log(10) - 10
spectre4 = np.concatenate([np.zeros([1,]), spectre4])
# 1/f noise
spectre5 = -1 * 10 * np.log(xvals[1:] * SAMP_FREQ) / np.log(10) - 30
spectre5 = np.concatenate([np.zeros([1,]), spectre5])

D_L = 0 # int(TIME_LEN * FREQ / (2 * np.pi) * 0.99)
D_L4 = int(SAMP_FREQ*0.00008)
D_L5 = int(SAMP_FREQ*0.0008)
D_H = NUM_SAMP // 128 # int(TIME_LEN * FREQ / (2 * np.pi) * 1.01)
plt.plot(xvals[D_L:D_H] * SAMP_FREQ, spectre1[D_L:D_H], 'r-')
plt.plot(xvals[D_L:D_H] * SAMP_FREQ, spectre3[D_L:D_H], 'g-')
plt.plot(xvals[D_L4:D_H] * SAMP_FREQ, spectre4[D_L4:D_H], 'b-')
plt.plot(xvals[D_L5:D_H] * SAMP_FREQ, spectre5[D_L5:D_H], 'm-')

plt.show()
