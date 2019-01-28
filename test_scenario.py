
from phase_noise_mod import *
from beacon_sim import *
import math


# -------------------- script parameters
LSp = 300. * 10 ** 6 # meters per second
wavelen_m = LSp / (868. * 10 ** 6)
num_frames = 7
recv_base_m = 0.2
max_delta = recv_base_m / LSp
PN_SC_MIN = 1. / 20000 # high frequency part in secons
PN_SC_MAX = 1. / 5 # low frequency part in seconds # both PN_SC_* and PHASE_DEV will affect phase noise
TX_PHASE_DEV = 2 * np.pi * 0.06      # in rad * herz
RX_PHASE_DEV = 2 * np.pi * 0.06      # in rad * herz
SINE_DEV = 2 * np.pi * 300
SLOPE_FRAC = 0.005 # one sided fraction
SAMP_FREQ = 4000000.
frame_duration = (2 ** 16) / SAMP_FREQ
frame_step = frame_duration / 2
b_time_diff = 0.1 * 10 ** -3
b_time_len  = 2.5 * 10 ** -3

# frequency bands
freq_band_radhz = 2 * np.pi * 70. * 10**3
#num_per_band = 250
#freq_step_radhz = freq_band_radhz / num_per_band
#freqs = np.linspace(freq_step_radhz / 2, freq_band_radhz - freq_step_radhz / 2, num_per_band)
freqs = [2 * np.pi * (-27500.),
         2 * np.pi * (-27500. + 250000.),
         2 * np.pi * (-27500. + 500000.),
         2 * np.pi * (-27500. + 750000.)]

reference_freq0 = 2 * np.pi * 1000000.
reference_freq1 = 2 * np.pi * 1500000.

file2write_ch1 = "ch1_out"
file2write_ch2 = "ch2_out"

# -------------------- script begin

scen = Scenario(frame_duration, frame_step, max_delta, wavelen_m)
idx = 0


bcn = Beacon(idx + 1, 499.,  1., 0., # b_id, x, y, z
             freqs[idx] - freq_band_radhz, freqs[idx] + freq_band_radhz, SINE_DEV, 10., # freq_low_radhz, freq_high_radhz, drift_radhz, power_dbm
             TX_PHASE_DEV, PN_SC_MIN, PN_SC_MAX, SLOPE_FRAC) # pn_dev_radhz, pn_sc_min, pn_sc_max, slope_frac

bcn.append_time_rep(TimeRep(0.    * 10**-3,  b_time_len,  frame_duration * 4))  # start_delay, duration, repeat_period
bcn.append_time_rep(TimeRep(0.001 * 10**-3 + b_time_diff,  b_time_len,  frame_duration * 4))  # start_delay, duration, repeat_period

scen.add_beacon(bcn)
idx += 1

bcn = Beacon(idx + 1, 499.,  1., 0., # b_id, x, y, z
             freqs[idx] - freq_band_radhz, freqs[idx] + freq_band_radhz, SINE_DEV, 10., # freq_low_radhz, freq_high_radhz, drift_radhz, power_dbm
             TX_PHASE_DEV, PN_SC_MIN, PN_SC_MAX, SLOPE_FRAC) # pn_dev_radhz, pn_sc_min, pn_sc_max, slope_frac

bcn.append_time_rep(TimeRep(frame_duration + 0.    * 10**-3,  b_time_len,  frame_duration * 4))  # start_delay, duration, repeat_period
bcn.append_time_rep(TimeRep(frame_duration + 0.001 * 10**-3 + b_time_diff,  b_time_len,  frame_duration * 4))  # start_delay, duration, repeat_period

scen.add_beacon(bcn)
idx += 1

bcn = Beacon(idx + 1, 499.,  1., 0., # b_id, x, y, z
             freqs[idx] - freq_band_radhz, freqs[idx] + freq_band_radhz, SINE_DEV, 10., # freq_low_radhz, freq_high_radhz, drift_radhz, power_dbm
             TX_PHASE_DEV, PN_SC_MIN, PN_SC_MAX, SLOPE_FRAC) # pn_dev_radhz, pn_sc_min, pn_sc_max, slope_frac

bcn.append_time_rep(TimeRep(2 * frame_duration + 0.    * 10**-3,  b_time_len,  frame_duration * 4))  # start_delay, duration, repeat_period
bcn.append_time_rep(TimeRep(2 * frame_duration + 0.001 * 10**-3 + b_time_diff,  b_time_len,  frame_duration * 4))  # start_delay, duration, repeat_period

scen.add_beacon(bcn)
idx += 1

bcn = Beacon(idx + 1, 499.,  1., 0., # b_id, x, y, z
             freqs[idx] - freq_band_radhz, freqs[idx] + freq_band_radhz, SINE_DEV, 10., # freq_low_radhz, freq_high_radhz, drift_radhz, power_dbm
             TX_PHASE_DEV, PN_SC_MIN, PN_SC_MAX, SLOPE_FRAC) # pn_dev_radhz, pn_sc_min, pn_sc_max, slope_frac

bcn.append_time_rep(TimeRep(3 * frame_duration +  0.    * 10**-3,  b_time_len,  frame_duration * 4))  # start_delay, duration, repeat_period
bcn.append_time_rep(TimeRep(3 * frame_duration +  0.001 * 10**-3 + b_time_diff,  b_time_len,  frame_duration * 4))  # start_delay, duration, repeat_period

scen.add_beacon(bcn)
idx += 1

# reference
bcn = Beacon(idx + 1, 500.,  1., 0., # b_id, x, y, z
             reference_freq0, reference_freq1, 0., 3., # freq_low_radhz, freq_high_radhz, drift_radhz, power_dbm
             TX_PHASE_DEV, PN_SC_MIN, PN_SC_MAX, SLOPE_FRAC) # pn_dev_radhz, pn_sc_min, pn_sc_max, slope_frac

bcn.append_time_rep(TimeRep(5.002 * 10**-3,  b_time_len,  frame_duration))  # start_delay, duration, repeat_period
bcn.append_time_rep(TimeRep(5.003 * 10**-3 + b_time_diff,  b_time_len,  frame_duration))  # start_delay, duration, repeat_period

scen.add_beacon(bcn)
idx += 1

scen.add_receiver(Receiver(500., 0., 0., # x, y, z
                           -recv_base_m / 2, 0., 0., -100., # ax, ay, az, sensitivity_dbm
                           RX_PHASE_DEV, PN_SC_MIN, PN_SC_MAX, #rx_pn_dev, rx_pn_sc_min, rx_pn_sc_max
                           SAMP_FREQ)) # samp_freq_hz


with open(file2write_ch1, "w") as outf_ch1:
    with open(file2write_ch2, "w") as outf_ch2:

        for fr_idx in range(num_frames):
            scen.transmit()
            sig = scen.receive(0)

            for idx in range(sig.shape[1]):
                outf_ch1.write("%f %f\n" % (np.real(sig[0, idx]), np.imag(sig[0, idx])))
                outf_ch2.write("%f %f\n" % (np.real(sig[1, idx]), np.imag(sig[1, idx])))

            # has to pass one more half frame
            scen.transmit()
