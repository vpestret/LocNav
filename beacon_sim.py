
from phase_noise_mod import *
import math

LSp = 300. * 10 ** 6 # meters per second

class GeomPoint:

    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z


class TimeRep:

    def __init__(self, start_delay, duration, repeat_period):
        self.start_delay = start_delay
        self.duration = duration
        self.repeat_period = repeat_period

    def register_owner(self, owner, regen_id):
        self.owner = owner
        self.regen_id = regen_id

    def get_signal_params(self, max_delta):
        """Creates and returns SignalParams structure self-sufficient to synthesize signal in every point"""
        assert self.owner is not None
        return self.owner.regen_params(self.regen_id, self.duration, max_delta)


class SignalParams:

    def __init__(self, x, y, z, fpn, dev, freq_radhz, ph_start, power_dbm, duration, fwn):
        GeomPoint.__init__(self, x, y, z)
        self.fpn = fpn
        self.dev = dev
        self.freq_radhz = freq_radhz
        self.ph_start = ph_start
        self.power_dbm = power_dbm
        self.duration = duration
        self.fwn = fwn


class Beacon(GeomPoint):

    def __init__(self, b_id, x, y, z, freq_low_radhz, freq_high_radhz, drift_radhz, power_dbm, pn_dev_radhz, pn_sc_min, pn_sc_max, slope_frac):
        GeomPoint.__init__(self, x, y, z)
        self.b_id = b_id
        self.freqs_radhz = []
        self.freqs_radhz.append(freq_low_radhz)
        self.freqs_radhz.append(freq_high_radhz)
        self.drift_radhz = drift_radhz
        self.power_dbm = power_dbm
        self.pn_dev_radhz = pn_dev_radhz
        self.pn_sc_min = pn_sc_min
        self.pn_sc_max = pn_sc_max
        self.fwn = interp1d(np.array([0., slope_frac, 1.-slope_frac, 1.]), np.array([0., 1., 1., 0.]), kind='linear')
        self.reps = []
        self.reps_regened = []
        self.freq_drift_add = 0. # will be updated each regen

    def append_time_rep(self, rep):
        assert len(self.reps) < 2
        rep.register_owner(self, len(self.reps))
        self.reps.append(rep)
        self.reps_regened.append(True)

    def regen_params(self, regen_id, duration, max_delta):
        # implemented yet only for two reps
        assert len(self.reps) == 1 or len(self.reps) == 2
        # model frequency modulation
        fpn = prepare_interp_function(duration, self.pn_sc_min, self.pn_sc_max, max_delta)
        # model drift
        if len(self.reps) == 1:
            self.freq_drift_add = np.random.normal(0., self.drift_radhz)
        elif self.reps_regened[0] and self.reps_regened[1]:
            self.freq_drift_add = np.random.normal(0., self.drift_radhz)
            self.reps_regened[0] = False
            self.reps_regened[1] = False
        freq_radhz = self.freqs_radhz[regen_id] + self.freq_drift_add
        self.reps_regened[regen_id] = True
        # phase start
        ph_start = np.random.uniform(0, 2 * np.pi)
        # return
        return SignalParams(self.x, self.y, self.z, fpn, self.pn_dev_radhz, freq_radhz, ph_start, self.power_dbm, duration, self.fwn)


class Receiver(GeomPoint):

    def __init__(self, x, y, z, ax, ay, az, sensitivity_dbm, rx_pn_dev, rx_pn_sc_min, rx_pn_sc_max, samp_freq_hz):
        GeomPoint.__init__(self, x, y, z)
        # direction from centre to first antenna which is antenna on negative angle -90 degree
        self.ax = ax
        self.ay = ay
        self.az = az
        self.avec = np.array([ax, ay, az])
        #
        self.sensitivity_dbm = sensitivity_dbm
        self.rx_pn_dev = rx_pn_dev
        self.rx_pn_sc_min = rx_pn_sc_min
        self.rx_pn_sc_max = rx_pn_sc_max
        self.samp_freq_hz = samp_freq_hz


class Scenario:

    def __init__(self, frame_duration, frame_step, max_delta, wavelen_m):
        self.frame_duration = frame_duration
        self.frame_step = frame_step
        self.max_delta = max_delta
        self.wavelen_m = wavelen_m
        self.beacons = []
        self.reps = dict()  # consists of TimeRep - copied from beacons one time when transmit starts WARNING this array has base which is frame_step ahead of self.base_time
        self.frame = dict() # consists of SignalParams
        self.receivers = []
        self.base_time = -1.


    def add_beacon(self, beacon):
        self.beacons.append(beacon)

        # do reps extraction from beacon
        assert len(beacon.reps) == 1 or len(beacon.reps) == 2
        for rep in beacon.reps:
            self.reps[rep.start_delay] = rep


    def add_receiver(self, receiver):
        self.receivers.append(receiver)


    def transmit(self):
        """Advance current time one frame"""
        assert len(self.beacons) != 0

        # advance self.frame
        if self.base_time >= 0.:
            # advance base time
            self.base_time += self.frame_step
            # we shall remove Signals which are not at least partially in [self.base_time, self.base_time + self.frame_duration) interval
            new_frame = dict()
            for signal_start in self.frame.keys():
                signal = self.frame[signal_start]
                if signal_start + signal.duration >=  self.frame_step:
                    # surviver shall be remained with subtracted start
                    new_frame[signal_start - self.frame_step] = signal
            self.frame = new_frame
        else:
            self.base_time = 0.

        # do transmit
        min_key = min(self.reps.keys())
        while min_key < self.frame_duration:
            rep = self.reps[min_key]
            self.frame[min_key] = rep.get_signal_params(self.max_delta)
            del self.reps[min_key]
            print("rep %d from beacon %d start = %f" % (rep.regen_id, rep.owner.b_id, self.base_time + min_key))
            self.reps[min_key + rep.repeat_period] = rep
            min_key = min(self.reps.keys())

        # remove frame_step time from each object inside self.reps
        new_reps = dict()
        for key in self.reps.keys():
            rep = self.reps[key]
            new_reps[key - self.frame_step] = rep
        self.reps = new_reps

        # print frame
        for signal_start in self.frame.keys():
            print("signal (%f, %f) freq %f hz" % (self.base_time + signal_start, self.base_time + signal_start + self.frame[signal_start].duration, self.frame[signal_start].freq_radhz / 2 / np.pi))
        print("frame transmited at base time  = %f" % (self.base_time))


    def receive(self, idx):
        """Get frame samples for a receiver with idx index"""
        assert idx < len(self.receivers)
        recv = self.receivers[idx]
        # prepare noise to fill with signal
        noise_dev = 10 ** (recv.sensitivity_dbm / 20) / math.sqrt(2)
        sig2ret = np.random.normal(0., noise_dev, [2, int(recv.samp_freq_hz *  self.frame_duration)]) + 1j * np.random.normal(0., noise_dev, [2, int(recv.samp_freq_hz *  self.frame_duration)])

        # generate receiver phase noise
        rx_fpn0 = prepare_interp_function(self.frame_duration * 4, recv.rx_pn_sc_min, recv.rx_pn_sc_max)
        rx_fpn1 = prepare_interp_function(self.frame_duration * 4, recv.rx_pn_sc_min, recv.rx_pn_sc_max)
        rx_fpn = (rx_fpn0, rx_fpn1)

        for signal_start in self.frame.keys():
            sig = self.frame[signal_start]

            # get delta and power from geometry
            rvec = np.array([sig.x - recv.x, sig.y - recv.y, sig.z - recv.z])
            dist_m = np.linalg.norm(rvec)
            tdelta = - 2 * np.dot(recv.avec, rvec) / dist_m / LSp
            # Friis formula
            path_loss = 20 * np.log(self.wavelen_m / 4. / np.pi / dist_m) / np.log(10)
            recv_pwr_dbm = sig.power_dbm + 6 + path_loss
            amp = 10 ** (recv_pwr_dbm / 20) # removed sqrt(2) since there will be a complex part
            # shift phase noise
            sh_rx_fpn = (lambda x : rx_fpn[0](x + signal_start + self.frame_duration), lambda x : rx_fpn[1](x + signal_start + self.frame_duration))
            # now we a ready to create signal
            recvd_signal, xvals = combine_pashe_noise(sig.duration, recv.samp_freq_hz, sig.freq_radhz, sig.ph_start, sig.fpn, sig.dev, tdelta, sh_rx_fpn, recv.rx_pn_dev)
            recvd_signal *= amp
            # do little windowing
            window = sig.fwn(xvals / sig.duration) # fwn is with normed arguments
            recvd_signal *= window

            # embedd signal if there is a place
            sig_offset = int(signal_start * recv.samp_freq_hz)
            sig_end = sig_offset + recvd_signal.shape[1]
            #print("  offset = %d, end = %d" % (sig_offset, sig_end))
            if sig_offset < sig2ret.shape[1] and sig_end > 0:
                recvd_start = -min(sig_offset, 0)
                sig_start = max(sig_offset, 0) # need to do this because negative values will wrap
                recvd_end = recvd_signal.shape[1] + min(sig2ret.shape[1] - sig_end, 0)
                sig2ret[:, sig_start:sig_end] += recvd_signal[:, recvd_start:recvd_end]

        return sig2ret
