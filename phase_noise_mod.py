import numpy as np
from scipy.interpolate import interp1d


# creates normed phase noise function with duration length and max_delta slack before
# and after. Phase noise is the interpolation between normaly distributed points divided at random time intervals.
# sc_min - is the minimum change interval and sc_max is the maxumum one.
def prepare_interp_function(duration, sc_min, sc_max, max_delta=0.):

    # put first sample
    xvals = [-max_delta]
    vals = [np.random.normal(0, 1.0)]

    while xvals[-1] < (duration + max_delta):
        xvals.append(xvals[-1] + np.random.uniform(sc_min, sc_max))
        vals.append(np.random.normal(0, 1.0))

    # ensure that whe have at least two values before next extrapolation step
    if len(xvals) == 1:
        xvals.append(duration + max_delta)
        vals.append(np.random.normal(0, 1.0))

    # lineary trim last value - there shall be at least 2 indexes
    vals[-1] = vals[-2] + (vals[-1] - vals[-2]) * ((duration + max_delta) - xvals[-2]) / (xvals[-1] - xvals[-2])
    xvals[-1] = duration + max_delta

    # go to numpy and use its power at full
    xvals = np.array(xvals)
    vals = np.array(vals)

    try:
        if xvals.size == 2:
            fpn = interp1d(xvals, vals, kind='linear')
        elif xvals.size == 3:
            fpn = interp1d(xvals, vals, kind='quadratic')
        elif xvals.size >= 4:
            fpn = interp1d(xvals, vals, kind='cubic')
        else:
            assert False
    except ValueError:
        print(xvals)

    return fpn


# combines phase noise with some central frequency and does sampling.
# gets phase noise interpolated function as tx_fpn together with deviation tx_pn_dev and sine_freq as the central frequency
# there is also rx_fpn noise
# if tdelta is not None then the dual mode is enabled with tdelta delay
def combine_pashe_noise(time_len, samp_freq, sine_freq, ph_start, tx_fpn, tx_pn_dev, tdelta=None, rx_fpn=None, rx_pn_dev=0.):
    num_samples = int(time_len * samp_freq)
    do_delayed = (tdelta is not None)

    # generate non-delayed and delayed tx phase noise
    xvals = np.linspace(0., time_len, num_samples + 1)[:-1]
    tx_pn_interp = tx_fpn(xvals)
    fpn_mean = np.sum(tx_pn_interp) / tx_pn_interp.size
    tx_pn_interp -= fpn_mean # subract mean value
    # delayed
    if do_delayed:
        xvals_d = np.linspace(tdelta, time_len + tdelta, num_samples + 1)[:-1]
        tx_pn_interp_d = tx_fpn(xvals_d)
        tx_pn_interp_d -= fpn_mean # subract mean value same as for non delayed function

    # generate rx phase noise
    if rx_fpn is not None:
        rx_pn_interp = rx_fpn[0](xvals)
        rx_pn_interp_d = rx_fpn[1](xvals)
    else:
        rx_pn_interp = np.zeros([num_samples,])
        rx_pn_interp_d = np.zeros([num_samples,])

    # combine all in phase
    phase = xvals * (sine_freq + tx_pn_interp * tx_pn_dev + rx_pn_interp * rx_pn_dev) + ph_start
    if do_delayed:
        phase_d = xvals_d * (sine_freq + tx_pn_interp_d * tx_pn_dev + rx_pn_interp_d * rx_pn_dev) + ph_start

    # compute complex exponent to enable negative frequencies
    signal = np.cos(phase) + 1j * np.sin(phase)
    if do_delayed:
        signal_d = np.cos(phase_d) + 1j * np.sin(phase_d)

    # return
    if not do_delayed:
        return signal, xvals # no reshape needed for backward compatibility
    else:
        return np.concatenate([signal.reshape([1, signal.size]), signal_d.reshape([1, signal_d.size])], axis = 0), xvals


# makes model signal with phase noise with single receiver
def make_phase_noise(time_len, samp_freq, sine_freq, sc_min, sc_max, dev, ph_start=0.):

    fpn = prepare_interp_function(time_len, sc_min, sc_max)

    return np.real(combine_pashe_noise(time_len, samp_freq, sine_freq, ph_start, fpn, dev)[0])


# makes model signal with phase noise with two receivers each having its receiver phase noise
def make_phase_noise_dual(time_len, samp_freq, sine_freq, sc_min, sc_max, tx_dev, tdelta, ph_start=0., rx_dev=0.):

    delta_samps = int(np.abs(tdelta * samp_freq)) + 1
    max_delta = delta_samps / samp_freq

    # generate transmitter phase noise
    tx_fpn = prepare_interp_function(time_len, sc_min, sc_max, max_delta)
    # generate receiver phase noise
    rx_fpn0 = prepare_interp_function(time_len, sc_min, sc_max)
    rx_fpn1 = prepare_interp_function(time_len, sc_min, sc_max)
    rx_fpn = (rx_fpn0, rx_fpn1)

    return np.real(combine_pashe_noise(time_len, samp_freq, sine_freq, ph_start, tx_fpn, tx_dev, tdelta, rx_fpn, rx_dev)[0])
