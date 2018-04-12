"""
This class provides various methods to process and transform measured raw signal
along with other helper functions
"""
from __future__ import division
import numpy as np
from scipy.signal import butter, filtfilt, sosfilt

from standards.iec_61260_1_2014 import REFERENCE_FREQUENCY as REFERENCE
from standards.iec_61260_1_2014 import NOMINAL_OCTAVE_CENTER_FREQUENCIES, NOMINAL_THIRD_OCTAVE_CENTER_FREQUENCIES
import standards.iec_61260_1_2014 #import index_of_frequency, exact_center_frequency, nominal_center_frequency, lower_frequency, upper_frequency

REFERENCE_PRESSURE = 2.0e-5

def get_exact_center_frequency(frequency=None, fraction=1, n=None, ref=REFERENCE):
    """Exact center frequency.

    :param frequency: Frequency within the band.
    :param fraction: Band designator.
    :param n: Index of band.
    :param ref: Reference frequency.
    :return: Exact center frequency for the given frequency or band index.

    .. seealso:: :func:`acoustics.standards.iec_61260_1_2014.exact_center_frequency`
    .. seealso:: :func:`acoustics.standards.iec_61260_1_2014.index_of_frequency`

    """
    if frequency is not None:
        n = standards.iec_61260_1_2014.index_of_frequency(frequency, fraction=fraction, ref=ref)
    return standards.iec_61260_1_2014.exact_center_frequency(n, fraction=fraction, ref=ref)


def get_nominal_center_frequency(frequency=None, fraction=1, n=None):
    """Nominal center frequency.

    :param frequency: Frequency within the band.
    :param fraction: Band designator.
    :param n: Index of band.
    :returns: The nominal center frequency for the given frequency or band index.

    .. seealso:: :func:`acoustics.standards.iec_61260_1_2014.exact_center_frequency`
    .. seealso:: :func:`acoustics.standards.iec_61260_1_2014.nominal_center_frequency`

    .. note:: Contrary to the other functions this function silently assumes 1000 Hz reference frequency.

    """
    center = get_exact_center_frequency(frequency, fraction, n)
    return standards.iec_61260_1_2014.nominal_center_frequency(center, fraction)


def get_lower_frequency(frequency=None, fraction=1, n=None, ref=REFERENCE):
    """Lower band-edge frequency.

    :param frequency: Frequency within the band.
    :param fraction: Band designator.
    :param n: Index of band.
    :param ref: Reference frequency.
    :returns: Lower band-edge frequency for the given frequency or band index.

    .. seealso:: :func:`acoustics.standards.iec_61260_1_2014.exact_center_frequency`
    .. seealso:: :func:`acoustics.standards.iec_61260_1_2014.lower_frequency`

    """
    center = get_exact_center_frequency(frequency, fraction, n, ref=ref)
    return standards.iec_61260_1_2014.lower_frequency(center, fraction)


def get_upper_frequency(frequency=None, fraction=1, n=None, ref=REFERENCE):
    """Upper band-edge frequency.

    :param frequency: Frequency within the band.
    :param fraction: Band designator.
    :param n: Index of band.
    :param ref: Reference frequency.
    :returns: Upper band-edge frequency for the given frequency or band index.

    .. seealso:: :func:`acoustics.standards.iec_61260_1_2014.exact_center_frequency`
    .. seealso:: :func:`acoustics.standards.iec_61260_1_2014.upper_frequency`

    """
    center = get_exact_center_frequency(frequency, fraction, n, ref=ref)
    return standards.iec_61260_1_2014.upper_frequency(center, fraction)


class Frequencies(object):
    """
    Object describing frequency bands.
    """

    def __init__(self, center, lower, upper, bandwidth=None):

        self.center = np.asarray(center)
        """
        Center frequencies.
        """

        self.lower = np.asarray(lower)
        """
        Lower frequencies.
        """

        self.upper = np.asarray(upper)
        """
        Upper frequencies.
        """

        self.bandwidth = np.asarray(bandwidth) if bandwidth is not None else np.asarray(self.upper) - np.asarray(self.lower)
        """
        Bandwidth.
        """

    def __iter__(self):
        for i in range(len(self.center)):
            yield self[i]

    def __len__(self):
        return len(self.center)

    def __str__(self):
        return str(self.center)

    def __repr__(self):
        return "Frequencies({})".format(str(self.center))

    def angular(self):
        """Angular center frequency in radians per second.
        """
        return 2.0 * np.pi * self.center



class EqualBand(Frequencies):
    """
    Equal bandwidth spectrum. Generally used for narrowband data.
    """

    def __init__(self, center=None, fstart=None, fstop=None, nbands=None, bandwidth=None):
        """

        :param center: Vector of center frequencies.
        :param fstart: First center frequency.
        :param fstop: Last center frequency.
        :param nbands: Amount of frequency bands.
        :param bandwidth: Bandwidth of bands.

        """

        if center is not None:
            try:
                nbands = len(center)
            except TypeError:
                center = [center]
                nbands = 1

            u = np.unique(np.diff(center).round(decimals=3))
            n = len(u)
            if n == 1:
                bandwidth = u
            elif n > 1:
                raise ValueError("Given center frequencies are not equally spaced.")
            else:
                pass
            fstart = center[0] #- bandwidth/2.0
            fstop = center[-1] #+ bandwidth/2.0
        elif fstart is not None and fstop is not None and nbands:
            bandwidth = (fstop - fstart) / (nbands-1)
        elif fstart is not None and fstop is not None and bandwidth:
            nbands = round((fstop - fstart) / bandwidth) + 1
        elif fstart is not None and bandwidth and nbands:
            fstop = fstart + nbands * bandwidth
        elif fstop is not None and bandwidth and nbands:
            fstart = fstop - (nbands-1) * bandwidth
        else:
            raise ValueError("Insufficient parameters. Cannot determine fstart, fstop, bandwidth.")

        center = fstart + np.arange(0, nbands) * bandwidth # + bandwidth/2.0
        upper  = fstart + np.arange(0, nbands) * bandwidth + bandwidth/2.0
        lower  = fstart + np.arange(0, nbands) * bandwidth - bandwidth/2.0

        super(EqualBand, self).__init__(center, lower, upper, bandwidth)

    def __getitem__(self, key):
        return type(self)(center=self.center[key], bandwidth=self.bandwidth)

    def __repr__(self):
        return "EqualBand({})".format(str(self.center))


class OctaveBand(Frequencies):
    """Fractional-octave band spectrum.
    """

    def __init__(self, center=None, fstart=None, fstop=None, nbands=None, fraction=1, reference=REFERENCE):

        if center is not None:
            try:
                nbands = len(center)
            except TypeError:
                center = [center]
            center = np.asarray(center)
            indices = standards.iec_61260_1_2014.index_of_frequency(center, fraction=fraction, ref=reference)
        elif fstart is not None and fstop is not None:
            nstart = standards.iec_61260_1_2014.index_of_frequency(fstart, fraction=fraction, ref=reference)
            nstop = standards.iec_61260_1_2014.index_of_frequency(fstop, fraction=fraction, ref=reference)
            indices = np.arange(nstart, nstop+1)
        elif fstart is not None and nbands is not None:
            nstart = standards.iec_61260_1_2014.index_of_frequency(fstart, fraction=fraction, ref=reference)
            indices = np.arange(nstart, nstart+nbands)
        elif fstop is not None and nbands is not None:
            nstop = standards.iec_61260_1_2014.index_of_frequency(fstop, fraction=fraction, ref=reference)
            indices = np.arange(nstop-nbands, nstop)
        else:
            raise ValueError("Insufficient parameters. Cannot determine fstart and/or fstop.")

        center = get_exact_center_frequency(None, fraction=fraction, n=indices, ref=reference)
        lower = get_lower_frequency(center, fraction=fraction)
        upper = get_upper_frequency(center, fraction=fraction)
        bandwidth = upper - lower
        nominal = get_nominal_center_frequency(None, fraction, indices)

        super(OctaveBand, self).__init__(center, lower, upper, bandwidth)

        self.fraction = fraction
        """Fraction of fractional-octave filter.
        """

        self.reference = reference
        """Reference center frequency.
        """

        self.nominal = nominal
        """Nominal center frequencies.
        """

    def __getitem__(self, key):
        return type(self)(center=self.center[key], fraction=self.fraction, reference=self.reference)

    def __repr__(self):
        return "OctaveBand({})".format(str(self.center))


class Signal(object):
    def ms(self, x):
        """Mean value of signal `x` squared.

        :param x: Dynamic quantity.
        :returns: Mean squared of `x`.

        """
        return (np.abs(x)**2.0).mean()

    def rms(self, x):
        """Root mean squared of signal `x`.

        :param x: Dynamic quantity.

        .. math:: x_{rms} = lim_{T \\to \\infty} \\sqrt{\\frac{1}{T} \int_0^T |f(x)|^2 \\mathrm{d} t }

        :seealso: :func:`ms`.

        """
        return np.sqrt(self.ms(x))

    def normalize(self, y, x=None):
        """normalize power in y to a (standard normal) white noise signal.

        Optionally normalize to power in signal `x`.

        #The mean power of a Gaussian with :math:`\\mu=0` and :math:`\\sigma=1` is 1.
        """
        #return y * np.sqrt( (np.abs(x)**2.0).mean() / (np.abs(y)**2.0).mean() )
        if x is not None:
            x = self.ms(x)
        else:
            x = 1.0
        return y * np.sqrt( x / self.ms(y) )
        #return y * np.sqrt( 1.0 / (np.abs(y)**2.0).mean() )

        ## Broken? Caused correlation in auralizations....weird!

    def window_scaling_factor(self, window, axis=-1):
        """
        Calculate window scaling factor.

        :param window: Window.

        When analysing broadband (filtered noise) signals it is common to normalize
        the windowed signal so that it has the same power as the un-windowed one.

        .. math:: S = \\sqrt{\\frac{\\sum_{i=0}^N w_i^2}{N}}

        """
        return np.sqrt((window*window).mean(axis=axis))

    def apply_window(self, x, window):
        """
        Apply window to signal.

        :param x: Instantaneous signal :math:`x(t)`.
        :param window: Vector representing window.

        :returns: Signal with window applied to it.

        .. math:: x_s(t) = x(t) / S

        where :math:`S` is the window scaling factor.

        .. seealso:: :func:`window_scaling_factor`.

        """
        s = self.window_scaling_factor(window) # Determine window scaling factor.
        n = len(window)
        windows = x//n  # Amount of windows.
        x = x[0:windows*n] # Truncate final part of signal that does not fit.
        #x = x.reshape(-1, len(window)) # Reshape so we can apply window.
        y = np.tile(window, windows)

        return x * y / s


    def bandpass_filter(self, lowcut, highcut, fs, order=8, output='sos'):
        """Band-pass filter.

        :param lowcut: Lower cut-off frequency
        :param highcut: Upper cut-off frequency
        :param fs: Sample frequency
        :param order: Filter order
        :param output: Output type. {'ba', 'zpk', 'sos'}. Default is 'sos'. See also :func:`scipy.signal.butter`.
        :returns: Returned value depends on `output`.

        A Butterworth filter is used.

        .. seealso:: :func:`scipy.signal.butter`.

        """
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        output = butter(order/2, [low, high], btype='band', output=output)
        return output


    def bandpass(self, signal, lowcut, highcut, fs, order=8, zero_phase=False):
        """Filter signal with band-pass filter.

        :param signal: Signal
        :param lowcut: Lower cut-off frequency
        :param highcut: Upper cut-off frequency
        :param fs: Sample frequency
        :param order: Filter order
        :param zero_phase: Prevent phase error by filtering in both directions (filtfilt)

        A Butterworth filter is used. Filtering is done with second-order sections.

        .. seealso:: :func:`bandpass_filter` for the filter that is used.

        """
        sos = self.bandpass_filter(lowcut, highcut, fs, order, output='sos')
        if zero_phase:
            return self._sosfiltfilt(sos, signal)
        else:
            return sosfilt(sos, signal)


    def bandstop(self, signal, lowcut, highcut, fs, order=8, zero_phase=False):
        """Filter signal with band-stop filter.

        :param signal: Signal
        :param lowcut: Lower cut-off frequency
        :param highcut: Upper cut-off frequency
        :param fs: Sample frequency
        :param order: Filter order
        :param zero_phase: Prevent phase error by filtering in both directions (filtfilt)

        """
        return self.lowpass(signal, lowcut, fs, order=(order//2), zero_phase=zero_phase) + self.highpass(signal, highcut, fs, order=(order//2), zero_phase=zero_phase)


    def lowpass(self, signal, cutoff, fs, order=4, zero_phase=False):
        """Filter signal with low-pass filter.

        :param signal: Signal
        :param fs: Sample frequency
        :param cutoff: Cut-off frequency
        :param order: Filter order
        :param zero_phase: Prevent phase error by filtering in both directions (filtfilt)

        A Butterworth filter is used. Filtering is done with second-order sections.

        .. seealso:: :func:`scipy.signal.butter`.

        """
        sos = butter(order, cutoff/(fs/2.0), btype='low', output='sos')
        if zero_phase:
            return self._sosfiltfilt(sos, signal)
        else:
            return sosfilt(sos, signal)


    def highpass(self, signal, cutoff, fs, order=4, zero_phase=False):
        """Filter signal with low-pass filter.

        :param signal: Signal
        :param fs: Sample frequency
        :param cutoff: Cut-off frequency
        :param order: Filter order
        :param zero_phase: Prevent phase error by filtering in both directions (filtfilt)

        A Butterworth filter is used. Filtering is done with second-order sections.

        .. seealso:: :func:`scipy.signal.butter`.

        """
        sos = butter(order, cutoff/(fs/2.0), btype='high', output='sos')
        if zero_phase:
            return self._sosfiltfilt(sos, signal)
        else:
            return sosfilt(sos, signal)


    def octave_filter(self, center, fs, fraction, order=8, output='sos'):
        """Fractional-octave band-pass filter.

        :param center: Centerfrequency of fractional-octave band.
        :param fs: Sample frequency
        :param fraction: Fraction of fractional-octave band.
        :param order: Filter order
        :param output: Output type. {'ba', 'zpk', 'sos'}. Default is 'sos'. See also :func:`scipy.signal.butter`.

        A Butterworth filter is used.

        .. seealso:: :func:`bandpass_filter`

        """
        ob = OctaveBand(center=center, fraction=fraction)
        return self.bandpass_filter(ob.lower[0], ob.upper[0], fs, order, output=output)


    def octavepass(self, signal, center, fs, fraction, order=8, zero_phase=False):
        """Filter signal with fractional-octave bandpass filter.

        :param signal: Signal
        :param center: Centerfrequency of fractional-octave band.
        :param fs: Sample frequency
        :param fraction: Fraction of fractional-octave band.
        :param order: Filter order
        :param zero_phase: Prevent phase error by filtering in both directions (filtfilt)

        A Butterworth filter is used. Filtering is done with second-order sections.

        .. seealso:: :func:`octave_filter`

        """
        sos = self.octave_filter(center, fs, fraction, order)
        if zero_phase:
            return self._sosfiltfilt(sos, signal)
        else:
            return sosfilt(sos, signal)

    def bandpass_frequencies(self, x, fs, frequencies, order=8, purge=False, zero_phase=False):
        """"Apply bandpass filters for frequencies

        :param x: Instantaneous signal :math:`x(t)`.
        :param fs: Sample frequency.
        :param frequencies: Frequencies. Instance of :class:`Frequencies`.
        :param order: Filter order.
        :param purge: Discard bands of which the upper corner frequency is above the Nyquist frequency.
        :param zero_phase: Prevent phase error by filtering in both directions (filtfilt)
        :returns: Tuple. First element is an instance of :class:`OctaveBand`. The second element an array.
        """
        if purge:
            frequencies = frequencies[frequencies.upper < fs/2.0]
        return frequencies, np.array([self.bandpass(x, band.lower, band.upper, fs, order, zero_phase=zero_phase) for band in frequencies])


    def bandpass_octaves(self, x, fs, frequencies=NOMINAL_OCTAVE_CENTER_FREQUENCIES, order=8, purge=False, zero_phase=False):
        """Apply 1/1-octave bandpass filters.

        :param x: Instantaneous signal :math:`x(t)`.
        :param fs: Sample frequency.
        :param frequencies: Frequencies.
        :param order: Filter order.
        :param purge: Discard bands of which the upper corner frequency is above the Nyquist frequency.
        :param zero_phase: Prevent phase error by filtering in both directions (filtfilt)
        :returns: Tuple. First element is an instance of :class:`OctaveBand`. The second element an array.

        .. seealso:: :func:`octavepass`
        """
        return self.bandpass_fractional_octaves(x, fs, frequencies, fraction=1, order=order, purge=purge, zero_phase=zero_phase)


    def bandpass_third_octaves(self, x, fs, frequencies=NOMINAL_THIRD_OCTAVE_CENTER_FREQUENCIES, order=8, purge=False, zero_phase=False):
        """Apply 1/3-octave bandpass filters.

        :param x: Instantaneous signal :math:`x(t)`.
        :param fs: Sample frequency.
        :param frequencies: Frequencies.
        :param order: Filter order.
        :param purge: Discard bands of which the upper corner frequency is above the Nyquist frequency.
        :param zero_phase: Prevent phase error by filtering in both directions (filtfilt)
        :returns: Tuple. First element is an instance of :class:`OctaveBand`. The second element an array.

        .. seealso:: :func:`octavepass`
        """
        return self.bandpass_fractional_octaves(x, fs, frequencies, fraction=3, order=order, purge=purge, zero_phase=zero_phase)


    def bandpass_fractional_octaves(self, x, fs, frequencies, fraction=None, order=8, purge=False, zero_phase=False):
        """Apply 1/N-octave bandpass filters.

        :param x: Instantaneous signal :math:`x(t)`.
        :param fs: Sample frequency.
        :param frequencies: Frequencies. Either instance of :class:`OctaveBand`, or array along with fs.
        :param order: Filter order.
        :param purge: Discard bands of which the upper corner frequency is above the Nyquist frequency.
        :param zero_phase: Prevent phase error by filtering in both directions (filtfilt)
        :returns: Tuple. First element is an instance of :class:`OctaveBand`. The second element an array.

        .. seealso:: :func:`octavepass`
        """
        if not isinstance(frequencies, Frequencies):
            frequencies = OctaveBand(center=frequencies, fraction=fraction)
        return self.bandpass_frequencies(x, fs, frequencies, order=order, purge=purge, zero_phase=zero_phase)


    def _sosfiltfilt(self, sos, x, axis=-1, padtype='odd', padlen=None, method='pad', irlen=None):
        """Filtfilt version using Second Order sections. Code is taken from scipy.signal.filtfilt and adapted to make it work with SOS.
        Note that broadcasting does not work.
        """
        from scipy.signal import sosfilt_zi
        from scipy.signal._arraytools import odd_ext, axis_slice, axis_reverse
        x = np.asarray(x)

        if padlen is None:
            edge = 0
        else:
            edge = padlen

        # x's 'axis' dimension must be bigger than edge.
        if x.shape[axis] <= edge:
            raise ValueError("The length of the input vector x must be at least "
                             "padlen, which is %d." % edge)

        if padtype is not None and edge > 0:
            # Make an extension of length `edge` at each
            # end of the input array.
            if padtype == 'even':
                ext = even_ext(x, edge, axis=axis)
            elif padtype == 'odd':
                ext = odd_ext(x, edge, axis=axis)
            else:
                ext = const_ext(x, edge, axis=axis)
        else:
            ext = x

        # Get the steady state of the filter's step response.
        zi = sosfilt_zi(sos)

        # Reshape zi and create x0 so that zi*x0 broadcasts
        # to the correct value for the 'zi' keyword argument
        # to lfilter.
        #zi_shape = [1] * x.ndim
        #zi_shape[axis] = zi.size
        #zi = np.reshape(zi, zi_shape)
        x0 = axis_slice(ext, stop=1, axis=axis)
        # Forward filter.
        (y, zf) = sosfilt(sos, ext, axis=axis, zi=zi * x0)

        # Backward filter.
        # Create y0 so zi*y0 broadcasts appropriately.
        y0 = axis_slice(y, start=-1, axis=axis)
        (y, zf) = sosfilt(sos, axis_reverse(y, axis=axis), axis=axis, zi=zi * y0)

        # Reverse y.
        y = axis_reverse(y, axis=axis)

        if edge > 0:
            # Slice the actual signal from the extended signal.
            y = axis_slice(y, start=edge, stop=-edge, axis=axis)

        return y


class Spectrum(object):
    def third_octaves(self, x, fs, density=False,
                  frequencies=NOMINAL_THIRD_OCTAVE_CENTER_FREQUENCIES,
                  ref=REFERENCE_PRESSURE):
        """Calculate level per 1/3-octave in frequency domain using the FFT.

        :param x: Instantaneous signal :math:`x(t)`.
        :param fs: Sample frequency.
        :param density: Power density instead of power.
        :returns: Tuple. First element is an instance of :class:`OctaveBand`. The second and third elements are arrays.

        .. note:: Based on power spectrum (FFT)

        .. seealso:: :attr:`acoustics.bands.THIRD_OCTAVE_CENTER_FREQUENCIES`

        .. note:: Exact center frequencies are always calculated.

        """
        fob = OctaveBand(center=frequencies, fraction=3)
        f, p = self.power_spectrum(x, fs)
        fnb = EqualBand(f)
        power = self.integrate_bands(p, fnb, fob)
        if density:
            power /= (fob.bandwidth/fnb.bandwidth)
        level = 10.0*np.log10(power / ref**2.0)
        return fob, power, level

    def octaves(self, x, fs, density=False,
                frequencies=NOMINAL_OCTAVE_CENTER_FREQUENCIES,
                ref=REFERENCE_PRESSURE):
        """Calculate level per 1/1-octave in frequency domain using the FFT.

        :param x: Instantaneous signal :math:`x(t)`.
        :param fs: Sample frequency.
        :param density: Power density instead of power.
        :param frequencies: Frequencies.
        :param ref: Reference value.
        :returns: Tuple. First element is an instance of :class:`OctaveBand`. The second and third elements are arrays.

        .. note:: Based on power spectrum (FFT)

        .. seealso:: :attr:`acoustics.bands.OCTAVE_CENTER_FREQUENCIES`

        .. note:: Exact center frequencies are always calculated.

        """
        fob = OctaveBand(center=frequencies, fraction=1)
        f, p = self.power_spectrum(x, fs)
        fnb = EqualBand(f)
        power = self.integrate_bands(p, fnb, fob)
        if density:
            power /= (fob.bandwidth/fnb.bandwidth)
        level = 10.0*np.log10(power / ref**2.0)
        return fob, power, level


    def fractional_octaves(self, x, fs, start=5.0, stop=16000.0, fraction=3, density=False):
        """Calculate level per 1/N-octave in frequency domain using the FFT. N is `fraction`.

        :param x: Instantaneous signal :math:`x(t)`.
        :param fs: Sample frequency.
        :param density: Power density instead of power.
        :returns: Tuple. First element is an instance of :class:`OctaveBand`. The second and third elements are arrays.

        .. note:: Based on power spectrum (FFT)

        .. note:: This function does *not* use nominal center frequencies.

        .. note:: Exact center frequencies are always calculated.
        """
        fob = OctaveBand(fstart=start, fstop=stop, fraction=fraction)
        f, p = self.power_spectrum(x, fs)
        fnb = EqualBand(f)
        power = self.integrate_bands(p, fnb, fob)
        if density:
            power /= (fob.bandwidth/fnb.bandwidth)
        level = 10.0*np.log10(power)
        return fob, power, level

    def amplitude_spectrum(self, x, fs, N=None):
        """
        Amplitude spectrum of instantaneous signal :math:`x(t)`.

        :param x: Instantaneous signal :math:`x(t)`.
        :param fs: Sample frequency :math:`f_s`.
        :param N: Amount of FFT bins.

        The amplitude spectrum gives the amplitudes of the sinusoidal the signal is built
        up from, and the RMS (root-mean-square) amplitudes can easily be found by dividing
        these amplitudes with :math:`\\sqrt{2}`.

        The amplitude spectrum is double-sided.

        """
        N = N if N else x.shape[-1]
        fr = np.fft.fft(x, n=N) / N
        f = np.fft.fftfreq(N, 1.0/fs)
        return np.fft.fftshift(f), np.fft.fftshift(fr, axes=[-1])


    def auto_spectrum(self, x, fs, N=None):
        """
        Auto-spectrum of instantaneous signal :math:`x(t)`.

        :param x: Instantaneous signal :math:`x(t)`.
        :param fs: Sample frequency :math:`f_s`.
        :param N: Amount of FFT bins.

        The auto-spectrum contains the squared amplitudes of the signal. Squared amplitudes
        are used when presenting data as it is a measure of the power/energy in the signal.

        .. math:: S_{xx} (f_n) = \\overline{X (f_n)} \\cdot X (f_n)

        The auto-spectrum is double-sided.

        """
        f, a = self.amplitude_spectrum(x, fs, N=N)
        return f, (a*a.conj()).real


    def power_spectrum(self, x, fs, N=None):
        """
        Power spectrum of instantaneous signal :math:`x(t)`.

        :param x: Instantaneous signal :math:`x(t)`.
        :param fs: Sample frequency :math:`f_s`.
        :param N: Amount of FFT bins.

        The power spectrum, or single-sided autospectrum, contains the squared RMS amplitudes of the signal.

        A power spectrum is a spectrum with squared RMS values. The power spectrum is
        calculated from the autospectrum of the signal.

        .. warning:: Does not include scaling to reference value!

        .. seealso:: :func:`auto_spectrum`

        """
        N = N if N else x.shape[-1]
        f, a = self.auto_spectrum(x, fs, N=N)
        a = a[..., N//2:]
        f = f[..., N//2:]
        a *= 2.0
        a[..., 0] /= 2.0    # DC component should not be doubled.
        if not N%2: # if not uneven
            a[..., -1] /= 2.0 # And neither should fs/2 be.
        return f, a


    def angle_spectrum(self, x, fs, N=None):
        """
        Phase angle spectrum of instantaneous signal :math:`x(t)`.

        :param x: Instantaneous signal :math:`x(t)`.
        :param fs: Sample frequency :math:`f_s`.
        :param N: Amount of FFT bins.

        This function returns a single-sided wrapped phase angle spectrum.

        .. seealso:: :func:`phase_spectrum` for unwrapped phase spectrum.

        """
        N = N if N else x.shape[-1]
        f, a = self.amplitude_spectrum(x, fs, N)
        a = np.angle(a)
        a = a[..., N//2:]
        f = f[..., N//2:]
        return f, a


    def phase_spectrum(self, x, fs, N=None):
        """
        Phase spectrum of instantaneous signal :math:`x(t)`.

        :param x: Instantaneous signal :math:`x(t)`.
        :param fs: Sample frequency :math:`f_s`.
        :param N: Amount of FFT bins.

        This function returns a single-sided unwrapped phase spectrum.

        .. seealso:: :func:`angle_spectrum` for wrapped phase angle.

        """
        f, a = self.angle_spectrum(x, fs, N=None)
        return f, np.unwrap(a)

    def density_spectrum(self, x, fs, N=None):
        """
        Density spectrum of instantaneous signal :math:`x(t)`.

        :param x: Instantaneous signal :math:`x(t)`.
        :param fs: Sample frequency :math:`f_s`.
        :param N: Amount of FFT bins.

        A density spectrum considers the amplitudes per unit frequency.
        Density spectra are used to compare spectra with different frequency resolution as the
        magnitudes are not influenced by the resolution because it is per Hertz. The amplitude
        spectra on the other hand depend on the chosen frequency resolution.

        """
        N = N if N else x.shape[-1]
        fr = np.fft.fft(x, n=N) / fs
        f = np.fft.fftfreq(N, 1.0/fs)
        return np.fft.fftshift(f), np.fft.fftshift(fr)


    def integrate_bands(self, data, a, b):
        """
        Reduce frequency resolution of power spectrum. Merges frequency bands by integration.

        :param data: Vector with narrowband powers.
        :param a: Instance of :class:`Frequencies`.
        :param b: Instance of :class:`Frequencies`.

        .. note:: Needs rewriting so that the summation goes over axis=1.

        """

        try:
            if b.fraction%a.fraction:
                raise NotImplementedError("Non-integer ratio of fractional-octaves are not supported.")
        except AttributeError:
            pass

        lower, _ = np.meshgrid(b.lower, a.center)
        upper, _ = np.meshgrid(b.upper, a.center)
        _, center= np.meshgrid(b.center, a.center)

        return ((lower < center) * (center <= upper) * data[...,None]).sum(axis=-2)
