"""
This class provides signal generators.
"""
import numpy as np
from scipy.signal import fftconvolve
from signal import Signal
try:
    from pyfftw.interfaces.numpy_fft import rfft, irfft, fft     # Performs much better than numpy's fftpack
except ImportError:                                    # Use monkey-patching np.fft perhaps instead?
    from numpy.fft import rfft, irfft, fft



class Generator(object):
    def __init__(self, fs=44100.0, duration=10.0, state=None):
        """
        Sample signal generator.

        :param fs: Sampling frequency.
        :param duration: Duration ot the signal to generate.
        :param state: State of a Pseudo Random Number Generator.
        :type state: :class:`np.random.RandomState`
        """
        self._fs = fs
        self._duration = duration
        self._N = fs * duration
        self._state = state
        self._signal = Signal()
        #self.probe_pulse = None

    def noise   (self, noise_type='white'):
        """Noise generator.

        :param noise_type: Type of noise.
        """
        _noise_generators = {
            'white'     : self.white,
            'pink'      : self.pink,
            'blue'      : self.blue,
            'brown'     : self.brown,
            'violet'    : self.violet,
            'sine_sweep': self.sine_sweep,
            }

        try:
            return _noise_generators[noise_type]()
        except KeyError:
            raise ValueError("Incorrect type.")


    def white(self):
        """
        White noise.

        White noise has a constant power density. It's narrowband spectrum is therefore flat.
        The power in white noise will increase by a factor of two for each octave band,
        and therefore increases with 3 dB per octave.
        """
        state = np.random.RandomState() if self._state is None else self._state
        return state.randn(self._N)


    def pink(self):
        """
        Pink noise.

        Pink noise has equal power in bands that are proportionally wide.
        Power density decreases with 3 dB per octave.
        """
        # This method uses the filter with the following coefficients.
        #b = np.array([0.049922035, -0.095993537, 0.050612699, -0.004408786])
        #a = np.array([1, -2.494956002, 2.017265875, -0.522189400])
        #return lfilter(B, A, np.random.randn(N))
        # Another way would be using the FFT
        #x = np.random.randn(N)
        #X = rfft(x) / N
        state = np.random.RandomState() if self._state is None else self._state
        uneven = self._N%2
        X = state.randn(self._N//2+1+uneven) + 1j * state.randn(self._N//2+1+uneven)
        S = np.sqrt(np.arange(len(X))+1.) # +1 to avoid divide by zero
        y = (irfft(X/S)).real
        if uneven:
            y = y[:-1]
        return self._signal.normalize(y)


    def blue(self):
        """
        Blue noise.

        Power increases with 6 dB per octave.
        Power density increases with 3 dB per octave.
        """
        state = np.random.RandomState() if self._state is None else self._state
        uneven = self._N%2
        X = state.randn(self._N//2+1+uneven) + 1j * state.randn(self._N//2+1+uneven)
        S = np.sqrt(np.arange(len(X)))# Filter
        y = (irfft(X*S)).real
        if uneven:
            y = y[:-1]
        return self._signal.normalize(y)


    def brown(self):
        """
        Violet noise.

        Power decreases with -3 dB per octave.
        Power density decreases with 6 dB per octave.

        """
        state = np.random.RandomState() if self._state is None else self._state
        uneven = self._N%2
        X = state.randn(self._N//2+1+uneven) + 1j * state.randn(self._N//2+1+uneven)
        S = (np.arange(len(X))+1)# Filter
        y = (irfft(X/S)).real
        if uneven:
            y = y[:-1]
        return self._signal.normalize(y)


    def violet(self):
        """
        Violet noise. Power increases with 6 dB per octave.

        Power increases with +9 dB per octave.
        Power density increases with +6 dB per octave.

        """
        state = np.random.RandomState() if self._state is None else self._state
        uneven = self._N%2
        X = state.randn(self._N//2+1+uneven) + 1j * state.randn(self._N//2+1+uneven)
        S = (np.arange(len(X)))# Filter
        y = (irfft(X*S)).real
        if uneven:
            y = y[:-1]
        return self._signal.normalize(y)


    def sine_sweep(self, f_start=50., f_stop=5000.):
        """
        Sine sweep. The total power remains constant per octave.
        The frequency of the signal is changed exponetially.

        :param f_start: Minimum frequency on which the sine sweep is initialized.
        :param f_stop: Maximum frequency of the sine sweep.

        The rate of change from f_start to f_stop is expontential.
        """
        def probe():
            def lin_freq(t):
                return w1*t + (w2-w1)/T * t*t / 2
            def log_freq(t):
                K = T * w1 / np.log(w2/w1)
                L = T / np.log(w2/w1)
                return K * (np.exp(t/L)-1.0)
            freqs = log_freq(range(int(T)))
            return np.sin(freqs)

        w1 = f_start / self._fs * 2*np.pi
        w2 = f_stop / self._fs * 2*np.pi
        T = self._N

        probe_pulse = probe()

        # Apply exponential signal on the beginning and the end of the probe signal.
        exp_window = 1-np.exp(np.linspace(0,-10, 5000))
        probe_pulse[:exp_window.shape[0]] *= exp_window
        probe_pulse[-exp_window.shape[0]:] *= exp_window[-1::-1]

        # This is what the value of K will be at the end (in dB):
        kend = 10**((-6*np.log2(w2/w1))/20)
        # dB to rational number.
        k = np.log(kend)/T

        # Making reverse probe impulse so that convolution will just
        # calculate dot product. Weighting it with exponent to acheive
        # 6 dB per octave amplitude decrease.
        reverse_pulse = probe_pulse[-1::-1] * \
            np.array(list(\
                map(lambda t: np.exp(float(t)*k), range(int(T)))\
                )\
            )

        # Now we have to normilze energy of result of dot product.
        # This is "naive" method but it just works.
        Frp =  fft(fftconvolve(reverse_pulse, probe_pulse))
        reverse_pulse /= np.abs(Frp[round(Frp.shape[0]/4)])

        return probe_pulse, reverse_pulse


    def estimate_impulse_response(self, room_response, reverse_pulse):

        I = fftconvolve( room_response, reverse_pulse, mode='full')
        I = I[probe_pulse.shape[0]:probe_pulse.shape[0]*2+1]

        peek_x = np.argmax( I, axis=0 )
        I = I[peek_x:]

        return I


    def simulate_measured_signal(self, generated_signal=None, attenuation=0.1, delay=0., noise_rms=0.05):
        """
        Simulate a measured signal by attenuating, delaying and adding noise to a generated signal.
        .. measured_signal = attenuation * generated_signal(-delay) + background_noise

        :param generated_signal: Noise and attenuation free signal that is generated and played :Numpy::'array'
        :param attenuation: factor by which the system (room) attenuates the generated signal, as received by a microphone.
        :param delay: delay the signal takes to be measured.
        :param noise_rms: the RMS of the white noise that is added to the signal.

        Return a simulation of the measured signal.
        """
        if generated_signal is None:
            raise ValueError("The generated noise and attenuation free signal has to be given")

        #ToDo: Add delay
        background_noise = noise_rms * np.random.random_sample(generated_signal.size)
        return (attenuation * generated_signal + background_noise)
