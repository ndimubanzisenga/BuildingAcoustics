"""
Building Acoustic Measurement
=============================
This module implements routines for the measurement of parameters to describe building acoustics
"""
from __future__ import division
from signal import Spectrum, OctaveBand, Signal
from scipy import stats
import numpy as np

class BuildingAcousticsMeasurement(object):
    def __init__(self, fs=44100, f_start=50., f_stop=5000., fraction=3):
        """
        :param fs: sampling frequency
        :param f_start: Minimum frequency on which the sine sweep is initialized.
        :param f_stop: Maximum frequency of the sine sweep.
        :param fraction: nth octave to be used
        """
        self._f_start = f_start
        self._f_stop = f_stop
        self._fraction = fraction
        self._fs = fs

        self.probe_signal = None
        self.response_signal = None
        self.octave_bands = OctaveBand(fstart=f_start, fstop=f_stop, fraction=fraction).center
        self.tx_room_spl = None
        self.rx_room_spl = None
        self.transmission_loss = None
        self.rx_room_background_noise_spl = None
        self.rx_room_reverberation_time = None
        self.ref_curve = None

    def compute_building_acoustics_parameters(self):
        """
        Calculate building acoustics parameters: The Transmitting room and Receiving room spl,
        the background noise level and reverberation time.
        """
        pass

    def verify_building_acoustics_regulation(self):
        pass

    def diagnose_defect(self):
        pass

    def compute_spl(self, signal, room):
        """
        Compute the sound pressure level.
        :param signal: Measured raw signal in the room.
        :param room: Room in which the signal measurement is made. Possible values are : {'tx', 'rx', 'noise_rx'}.
        """
        if (room is not 'tx') and (room is not 'rx') and (room is not 'noise_rx'):
            raise ValueError("Specified room type should be either 'rx' or 'tx' or 'noise_rx'")

        self.response_signal = signal
        f_start = self._f_start
        f_stop = self._f_stop
        fraction = self._fraction
        octave_bands = self.octave_bands

        spectrum = Spectrum()
        #frequencies = OctaveBand(fstart=f_start, fstop=f_stop, fraction=fraction)
        _, octaves_power_levels = spectrum.third_octaves(signal, self._fs, frequencies=octave_bands)
        #self.octave_bands = frequencies.center
        if room == 'tx':
            self.tx_room_spl = octaves_power_levels
        elif room == 'rx':
            self.rx_room_spl = octaves_power_levels
        elif room == 'noise_rx':
            self.rx_room_background_noise_spl = octaves_power_levels

        return

    def compute_transmission_loss(self):
        """
        Compute the transmission loss, given the spl in the Transmitting and Receiving room has been computed.
        """
        if (self.rx_room_spl is None) or (self.tx_room_spl is None):
            raise ValueError("First compute the Receiving room and Transmitting room SPL.")

        self.transmission_loss = self.tx_room_spl - self.rx_room_spl

        return

    def get_reference_curve(self, Rw_nominal, f_start=None, f_stop=None, fraction=3, weighting=None):
        """
        Calculate the reference curve from a single number Rw nominal value.

        :param Rw_nominal: single number descriptor threshold defined by the building acoustics regulations.
        :param weigthing: the weigthing to apply.

        Return the reference rw curve for the specified frequency range
        """
        if (f_start is not None) and (f_stop is not None):
            frequencies = OctaveBand(fstart=f_start, fstop=f_stop, fraction=fraction)
            octave_bands = frequencies.center
        elif (self.octave_bands is not None):
            octave_bands = self.octave_bands
        else:
            raise ValueError("The frequency range to consider must be set")

        ref_curve = None

        return ref_curve

    def rw_curve(self, transmission_loss=None):
        """
        Calculate the reference curve of :math:`Rw` from a NumPy array `transmission_loss` with third
        octave data between 100 Hz and 3.15 kHz.

        :param transmission_loss: Transmission Loss

        """
        if (transmission_loss is not None):
            t = transmission_loss
        elif (self.transmission_loss is not None):
            t = self.transmission_loss
        else:
            raise ValueError("The transmission loss curve must be set")

        ref_curve = np.array([[0, 3, 6, 9, 12, 15, 18, 19, 20, 21, 22, 23, 23, 23, 23, 23]]) #ToDo: Function to compute the reference curve for a dynamic freq. range
        # ref_curve = self.get_reference_curve(Rw_nominal)
        residuals_sum = 0

        while residuals_sum > -32:
            ref_curve += 1
            diff = t - ref_curve
            residuals = np.clip(diff, np.min(diff), 0)
            residuals_sum = np.sum(residuals)
        ref_curve -= 1
        self.ref_curve = ref_curve

        return


    def compute_reverberation_time(self, signal=None, fs=None, method='impulse'):
        """
        Compute reverberation time.

        :param signal: signal to use for the computation of the reverberation time.
                This can be a measured impulse response or raw sound pressure according to the chosen method.
        :param fs: Sampling frequency of the measured signal.
        :param method: Method to use for the computation of the reverberation time :{'impulse',}

        """
        if (method is not 'impulse'):
            return ValueError("Possible value of the method argument are : {'impulse',}")

        reverberation_time = None
        if (method is 'impulse'):
            reverberation_time = self.t60_impulse(measured_impulse_response=signal, fs=fs)

        return reverberation_time


    def t60_impulse(self, measured_impulse_response=None, fs=None, rt='t30'):
        """
        Reverberation time from a measured impulse response.

        :param measured_impulse_response: Numpy array of the measured impulse response.
        :param rt: Reverberation time estimator. It accepts `'t30'`, `'t20'`, `'t10'` and `'edt'`.
        :returns: Reverberation time :math:`T_{60}`

        """
        if (measured_impulse_response is None):
            if (self.probe_signal is not None) and (self.response_signal is not None):
                # ToDo: Derive the impulse respnse from the probe signal and the measured response signal
                #       Currently the impulse response is just equated to the measured response signal.
                measured_impulse_response = self.response_signal
            else:
                raise ValueError("The measused impulse response must be set. Otherwise the probe and response\
                                   signals must be set in order to derive the impulse response")

        if (fs is None):
            if (self._fs is not None):
                fs = self._fs
            else:
                raise ValueError(" The sampling frequency of the measured impulse response must be set")

        if (rt is not 't30') and (rt is not 't20') and (rt is not 't10') and (rt is not 'edt'):
            raise ValueError("Possible values of the rt argument are {'t30', 't20', 't10', 'edt'}")

        bands = OctaveBand(center=self.octave_bands, fraction=self._fraction)
        signal_processor = Signal()
        rt = rt.lower()

        if rt == 't30':
            init = -5.0
            end = -35.0
            factor = 2.0
        elif rt == 't20':
            init = -5.0
            end = -25.0
            factor = 3.0
        elif rt == 't10':
            init = -5.0
            end = -15.0
            factor = 6.0
        elif rt == 'edt':
            init = 0.0
            end = -10.0
            factor = 6.0

        t60 = np.zeros(bands.center.size)

        for i in range(bands.center.size):
            # Filtering signal
            filtered_signal = signal_processor.bandpass(measured_impulse_response, bands.lower[i], bands.upper[i], fs, order=8)
            abs_signal = np.abs(filtered_signal) / np.max(np.abs(filtered_signal))

            # Schroeder integration
            sch = np.cumsum(abs_signal[::-1]**2)[::-1]
            sch_db = 10.0 * np.log10(sch / np.max(sch))

            # Linear regression
            sch_init = sch_db[np.abs(sch_db - init).argmin()]
            sch_end = sch_db[np.abs(sch_db - end).argmin()]
            init_sample = np.where(sch_db == sch_init)[0][0]
            end_sample = np.where(sch_db == sch_end)[0][0]
            x = np.arange(init_sample, end_sample + 1) / fs
            y = sch_db[init_sample: end_sample + 1]
            m, b, r_value, p_value, std_err = stats.linregress(x,y) # y = m*x + b

            # Reverberation time (T30, T20, T10 or EDT)
            db_regress_init = (init - b) / m
            db_regress_end = (end - b) / m
            t60[i] = factor * (db_regress_end - db_regress_init)
        return t60
