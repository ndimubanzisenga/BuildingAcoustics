"""
Building Acoustic Measurement
=============================
This module implements routines for the measurement of parameters to describe building acoustics
"""
from __future__ import division
from signal import Spectrum, OctaveBand, Signal
from scipy import stats
import numpy as np

T_0 = 0.5 # Reference reverberation time

class AcousticParameters(object):
    def __init__(self, bands_number):
        self.L = np.zeros(bands_number)
        self.T = np.zeros(bands_number)
        self.Ln = np.zeros(bands_number)
        self.L_sigma = np.zeros(bands_number)
        self.T_sigma = np.zeros(bands_number)
        self.Ln_sigma = np.zeros(bands_number)
        self._L_count = 0
        self._T_count = 0

    def average_L(self, L):
        N = self._L_count
        self.L = (self.L * N + L) / (N + 1)
        self._L_count = self._L_count + 1
        # ToDo: Set sigma

    def average_T(self, T):
        N = self._T_count
        self.T = (self.T * N + T) / (N + 1)
        self._T_count = self._T_count + 1
        # ToDo: Set sigma


class BuildingAcousticsMeasurement(object):
    def __init__(self, fs=44100, f_start=50., f_stop=5000., fraction=3):
        ### ToDo Generalize to multiple channels measurement
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
        self._room_acoustic_params = None
        self._rooms_measurements = list()

        self.probe_signal = None
        self.response_signal = None
        self.octave_bands = OctaveBand(fstart=f_start, fstop=f_stop, fraction=fraction).center
        self.tx_room_spl = None
        self.rx_room_spl = None
        self.reverberation_time = None
        self.transmission_loss = None
        self.ref_curve = None

        self.initialize_room_measurement()


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

    def initialize_room_measurement(self):
        self._rooms_measurements = list()
        bands_number = self.octave_bands.size
        tx_room_acoustic_params = AcousticParameters(bands_number)
        rx_room_acoustic_params = AcousticParameters(bands_number)
        self._rooms_measurements.append(tx_room_acoustic_params)
        self._rooms_measurements.append(rx_room_acoustic_params)
        self.update_attributes()
        #self._room_acoustic_params = AcousticParameters(bands_number)
        return

    def finalize_room_measurement(self):
        self._rooms_measurements.append(self._room_acoustic_params)
        return

    def update_attributes(self):
        self.tx_room_spl = self._rooms_measurements[0].L
        self.rx_room_spl = self._rooms_measurements[1].L
        self.reverberation_time = self._rooms_measurements[1].T
        return

    def compute_spl(self, room, signal):
        """
        Compute the sound pressure level.
        :param signal: Measured raw signal in the room.
        :param room: Room in which the signal measurement is made. Possible values are : {'tx', 'rx', 'noise_rx'}.
        """
        if (room is not 'tx') and (room is not 'rx'):
            raise ValueError("Room parameter should take values {'tx', 'rx'}")

        self.response_signal = signal
        f_start = self._f_start
        f_stop = self._f_stop
        fraction = self._fraction
        octave_bands = self.octave_bands

        spectrum = Spectrum()
        _, octaves_power_levels = spectrum.third_octaves(signal, self._fs, frequencies=octave_bands)
        if room is 'tx':
            self._rooms_measurements[0].average_L(octaves_power_levels)
        elif room is 'rx':
            self._rooms_measurements[1].average_L(octaves_power_levels)
        self.update_attributes()
        #self._room_acoustic_params.average_L(octaves_power_levels)
        return

    def compute_reverberation_time(self, room, signal=None, fs=None, method='impulse', args=None):
        """
        Compute reverberation time.

        :param signal: signal to use for the computation of the reverberation time.
                This can be a measured impulse response or raw sound pressure according to the chosen method.
        :param fs: Sampling frequency of the measured signal.
        :param method: Method to use for the computation of the reverberation time :{'impulse',}
        :param args: method specific arguments

        """
        if (method is not 'impulse'):
            return ValueError("Possible value of the method argument are : {'impulse',}")
        if (room is not 'tx') and (room is not 'rx'):
            raise ValueError("Room parameter should take values {'tx', 'rx'}")

        reverberation_time = None
        if (method is 'impulse'):
            if args is not None:
                reverberation_time = self.t60_impulse(measured_impulse_response=signal, fs=fs, rt=args)
            else:
                reverberation_time = self.t60_impulse(measured_impulse_response=signal, fs=fs)

        if room is 'tx':
            self._rooms_measurements[0].average_T(reverberation_time)
        elif room is 'rx':
            self._rooms_measurements[1].average_T(reverberation_time)
        self.update_attributes()
        #self._room_acoustic_params.average_T(reverberation_time)
        return

    def DnT(self):
        D = self.compute_transmission_loss()
        T = self._rooms_measurements[1].T
        Ln = self._rooms_measurements[1].Ln

        DnT = D + 10 * np.log10(T / T_0)

        return DnT

    def compute_transmission_loss(self):
        """
        Compute the transmission loss, given the spl in the Transmitting and Receiving room has been computed.
        """
        if (len(self._rooms_measurements) is not 2):
            raise ValueError("Measurements from both the Transmitting and Receiving rooms are not avaliable")
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

        if (rt is not 't30') and (rt is not 't20') and (rt is not 't10') and (rt is not 'edt') and (rt is not 'opt'):
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
        elif rt == 'opt':
            init = -1.0
            end = -5.0
            factor = 60 / (init - end)

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
