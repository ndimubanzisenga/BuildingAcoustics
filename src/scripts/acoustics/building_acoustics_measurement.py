"""
Building Acoustic Measurement
=============================
This module implements routines for the measurement of parameters to describe building acoustics
"""
from __future__ import division
from signal import Spectrum, OctaveBand, Signal
from scipy import stats
import numpy as np

# ROOT_DIR = ""
from sys import platform
if platform == "win32":
    ROOT_DIR = 'C:/Users/sengan/Documents/Projects/BuildingAcoustics/'
elif platform == "linux2":
    ROOT_DIR = '../../../'


def load_regulations():
    import json
    regulations_file = ROOT_DIR + '/data/building_acoustics_regulations_germany.json'
    data = None

    with open(regulations_file) as json_data_file:
        data = json.load(json_data_file)
    return data


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
        # ToDo Generalize to multiple channels measurement
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

        self.octave_bands = OctaveBand(fstart=f_start, fstop=f_stop, fraction=fraction).center
        self.tx_room_spl = None
        self.rx_room_spl = None
        self.reverberation_time = None

        self.initialize_room_measurement()

    def initialize_room_measurement(self):
        self._rooms_measurements = list()
        bands_number = self.octave_bands.size
        tx_room_acoustic_params = AcousticParameters(bands_number)
        rx_room_acoustic_params = AcousticParameters(bands_number)
        self._rooms_measurements.append(tx_room_acoustic_params)
        self._rooms_measurements.append(rx_room_acoustic_params)
        self.update_attributes()
        self._regulatations = load_regulations()
        #self._room_acoustic_params = AcousticParameters(bands_number)

        return

    def finalize_room_measurement(self):
        self._rooms_measurements.append(self._room_acoustic_params)
        return

    def compute_building_acoustics_parameters(self):
        """
        Calculate building acoustics parameters: The Transmitting room and Receiving room spl,
        the background noise level and reverberation time.
        """
        pass

    def verify_building_acoustics_regulation(self, Rw, tolerance, building_use, building_type, test_element_type):
        """
        Verify whether the measured building performance complies with the building acoustics regulations.
        :param Rw: measured  sound reduction index single number.
        :param tolerance: tolerance in dB by which the measured Rw can deviate from the nominal Rw.
        :param building_use: Usage of the building being tested. Possible values are : {'ResidentialAndOffice', 'NonResidential'}.
        :param building_type: Type of the building being tested. Possible values are : {'MultiStorey', 'DetachedHouse',\
                              'Hotel', 'Hospital', 'School' }.
        :param test_element_type: Type of the building element under test. Possible values : {'Ceiling', 'Wall', 'Door'}

        :return status: Status of the building element under test in regards to the building acoustics regulation.
        """
        status = None
        Rw_nominal = self._regulatations[building_use][building_type][test_element_type]
        diff = Rw - Rw_nominal
        if (diff > 0) and (abs(diff) > tolerance):
            status = 5
        elif (diff > 0) and (abs(diff) < tolerance):
            status = 0
        elif (diff < 0) and (abs(diff) < tolerance):
            status = 0
        elif (diff < 0) and (abs(diff) > tolerance):
            status = -5

        return status

    def diagnose_defect(self):
        pass

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

        f_start = self._f_start
        f_stop = self._f_stop
        fraction = self._fraction
        octave_bands = self.octave_bands

        spectrum = Spectrum()
        _, octaves_power, octaves_power_levels = spectrum.third_octaves(signal, self._fs, frequencies=octave_bands)
        if room is 'tx':
            self._rooms_measurements[0].average_L(octaves_power_levels)
        elif room is 'rx':
            self._rooms_measurements[1].average_L(octaves_power_levels)
        self.update_attributes()
        # self._room_acoustic_params.average_L(octaves_power_levels)
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
        # self._room_acoustic_params.average_T(reverberation_time)
        return

    def DnT(self, T_0=0.5):
        """
        Calculate the Standardized Sound Level Difference (D_nT).
        """
        D = self.tx_room_spl - self.rx_room_spl
        Ln = self._rooms_measurements[1].Ln
        T = self.reverberation_time
        if np.alltrue(T == 0):
            # If reverberation_time has not been estimated set its value to 0.5 sec for all frequency bands
            # ToDo: Better logic to check whether reverberation_time has been estimated
            T += T_0
        D_nT = D + 10 * np.log10(T / T_0)

        return D_nT

    def R(self, S, V):
        """
        Calculate the Apparent Sound Reduction Index (R').

        :param S: area of the common partition between the source and the receiving room in {m^2}.
        :param V: volume of the receiving room in {m^3}.
        """
        D = self.tx_room_spl - self.rx_room_spl
        Ln = self._rooms_measurements[1].Ln
        T = self.reverberation_time
        T_ref = 0.5
        # If reverberation_time has not estimated for a frequency band, set its value to 0.5 sec. This also avoids division by zero.
        # ToDo: investigate which value is best as default value instead of 0.5
        for i in xrange(T.size):
            if T[i] == 0:
                T[i] = T_ref
        A = 0.16 * (V / T)
        R = D + 10 * np.log10(S / A)

        return R

    def compute_single_number(self, transmission_loss):
        """
        Calculate the reference curve of :math:`Rw` from a NumPy array `transmission_loss` with third
        octave data between 100 Hz and 3.15 kHz.

        :param transmission_loss: Transmission Loss

        """
        t = transmission_loss

        # ToDo: Function to compute the reference curve for a dynamic freq. range
        ref_curve = np.array([[0, 3, 6, 9, 12, 15, 18, 19, 20, 21, 22, 23, 23, 23, 23, 23]])

        # Workaround to handle actave bands beyond 3150 Hz center.
        # ToDo: Handle  center frequencies beyond the range [100., 3150.] for octave and third_octaves.
        if t.size > ref_curve.size:
            t = t[:ref_curve.size]
        elif t.size < ref_curve.size:
            ref_curve = ref_curve[:t.size]

        # Check which direction to move the ref_curve
        direction = np.sign(np.sum(t - ref_curve))

        residuals_sum = 0

        if direction > 0:
            # Move the ref_curve updward towards the transmission_loss curve
            while residuals_sum > -32:
                ref_curve += 1
                diff = t - ref_curve
                residuals = direction * np.clip(diff, np.min(diff), 0)
                residuals_sum = np.sum(residuals)
        else:
            # Move the ref_curve downward towards the transmission_loss curve
            while residuals_sum < -32:
                ref_curve -= 1
                diff = t - ref_curve
                residuals = direction * np.clip(diff, np.min(diff), 0)
                residuals_sum = np.sum(residuals)
        ref_curve -= 1

        return ref_curve[0, 7], ref_curve

    def compute_adaptation_terms(self, tl, single_number):
        """
        Calculate the adaptation term as defined in ISO 717-1.

        :param tl: Transmission loss
        :param single_number: Measured single number quantity. Can be DnT,w or Rw.
        """
        c = int(round(self.rw_c(tl) - single_number))
        c_tr = int(round(self.rw_ctr(tl) - single_number))
        return c, c_tr

    def rw_c(self, tl):
        """
        Calculate :math:`R_W + C` from a NumPy array `tl` with third octave data
        between 100 Hz and 3.15 kHz.

        :param tl: Transmission Loss
        """
        k = np.array([-29, -26, -23, -21, -19, -17, -15, -13, -12, -11, -10, -9,
                      -9, -9, -9, -9])
        a = -10 * np.log10(np.sum(10**((k - tl) / 10)))
        return a

    def rw_ctr(self, tl):
        """
        Calculate :math:`R_W + C_{tr}` from a NumPy array `tl` with third octave
        data between 100 Hz and 3.15 kHz.

        :param tl: Transmission Loss
        """
        k_tr = np.array([-20, -20, -18, -16, -15, -14, -13, -12, -11, -9, -8, -9,
                         -10, -11, -13, -15])
        a_tr = -10 * np.log10(np.sum(10**((k_tr - tl) / 10)))
        return a_tr

    def t60_impulse(self, measured_impulse_response=None, fs=None, rt='t30', test_octave_band=0):
        """
        Reverberation time from a measured impulse response.

        :param measured_impulse_response: Numpy array of the measured impulse response.
        :param rt: Reverberation time estimator. It accepts `'t30'`, `'t20'`, `'t10'` and `'edt'`.
        :returns: Reverberation time :math:`T_{60}`

        """
        if (measured_impulse_response is None):
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
            filtered_signal = signal_processor.bandpass(
                measured_impulse_response, bands.lower[i], bands.upper[i], fs, order=8)
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
            m, b, r_value, p_value, std_err = stats.linregress(x, y)  # y = m*x + b

            # Reverberation time (T30, T20, T10 or EDT)
            db_regress_init = (init - b) / m
            db_regress_end = (end - b) / m
            t60[i] = factor * (db_regress_end - db_regress_init)

            if i == test_octave_band:
                y_ = m * x + b
                self.bandpass_filtered_ir = filtered_signal
                self.schroeder_curve = sch
                self.regression_result = [x, y, y_]

        return t60
