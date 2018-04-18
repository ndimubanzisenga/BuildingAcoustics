# -*- coding: iso-8859-15 -*-
import Ly
import lys
import sys

import os
import time
import datetime
import numpy as np
from scipy.interpolate import UnivariateSpline

ROOT_DIR = 'C:/Users/sengan/Documents/Projects/BuildingAcoustics/'
BUILDING_TYPES = {0: 'MultiStorey', 1: 'DetachedHouse', 2: 'Hotel', 3: 'Hospital', 4: 'School'}
ELEMENT_TYPES = {0: 'Ceiling', 1: 'Wall', 2: 'Door'}

sys.path.append(ROOT_DIR + 'src/scripts')
from acoustics.building_acoustics_measurement import BuildingAcousticsMeasurement
from acoustics.generator import Generator


def log_data(fileName, data):
    '''
    Save data to file.
    If data is of type numpy.ndarray, save it using numpy.savetxt.
    If data is string, save it using standard python file handle.
    '''
    if not os.path.exists(os.path.dirname(fileName)):
        os.makedirs(os.path.dirname(fileName))

    if type(data) is str:
        text_file = open(fileName, 'w')
        text_file.write(data)
        text_file.close()
    elif type(data) is np.ndarray:
        np.savetxt(fileName, data)
    else:
        raise ValueError("Can't log data of unknown type.")
    return

# Module variables initialization


class info(object):
    def __init__(self):
        # Variables for settings
        # Click on the help button to get more information.
        # Example variables
        pass


# Temporary variables, not saved with the worksheet
class pvar(object):
    def __init__(self, probe_signal_duration=10., probe_signal_freq_l=100., probe_signal_freq_h=10000.,
                 low_octave_band=100., high_octave_band=3150., fraction=3, data_acquisition_delay=2, measurement_description=''):
        # Working variables
        # Click on the help button to get more information.
        # Example variables
        self.low_octave_band = low_octave_band
        self.high_octave_band = high_octave_band
        self.fraction = fraction
        self.probe_signal_duration = probe_signal_duration
        self.probe_signal_freq_l = probe_signal_freq_l
        self.probe_signal_freq_h = probe_signal_freq_h
        self.data_acquisition_delay = data_acquisition_delay
        self.measurement_description =  measurement_description
        self.initialize_test()

    def get_sampling_frequency(self):
        timebase_id = Ly.GetTimeBaseIDByName('Driver')
        sample_distance = Ly.GetTimeBaseSampleDistance(timebase_id)
        sampling_frequency = 1.0 / float(sample_distance)

        return sampling_frequency

    def initialize_test_signal(self, signal_type='sine_sweep'):
        self.sampling_frequency = self.get_sampling_frequency()

        self.signal_type = signal_type
        self.generator = Generator(fs=self.sampling_frequency, duration=self.probe_signal_duration)
        if signal_type == 'sine_sweep':
            self.probe_signal, self.reverse_signal = self.generator.noise(signal_type, args=[self.probe_signal_freq_l, self.probe_signal_freq_h])
        else:
            self.probe_signal = self.generator.noise(signal_type)

        return

    def initialize_measurement(self):
        self.room_response = None
        self.tx_room_measured = False
        self.rx_room_measured = False
        self.block_count = 0
        self.measurement_count = 0
        self.building_acoustics_measurement = BuildingAcousticsMeasurement(fs=self.sampling_frequency, f_start=self.low_octave_band,
                                                                           f_stop=self.high_octave_band, fraction=self.fraction)
        return

    def initialize_test(self):
        self.initialize_test_signal()
        self.initialize_measurement()

        return


class pscript(lys.mclass):
    def __init__(self, magic):
        print("## Initializing module.... ##")
        self.info = info()
        self.pvar = pvar()
        self.log_dir = None
        self.log_file_name = None
        print("## Initialized module.... ##")

    def Create(self):
        # Module initialization
        # Click on the help button to get more information.

        return True

    def Delete(self):
        # Tidy up on deletion of the module (if needed)
        # Click on the help button to get more information.
        pass

    def DlgInit(self, dlg):
        # Initialization of settings dialog
        # Click on the help button to get more information.

        # Set dialog title
        dlg.title = "Script (Bulding Acoustic Meas.)"

        # Determine the number of channels, current channel and
        # maximum number of channels
        # (Covers 1:1 channel relation with at least one input.)
        self.DlgNumChannels = self.NumInChannel
        self.DlgMaxChannels = Ly.MAX_CHANNELS

        # Setup dialog parameters
        dm = lys.DialogManager(self, dlg)
        dm.SelectModulePage()
        dm.AppendFloat("Param sampling_frequency", self.pvar.sampling_frequency,
                       "Sampling frequency with which the signal is to be generated and measured")
        dm.AppendFloat("Param low_octave_band", self.pvar.low_octave_band, "Lowest octave band center frequency")
        dm.AppendFloat("Param high_octave_band", self.pvar.high_octave_band, "Highest octave band center frequency")
        dm.AppendFloat("Param fraction", self.pvar.fraction, "Octave band fraction")
        dm.AppendFloat("Param probe_signal_duration", self.pvar.probe_signal_duration, "Duration of the probe signal")
        dm.AppendFloat("Param probe_signal_freq_l", self.pvar.probe_signal_freq_l,
                       "Lowest frequency of the probe signal")
        dm.AppendFloat("Param probe_signal_freq_h", self.pvar.probe_signal_freq_h,
                       "Highest frequency of the probe signal")
        dm.AppendFloat("Param data_acquisition_delay", self.pvar.data_acquisition_delay,
                       "Delay in blocks after which to start data acquisition")

    def DlgOk(self, dlg):
        # Get values of dialog parameters
        # Click on the help button to get more information.
        dom = lys.DialogOkManager(dlg)
        dom.SelectModulePage()

        # self.pvar.sampling_frequency = dom.GetValue("Param sampling_frequency")
        self.pvar.low_octave_band = dom.GetValue("Param low_octave_band")
        self.pvar.high_octave_band = dom.GetValue("Param high_octave_band")
        self.pvar.fraction = dom.GetValue("Param fraction")
        self.pvar.probe_signal_duration = dom.GetValue("Param probe_signal_duration")
        self.pvar.probe_signal_freq_l = dom.GetValue("Param probe_signal_freq_l")
        self.pvar.probe_signal_freq_h = dom.GetValue("Param probe_signal_freq_h")
        self.pvar.data_acquisition_delay = dom.GetValue("Param data_acquisition_delay")

        self.pvar.initialize_test()

        # Configure Inputs and Outputs
        # (Covers 1:1 channel relation with at least one input.
        # You need to adjust this section if you have chosen another relation
        # setting. You can find more information how to do this in the help)
        self.SetConnectors(self.DlgNumChannels, self.DlgNumChannels)

    def DlgCancel(self, dlg):
        # Cancel button clicked.
        # Click on the help button to get more information.
        pass

    def Save(self):
        # Prepare data before worksheet will be saved (if needed)
        # Click on the help button to get more information.
        pass

    def Load(self):
        # Prepare data after worksheet has been loaded (if needed)
        # Click on the help button to get more information.
        pass

    def Start(self):
        # Initialize variables on start of measurement (if needed)
        # Click on the help button to get more information.
        # self.pvar.initialize_measurement()

        # Initialize global variables
        Ly.SetVar(2, 0.) # DnT,w
        Ly.SetVar(3, 0.) # R'w
        Ly.SetVar(9, 5.) # violation_status
        Ly.SetVar(13, 0.) # Source room selector
        Ly.SetVar(14, 0.) # Receiving room selector

        # Handle the mismatch of sampling_frequency when worksheet is loaded.
        # ToDo: Find permanent solution.
        # print (" Start - before fix:: FS = {0}").format(self.pvar.sampling_frequency)
        if (self.pvar.sampling_frequency != self.pvar.get_sampling_frequency()):
            self.pvar.initialize_test()
        else:
            self.pvar.initialize_measurement()
        # print (" Start - after fix:: FS = {0}").format(self.pvar.sampling_frequency)

        self._LOG_DATA = Ly.GetVar(12)

        # Log measurement description data to file
        if self._LOG_DATA:
            ts = time.time()
            time_stamp = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d-%H%M%S')
            self.log_dir = ('{0}/data/DasylabTests/{1}').format(ROOT_DIR, time_stamp)

            description_file_name = self.log_dir + '/MeasurementDescription.dd'
            measurement_description = str(self.pvar.measurement_description)
            measurement_description += ('Sampling Frequency : {0}\n').format(self.pvar.sampling_frequency)
            measurement_description += ('Signal Duration : {0}\n').format(self.pvar.probe_signal_duration)
            measurement_description += ('Noise Type : {0}\n').format(self.pvar.signal_type)
            measurement_description += ('Lowest Generated Frequency : {0}\n').format(self.pvar.probe_signal_freq_l)
            measurement_description += ('Highest Generated Frequency : {0}\n').format(self.pvar.probe_signal_freq_h)
            measurement_description += ('Lowest Octave Band : {0}\n').format(self.pvar.low_octave_band)
            measurement_description += ('Highest Octave Band : {0}\n').format(self.pvar.high_octave_band)
            log_data(description_file_name, measurement_description)

        return True

    def Stop(self):
        # Tidy up on stop of measurement (if needed)
        # Click on the help button to get more information.
        pass

    def SetupFifo(self, channel):
        # Setup flags, types and max. block size of a channel (if needed)
        # Click on the help button to get more information.
        pass

    def ProcessValue(self):
        # Process single value
        # Click on the help button to get more information - especially
        # if you have chosen a user defined setting because you very likely
        # need adjustments in this case!
        pass

    def ProcessData(self):
        # Process data blocks
        # Click on the help button to get more information - especially
        # if you have chosen a user defined setting because you very likely
        # need adjustments in this case!

        def numpyToDasylab(outBuffer, referenceBuffer, numpyDataArray):
            """
            Convert a numpy::array to a Dasylab buffer.
            :param outBuffer: Reference to buffere to be created.
            :param referenceBuffer: Buffer to be used as reference to assign the time base properties to the new bufffer.
            :param numpyDataArray: numpy::array to fill the new buffer.
            """
            outBuffer.BlockSize = referenceBuffer.BlockSize
            outBuffer.StartTime = referenceBuffer.StartTime
            outBuffer.SampleDistance = referenceBuffer.SampleDistance

            blockSize = outBuffer.BlockSize
            dataLength = numpyDataArray.size

            if (outBuffer is None):
                self.ShowWarning("Could not get output block! Stopped!")
                Ly.StopExperiment()
                return
            if (blockSize < dataLength):
                numpyDataArray = numpyDataArray[:blockSize]

            elif (blockSize > dataLength):
                numpyDataArray = np.append(numpyDataArray, np.zeros(blockSize - dataLength))

            for i in xrange(numpyDataArray.shape[0]):
                outBuffer[i] = numpyDataArray[i]
            outBuffer.Release()
            return

        def dasylabToNumpy(inBuffer):
            """
            Convert a Dasylab buffer to a numpy array
            :param inBuffer: Buffer to convert into a numpy::array

            return numpy::array
            """
            data = list()
            for i in xrange(inBuffer.BlockSize):
                data.append(inBuffer[i])
            return (np.asarray(data))

        def dilateArray(oldArray, newLength):
            """
            Dilate a an array so as to increase its size.
            :param oldArray: numpy::array to dilate.
            :param newLength: Length the array is to be delated to.
            """
            # ToDo: What happens when newLength < oldArray.size ?
            oldIndices = np.arange(0, oldArray.size)
            newIndices = np.linspace(0, (oldArray.size - 1), newLength)
            spl = UnivariateSpline(oldIndices, oldArray, k=1, s=0)
            newArray = spl(newIndices)
            return newArray

        # All inputs available?
        for i in range(self.NumInChannel):
            if (self.GetInputBlock(i) is None):
                return True

        # Get input blocks
        delay = self.pvar.data_acquisition_delay  # In blocks
        acquired_data_buffer_0 = self.GetInputBlock(0)
        acquired_data_buffer_1 = self.GetInputBlock(1)
        acquired_data_buffer_2 = self.GetInputBlock(2)
        acquired_data_buffer_3 = self.GetInputBlock(3)
        acquired_data_buffer_4 = self.GetInputBlock(4)
        acquired_data_buffer_5 = self.GetInputBlock(5)
        acquired_data_buffer_6 = self.GetInputBlock(6)

        block_size = acquired_data_buffer_0.BlockSize
        sequence_length = self.pvar.probe_signal.size

        is_data_acquisition_done = Ly.GetVar(1)
        if not is_data_acquisition_done:
            # Process data. Acquires data with a delay of 'delay' blocks, since the probe signal is played after the first block.
            # Hence the time it takes to play and start recording the signal is taken into account
            if (block_size * (self.pvar.block_count - delay)) < sequence_length:
                acquired_data_block = dasylabToNumpy(acquired_data_buffer_0)
                output_data_block = None

                if (sequence_length - block_size * self.pvar.block_count) < block_size:
                    # Check if this is the last block of the generated signal
                    # Check if the last block of the generated signal has fewer samples than the block_size
                    if (sequence_length - block_size * self.pvar.block_count) <= 0:
                        # In case the whole probe signal has been played
                        output_data_block = np.zeros(block_size)
                    else:
                        # If this is the last block in the probe signal
                        output_data_block = self.pvar.probe_signal[(block_size * self.pvar.block_count):]
                    n_samples_last_block = sequence_length - block_size * (self.pvar.block_count - delay)
                    if self.pvar.block_count < delay:
                        # If the probe signal length is less than the block_size. Hence the signal
                        # can't be delayed further
                        n_samples_last_block = sequence_length
                    acquired_data_block = acquired_data_block[:n_samples_last_block]
                else:
                    # Case where the generated signal block is a equal to the block_size
                    output_data_block = self.pvar.probe_signal[(
                        block_size * self.pvar.block_count):(block_size * (self.pvar.block_count + 1))]

                probe_signal_out_buffer = self.GetOutputBlock(0)
                numpyToDasylab(probe_signal_out_buffer, acquired_data_buffer_0, output_data_block)

                # Store acquired signal. Storage starts after a delay since the probe signal isn't played yet.
                if (self.pvar.block_count is delay):
                    self.pvar.measurement_count += 1
                    self.pvar.room_response = acquired_data_block
                    if self._LOG_DATA:
                        self.log_file_name = ('{0}/Measurement-{1}.data').format(self.log_dir, self.pvar.measurement_count)

                elif (self.pvar.block_count > delay):
                    self.pvar.room_response = np.append(self.pvar.room_response, acquired_data_block)
                self.pvar.block_count = self.pvar.block_count + 1
            else:

                impulse_response = self.pvar.generator.estimate_impulse_response(self.pvar.room_response[0:], self.pvar.reverse_signal)
                # Trim IR to 1 sec
                # ToDo: make this duration dynamic
                impulse_response = impulse_response[:int(self.pvar.sampling_frequency * 1)]

                if Ly.GetVar(13):
                    # Source room selected
                    self.pvar.building_acoustics_measurement.compute_spl('tx', self.pvar.room_response)
                    self.pvar.building_acoustics_measurement.compute_reverberation_time(
                        'tx', signal=impulse_response, args='t10')
                    self.pvar.tx_room_measured = True

                elif Ly.GetVar(14):
                    # Receiving room selected
                    self.pvar.building_acoustics_measurement.compute_spl('rx', self.pvar.room_response)
                    self.pvar.building_acoustics_measurement.compute_reverberation_time(
                        'rx', signal=impulse_response, args='t10')
                    self.pvar.rx_room_measured = True

                # If both source and receiving rooms have been measured, output results
                # ToDo: Result switch in Dasylab to enable output of results
                if self.pvar.tx_room_measured and self.pvar.rx_room_measured:
                    # Calculate DnT,w
                    D_nT = self.pvar.building_acoustics_measurement.DnT()
                    D_nT_w, ref_curve = self.pvar.building_acoustics_measurement.compute_single_number(D_nT)
                    Ly.SetVar(2, D_nT_w)

                    # Calculate R'w
                    S = Ly.GetVar(4)
                    V = Ly.GetVar(5)
                    R = self.pvar.building_acoustics_measurement.R(S, V)
                    R_w, ref_curve = self.pvar.building_acoustics_measurement.compute_single_number(R)
                    Ly.SetVar(3, R_w)

                    # Check if building regulations are satisfied. Currently only using R'w.
                    building = int(Ly.GetVar(6))
                    test_element = int(Ly.GetVar(7))
                    tolerance = Ly.GetVar(9)
                    # ToDo: remove this hack, and Generalize to get building usage
                    if building < 2:
                        building_use = 'ResidentialAndOffice'
                    else:
                        building_use = 'NonResidential'

                    violation_status = self.pvar.building_acoustics_measurement.verify_building_acoustics_regulation(
                        R_w, tolerance, building_use, BUILDING_TYPES[building], ELEMENT_TYPES[test_element])
                    Ly.SetVar(9, violation_status)

                    # Output results.
                    D_nT = dilateArray(D_nT, block_size)
                    D_nT_out_buffer = self.GetOutputBlock(2)
                    numpyToDasylab(D_nT_out_buffer, acquired_data_buffer_0, D_nT)

                    R = dilateArray(R, block_size)
                    R_out_buffer = self.GetOutputBlock(3)
                    numpyToDasylab(R_out_buffer, acquired_data_buffer_0, R)

                    ref_curve = dilateArray(ref_curve, block_size)
                    ref_curve_out_buffer = self.GetOutputBlock(4)
                    numpyToDasylab(ref_curve_out_buffer, acquired_data_buffer_0, ref_curve)

                    # Log acquired Room Response to file
                    if (self._LOG_DATA) and (self.log_file_name is not None):
                        log_data(self.log_file_name, self.pvar.room_response)

                # Output computed building acoustic parameters
                octave_bands = self.pvar.building_acoustics_measurement.octave_bands
                reverberation_time = self.pvar.building_acoustics_measurement.reverberation_time
                tx_room_spl = self.pvar.building_acoustics_measurement.tx_room_spl
                rx_room_spl = self.pvar.building_acoustics_measurement.rx_room_spl

                # reverberation_time = dilateArray(reverberation_time, block_size)
                # reverberation_time_out_buffer = self.GetOutputBlock(1)
                # numpyToDasylab(reverberation_time_out_buffer, acquired_data_buffer_0, reverberation_time)

                octave_bands = dilateArray(octave_bands, block_size)
                octave_bands_out_buffer = self.GetOutputBlock(1)
                numpyToDasylab(octave_bands_out_buffer, acquired_data_buffer_0, octave_bands)

                tx_room_spl = dilateArray(tx_room_spl, block_size)
                tx_room_spl_out_buffer = self.GetOutputBlock(5)
                numpyToDasylab(tx_room_spl_out_buffer, acquired_data_buffer_0, tx_room_spl)

                rx_room_spl = dilateArray(rx_room_spl, block_size)
                rx_room_spl_out_buffer = self.GetOutputBlock(6)
                numpyToDasylab(rx_room_spl_out_buffer, acquired_data_buffer_0, rx_room_spl)

                # Terminate acquisition
                # ToDo: Disable room selector global variables as well. But first fix the the result enabling.
                self.pvar.block_count = 0  # Reset probe signal processed block count
                Ly.SetVar(1, 5.0)  # Terminate data acquisition

        acquired_data_buffer_0.Release()
        acquired_data_buffer_1.Release()
        acquired_data_buffer_2.Release()
        acquired_data_buffer_3.Release()
        acquired_data_buffer_4.Release()
        acquired_data_buffer_5.Release()
        acquired_data_buffer_6.Release()
        return True
