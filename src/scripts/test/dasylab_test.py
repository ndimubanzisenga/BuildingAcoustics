# -*- coding: iso-8859-15 -*-
import Ly
import lys
import sys

import os
import time
import datetime
import numpy as np
from scipy.interpolate import UnivariateSpline

LOG_DATA = False
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

        timebase_id = Ly.GetTimeBaseIDByName('Driver')
        sample_distance = Ly.GetTimeBaseSampleDistance(timebase_id)
        sampling_frequency = 1.0 / float(sample_distance)

        self.sampling_frequency = sampling_frequency
        self.low_octave_band = 100.
        self.high_octave_band = 3150.
        self.fraction = 3
        self.probe_signal_duration = 10.
        self.probe_signal_freq_l = 100.
        self.probe_signal_freq_h = 10000.
        self.data_acquisition_delay = 2


# Temporary variables, not saved with the worksheet
class pvar(object):
    def __init__(self, sampling_frequency, probe_signal_duration, probe_signal_freq_l=100., probe_signal_freq_h=15000.,
                 low_octave_band=100., high_octave_band=3150., fraction=3, description=''):
        # Working variables
        # Click on the help button to get more information.
        # Example variables
        self.generator = Generator(fs=sampling_frequency, duration=probe_signal_duration)
        noise_type = 'sine_sweep'

        if noise_type == 'sine_sweep':
            self.probe_signal, self.reverse_signal = self.generator.noise(
                noise_type, [probe_signal_freq_l, probe_signal_freq_h])
        else:
            self.probe_signal = self.generator.noise(noise_type)
        self.room_response = None
        self.building_acoustics_measurement = BuildingAcousticsMeasurement(fs=sampling_frequency, f_start=low_octave_band,
                                                                           f_stop=high_octave_band, fraction=fraction)
        self.block_count = 0
        self.measurement_count = 0
        self.tx_room_measured = False
        self.rx_room_measured = False
        self.log_file_name = None

        if LOG_DATA:
            ts = time.time()
            time_stamp = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d-%H%M%S')
            self.log_dir = ('{0}/data/DasylabTests/{1}').format(ROOT_DIR, time_stamp)

            description_file_name = self.log_dir + '/MeasurementDescription.dd'
            measurement_description = str(description)
            measurement_description += ('Sampling Frequency : {0}\n').format(sampling_frequency)
            measurement_description += ('Signal Duration : {0}\n').format(probe_signal_duration)
            measurement_description += ('Noise Type : {0}\n').format(noise_type)
            measurement_description += ('Lowest Generated Frequency : {0}\n').format(probe_signal_freq_l)
            measurement_description += ('Highest Generated Frequency : {0}\n').format(probe_signal_freq_h)
            measurement_description += ('Lowest Octave Band : {0}\n').format(low_octave_band)
            measurement_description += ('Highest Octave Band : {0}\n').format(high_octave_band)
            log_data(description_file_name, measurement_description)


class pscript(lys.mclass):
    def __init__(self, magic):
        print("## Initializing module.... ##")
        self.info = info()
        self.pvar = pvar(self.info.sampling_frequency, self.info.probe_signal_duration, self.info.probe_signal_freq_l,
                         self.info.probe_signal_freq_h, self.info.low_octave_band, self.info.high_octave_band, self.info.fraction)
        Ly.SetVar(2, 0.)
        Ly.SetVar(3, 0.)
        Ly.SetVar(9, 0.)
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
        # (Covers 1:1 channel relation with at least one input.
        # You need to adjust this section if you have chosen another relation
        # setting. You can find more information how to do this in the help)
        self.DlgNumChannels = self.NumInChannel
        self.DlgMaxChannels = Ly.MAX_CHANNELS
        # Setup dialog parameters
        dm = lys.DialogManager(self, dlg)
        dm.SelectModulePage()
        dm.AppendFloat("Param sampling_frequency", self.info.sampling_frequency,
                       "Sampling frequency with which the signal is to be generated and measured")
        dm.AppendFloat("Param low_octave_band", self.info.low_octave_band, "Lowest octave band center frequency")
        dm.AppendFloat("Param high_octave_band", self.info.high_octave_band, "Highest octave band center frequency")
        dm.AppendFloat("Param fraction", self.info.fraction, "Octave band fraction")
        dm.AppendFloat("Param probe_signal_duration", self.info.probe_signal_duration, "Duration of the probe signal")
        dm.AppendFloat("Param probe_signal_freq_l", self.info.probe_signal_freq_l,
                       "Lowest frequency of the probe signal")
        dm.AppendFloat("Param probe_signal_freq_h", self.info.probe_signal_freq_h,
                       "Highest frequency of the probe signal")
        dm.AppendFloat("Param data_acquisition_delay", self.info.data_acquisition_delay,
                       "Delay in blocks after which to start data acquisition")

    def DlgOk(self, dlg):
        # Get values of dialog parameters
        # Click on the help button to get more information.
        dom = lys.DialogOkManager(dlg)
        dom.SelectModulePage()
        self.info.sampling_frequency = dom.GetValue("Param sampling_frequency")
        self.info.low_octave_band = dom.GetValue("Param low_octave_band")
        self.info.high_octave_band = dom.GetValue("Param high_octave_band")
        self.info.fraction = dom.GetValue("Param fraction")
        self.info.probe_signal_duration = dom.GetValue("Param probe_signal_duration")
        self.info.probe_signal_freq_l = dom.GetValue("Param probe_signal_freq_l")
        self.info.probe_signal_freq_h = dom.GetValue("Param probe_signal_freq_h")
        self.info.data_acquisition_delay = dom.GetValue("Param data_acquisition_delay")

        self.pvar = pvar(self.info.sampling_frequency, self.info.probe_signal_duration, self.info.probe_signal_freq_l,
                         self.info.probe_signal_freq_h, self.info.low_octave_band, self.info.high_octave_band, self.info.fraction)

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
        print("## Reinitializing module.... ##")
        self.pvar.building_acoustics_measurement.initialize_room_measurement()
        self.pvar.room_response = None
        self.pvar.block_count = 0
        print("## Reinitialized module.... ##")

        return True

    def Stop(self):
        # Tidy up on stop of measurement (if needed)
        # Click on the help button to get more information.
        pass

    def SetupFifo(self, channel):
        # Setup flags, types and max. block size of a channel (if needed)
        # Click on the help button to get more information.
        pass

    def ProcessValue(self, v, c, N):
        # Process single value
        # Click on the help button to get more information - especially
        # if you have chosen a user defined setting because you very likely
        # need adjustments in this case!

        # r = math.exp((v+c) / N)
        r = self.pvar.estimator.probe_pulse[c + (N * self.pvar.block_count)]
        return r

    def ProcessData(self):
        # Process data blocks
        # Click on the help button to get more information - especially
        # if you have chosen a user defined setting because you very likely
        # need adjustments in this case!

        def numpyToDasylab(outBuffer, referenceBuffer, numpyDataArray):
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
            data = list()
            for i in xrange(inBuffer.BlockSize):
                data.append(inBuffer[i])
            return (np.asarray(data))

        def dilateArray(oldArray, newLength):
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
        delay = self.info.data_acquisition_delay  # In blocks
        acquired_data_buffer = self.GetInputBlock(0)
        room_selector_buffer = self.GetInputBlock(1)
        room_selector_buffer_1 = self.GetInputBlock(2)
        room_selector_buffer_2 = self.GetInputBlock(3)
        room_selector_buffer_3 = self.GetInputBlock(4)
        room_selector_buffer_4 = self.GetInputBlock(5)

        block_size = acquired_data_buffer.BlockSize
        sequence_length = self.pvar.probe_signal.size

        is_data_acquisition_done = Ly.GetVar(1)
        if not is_data_acquisition_done:
            # Process data. Acquires data with a delay of 'delay' blocks, since the probe signal is played after the first block.
            # Hence the time it takes to play and start recording the signal is taken into account
            if (block_size * (self.pvar.block_count - delay)) < sequence_length:
                acquired_data_block = dasylabToNumpy(acquired_data_buffer)
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
                numpyToDasylab(probe_signal_out_buffer, acquired_data_buffer, output_data_block)

                # Store acquired signal. Storage starts after a delay since the probe signal isn't played yet.
                if (self.pvar.block_count is delay):
                    self.pvar.measurement_count += 1
                    self.pvar.room_response = acquired_data_block
                    if LOG_DATA:
                        self.log_file_name = ('{0}/TestData-{1}.data').format(self.pvar.log_dir,
                                                                              self.pvar.measurement_count)

                elif (self.pvar.block_count > delay):
                    self.pvar.room_response = np.append(self.pvar.room_response, acquired_data_block)
                self.pvar.block_count = self.pvar.block_count + 1
            else:
                Ly.SetVar(1, 5.0)  # Terminate data acquisition
                self.pvar.block_count = 0  # Reset probe signal output block count
                impulse_response = self.pvar.generator.estimate_impulse_response(
                    self.pvar.room_response[0:], self.pvar.reverse_signal)
                impulse_response = impulse_response[:self.info.sampling_frequency * 1]

                if (LOG_DATA) and (self.log_file_name is not None):
                    # ToDo add swtich within Dasylab
                    log_data(self.log_file_name, self.pvar.room_response)

                if room_selector_buffer[0] > 0:
                    # Transmitting room selected
                    self.pvar.building_acoustics_measurement.compute_spl('tx', self.pvar.room_response)
                    self.pvar.building_acoustics_measurement.compute_reverberation_time(
                        'tx', signal=impulse_response, args='t10')
                    self.pvar.tx_room_measured = True
                elif room_selector_buffer[0] < 0:
                    # Receiving room selected
                    self.pvar.building_acoustics_measurement.compute_spl('rx', self.pvar.room_response)
                    self.pvar.building_acoustics_measurement.compute_reverberation_time(
                        'rx', signal=impulse_response, args='t10')
                    self.pvar.rx_room_measured = True
                else:
                    # No room selected
                    print("No room selected")

                if self.pvar.tx_room_measured and self.pvar.rx_room_measured:
                    D_nT = self.pvar.building_acoustics_measurement.DnT()
                    D_nT_w, _ = self.pvar.building_acoustics_measurement.compute_single_number(D_nT)
                    Ly.SetVar(2, D_nT_w)

                    S = Ly.GetVar(4)
                    V = Ly.GetVar(5)
                    R = self.pvar.building_acoustics_measurement.R(S, V)
                    R_w, ref_curve = self.pvar.building_acoustics_measurement.compute_single_number(R)
                    Ly.SetVar(3, R_w)

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

                    D_nT = dilateArray(D_nT, block_size)
                    D_nT_out_buffer = self.GetOutputBlock(1)
                    numpyToDasylab(D_nT_out_buffer, acquired_data_buffer, D_nT)

                    ref_curve = dilateArray(ref_curve, block_size)
                    ref_curve_out_buffer = self.GetOutputBlock(5)
                    numpyToDasylab(ref_curve_out_buffer, acquired_data_buffer, ref_curve)

                reverberation_time = self.pvar.building_acoustics_measurement.reverberation_time
                tx_room_spl = self.pvar.building_acoustics_measurement.tx_room_spl
                rx_room_spl = self.pvar.building_acoustics_measurement.rx_room_spl
                octave_bands = self.pvar.building_acoustics_measurement.octave_bands

                # print("Reverberation time:\n{0}").format(reverberation_time)
                # print("Tx Room SPL:\n{0}").format(tx_room_spl)
                # print("Rx Room SPL:\n{0}").format(rx_room_spl)
                # print("Octave bands \n{0}").format(octave_bands)

                # reverberation_time = dilateArray(reverberation_time, block_size)
                # reverberation_time_out_buffer = self.GetOutputBlock(1)
                # numpyToDasylab(reverberation_time_out_buffer, acquired_data_buffer, reverberation_time)

                tx_room_spl = dilateArray(tx_room_spl, block_size)
                reverberation_time_out_buffer = self.GetOutputBlock(2)
                numpyToDasylab(reverberation_time_out_buffer, acquired_data_buffer, tx_room_spl)

                rx_room_spl = dilateArray(rx_room_spl, block_size)
                reverberation_time_out_buffer = self.GetOutputBlock(3)
                numpyToDasylab(reverberation_time_out_buffer, acquired_data_buffer, rx_room_spl)

                octave_bands = dilateArray(octave_bands, block_size)
                octave_bands_out_buffer = self.GetOutputBlock(4)
                numpyToDasylab(octave_bands_out_buffer, acquired_data_buffer, octave_bands)

        acquired_data_buffer.Release()
        room_selector_buffer.Release()
        room_selector_buffer_1.Release()
        room_selector_buffer_2.Release()
        room_selector_buffer_3.Release()
        room_selector_buffer_4.Release()
        return True
