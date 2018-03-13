# -*- coding: iso-8859-15 -*-
import Ly
import lys
import sys

import numpy as np

sys.path.append('C:/Users/sengan/Documents/Projects/BuildingAcoustics/src/scripts')
from acoustics.building_acoustics_measurement import BuildingAcousticsMeasurement
from acoustics.generator import Generator

# Module variables initialization
class info(object):
    def __init__(self):
        # Variables for settings
        # Click on the help button to get more information.
        # Example variables

        timebase_id = Ly.GetTimeBaseIDByName('Driver')
        sample_distance = Ly.GetTimeBaseSampleDistance(timebase_id)
        sampling_frequency = 1.0 /  float(sample_distance)

        self.sampling_frequency = sampling_frequency
        self.low_octave_band = 100.
        self.high_octave_band = 3150.
        self.fraction = 3
        self.probe_signal_duration = 10.
        self.probe_signal_freq_l = 100.
        self.probe_signal_freq_h = 15000.
        self.data_acquisition_delay = 2

# Temporary variables, not saved with the worksheet
class pvar(object):
    def __init__(self, sampling_frequency, probe_signal_duration, probe_signal_freq_l=100., probe_signal_freq_h=15000.,\
                low_octave_band=100., high_octave_band=3150., fraction=3):
        # Working variables
        # Click on the help button to get more information.
        # Example variables
        self.generator = Generator(fs=sampling_frequency, duration=probe_signal_duration)
        noise_type = 'sine_sweep'

        if noise_type == 'sine_sweep':
            self.probe_signal, self.reverse_signal = self.generator.noise(noise_type, [probe_signal_freq_l, probe_signal_freq_h])
        else:
            self.probe_signal = self.generator.noise(noise_type)
        self.room_response = None
        self.building_acoustics_measurement = BuildingAcousticsMeasurement(fs= sampling_frequency, f_start=low_octave_band,\
                                                                            f_stop=high_octave_band, fraction=fraction)
        self.block_count = 0

class pscript(lys.mclass):
    def __init__(self, magic):
        #self.RegisterTimeBase("NoiseGenerator", 800, "Noise generator for building sound insulation test ", UpdateTimebase)
        #timebase_id = 800
        #blockSize = 10000
        #sampleDistance = 1 / float(blockSize * 5)
        #Ly.SetTimeBase(timebase_id, blocksize, sampleDistance)
        print("## Initializing module.... ##")
        self.info = info()
        self.pvar = pvar(self.info.sampling_frequency, self.info.probe_signal_duration, self.info.probe_signal_freq_l,\
                         self.info.probe_signal_freq_h, self.info.low_octave_band, self.info.high_octave_band, self.info.fraction)
        print("## Initialized module.... ##")

    def Create (self):
        # Module initialization
        # Click on the help button to get more information.

        return True

    def Delete (self):
        # Tidy up on deletion of the module (if needed)
        # Click on the help button to get more information.
        pass

    def DlgInit (self, dlg):
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
        dm.AppendFloat("Param sampling_frequency", self.info.sampling_frequency,\
                        "Sampling frequency with which the signal is to be generated and measured")
        dm.AppendFloat("Param low_octave_band", self.info.low_octave_band, "Lowest octave band center frequency")
        dm.AppendFloat("Param high_octave_band", self.info.high_octave_band, "Highest octave band center frequency")
        dm.AppendFloat("Param fraction", self.info.fraction, "Octave band fraction")
        dm.AppendFloat("Param probe_signal_duration", self.info.probe_signal_duration, "Duration of the probe signal")
        dm.AppendFloat("Param probe_signal_freq_l", self.info.probe_signal_freq_l, "Lowest frequency of the probe signal")
        dm.AppendFloat("Param probe_signal_freq_h", self.info.probe_signal_freq_h, "Highest frequency of the probe signal")
        dm.AppendFloat("Param data_acquisition_delay", self.info.data_acquisition_delay,\
                        "Delay in blocks after which to start data acquisition")

    def DlgOk (self, dlg):
        # Get values of dialog parameters
        # Click on the help button to get more information.
        dom = lys.DialogOkManager(dlg)
        dom.SelectModulePage()
        self.info.low_octave_band = dom.GetValue("Param low_octave_band")
        self.info.high_octave_band = dom.GetValue("Param high_octave_band")
        self.info.fraction = dom.GetValue("Param fraction")
        self.info.probe_signal_duration = dom.GetValue("Param probe_signal_duration")
        self.info.probe_signal_freq_l = dom.GetValue("Param probe_signal_freq_l")
        self.info.probe_signal_freq_h = dom.GetValue("Param probe_signal_freq_h")
        self.info.data_acquisition_delay = dom.GetValue("Param data_acquisition_delay")

        self.pvar = pvar(self.info.sampling_frequency, self.info.probe_signal_duration, self.info.probe_signal_freq_l,\
                         self.info.probe_signal_freq_h, self.info.low_octave_band, self.info.high_octave_band, self.info.fraction)

        # Configure Inputs and Outputs
        # (Covers 1:1 channel relation with at least one input.
        # You need to adjust this section if you have chosen another relation
        # setting. You can find more information how to do this in the help)
        self.SetConnectors(self.DlgNumChannels, self.DlgNumChannels)

    def DlgCancel (self, dlg):
        # Cancel button clicked.
        # Click on the help button to get more information.
        pass

    def Save (self):
        # Prepare data before worksheet will be saved (if needed)
        # Click on the help button to get more information.
        pass

    def Load (self):
        # Prepare data after worksheet has been loaded (if needed)
        # Click on the help button to get more information.
        pass

    def Start (self):
        # Initialize variables on start of measurement (if needed)
        # Click on the help button to get more information.
        print("Reinitializing module....")
        self.pvar.block_count = 0

        return True

    def Stop (self):
        # Tidy up on stop of measurement (if needed)
        # Click on the help button to get more information.
        pass

    def SetupFifo (self, channel):
        # Setup flags, types and max. block size of a channel (if needed)
        # Click on the help button to get more information.
        pass

    def ProcessValue (self, v, c, N):
        # Process single value
        # Click on the help button to get more information - especially
        # if you have chosen a user defined setting because you very likely
        # need adjustments in this case!

        # r = math.exp((v+c) / N)
        r = self.pvar.estimator.probe_pulse[c + (N * self.pvar.block_count)]
        return r

    def ProcessData (self):
        # Process data blocks
        # Click on the help button to get more information - especially
        # if you have chosen a user defined setting because you very likely
        # need adjustments in this case!

        def numpyToDasylab(outBuffer, numpyDataArray):
            blockSize = outBuffer.BlockSize
            dataLength = numpyDataArray.size

            if (outBuffer == None):
                self.ShowWarning("Could not get output block! Stopped!")
                Ly.StopExperiment()
                return
            if (blockSize < dataLength):
                numpyDataArray = numpyDataArray[:blockSize]

            elif (blockSize > dataLength):
                numpyDataArray = np.append(numpyDataArray, np.zeros(blockSize - dataLength))

            for i in xrange(numpyDataArray.shape[0]):
                outBuffer[i] = numpyDataArray[i]
            return

        def dasylabToNumpy(inBuffer):
            data =  list()
            for i in xrange(inBuffer.BlockSize):
                data.append(inBuffer[i])
            return (np.asarray(data))

        # All inputs available?
        for i in range(self.NumInChannel):
            if (self.GetInputBlock(i) == None):
                return True

        # Get input blocks
        channel_data = 0
        channel_selector = 1
        delay = self.info.data_acquisition_delay # In blocks
        acquired_data_buffer = self.GetInputBlock(channel_data)
        room_selector_buffer = self.GetInputBlock(channel_selector)

        block_size = acquired_data_buffer.BlockSize
        sequence_length = self.pvar.probe_signal.size

        is_data_acquisition_done = Ly.GetVar(1)
        if not is_data_acquisition_done:
            # Process data. Acquires data with a delay of one block, since the probe signal is played after the first block
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
                        # If the probe signal length is less than the block_size. Hence the signal can't be delayed further
                        n_samples_last_block = sequence_length
                    acquired_data_block = acquired_data_block[:n_samples_last_block]
                else:
                    # Case where the generated signal block is a equal to the block_size
                    output_data_block = self.pvar.probe_signal[(block_size * self.pvar.block_count):(block_size * (self.pvar.block_count+1))]

                probe_signal_out_buffer = self.GetOutputBlock(channel_data)
                probe_signal_out_buffer.BlockSize = acquired_data_buffer.BlockSize
                probe_signal_out_buffer.StartTime = acquired_data_buffer.StartTime
                probe_signal_out_buffer.SampleDistance = acquired_data_buffer.SampleDistance

                numpyToDasylab(probe_signal_out_buffer, output_data_block)

                probe_signal_out_buffer.Release()
                # Store acquired signal. The fist block is skipped since the probe signal isn't played yet
                if (self.pvar.block_count is delay):
                    self.pvar.room_response = acquired_data_block
                elif (self.pvar.block_count > delay):
                    self.pvar.room_response = np.append(self.pvar.room_response, acquired_data_block)
                self.pvar.block_count = self.pvar.block_count + 1
            else:
                Ly.SetVar(1, 5.0) # Terminate data acquisition
                self.pvar.block_count = 0 # Reset probe signal output block count
                print("Probe Signal : {0}").format(self.pvar.probe_signal.shape)
                print("Room response : {0}").format(self.pvar.room_response.shape)

                impulse_response = self.pvar.generator.estimate_impulse_response(self.pvar.room_response[0:], self.pvar.reverse_signal)
                #np.savetxt('C:/Users/sengan/Documents/Projects/BuildingAcoustics/data/impulse_response.log', impulse_response)
                print ("Impulse response : {0}").format(impulse_response.shape)
                if room_selector_buffer[0] > 0:
                    # Transmitting room selected
                    print("Transmitting room selected")
                    self.pvar.building_acoustics_measurement.compute_spl('tx', self.pvar.room_response)
                    self.pvar.building_acoustics_measurement.compute_reverberation_time('tx', impulse_response)
                elif room_selector_buffer[0] < 0:
                    # Receiving room selected
                    print("Receiving room selected")
                    self.pvar.building_acoustics_measurement.compute_spl('rx', self.pvar.room_response)
                    self.pvar.building_acoustics_measurement.compute_reverberation_time('rx', impulse_response)
                else:
                    # No room selected
                    print("No room selected")
                print("Reverberation time:\n{0}").format(self.pvar.building_acoustics_measurement.reverberation_time)

        acquired_data_buffer.Release()
        room_selector_buffer.Release()
        return True
