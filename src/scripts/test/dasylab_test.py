# -*- coding: iso-8859-15 -*-
import Ly
import lys
import sys

import numpy as np
from scipy.fftpack import fft, ifft
from scipy.signal import fftconvolve, convolve
import math

class RoomResponseEstimator(object):
    """
    Gives probe impulse, gets response and calculate impulse response.
    Method from paper: "Simultaneous measurement of impulse response and distortion with a swept-sine technique"
    Angelo Farina
    """

    def __init__(self, duration=10.0, low=100.0, high=15000.0, Fs=44100.0):
        self.Fs = Fs

        # Total length in samples
        self.T = Fs*duration
        self.w1 = low / self.Fs * 2*np.pi
        self.w2 = high / self.Fs * 2*np.pi

        self.probe_pulse = self.probe()
        #self.probe_pulse = np.random.random_sample(self.T)

        # Apply exponential signal on the beginning and the end of the probe signal.
        exp_window = 1-np.exp(np.linspace(0,-10, 5000))
        self.probe_pulse[:exp_window.shape[0]] *= exp_window
        self.probe_pulse[-exp_window.shape[0]:] *= exp_window[-1::-1]

        # This is what the value of K will be at the end (in dB):
        kend = 10**((-6*np.log2(self.w2/self.w1))/20)
        # dB to rational number.
        k = np.log(kend)/self.T

        # Making reverse probe impulse so that convolution will just
        # calculate dot product. Weighting it with exponent to acheive
        # 6 dB per octave amplitude decrease.
        self.reverse_pulse = self.probe_pulse[-1::-1] * \
            np.array(list(\
                map(lambda t: np.exp(float(t)*k), range(int(self.T)))\
                )\
            )

        # Now we have to normilze energy of result of dot product.
        # This is "naive" method but it just works.
        corr = fftconvolve(self.reverse_pulse, self.probe_pulse)
        Frp =  fft(corr)
        self.reverse_pulse /= np.abs(Frp[round(Frp.shape[0]/4)])

    def probe(self):

        w1 = self.w1
        w2 = self.w2
        T = self.T

        # page 5
        def lin_freq(t):
            return w1*t + (w2-w1)/T * t*t / 2

        # page 6
        def log_freq(t):
            K = T * w1 / np.log(w2/w1)
            L = T / np.log(w2/w1)
            return K * (np.exp(t/L)-1.0)

        freqs = log_freq(range(int(T)))
        impulse = np.sin(freqs)
        return impulse

    def estimate(self, response):

        I = fftconvolve( response, self.reverse_pulse, mode='full')
        I = I[self.probe_pulse.shape[0]:self.probe_pulse.shape[0]*2+1]

        peek_x = np.argmax( I, axis=0 )
        I = I[peek_x:]

        return I

# Module variables initialization
class info(object):
    def __init__(self):
        # Variables for settings
        # Click on the help button to get more information.
        # Example variables
        self.m_a = 1.0
        self.c_b = [0.1, 1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1, 9.1, 10.1, 11.1, 12.1, 13.1, 14.1, 15.1]
        self.c_c = [0.0 for n in range(16)]

# Temporary variables, not saved with the worksheet
class pvar(object):
    def __init__(self):
        # Working variables
        # Click on the help button to get more information.
        # Example variables
        duration = 10.0
        lowfreq = 100.0
        highfreq = 15000.0
        timebase_id = Ly.GetTimeBaseIDByName('Driver')
        auto_blocksize = False
        freq_format = 2
        var_blocksize = 0
        var_sample_rate = 0

        sample_distance = Ly.GetTimeBaseSampleDistance(timebase_id)
        sampling_frequency = 1.0 /  float(sample_distance)
        block_size = sampling_frequency
        print("Sampling Freq : {0}, Block Size : {1}").format(sampling_frequency, block_size)

        #Ly.UpdateTimeBase(timebase_id, auto_blocksize, block_size, freq_format, sample_distance, var_blocksize, var_sample_rate)
        self.response = None
        self.room_response = None
        self.m_m = 0.0
        self.c_n = [1.0 for n in range(16)]

        self.estimator = RoomResponseEstimator(duration, lowfreq, highfreq, sampling_frequency)
        self.blockCount = 0

class pscript(lys.mclass):
    def __init__(self, magic):
        #self.RegisterTimeBase("NoiseGenerator", 800, "Noise generator for building sound insulation test ", UpdateTimebase)
        #timebase_id = 800
        #blockSize = 10000
        #sampleDistance = 1 / float(blockSize * 5)
        #Ly.SetTimeBase(timebase_id, blocksize, sampleDistance)

        self.info=info()
        self.pvar=pvar()
        print("Initializing module....")

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
        dlg.title = "Script (Processing Data)"
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
        dm.AppendFloat("Param a", self.info.m_a, "Example: variable m_a in module scope")
        dm.SelectChannelPage()
        dm.AppendFloat("Param b", self.info.c_b, "Example: variable c_b in channel scope")
        dm.AppendFloat("Param c", self.info.c_c, "Example: variable c_c in channel scope")

    def DlgOk (self, dlg):
        # Get values of dialog parameters
        # Click on the help button to get more information.
        dom = lys.DialogOkManager(dlg)
        dom.SelectModulePage()
        self.info.m_a = dom.GetValue("Param a")
        dom.SelectChannelPage()
        self.info.c_b = dom.GetValue("Param b")
        self.info.c_c = dom.GetValue("Param c")

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
        self.pvar.blockCount = 0

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
        r = self.pvar.estimator.probe_pulse[c + (N * self.pvar.blockCount)]
        return r

    def ProcessData (self):
        # Process data blocks
        # Click on the help button to get more information - especially
        # if you have chosen a user defined setting because you very likely
        # need adjustments in this case!

        def numpyToDasylab(outChannel, inBuffer, numpyDataArray):
            outBuffer = self.GetOutputBlock(outChannel)
            blockSize = inBuffer.BlockSize
            dataLength = numpyDataArray.shape[0]

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

            outBuffer.StartTime = inBuffer.StartTime
            outBuffer.SampleDistance = inBuffer.SampleDistance
            outBuffer.BlockSize = inBuffer.BlockSize
            outBuffer.Release()

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

        # Process data
        #for channel in xrange(self.NumInChannel):
        channel0 = 0
        channel1 = 1
        inBuffer = self.GetInputBlock(channel0)
        blockSize = inBuffer.BlockSize
        sequenceLength = self.pvar.estimator.probe_pulse.shape[0]
        blocksNumber = sequenceLength / blockSize

        if (blockSize * self.pvar.blockCount) < sequenceLength:
            #print("Block : {0}  of  :  {1} . Sequence length : {2} . Input block size  : {3}. Input sampling freq : {4}").format(self.pvar.blockCount, blocksNumber, sequenceLength, blockSize, (1/inBuffer.SampleDistance))
            inBlockData = dasylabToNumpy(inBuffer)
            outBlockData = None

            # Output generated sequence
            if (sequenceLength - blockSize * self.pvar.blockCount) < blockSize:
                # Case where the generated sequence length is not a multiple of the block size
                outBlockData = self.pvar.estimator.probe_pulse[(blockSize * self.pvar.blockCount):]
                inBlockData = inBlockData[:outBlockData.shape[0]]
            else:
                # Case where the generated sequence length is a multiple od the block size
                outBlockData = self.pvar.estimator.probe_pulse[(blockSize * self.pvar.blockCount):(blockSize * (self.pvar.blockCount+1))]

            numpyToDasylab(channel0, inBuffer, outBlockData)

            # Store acquired signal
            if (self.pvar.blockCount is 0):
                self.pvar.response = inBlockData
            else:
                self.pvar.response = np.append(self.pvar.response, inBlockData)

        else:
            print("Computing room response....")
            inBuffer = self.GetInputBlock(channel1)
            self.room_response = self.pvar.estimator.estimate(self.pvar.response)
            np.savetxt('C:/Users/sengan/Documents/Test/src/room_response.log', self.room_response)
            numpyToDasylab(channel1, inBuffer, self.room_response[:blockSize])
            print("Room response length : {0}").format(self.room_response.shape)
            Ly.StopExperiment()
            return True

        print ("Block : {0} . Sequence length : {1} . Response length : {2}").format(self.pvar.blockCount, self.pvar.estimator.probe_pulse.shape, self.pvar.response.shape)
        self.pvar.blockCount = self.pvar.blockCount + 1
        inBuffer.Release()

        return True
