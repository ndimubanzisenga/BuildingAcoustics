from __future__ import division
import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt

import wavio
import subprocess
import os
import time
import datetime

from acoustics.generator import Generator
from acoustics.signal import OctaveBand, Signal, Spectrum
from acoustics.building_acoustics_measurement import BuildingAcousticsMeasurement

ROOT_DIR = 'C:/Users/sengan/Documents/Projects/BuildingAcoustics/'
ts = time.time()
time_stamp = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d-%H%M%S')
log_dir = ('{0}/data/StandaloneTests/{1}/').format(ROOT_DIR, time_stamp)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
result_dir = ROOT_DIR + 'data/results/' + time_stamp + '/'
if not os.path.exists(result_dir):
    os.makedirs(result_dir)
DEBUG = True

def load_wavfile(file_name):
    rr = wavio.read(file_name)
    data = rr.data[0:,0]
    data = data / math.pow(2.0, rr.sampwidth*8-1)
    return data

def plot_data(y_data, x_data=None, title=None, y_label=None, x_label=None, scale='linear', args=None):
    fig, ax = plt.subplots()
    if x_data is None:
        ax.plot(y_data)
    else:
        ax.plot(x_data, y_data)
    ax.set_xscale(scale)
    if args is not None:
        ax.set_xticks(args)
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid()
    file_name = result_dir + title
    fig.savefig(file_name)

class Test(object):
    def __init__(self, reused_file=None, fs=44100., duration=10.0, n_channels=1, f_start=100., f_stop=5000., fraction=3, noise_type='sine_sweep'):
        ### ToDo: Deistinguish between f_start, f_stop for generator and for analysis
        self._fs = fs
        self.probe_signal, self.reverse_filter = self.generate_probe_signal(fs, duration, 50., 8000., 'sine_sweep')
        if reused_file is not None:
            self.room_response = load_wavfile(reused_file)
        else:
            self.room_response = self.simulate_rr(fs)

        self.impulse_response = self.compute_ir()
        self.spl, self.reverberation_time, self.octave_bands = self.compute_acoustic_parameters(f_start, f_stop, fs)

        self.test_acoustic_parameters_measurement()
        #self.test_generator()
        plt.show()

    def generate_probe_signal(self, fs, duration, f_start, f_stop, noise_type):
        self._gen = Generator(fs, duration)
        return self._gen.noise(noise_type, [f_start, f_stop])

    def simulate_rr(self, fs):
        # Store probe signal on a wav file
        ps_file = log_dir + "probe_signal.wav"
        #ps = np.append(np.zeros(44100), self.probe_signal) # add silence before sequence
        wavio.write(ps_file, ps, fs, sampwidth=2)

        # Play probe signal and record microphones input simultaneously.
        rr_file = log_dir + "room_response.wav"
        reccommand = "sox -t waveaudio -q --clobber -r 44100 -b 16 -D -c 2 -d {0} trim 0 10".format(rr_file)
        reccommand = reccommand.split(" ")
        prec = subprocess.Popen(reccommand, shell=True)

        playcommand = "sox -q {0} -t waveaudio -d".format(ps_file)
        playcommand = playcommand.split(" ")
        pplay = subprocess.Popen(playcommand, shell=True)

        pplay.wait()
        prec.wait()

        # Read stored room response
        rr = wavio.read(rr_file)
        room_response = rr.data[0:,0]
        room_response = room_response / math.pow(2.0, rr.sampwidth*8-1)

        return room_response

    def compute_ir(self):
        return self._gen.estimate_impulse_response(self.room_response, self.reverse_filter)

    def compute_acoustic_parameters(self, f_start, f_stop, fs):
        building_acoustics_measurement = BuildingAcousticsMeasurement(f_start=f_start, f_stop=f_stop, fraction=3)
        building_acoustics_measurement.compute_spl('rx', self.room_response)
        building_acoustics_measurement.compute_reverberation_time('rx', self.impulse_response, fs, 'impulse', 't10')

        return building_acoustics_measurement.rx_room_spl, building_acoustics_measurement.reverberation_time,\
               building_acoustics_measurement.octave_bands

    def test_generator(self):
        f_start = 50.
        f_stop = 8000.
        fraction = 3
        white_noise = self._gen.noise('white')
        pink_noise = self._gen.noise('pink')
        sine_sweep, inverse_filter = self._gen.noise('sine_sweep', [f_start, f_stop])

        spectrum = Spectrum()
        frequencies = OctaveBand(fstart=f_start, fstop=f_stop, fraction=fraction)
        _, white_noise_octaves_sxx = spectrum.third_octaves(white_noise, self._fs, frequencies=frequencies.center, density=False)
        _, pink_noise_octaves_sxx = spectrum.third_octaves(pink_noise, self._fs, frequencies=frequencies.center, density=False)
        _, sine_sweep_octaves_sxx = spectrum.third_octaves(sine_sweep, self._fs, frequencies=frequencies.center, density=False)
        _, inverse_filter_octaves_sxx = spectrum.third_octaves(inverse_filter, self._fs, frequencies=frequencies.center, density=False)

        white_noise_fft = abs(np.fft.rfft(white_noise))
        pink_noise_fft = abs(np.fft.rfft(pink_noise))
        sine_sweep_fft = abs(np.fft.rfft(sine_sweep))
        inverse_filter_fft = abs(np.fft.rfft(inverse_filter))

        octave_ticks = np.asarray(frequencies.center, dtype=np.int)[::2]
        plot_data(y_data=white_noise_octaves_sxx, x_data=frequencies.center, title='White Noise Power Spectrum',\
                  y_label='Sound pressure mean square [Pa^2]', x_label='Frequency [Hz]', scale='log', args=octave_ticks)
        plot_data(y_data=pink_noise_octaves_sxx, x_data=frequencies.center, title='Pink Noise Power Spectrum',\
                  y_label='Sound pressure mean square [Pa^2]', x_label='Frequency [Hz]', scale='log', args=octave_ticks)
        plot_data(y_data=sine_sweep_octaves_sxx, x_data=frequencies.center, title='Sine Sweep Power Spectrum',\
                  y_label='Sound pressure mean square [Pa^2]', x_label='Frequency [Hz]', scale='log', args=octave_ticks)
        plot_data(y_data=inverse_filter_octaves_sxx, x_data=frequencies.center, title='Inverse Filter Power Spectrum',\
                  y_label='Sound pressure mean square[Pa^2]', x_label='Frequency [Hz]', scale='log', args=octave_ticks)

        N = int(f_stop  * white_noise_fft.size * 2/self._fs) # index corresponding to f_stop
        fft_freq = np.fft.fftfreq(white_noise.size, 1/self._fs)
        fft_freq = fft_freq[:N]
        plot_data(y_data=white_noise_fft[:N], x_data=fft_freq, title='White Noise Spectrum', y_label='Sound pressure [Pa]',\
                  x_label='Frequency [Hz]', scale='linear')
        plot_data(y_data=pink_noise_fft[:N], x_data=fft_freq, title='Pink Noise Spectrum', y_label='Sound pressure [Pa]',\
                  x_label='Frequency [Hz]', scale='linear')
        plot_data(y_data=sine_sweep_fft[:N], x_data=fft_freq, title='Sine Sweep Spectrum', y_label='Sound pressure [Pa]',\
                  x_label='Frequency [Hz]', scale='linear')
        plot_data(y_data=inverse_filter_fft[:N], x_data=fft_freq, title='Inverse Filter Spectrum', y_label='Sound pressure [Pa]',\
                  x_label='Frequency [Hz]', scale='linear')



    def test_acoustic_parameters_measurement(self):
        ### ToDo: Deistinguish between f_start, f_stop for generator and for analysis
        f_start = 100.
        f_stop = 5000.
        fraction = 3
        test_octave_band = 0
        test_duration = 2
        test_channel = 0
        T = int(self._fs * test_duration)
        t = np.arange(self.probe_signal.size)/self._fs
        t = t[:T]

        building_acoustics_measurement = BuildingAcousticsMeasurement(f_start=f_start, f_stop=f_stop, fraction=3)
        octave_bands = building_acoustics_measurement.octave_bands

        ir = self.impulse_response[:T]
        building_acoustics_measurement.t60_impulse(ir, fs=self._fs, rt='t10', test_octave_band=0)
        test_bandpass_filtered_ir = building_acoustics_measurement.bandpass_filtered_ir
        test_bandpass_filtered_ir_db = 10 * np.log10(abs(test_bandpass_filtered_ir) / test_bandpass_filtered_ir.max())
        schroeder_curve = building_acoustics_measurement.schroeder_curve
        schroeder_curve_db = 10 * np.log10(schroeder_curve / schroeder_curve.max())
        x, y, y_ = building_acoustics_measurement.regression_result


        octave_ticks = np.asarray(octave_bands, dtype=np.int)[::2]
        plot_data(y_data=self.impulse_response[:T], x_data=t, title='Impulse Response',\
                  y_label='Sound pressure [Pa]', x_label='Time [sec]', scale='linear', args=None)
        #title = 'Filtered IR at center freq {0} Hz, in a 1/{1} octave band'.format(octave_bands[test_octave_band], fraction)
        plot_data(y_data=test_bandpass_filtered_ir, x_data=t, title='Filtered IR', y_label='Sound pressure [Pa]', x_label='Time [sec]',\
                  scale='linear', args=None)
        plot_data(y_data=test_bandpass_filtered_ir_db, x_data=t, title='Filtered IR - dB', y_label='Sound pressure Level [dB]', x_label='Time [sec]',\
                  scale='linear', args=None)
        plot_data(y_data=schroeder_curve, x_data=t, title='Schroeder curve', y_label='Sound pressure [Pa]', x_label='Time [sec]',\
                  scale='linear', args=None)
        plot_data(y_data=schroeder_curve_db, x_data=t, title='Schroeder curve - dB', y_label='Sound pressure Level [dB]', x_label='Time [sec]',\
                  scale='linear', args=None)

        plt.figure()
        plt.plot(t, test_bandpass_filtered_ir_db)
        plt.plot(t, schroeder_curve_db)
        plt.xlabel('Time [sec]')
        plt.ylabel('Sound pressure Level [dB]')
        plt.grid()

        plt.figure()
        plt.scatter(x[::100], y[::100], marker='+', c='r')
        plt.plot(x, y_)
        plt.title('Decay curve fitting')
        plt.xlabel('Time [sec]')
        plt.ylabel('Sound pressure Level [dB]')
        plt.grid()

        plot_data(y_data=self.spl, x_data=octave_bands, title='Room SPL', y_label='Sound Pressure Level [dB]', x_label='Frequency [Hz]',\
                  scale='log', args=octave_ticks)
        plot_data(y_data=self.reverberation_time, x_data=octave_bands, title='Reverberation time', y_label='Reverberation time [sec]',\
                  x_label='Frequency [Hz]', scale='log', args=octave_ticks)

        return


def main():
    rr_log_file = ROOT_DIR+'data/StandaloneTests/2018-04-08-144430/room_response.wav'
    t = Test(reused_file=rr_log_file)
    print("## Ended Successfully ##")

if __name__ == "__main__":
    main()
