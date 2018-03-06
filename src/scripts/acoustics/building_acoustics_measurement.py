"""
Building Acoustic Measurement
=============================
This module implements routines for the measurement of parameters to describe building acoustics
"""

class BuildingAcousticsMeasurement(object):
    def __init__(self, fs=44100, f_start=50., f_stop=5000.):
        self.probe_signal = None
        self.response_signal = None
        self.tx_room_spl = None
        self.rx_room_spl = None
        self.rx_room_background_noise_spl = None
        self.rx_room_reverberation_time = None

    def compute_building_acoustics_parameters(self):
        """
        Calculate building acoustics parameters: Rx and Tx rooms spl,
        background noise and reverberation time.
        """
        pass

    def verify_building_acoustics_regulation(self):
        pass

    def diagnose_defect(self):
        pass

    def compute_spl(self):
        pass

    def compute_reverberation_time(self):
        pass
