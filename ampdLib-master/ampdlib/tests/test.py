# -*- coding: utf-8 -*-
__author__      = "Luca Cerina"
__copyright__   = "Copyright 2023, Luca Cerina"
__email__       = "lccerina@gmail.com"

import unittest
import numpy as np
from scipy.io import loadmat
import ampdlib
import matplotlib.pyplot as plt

class TestLibrary(unittest.TestCase):
    def setUp(self) -> None:
        self.test_data = loadmat('ampdLib-master/ampdlib/tests/data.mat')
        N = 30000
        self.input_data = self.test_data['ecg_signal'][0:N,0].flatten()
        self.real_peaks = self.test_data['real_peaks'][0]
        return super().setUp()

    def test_detection(self):
        # Run the AMPD peak detection
        ampd_peaks = ampdlib.ampd_fast(self.input_data, window_length=2000, hop_length=1000)

        # Calculate error
     # Make sure ampd_peaks and real_peaks have the same length or handle mismatches
        min_len = min(len(ampd_peaks), len(self.real_peaks))
        error_peaks = np.sum((ampd_peaks[:min_len] - self.real_peaks[:min_len]) != 0)
        self.assertEqual(error_peaks, 0)

    # Filter real_peaks to ensure all indices are valid for plotting
        valid_real_peaks = self.real_peaks[self.real_peaks < len(self.input_data)]

    # Plotting section
        plt.figure(figsize=(12, 6))
        plt.plot(self.input_data, label='ECG Signal')
        plt.plot(ampd_peaks, self.input_data[ampd_peaks], 'ro', label='Detected Peaks')
        plt.plot(valid_real_peaks, self.input_data[valid_real_peaks], 'gx', label='Real Peaks')

        plt.title('AMPD Peak Detection Results')
        plt.xlabel('Sample Index')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid(True)
        plt.show()

    def test_no_hop_length(self):
        ampd_peaks = ampdlib.ampd_fast(self.input_data, window_length=2000)
        error_peaks = np.sum((ampd_peaks - self.real_peaks[0:ampd_peaks.shape[0]]) != 0)
        self.assertEqual(error_peaks, 0)

    def test_large_window_length(self):
        ampd_peaks = ampdlib.ampd_fast(self.input_data, window_length=len(self.input_data)+1)
        error_peaks = np.sum((ampd_peaks - self.real_peaks[0:ampd_peaks.shape[0]]) != 0)
        self.assertEqual(error_peaks, 0)

    def test_order_warnings(self):
        input_data = self.input_data[:1282]
        with self.assertWarns(UserWarning):
            ampdlib.ampd_fast_sub(input_data, order=4, verbose=True)

    def test_pool(self):
        ampd_peaks = ampdlib.ampd_pool(self.input_data, window_length=2000, hop_length=1000, verbose=True)
        error_peaks = np.sum((ampd_peaks - self.real_peaks[0:ampd_peaks.shape[0]]) != 0)
        self.assertEqual(error_peaks, 0)

    def test_assertions(self):
        with self.assertRaises(AssertionError):
            ampdlib.ampd(self.input_data, lsm_limit=-1)
        with self.assertRaises(AssertionError):
            ampdlib.ampd(self.input_data, lsm_limit=6/len(self.input_data)-1e-8)
        with self.assertRaises(AssertionError):
            ampdlib.ampd(self.input_data, lsm_limit=2)
        with self.assertRaises(AssertionError):
            ampdlib.ampd_fast(self.input_data, 2000, -1)
        with self.assertRaises(AssertionError):
            ampdlib.ampd_fast(self.input_data, 2000, 2100)
        with self.assertRaises(AssertionError):
            ampdlib.ampd_fast_sub(self.input_data, order=0)
        with self.assertRaises(AssertionError):
            ampdlib.ampd_fast_sub(self.input_data, lsm_limit=2)  
        with self.assertRaises(AssertionError):
            ampdlib.ampd_pool(self.input_data, 2000, -1)  
        with self.assertRaises(AssertionError):
            ampdlib.ampd_pool(self.input_data, 2000, 2100)  
        with self.assertRaises(AssertionError):
            ampdlib.ampd_pool(self.input_data, 2000, lsm_limit=2)  
        with self.assertRaises(AssertionError):
            ampdlib.ampd_pool(self.input_data, 2000, nr_workers=0)

class TestOptimalSize(unittest.TestCase):
    def test_assertions(self):
        with self.assertRaises(AssertionError):
            ampdlib.get_optimal_size(1000)
        with self.assertRaises(AssertionError):
            ampdlib.get_optimal_size(1000, -1)
        with self.assertRaises(AssertionError):
            ampdlib.get_optimal_size(1000, 2000)
        with self.assertRaises(AssertionError):
            ampdlib.get_optimal_size(1000, None, -1)
        with self.assertRaises(AssertionError):
            ampdlib.get_optimal_size(1000, None, 2)
        with self.assertRaises(AssertionError):
            ampdlib.get_optimal_size(1000, None, 1, -1)
    
    def test_scale(self):
        scale, lsm, _ = ampdlib.get_optimal_size(1000, None, 0.1)
        self.assertEqual(scale, 100)

    def test_lsm(self):
        scale, lsm, _ = ampdlib.get_optimal_size(1000, 100, None)
        self.assertEqual(lsm, 0.1)

    def test_warning(self):
        with self.assertWarns(UserWarning):
            scale, lsm, _ = ampdlib.get_optimal_size(1000, 1, None)

if __name__ == "__main__":
    # Normally, you'd run tests from the command line, but you can also do this:
    # unittest.main()

    # For demonstration, you can instantiate the test class and run test_detection directly
    # if you want to see the plot (comment out unittest.main() above).
    test_suite = unittest.TestSuite()
    test_suite.addTest(TestLibrary('test_detection'))
    runner = unittest.TextTestRunner()
    runner.run(test_suite)
