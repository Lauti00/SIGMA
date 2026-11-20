import unittest

import numpy as np

from core.metrics import mean_squared_error, signal_to_noise_ratio_db


class MetricsTests(unittest.TestCase):
    def test_mse_and_snr_for_known_signals(self):
        clean = np.array([1.0, -1.0, 1.0, -1.0])
        estimate = np.array([0.5, -0.5, 1.0, -1.0])
        self.assertAlmostEqual(mean_squared_error(clean, estimate), 0.125)
        self.assertAlmostEqual(signal_to_noise_ratio_db(clean, estimate), 10 * np.log10(8))

    def test_identical_signal_has_infinite_snr_and_zero_mse(self):
        clean = np.array([0.2, -0.3, 0.1])
        self.assertEqual(mean_squared_error(clean, clean), 0.0)
        self.assertEqual(signal_to_noise_ratio_db(clean, clean), float("inf"))

    def test_invalid_reference_pairs_are_rejected(self):
        with self.assertRaises(ValueError):
            mean_squared_error([1, 2], [1])
        with self.assertRaises(ValueError):
            signal_to_noise_ratio_db([0, 0], [0.1, -0.1])
        with self.assertRaises(ValueError):
            mean_squared_error([[1, 2]], [[1, 2]])


if __name__ == "__main__":
    unittest.main()
