import unittest

import numpy as np
from scipy.signal import freqz, sosfilt, sosfilt_zi, sosfreqz

from core.fir import FILTER_TYPES, FIR_DIRECT_MAX_SAMPLES, apply_fir, design_fir
from core.iir import IIR_FAMILIES, IIR_MANUAL_MAX_SAMPLES, apply_iir, design_iir


class FilterTests(unittest.TestCase):
    def setUp(self):
        self.sample_rate = 8000
        times = np.arange(2048) / self.sample_rate
        self.signal = np.sin(2 * np.pi * 440 * times)
        self.cutoff_for = {
            "Pasabajo": [1000],
            "Pasaalto": [1000],
            "Pasabanda": [300, 1000],
            "Rechaza banda": [300, 1000],
        }

    def test_all_fir_filters_process_and_preserve_length(self):
        for filter_name in FILTER_TYPES:
            with self.subTest(filter=filter_name):
                coefficients = design_fir(filter_name, self.sample_rate, 64, self.cutoff_for[filter_name])
                result = apply_fir(self.signal, coefficients)
                self.assertEqual(result.shape, self.signal.shape)
                self.assertTrue(np.all(np.isfinite(result)))

    def test_all_iir_filters_process_and_preserve_length(self):
        for family in IIR_FAMILIES:
            for filter_name in FILTER_TYPES:
                with self.subTest(family=family, filter=filter_name):
                    coefficients = design_iir(
                        filter_name,
                        self.sample_rate,
                        4,
                        self.cutoff_for[filter_name],
                        family=family,
                    )
                    result = apply_iir(self.signal, coefficients)
                    self.assertEqual(result.shape, self.signal.shape)
                    self.assertTrue(np.all(np.isfinite(result)))

    def test_iir_families_validate_required_ripple_parameters(self):
        with self.assertRaises(ValueError):
            design_iir(
                "Pasabajo",
                self.sample_rate,
                4,
                [1000],
                family="Chebyshev I",
                passband_ripple_db=0,
            )
        with self.assertRaises(ValueError):
            design_iir(
                "Pasabajo",
                self.sample_rate,
                4,
                [1000],
                family="Chebyshev II",
                stopband_attenuation_db=0,
            )
        with self.assertRaises(ValueError):
            design_iir(
                "Pasabajo",
                self.sample_rate,
                4,
                [1000],
                family="Elíptico",
                passband_ripple_db=1,
                stopband_attenuation_db=1,
            )
        with self.assertRaises(ValueError):
            design_iir(
                "Pasabajo",
                self.sample_rate,
                4,
                [1000],
                family="IIR no reconocido",
            )

    def test_manual_iir_matches_scipy_reference(self):
        rng = np.random.default_rng(1234)
        signal = rng.normal(0, 0.2, 300)
        for family in IIR_FAMILIES:
            for filter_name in FILTER_TYPES:
                with self.subTest(family=family, filter=filter_name):
                    sections = design_iir(
                        filter_name,
                        self.sample_rate,
                        5,
                        self.cutoff_for[filter_name],
                        family=family,
                    )
                    initial_state = sosfilt_zi(sections) * signal[0]
                    reference = sosfilt(sections, signal, zi=initial_state)[0]
                    result = apply_iir(signal, sections)
                    np.testing.assert_allclose(
                        result,
                        reference,
                        rtol=1e-10,
                        atol=1e-12,
                    )

    def test_long_fir_uses_fft_convolution_and_matches_direct_convolution(self):
        signal = np.sin(2 * np.pi * 440 * np.arange(FIR_DIRECT_MAX_SAMPLES + 1) / self.sample_rate)
        coefficients = design_fir("Pasabajo", self.sample_rate, 8, [1000])
        result = apply_fir(signal, coefficients)
        full_reference = np.convolve(signal, coefficients, mode="full")
        start = (len(coefficients) - 1) // 2
        reference = full_reference[start : start + len(signal)]
        np.testing.assert_allclose(result, reference, rtol=1e-10, atol=1e-10)

    def test_long_iir_uses_scipy_and_matches_scipy_reference(self):
        signal = np.sin(2 * np.pi * 440 * np.arange(IIR_MANUAL_MAX_SAMPLES + 1) / self.sample_rate)
        sections = design_iir("Pasabajo", self.sample_rate, 4, [1000])
        initial_state = sosfilt_zi(sections) * signal[0]
        reference = sosfilt(sections, signal, zi=initial_state)[0]
        result = apply_iir(signal, sections)
        np.testing.assert_allclose(result, reference, rtol=1e-12, atol=1e-12)

    def test_invalid_cutoff_values_are_rejected(self):
        with self.assertRaises(ValueError):
            design_fir("Pasabajo", self.sample_rate, 64, [0])
        with self.assertRaises(ValueError):
            design_iir("Pasabanda", self.sample_rate, 4, [1000, 300])
        with self.assertRaises(ValueError):
            design_iir("Pasabajo", self.sample_rate, 4, [5000])

    def test_fir_odd_order_is_adjusted_to_even_order(self):
        coefficients = design_fir("Pasabajo", self.sample_rate, 63, [1000])
        self.assertEqual(len(coefficients), 65)

    def test_fir_and_iir_frequency_responses_attenuate_expected_bands(self):
        cases = {
            "Pasabajo": ([800], 200, 2500),
            "Pasaalto": ([800], 2500, 200),
            "Pasabanda": ([400, 900], 600, 1800),
            "Rechaza banda": ([400, 900], 1800, 600),
        }
        for filter_name, (cutoffs, pass_hz, stop_hz) in cases.items():
            with self.subTest(family="FIR", filter=filter_name):
                coefficients = design_fir(filter_name, self.sample_rate, 100, cutoffs)
                _, pass_response = freqz(coefficients, worN=[pass_hz], fs=self.sample_rate)
                _, stop_response = freqz(coefficients, worN=[stop_hz], fs=self.sample_rate)
                self.assertGreater(abs(pass_response[0]), 0.75)
                self.assertLess(abs(stop_response[0]), 0.15)

            for family in IIR_FAMILIES:
                with self.subTest(family=family, filter=filter_name):
                    sections = design_iir(
                        filter_name,
                        self.sample_rate,
                        4,
                        cutoffs,
                        family=family,
                    )
                    _, pass_response = sosfreqz(
                        sections,
                        worN=[pass_hz],
                        fs=self.sample_rate,
                    )
                    _, stop_response = sosfreqz(
                        sections,
                        worN=[stop_hz],
                        fs=self.sample_rate,
                    )
                    self.assertGreater(abs(pass_response[0]), 0.75)
                    self.assertLess(abs(stop_response[0]), 0.15)


if __name__ == "__main__":
    unittest.main()
