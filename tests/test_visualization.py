import unittest

import numpy as np

from visualization.interactive_plots import (
    spectrum_figure,
    time_comparison_figure,
)


class ComparisonPlotTests(unittest.TestCase):
    def setUp(self):
        self.sample_rate = 8000
        self.signal = np.sin(
            2 * np.pi * 440 * np.arange(1024) / self.sample_rate
        )
        self.frequencies = np.fft.rfftfreq(
            len(self.signal),
            1 / self.sample_rate,
        )
        self.amplitude = np.abs(np.fft.rfft(self.signal))

    def assert_three_aligned_panels(self, figure):
        self.assertEqual(len(figure.data), 3)
        self.assertEqual(
            [trace.yaxis for trace in figure.data],
            ["y", "y2", "y3"],
        )
        self.assertEqual(figure.layout.yaxis2.matches, "y")
        self.assertEqual(figure.layout.yaxis3.matches, "y")

    def test_time_comparison_separates_signals_and_keeps_one_range_slider(self):
        figure = time_comparison_figure(
            self.signal,
            self.sample_rate,
            processed=self.signal * 0.5,
            reference=self.signal * 0.75,
            separate=True,
        )

        self.assert_three_aligned_panels(figure)
        self.assertFalse(figure.layout.xaxis.rangeslider.visible)
        self.assertTrue(figure.layout.xaxis3.rangeslider.visible)

    def test_spectrum_comparison_separates_signals_on_shared_scale(self):
        figure = spectrum_figure(
            self.frequencies,
            self.amplitude,
            processed=(self.frequencies, self.amplitude * 0.5),
            reference=(self.frequencies, self.amplitude * 0.75),
            separate=True,
        )

        self.assert_three_aligned_panels(figure)


if __name__ == "__main__":
    unittest.main()
