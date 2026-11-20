import io
import unittest

import numpy as np
from scipy.io import wavfile

from core.audio import read_wav_file


class AudioReadTests(unittest.TestCase):
    def test_reads_mono_integer_wav_and_normalizes(self):
        original = np.array([0, 16384, -16384], dtype=np.int16)
        stream = io.BytesIO()
        wavfile.write(stream, 8000, original)
        stream.seek(0)

        class Uploaded:
            def getvalue(self):
                return stream.getvalue()

        sample_rate, audio, channels = read_wav_file(Uploaded())
        self.assertEqual(sample_rate, 8000)
        self.assertEqual(channels, 1)
        self.assertAlmostEqual(audio[1], 0.5, places=3)

    def test_stereo_is_averaged_to_mono(self):
        original = np.array([[1000, -1000], [2000, 2000]], dtype=np.int16)
        stream = io.BytesIO()
        wavfile.write(stream, 8000, original)
        stream.seek(0)

        class Uploaded:
            def getvalue(self):
                return stream.getvalue()

        _, audio, channels = read_wav_file(Uploaded())
        self.assertEqual(channels, 2)
        self.assertEqual(audio.shape, (2,))
        self.assertAlmostEqual(audio[0], 0.0)
        self.assertGreater(audio[1], 0.0)


if __name__ == "__main__":
    unittest.main()
