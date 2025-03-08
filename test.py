import unittest
from src.GatingAlgo import SingleGaussian, DoubleGaussian

class TestSPADSimulateEngine(unittest.TestCase):
    def test_single_gaussian_coates_estimation(self):
        engine = SingleGaussian(num_bins=100, pulse_pos=50, pulse_width=5, signal_strength=0.5, bg_strength=0.1, cycles=1000)
        coates_estimation = engine.coates_estimation()
        self.assertIsNotNone(coates_estimation)
        self.assertEqual(len(coates_estimation), 100)

    def test_double_gaussian_coates_estimation(self):
        engine = DoubleGaussian(num_bins=100, pulse_pos1=40, pulse_pos2=70, pulse_width1=5, pulse_width2=5, signal_strength=0.5, bg_strength=0.1, cycles=1000)
        coates_estimation = engine.coates_estimation()
        self.assertIsNotNone(coates_estimation)
        self.assertEqual(len(coates_estimation), 100)

if __name__ == '__main__':
    unittest.main()