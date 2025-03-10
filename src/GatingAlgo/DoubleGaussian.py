import numpy as np
from .SPADSimulateEngine import SPADSimulateEngine


class DoubleGaussian(SPADSimulateEngine):
    def __init__(self, num_bins=100, pulse_pos1=40, pulse_pos2=70, pulse_width1=5, pulse_width2=5, signal_strength1=0.5, signal_strength2=0.5, bg_strength=0.1, cycles=1000):
        self._pulse_pos1 = pulse_pos1
        self._pulse_pos2 = pulse_pos2
        self._pulse_width1 = pulse_width1
        self._pulse_width2 = pulse_width2
        self._signal_strength1 = signal_strength1
        self._signal_strength2 = signal_strength2
        self._bg_strength = bg_strength
        super().__init__(num_bins, cycles)

    def update_flux(self):
        """
        生成光场光通量（双高斯脉冲信号 + 均匀背景）
        """
        x = np.arange(self._num_bins)
        pulse1 = self._signal_strength1 * np.exp(-0.5 * ((x - self._pulse_pos1) / self._pulse_width1)**2)
        pulse2 = self._signal_strength2 * np.exp(-0.5 * ((x - self._pulse_pos2) / self._pulse_width2)**2)
        background = self._bg_strength * np.ones(self._num_bins)
        self._flux = pulse1 + pulse2 + background
