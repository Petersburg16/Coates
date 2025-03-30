from traitlets import default
from .SPADSimulateEngine import SPADSimulateEngine,SPADDataContainer
import numpy as np


from .DualGaussianViewer import DualGaussianViewer 
default_double_gaussian_params = {
        "pulse_pos1": 40,
        "pulse_pos2": 70,
        "pulse_width1": 5,
        "pulse_width2": 5,
        "signal_strength1": 0.5,
        "signal_strength2": 0.5,
        "bg_strength": 0.01,
        "total_strength": 1,
}
class DualGaussian(SPADSimulateEngine):
    def __init__(
        self,
        double_gaussian_params: dict = default_double_gaussian_params,
        gate_info: tuple = (0, 100),
        cycles: int = 1000,
        simulate_field: int = 200
    ):
        self.double_gaussian_flux_params = double_gaussian_params
        self._simulate_field = simulate_field

        super().__init__(gate_info, cycles, simulate_field)
        self.data.resolution_factor = 10


    def generate_flux(self):
        self.data.flux=self.double_gaussian_flux(self._simulate_field, resolution_factor=1, **self.double_gaussian_flux_params)
        self.data.smooth_flux=self.double_gaussian_flux(self._simulate_field, self.data.resolution_factor, **self.double_gaussian_flux_params)


    def create_interactive_plot(self):
        """创建交互式图形并启动Dash服务器"""
        app = DualGaussianViewer.create_dash_app(self.data, self, title="双高斯SPAD模拟器")
        app.run(debug=True)

        
    def update_simulation_params(self, new_params):
        """更新模拟参数并重新生成所有数据"""
        self.double_gaussian_flux_params.update(new_params)
        self.generate_flux()
        self.update_ideal_histogram()
        self.update_simulated_histogram()
        self.update_coates_estimation()
    
    @staticmethod
    def double_gaussian_flux(simulate_field, resolution_factor, pulse_pos1, pulse_pos2, pulse_width1, pulse_width2, signal_strength1, signal_strength2, bg_strength, total_strength):

        x = np.linspace(0, simulate_field - 1, simulate_field * resolution_factor)
        pulse1 = signal_strength1 * np.exp(-0.5 * ((x - pulse_pos1) / pulse_width1)**2)
        pulse2 = signal_strength2 * np.exp(-0.5 * ((x - pulse_pos2) / pulse_width2)**2)
        background = bg_strength * np.ones(len(x))
        flux = pulse1 + pulse2 + background

        return SPADSimulateEngine.normalize_flux(flux) * total_strength*resolution_factor
    


