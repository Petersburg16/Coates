from dataclasses import dataclass
from math import exp
from scipy.special import log1p
import numpy as np

@dataclass
class SPADDataContainer:
    """
    SPAD数据容器类，用于存储SPAD模拟过程生成的数据
    主要保存的是最终需要被展示的数据，并提供一些动态方法来更新数据

    flux只是用来生成直方图的数据，numbin是对应bin的分辨率，固定长度lighten_space(因此门控第二个位置不能超过4096)
    """

    # SPAD模拟对应真实数据的参数
    # gate_length->numbins
    # exposure->cycles

    gate_info:tuple=(0,100)
    exposure:int=1000
    @property
    def gate_length(self)->int:
        return self.gate_info[1] - self.gate_info[0]


    flux:np.ndarray=None
    smooth_flux:np.ndarray=None
    ideal_histogram:np.ndarray=None
    spad_histogram:np.ndarray=None
    coates_histogram:np.ndarray=None

    @property
    def spad_histogram_without_overflow(self)->np.ndarray:
        return self.spad_histogram[:-1]


class SPADSimulateEngine:
    def __init__(self,num_bins:int=100,cycles:int=1000):
        self._num_bins = num_bins
        self._cycles = cycles
        self.spad_data= SPADDataContainer()


    def generate_flux(self):


        self._singnal_strength = 1.0
        self._pulse_pos = 50.0
        self._pulse_width = 3.0
        self._bg_strength = 0.01

        self._total_strength=1

        x = np.arange(self._num_bins)
        pulse = self._singnal_strength * np.exp(-0.5 * ((x - self._pulse_pos) / self._pulse_width)**2)
        background = self._bg_strength * np.ones(self._num_bins)
        self.spad_data.flux=self.normalize_flux(pulse + background)*self._total_strength


    def update_ideal_histogram(self):
        self.spad_data.ideal_histogram = self.generate_ideal_histogram(self.spad_data.flux,self._cycles)

    def update_simulated_histogram(self):
        self.spad_data.spad_histogram = self.SPAD_simulation(self.spad_data.flux,self._cycles)

    def update_coates_estimation(self):
        self.spad_data.coates_histogram =self.coates_estimator(self.spad_data.spad_histogram,self._cycles)
        


    def plot_test(self)-> None:
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=np.arange(self._num_bins), y=self.spad_data.flux*self._cycles, mode='lines', name='Flux'))
        fig.add_trace(go.Bar(x=np.arange(self._num_bins), y=self.spad_data.ideal_histogram, name='Ideal Histogram'))
        fig.add_trace(go.Bar(x=np.arange(self._num_bins), y=self.spad_data.spad_histogram_without_overflow, name='Simulated Histogram'))
        fig.add_trace(go.Bar(x=np.arange(self._num_bins), y=self.spad_data.coates_histogram, name='Coates Histogram'))
        fig.show()
    

    
# -------------------------------------------------------------------
# 静态工具方法（允许外部直接调用）
# -------------------------------------------------------------------
    @staticmethod
    def generate_ideal_histogram(flux:np.ndarray,cycles:int)-> np.ndarray:
        num_bins=len(flux)
        histogram = np.zeros(num_bins)
        detection_probabilities = 1 - np.exp(-flux)
        for i in range(num_bins):
            histogram[i] = np.random.poisson(detection_probabilities[i]*cycles)
        histogram=np.round(histogram).astype(int)
        return histogram
    
    
    @staticmethod
    def SPAD_simulation(flux:np.ndarray,cycles:int)-> np.ndarray:
        num_bins = len(flux)
        histogram = np.zeros(num_bins + 1, dtype=int)  # 最后一个是溢出仓
        detection_probabilities = 1 - np.exp(-flux)

        for _ in range(cycles):
            detected = False
            for current_bin in range(num_bins):
                q_i = detection_probabilities[current_bin]
                if np.random.rand() < q_i:
                    histogram[current_bin] += 1
                    detected = True
                    break  
            if not detected:
                histogram[-1] += 1  
        return histogram
    
    @staticmethod
    def normalize_flux(flux:np.ndarray)-> np.ndarray:
        normalized_flux = flux / np.sum(flux)
        return normalized_flux

    @staticmethod
    def normalize_histogram(histogram:np.ndarray, cycles:int)-> np.ndarray:
        normalized_histogram = np.round(histogram / np.sum(histogram)*cycles).astype(int)
        return normalized_histogram
    

    


    @staticmethod 
    def safe_log(x):
        """数值稳定的对数计算 -ln(1-x)"""
        if x >= 1.0:
            return np.inf  # 处理无穷大
        elif x < 1e-15:    # 小值时泰勒展开
            return x + x**2/2 + x**3/3 + x**4/4
        else:
            return -log1p(-x)  # 等价于 -ln(1-x) 的稳定计算

    @staticmethod
    def coates_estimator(histogram:np.ndarray,cycles:int)-> np.ndarray:
        """
        Coates估计器的数值稳定实现
        注意输入的光子直方图histogram的长度是B+1，最后一个是溢出仓
        """
        safe_log_vec = np.vectorize(SPADSimulateEngine.safe_log, otypes=[np.longdouble])
        
        # 转换输入参数为高精度类型
        N_i = histogram[:-1].astype(np.longdouble)
        total_counts = np.sum(histogram).astype(np.longdouble)
        
        """
        if histogram[-1] == 0:
            total_counts += 1  # 防止除零错误
        """
        
        # 计算分母序列
        cumulative_loss = np.zeros_like(N_i, dtype=np.longdouble)
        denominator = np.zeros_like(N_i, dtype=np.longdouble)
        denominator[0] = total_counts
        
        for i in range(1, len(N_i)):
            cumulative_loss[i] = cumulative_loss[i-1] + N_i[i-1]
            denominator[i] = total_counts - cumulative_loss[i]
        
        # 处理分母非正值
        denominator = np.maximum(denominator, 1e-15)
        P = np.divide(N_i, denominator, where=~np.isnan(denominator), out=np.full_like(denominator, np.nan))
        S = safe_log_vec(P)
        # 将负值和 NaN 设置为 0
        S = np.where(np.isnan(S) | (S < 0), 0, S)

        # 将 inf 替换为 S 中其他有效值的最大值
        if np.isinf(S).any():
            max_valid_value = np.max(S[~np.isinf(S)])  # 找到 S 中非 inf 的最大值
            S = np.where(np.isinf(S), max_valid_value, S)
            
        coates_histogram = S*cycles
        return np.round(coates_histogram).astype(int)

        
        

    
def test():
    spad_simulator = SPADSimulateEngine()
    spad_simulator.generate_flux()
    spad_simulator.update_ideal_histogram()
    spad_simulator.update_simulated_histogram()
    spad_simulator.update_coates_estimation()
    spad_simulator.plot_test()

if __name__ == "__main__":
    test()
