import numpy as np
from scipy.special import log1p
import plotly.graph_objects as go

class SPADSimulateEngine:
    def __init__(self, num_bins=100, signal_strength=0.5, bg_strength=0.1, cycles=1000):
        self._num_bins = num_bins
        self._signal_strength = signal_strength
        self._bg_strength = bg_strength
        self._cycles = cycles

        self.update_flux()
        self.update_simulated_ideal_histogram()
        self.update_detection_probabilities()
        self.update_simulated_histogram()
        self.update_coates_estimation()

    def update_flux(self):
        """
        生成光场光通量（需要在子类中实现）
        """
        raise NotImplementedError("Subclasses should implement this method")

    def update_detection_probabilities(self):
        """
        计算每个bin的光子检测概率Pi = 1 - e^{-r_i}
        """
        self._detection_probabilities = 1 - np.exp(-self._flux)

    def update_simulated_histogram(self):
        """
        模拟SPAD检测（包含堆积效应）
        """
        num_bins = len(self._flux)
        histogram = np.zeros(num_bins + 1, dtype=int)  # 最后一个是溢出仓
        
        for _ in range(self._cycles):
            detected = False
            for current_bin in range(num_bins):
                q_i = self._detection_probabilities[current_bin]
                
                if np.random.rand() < q_i:
                    histogram[current_bin] += 1
                    detected = True
                    break  # 检测到光子，立即结束当前周期
            
            if not detected:
                histogram[-1] += 1  # 未检测到光子，记录溢出仓
        
        self._simulated_histogram = histogram
    def update_coates_estimation(self):
        """
        使用Coates估计器还原真实光通量，并生成相应的直方图
        """
        coates_flux = self.coates_estimator(self._simulated_histogram)
        expected_counts = coates_flux * self._cycles
        expected_counts = np.clip(expected_counts, 0, None)  # 确保值非负

        raw_histogram = np.random.poisson(expected_counts)
        total_counts = np.sum(raw_histogram)
        
        if total_counts > 0:
            self._coates_estimation = (raw_histogram / total_counts) * self._cycles
            self._coates_estimation = np.round(self._coates_estimation).astype(int)
        else:
            self._coates_estimation = raw_histogram


    def update_simulated_ideal_histogram(self):
        """
        模拟理想泊松累积（无堆积效应），并归一化以确保总次数为_cycles
        """
        expected_counts = self._flux * self._cycles
        raw_histogram = np.random.poisson(expected_counts)
        total_counts = np.sum(raw_histogram)
        
        if total_counts > 0:
            self._ideal_histogram = (raw_histogram / total_counts) * self._cycles
            self._ideal_histogram = np.round(self._ideal_histogram).astype(int)
        else:
            self._ideal_histogram = raw_histogram

    def generate_flux(self):
        """
        生成光场光通量（需要在子类中实现）
        """
        self.update_flux()
        return self._flux

    def simulate_ideal_poisson(self):
        """
        模拟理想泊松累积（无堆积效应）
        """
        self.update_simulated_ideal_histogram()
        return self._ideal_histogram

    def calculate_detection_probability(self):
        """
        计算每个bin的光子检测概率Pi = 1 - e^{-r_i}
        """
        self.update_detection_probabilities()
        return self._detection_probabilities

    def simulate_sync_spad(self):
        """
        模拟SPAD检测（包含堆积效应）
        """
        self.update_simulated_histogram()
        return self._simulated_histogram
    def coates_estimation(self):
        """
        模拟SPAD检测（包含堆积效应）
        """
        self.update_coates_estimation()
        return self._coates_estimation
    
    def plot_hist_plotly(self):
        fig = go.Figure()

        # 理想泊松直方图
        fig.add_trace(go.Bar(
            x=list(range(len(self._ideal_histogram))),
            y=self._ideal_histogram,
            name='Ideal Poisson Accumulation (No Pile-up)',
            marker_color='#1f77b4'
        ))

        # 同步SPAD检测直方图
        fig.add_trace(go.Bar(
            x=list(range(len(self._simulated_histogram)-1)),
            y=self._simulated_histogram[:-1],
            name='Sync SPAD Detection Histogram',
            marker_color='#ff7f0e'
        ))

        fig.update_layout(
            title='Photon Counting Histogram',
            xaxis_title='Time Bin',
            yaxis_title='Photon Counts',
            barmode='group',
            legend_title_text='Histogram Type',
            font=dict(
                family="Arial, sans-serif",
                size=18,
                color="Black"
            ),
            title_font=dict(
                family="Arial, sans-serif",
                size=22,
                color="Black"
            ),
            legend=dict(
                x=0.99,
                y=0.99,
                xanchor='right',
                yanchor='top',
                bgcolor='rgba(255,255,255,0.5)',
                bordercolor='rgba(0,0,0,0.5)',
                borderwidth=1
            )
        )
        fig.show()
        

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
    def coates_estimator(histogram):
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
        valid_S = S[~np.isnan(S) & (S >= 0)]
        return valid_S
        
        
 
class SingleGaussian(SPADSimulateEngine):
    def __init__(self, num_bins=100, pulse_pos=50, pulse_width=5, signal_strength=0.5, bg_strength=0.1, cycles=1000):
        self._pulse_pos = pulse_pos
        self._pulse_width = pulse_width
        super().__init__(num_bins, signal_strength, bg_strength, cycles)

    def update_flux(self):
        """
        生成光场光通量（单高斯脉冲信号 + 均匀背景）
        """
        x = np.arange(self._num_bins)
        pulse = self._signal_strength * np.exp(-0.5 * ((x - self._pulse_pos) / self._pulse_width)**2)
        background = self._bg_strength * np.ones(self._num_bins)
        self._flux = pulse + background
        
class DoubleGaussian(SPADSimulateEngine):
    def __init__(self, num_bins=100, pulse_pos1=40, pulse_pos2=70, pulse_width1=5, pulse_width2=5, signal_strength=0.5, bg_strength=0.1, cycles=1000):
        self._pulse_pos1 = pulse_pos1
        self._pulse_pos2 = pulse_pos2
        self._pulse_width1 = pulse_width1
        self._pulse_width2 = pulse_width2
        super().__init__(num_bins, signal_strength, bg_strength, cycles)

    def update_flux(self):
        """
        生成光场光通量（双高斯脉冲信号 + 均匀背景）
        """
        x = np.arange(self._num_bins)
        pulse1 = self._signal_strength * np.exp(-0.5 * ((x - self._pulse_pos1) / self._pulse_width1)**2)
        pulse2 = self._signal_strength * np.exp(-0.5 * ((x - self._pulse_pos2) / self._pulse_width2)**2)
        background = self._bg_strength * np.ones(self._num_bins)
        self._flux = pulse1 + pulse2 + background
