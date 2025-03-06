import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']

class GatingAlgorithm:
    def __init__(self, num_bins=100, pulse_pos=50, pulse_width=5, signal_strength=0.5, bg_strength=0.1, cycles=1000):
        self._num_bins = num_bins
        self._pulse_pos = pulse_pos
        self._pulse_width = pulse_width
        self._signal_strength = signal_strength
        self._bg_strength = bg_strength
        self._cycles = cycles

        self.update_flux()
        self.update_simulated_ideal_histogram()
        self.update_detection_probabilities()
        self.update_simulated_histogram()
        

    def update_flux(self):
        """
        生成光场光通量（脉冲信号 + 均匀背景）
        """
        x = np.arange(self._num_bins)
        pulse = self._signal_strength * np.exp(-0.5 * ((x - self._pulse_pos) / self._pulse_width)**2)
        background = self._bg_strength * np.ones(self._num_bins)
        self._flux = pulse + background

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
    def update_simulated_ideal_histogram(self):
        """
        模拟理想泊松累积（无堆积效应）
        """
        expected_counts = self._flux * self._cycles
        self._ideal_histogram = np.random.poisson(expected_counts)

        
    def generate_flux(self):
        """
        生成光场光通量（脉冲信号 + 均匀背景）
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

    def plot_sync_results(self):
        """可视化结果（新增Pi曲线）"""
        plt.figure(figsize=(12, 12))
        
        # 新增子图位置 1: 检测概率曲线
        plt.subplot(5, 1, 1)
        plt.plot(self._detection_probabilities, 'm-', linewidth=2)
        plt.title('Photon Detection Probability $P_i = 1 - e^{-r_i}$')
        plt.xlabel('Time Bin')
        plt.ylabel('Probability')
        plt.grid(True)
        
        # 调整原有子图位置
        # 原始光场（位置2）
        plt.subplot(5, 1, 2)
        plt.plot(self._flux, 'b-', label='True Flux')
        plt.title('True Photon Flux (Signal + Background)')
        plt.xlabel('Time Bin')
        plt.ylabel('Photon Rate')
        plt.grid(True)
        
        # 理想泊松直方图（位置3）
        plt.subplot(5, 1, 3)
        plt.bar(range(len(self._ideal_histogram)), self._ideal_histogram, color='g', alpha=0.7)
        plt.title('Ideal Poisson Accumulation (No Pile-up)')
        plt.xlabel('Time Bin')
        plt.ylabel('Photon Counts')
        plt.grid(True)
        
        # 同步SPAD检测直方图（位置4）
        plt.subplot(5, 1, 4)
        plt.bar(range(len(self._simulated_histogram)-1), self._simulated_histogram[:-1], color='r', alpha=0.7)
        plt.title('Sync SPAD Detection (Main Bins)')
        plt.xlabel('Time Bin')
        plt.ylabel('Photon Counts')
        plt.grid(True)
        
        # 溢出仓（位置5）
        plt.subplot(5, 1, 5)
        plt.bar(['Overflow'], [self._simulated_histogram[-1]], color='purple', alpha=0.7)
        plt.title('Overflow Bin (No Photon Detected)')
        plt.ylabel('Cycle Counts')
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
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
        
        
        
    def plot_hist(self):
        plt.figure(figsize=(6, 4))
        # 理想泊松直方图
        plt.subplot(2, 1, 1)
        plt.bar(range(len(self._ideal_histogram)), self._ideal_histogram, color='#1f77b4', alpha=0.7)
        plt.title("理想泊松累积直方图(无堆积效应)")
        plt.xlabel('时间仓')
        plt.ylabel('光子计数')
        plt.grid(True)
        
        # 同步SPAD检测直方图
        plt.subplot(2, 1, 2) 
        plt.bar(range(len(self._simulated_histogram)-1), self._simulated_histogram[:-1], color='#ff7f0e', alpha=0.7)
        plt.title('同步SPAD探测直方图')
        plt.xlabel('时间仓')
        plt.ylabel('光子计数')
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
        

