import numpy as np
import matplotlib.pyplot as plt

class GatingAlgorithm:
    def __init__(self, num_bins=100, pulse_pos=50, pulse_width=5, signal_strength=0.5, bg_strength=0.1):
        self.num_bins = num_bins
        self.pulse_pos = pulse_pos
        self.pulse_width = pulse_width
        self.signal_strength = signal_strength
        self.bg_strength = bg_strength

        self.flux = self.generate_flux()

    def generate_flux(self):
        """
        生成光场光通量（脉冲信号 + 均匀背景）
        
        参数：
            num_bins: 时间仓数量
            pulse_pos: 脉冲中心位置（bin索引）
            pulse_width: 脉冲高斯分布的标准差
            signal_strength: 脉冲峰值强度（光子数/仓）
            bg_strength: 背景光强度（光子数/仓）
            
        返回：
            flux: 光通量波形（每个仓的光子到达率r_i）
        """
        x = np.arange(self.num_bins)
        pulse = self.signal_strength * np.exp(-0.5 * ((x - self.pulse_pos) / self.pulse_width)**2)
        background = self.bg_strength * np.ones(self.num_bins)
        return pulse + background

    def simulate_ideal_poisson(self, num_cycles=1000):
        """
        模拟理想泊松累积（无堆积效应）
        
        参数：
            num_cycles: 激光周期数
            
        返回：
            histogram: 理想泊松直方图（每个仓的独立泊松计数）
        """
        expected_counts = self.flux * num_cycles
        Pi = np.random.poisson(expected_counts)
        return Pi

    def calculate_detection_probability(self):
        """
        计算每个bin的光子检测概率Pi = 1 - e^{-r_i}
        (对应论文中的q_i参数)
        """
        return 1 - np.exp(-self.flux)

    def simulate_sync_spad(self, num_cycles=1000):
        """
        模拟SPAD检测（包含堆积效应）
        
        参数：
            num_cycles: 激光周期数
            dead_time: 死区时间（仓数）
            
        返回：
            histogram: 实际检测到的光子直方图
        """
        num_bins = len(self.flux)
        histogram = np.zeros(num_bins + 1, dtype=int)  # 最后一个是溢出仓
        
        for _ in range(num_cycles):
            detected = False
            for current_bin in range(num_bins):
                # 计算检测概率 q_i = 1 - e^{-r_i}
                r_i = self.flux[current_bin]
                q_i = 1 - np.exp(-r_i)
                
                if np.random.rand() < q_i:
                    histogram[current_bin] += 1
                    detected = True
                    break  # 检测到光子，立即结束当前周期
            
            if not detected:
                histogram[-1] += 1  # 未检测到光子，记录溢出仓
        
        return histogram

    def plot_sync_results(self, ideal_hist, sync_hist, Pi):
        """可视化结果（新增Pi曲线）"""
        plt.figure(figsize=(12, 12))
        
        # 新增子图位置 1: 检测概率曲线
        plt.subplot(5, 1, 1)
        plt.plot(Pi, 'm-', linewidth=2)
        plt.title('Photon Detection Probability $P_i = 1 - e^{-r_i}$')
        plt.xlabel('Time Bin')
        plt.ylabel('Probability')
        plt.grid(True)
        
        # 调整原有子图位置
        # 原始光场（位置2）
        plt.subplot(5, 1, 2)
        plt.plot(self.flux, 'b-', label='True Flux')
        plt.title('True Photon Flux (Signal + Background)')
        plt.xlabel('Time Bin')
        plt.ylabel('Photon Rate')
        plt.grid(True)
        
        # 理想泊松直方图（位置3）
        plt.subplot(5, 1, 3)
        plt.bar(range(len(ideal_hist)), ideal_hist, color='g', alpha=0.7)
        plt.title('Ideal Poisson Accumulation (No Pile-up)')
        plt.xlabel('Time Bin')
        plt.ylabel('Photon Counts')
        plt.grid(True)
        
        # 同步SPAD检测直方图（位置4）
        plt.subplot(5, 1, 4)
        plt.bar(range(len(sync_hist)-1), sync_hist[:-1], color='r', alpha=0.7)
        plt.title('Sync SPAD Detection (Main Bins)')
        plt.xlabel('Time Bin')
        plt.ylabel('Photon Counts')
        plt.grid(True)
        
        # 溢出仓（位置5）
        plt.subplot(5, 1, 5)
        plt.bar(['Overflow'], [sync_hist[-1]], color='purple', alpha=0.7)
        plt.title('Overflow Bin (No Photon Detected)')
        plt.ylabel('Cycle Counts')
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
