import numpy as np
from .SPADSimulateEngine import SPADSimulateEngine
import plotly.graph_objects as go
from scipy.optimize import minimize


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
        
    def estimate_parameters_mle(self):
        """
        使用最大似然估计从Coates估计结果中推断双高斯原始参数
        
        返回:
            dict: 包含估计的参数 (pulse_pos1, pulse_pos2, pulse_width1, pulse_width2, 
                  signal_strength1, signal_strength2, bg_strength)
        """
        # 确保已有Coates估计结果
        if not hasattr(self, '_coates_estimation'):
            self.update_coates_estimation()
        
        # 提取Coates估计的光子数分布
        estimated_flux = self._coates_estimation / self._cycles
        
        # 定义参数约束 
        # 参数顺序: [pulse_pos1, pulse_pos2, pulse_width1, pulse_width2, signal_strength1, signal_strength2, bg_strength]
        bounds = [
            (0, self._num_bins - 1),      # pulse_pos1 在时间窗口内
            (0, self._num_bins - 1),      # pulse_pos2 在时间窗口内
            (0.1, self._num_bins/5),      # pulse_width1 合理范围
            (0.1, self._num_bins/5),      # pulse_width2 合理范围
            (0.01, 5.0),                  # signal_strength1 合理范围
            (0.01, 5.0),                  # signal_strength2 合理范围
            (0.001, 1.0)                  # bg_strength 合理范围
        ]
        
        # 定义目标函数 (负对数似然函数)
        def negative_log_likelihood(params):
            pos1, pos2, width1, width2, strength1, strength2, bg = params
            
            # 生成候选模型的光通量
            x = np.arange(self._num_bins)
            pulse1 = strength1 * np.exp(-0.5 * ((x - pos1) / width1)**2)
            pulse2 = strength2 * np.exp(-0.5 * ((x - pos2) / width2)**2)
            model_flux = pulse1 + pulse2 + bg
            
            # 计算负对数似然 (泊松分布)
            # 避免对数中的零值或负值
            model_flux = np.maximum(model_flux, 1e-10)  
            
            # 使用泊松似然计算，对于每个bin计算似然并求和
            neg_log_likelihood = np.sum(model_flux - estimated_flux * np.log(model_flux))
            
            return neg_log_likelihood
        
        # 初始猜测值，使用类的当前参数
        initial_guess = [
            self._pulse_pos1,
            self._pulse_pos2, 
            self._pulse_width1,
            self._pulse_width2,
            self._signal_strength1,
            self._signal_strength2,
            self._bg_strength
        ]
        
        # 执行优化
        result = minimize(
            negative_log_likelihood,
            initial_guess,
            method='L-BFGS-B',  # 边界约束优化算法
            bounds=bounds
        )
        
        # 提取结果
        estimated_pos1, estimated_pos2, estimated_width1, estimated_width2, estimated_strength1, estimated_strength2, estimated_bg = result.x
        
        # 创建结果字典
        mle_params = {
            'pulse_pos1': estimated_pos1,
            'pulse_pos2': estimated_pos2,
            'pulse_width1': estimated_width1,
            'pulse_width2': estimated_width2,
            'signal_strength1': estimated_strength1,
            'signal_strength2': estimated_strength2,
            'bg_strength': estimated_bg,
            'optimization_success': result.success,
            'optimization_message': result.message
        }
        
        # 在返回前打印结果到终端
        print("原始参数:")
        print(f"  脉冲1位置: {self._pulse_pos1}, 脉冲1宽度: {self._pulse_width1}")
        print(f"  脉冲2位置: {self._pulse_pos2}, 脉冲2宽度: {self._pulse_width2}")
        print(f"  信号1强度: {self._signal_strength1}, 信号2强度: {self._signal_strength2}")
        print(f"  背景强度: {self._bg_strength}")
        print("\nMLE估计参数:")
        print(f"  脉冲1位置: {mle_params['pulse_pos1']:.2f}, 脉冲1宽度: {mle_params['pulse_width1']:.2f}")
        print(f"  脉冲2位置: {mle_params['pulse_pos2']:.2f}, 脉冲2宽度: {mle_params['pulse_width2']:.2f}")
        print(f"  信号1强度: {mle_params['signal_strength1']:.2f}, 信号2强度: {mle_params['signal_strength2']:.2f}")
        print(f"  背景强度: {mle_params['bg_strength']:.2f}")
        print(f"  优化成功: {mle_params['optimization_success']}")
        
        # 返回估计的参数
        return mle_params
    
    def update_mle_params(self):
        """
        更新参数估计结果
        """
        self._mle_params = self.estimate_parameters_mle()
        
    def plot_mle_comparison(self):
        """
        绘制原始光通量、MLE估计和Coates估计的比较图
        """
        # 获取MLE估计的参数
        self.update_mle_params()
        mle_params=self._mle_params
        
        # 生成原始x轴数据点
        x_bins = np.arange(self._num_bins)
        
        # 创建高分辨率x轴数据用于平滑曲线
        x_smooth = np.linspace(0, self._num_bins - 1, self._num_bins * 10)
        
        # 使用已计算的归一化flux
        original_flux_normalized = self._normalized_flux
        
        # MLE估计的光通量 (高分辨率)
        mle_pulse1_smooth = mle_params['signal_strength1'] * np.exp(-0.5 * ((x_smooth - mle_params['pulse_pos1']) / mle_params['pulse_width1'])**2)
        mle_pulse2_smooth = mle_params['signal_strength2'] * np.exp(-0.5 * ((x_smooth - mle_params['pulse_pos2']) / mle_params['pulse_width2'])**2)
        mle_flux_smooth = mle_pulse1_smooth + mle_pulse2_smooth + mle_params['bg_strength']
        
        # Coates估计的光通量
        coates_flux = self._coates_estimation / self._cycles
        
        # 计算归一化参数
        total_flux = np.sum(self._flux)
        signal_strength1_normalized = self._signal_strength1 / total_flux
        signal_strength2_normalized = self._signal_strength2 / total_flux
        bg_strength_normalized = self._bg_strength / total_flux
        
        # 计算原始模型的高分辨率版本
        original_pulse1_smooth = self._signal_strength1 * np.exp(-0.5 * ((x_smooth - self._pulse_pos1) / self._pulse_width1)**2)
        original_pulse2_smooth = self._signal_strength2 * np.exp(-0.5 * ((x_smooth - self._pulse_pos2) / self._pulse_width2)**2)
        original_flux_smooth = original_pulse1_smooth + original_pulse2_smooth + self._bg_strength
        original_flux_smooth_normalized = original_flux_smooth / np.sum(original_flux_smooth) * np.sum(self._normalized_flux)
        
        # 创建图表
        fig = go.Figure()
        
        # 添加归一化原始光通量曲线 (平滑版本)
        fig.add_trace(go.Scatter(
            x=x_smooth, 
            y=original_flux_smooth_normalized,
            mode='lines',
            name='Normalized Original Flux',
            line=dict(color='#1f77b4', width=3)
        ))
        
        # 添加MLE估计的光通量曲线 (平滑版本)
        fig.add_trace(go.Scatter(
            x=x_smooth, 
            y=mle_flux_smooth,
            mode='lines',
            name='MLE Estimated Flux',
            line=dict(color='#ff7f0e', width=3)
        ))
        
        # 添加Coates估计点
        fig.add_trace(go.Scatter(
            x=x_bins[:len(coates_flux)], 
            y=coates_flux,
            mode='markers',
            name='Coates Estimation',
            marker=dict(color='#2ca02c', size=8)
        ))
        
        # 设置图表布局 - 保持与SingleGaussian类一致的样式
        fig.update_layout(
            title='Normalized Flux Comparison',
            xaxis_title='Time Bin',
            yaxis_title='Normalized Photon Flux',
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
                borderwidth=1,
                font=dict(
                    size=12
                )
            )
        )

        # 添加参数信息到图表，显示归一化后的参数
        parameter_text = (
            f"Real: pos1={self._pulse_pos1:.1f}, pos2={self._pulse_pos2:.1f}, "
            f"width1={self._pulse_width1:.1f}, width2={self._pulse_width2:.1f}<br>"
            f"strengths={signal_strength1_normalized:.2f}/{signal_strength2_normalized:.2f}, bg={bg_strength_normalized:.2f}<br>"
            f"MLE: pos1={mle_params['pulse_pos1']:.1f}, pos2={mle_params['pulse_pos2']:.1f}, "
            f"width1={mle_params['pulse_width1']:.1f}, width2={mle_params['pulse_width2']:.1f}<br>"
            f"strengths={mle_params['signal_strength1']:.2f}/{mle_params['signal_strength2']:.2f}, bg={mle_params['bg_strength']:.2f}"
        )
        
        fig.add_annotation(
            xref="paper", yref="paper",
            x=0.01, y=0.98,
            text=parameter_text,
            showarrow=False,
            font=dict(size=14),
            bgcolor="rgba(255,255,255,0.7)",
            bordercolor="black",
            borderwidth=1,
            align="left"
        )
        
        fig.show()

    def plot_combined_hist_mle(self):
        """
        绘制直方图和MLE估计结果的合并图
        """
        # 创建图表
        fig = go.Figure()
        
        # --- 添加直方图部分 ---
        # 理想泊松直方图
        fig.add_trace(go.Bar(
            x=list(range(len(self._ideal_histogram))),
            y=self._ideal_histogram,
            name='Ideal Poisson Histogram',
            marker_color='rgba(31, 119, 180, 1.0)',
            opacity=1.0
        ))

        # Coates估计直方图
        fig.add_trace(go.Bar(
            x=list(range(len(self._coates_estimation))),
            y=self._coates_estimation,
            name='Coates Estimation Histogram',
            marker_color='rgba(44, 160, 44, 1.0)',
            opacity=1.0
        ))

        # 同步SPAD检测直方图
        fig.add_trace(go.Bar(
            x=list(range(len(self._simulated_histogram)-1)),
            y=self._simulated_histogram[:-1],
            name='SPAD Detection Histogram',
            marker_color='rgba(255, 127, 14, 1.0)',
            opacity=1.0
        ))
        
        # 获取MLE估计的参数
        self.update_mle_params()
        mle_params=self._mle_params
        
        # 生成标准x轴数据
        x_bins = np.arange(self._num_bins)
        
        # 生成高分辨率x轴数据用于平滑曲线
        x_smooth = np.linspace(0, self._num_bins - 1, self._num_bins * 10)
        
        # 归一化参数计算
        total_flux = np.sum(self._flux)
        signal_strength1_normalized = self._signal_strength1 / total_flux * self._cycles
        signal_strength2_normalized = self._signal_strength2 / total_flux * self._cycles
        bg_strength_normalized = self._bg_strength / total_flux * self._cycles
        
        # 原始光通量曲线（高分辨率平滑版本）
        original_pulse1_smooth = signal_strength1_normalized * np.exp(-0.5 * ((x_smooth - self._pulse_pos1) / self._pulse_width1)**2)
        original_pulse2_smooth = signal_strength2_normalized * np.exp(-0.5 * ((x_smooth - self._pulse_pos2) / self._pulse_width2)**2)
        original_flux_smooth = original_pulse1_smooth + original_pulse2_smooth + bg_strength_normalized
        
        fig.add_trace(go.Scatter(
            x=x_smooth, 
            y=original_flux_smooth,
            mode='lines',
            name='Original Flux Model',
            line=dict(color='#4C72B0', width=3)  # 科研绘图深蓝色
        ))
        
        # MLE估计的光通量曲线（高分辨率平滑版本）
        # 缩放MLE估计结果到直方图尺度
        mle_signal1_scaled = mle_params['signal_strength1'] * self._cycles
        mle_signal2_scaled = mle_params['signal_strength2'] * self._cycles
        mle_bg_scaled = mle_params['bg_strength'] * self._cycles
        
        mle_pulse1_smooth = mle_signal1_scaled * np.exp(-0.5 * ((x_smooth - mle_params['pulse_pos1']) / mle_params['pulse_width1'])**2)
        mle_pulse2_smooth = mle_signal2_scaled * np.exp(-0.5 * ((x_smooth - mle_params['pulse_pos2']) / mle_params['pulse_width2'])**2)
        mle_flux_smooth = mle_pulse1_smooth + mle_pulse2_smooth + mle_bg_scaled
        
        fig.add_trace(go.Scatter(
            x=x_smooth, 
            y=mle_flux_smooth,
            mode='lines',
            name='MLE Estimated Model',
            line=dict(color='#C44E52', width=3)  # 科研绘图红色
        ))
        
        # 设置图表布局
        fig.update_layout(
            title='Photon Counting Histogram with MLE Double Gaussian Model',
            xaxis_title='Time Bin',
            yaxis_title='Photon Counts',
            barmode='group',
            legend_title_text='Data Type',
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
                bgcolor='rgba(255,255,255,0.7)',
                bordercolor='rgba(0,0,0,0.5)',
                borderwidth=1,
                font=dict(
                    size=12
                )
            )
        )
        
        # 添加参数信息到图表
        parameter_text = (
            f"Original: pos1={self._pulse_pos1:.1f}, pos2={self._pulse_pos2:.1f}, "
            f"width1={self._pulse_width1:.1f}, width2={self._pulse_width2:.1f}<br>"
            f"MLE: pos1={mle_params['pulse_pos1']:.1f}, pos2={mle_params['pulse_pos2']:.1f}, "
            f"width1={mle_params['pulse_width1']:.1f}, width2={mle_params['pulse_width2']:.1f}<br>"
            f"MLE Success: {mle_params['optimization_success']}"
        )
        
        fig.add_annotation(
            xref="paper", yref="paper",
            x=0.01, y=0.98,
            text=parameter_text,
            showarrow=False,
            font=dict(size=14),
            bgcolor="rgba(255,255,255,0.7)",
            bordercolor="black",
            borderwidth=1,
            align="left"
        )
        
        fig.show()


