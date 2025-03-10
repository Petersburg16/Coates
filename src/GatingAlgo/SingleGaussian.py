import numpy as np
import plotly.graph_objects as go
from scipy.optimize import minimize
from .SPADSimulateEngine import SPADSimulateEngine

class SingleGaussian(SPADSimulateEngine):
    def __init__(self, num_bins=100, pulse_pos=50, pulse_width=5, signal_strength=0.5, bg_strength=0.1, cycles=1000):
        self._pulse_pos = pulse_pos
        self._pulse_width = pulse_width
        self._signal_strength = signal_strength
        self._bg_strength = bg_strength
        super().__init__(num_bins, cycles)

    def update_flux(self):
        """
        生成光场光通量（单高斯脉冲信号 + 均匀背景）
        """
        x = np.arange(self._num_bins)
        pulse = self._signal_strength * np.exp(-0.5 * ((x - self._pulse_pos) / self._pulse_width)**2)
        background = self._bg_strength * np.ones(self._num_bins)
        self._flux = pulse + background
    
    def estimate_parameters_mle(self):
        """
        使用最大似然估计从Coates估计结果中推断原始高斯参数
        
        返回:
            dict: 包含估计的参数 (pulse_pos, pulse_width, signal_strength, bg_strength)
        """
        from scipy.optimize import minimize
        
        # 确保已有Coates估计结果
        if not hasattr(self, '_coates_estimation'):
            self.update_coates_estimation()
        
        # 提取Coates估计的光子数分布
        estimated_flux = self._coates_estimation / self._cycles
        
        # 定义参数约束 (参数顺序: [pulse_pos, pulse_width, signal_strength, bg_strength])
        bounds = [
            (0, self._num_bins - 1),      # pulse_pos 在时间窗口内
            (0.1, self._num_bins / 5),    # pulse_width 合理范围
            (0.01, 5.0),                  # signal_strength 合理范围
            (0.001, 1.0)                  # bg_strength 合理范围
        ]
        
        # 定义目标函数 (负对数似然函数)
        def negative_log_likelihood(params):
            pos, width, strength, bg = params
            
            # 生成候选模型的光通量
            x = np.arange(self._num_bins)
            pulse = strength * np.exp(-0.5 * ((x - pos) / width)**2)
            model_flux = pulse + bg
            
            # 计算负对数似然 (泊松分布)
            # 避免对数中的零值或负值
            model_flux = np.maximum(model_flux, 1e-10)  
            
            # 使用泊松似然计算，对于每个bin计算似然并求和
            neg_log_likelihood = np.sum(model_flux - estimated_flux * np.log(model_flux))
            
            return neg_log_likelihood
        
        # 初始猜测值，使用类的当前参数
        initial_guess = [
            self._pulse_pos, 
            self._pulse_width,
            self._signal_strength,
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
        estimated_pos, estimated_width, estimated_strength, estimated_bg = result.x
        
        # 返回估计的参数
        return {
            'pulse_pos': estimated_pos,
            'pulse_width': estimated_width,
            'signal_strength': estimated_strength,
            'bg_strength': estimated_bg,
            'optimization_success': result.success,
            'optimization_message': result.message
        }
    def plot_mle_comparison(self):

        
        # 获取MLE估计的参数
        mle_params = self.estimate_parameters_mle()
        
        # 生成x轴数据
        x = np.arange(self._num_bins)
        
        # 归一化原始光通量
        original_pulse = self._signal_strength * np.exp(-0.5 * ((x - self._pulse_pos) / self._pulse_width)**2)
        original_background = self._bg_strength * np.ones(self._num_bins)
        original_flux_raw = original_pulse + original_background
        
        # 使用已计算的归一化flux
        original_flux_normalized = self._normalized_flux
        
        # MLE估计的光通量
        mle_pulse = mle_params['signal_strength'] * np.exp(-0.5 * ((x - mle_params['pulse_pos']) / mle_params['pulse_width'])**2)
        mle_flux = mle_pulse + mle_params['bg_strength']
        
        # Coates估计的光通量
        coates_flux = self._coates_estimation / self._cycles
        
        # 创建图表
        fig = go.Figure()
        
        # 添加归一化原始光通量曲线
        fig.add_trace(go.Scatter(
            x=x, 
            y=original_flux_normalized,
            mode='lines',
            name='Normalized Original Flux',
            line=dict(color='blue', width=2)
        ))
        
        # 添加MLE估计的光通量曲线
        fig.add_trace(go.Scatter(
            x=x, 
            y=mle_flux,
            mode='lines',
            name='MLE Estimated Flux',
            line=dict(color='red', width=2)
        ))
        
        # 添加Coates估计点
        fig.add_trace(go.Scatter(
            x=x[:len(coates_flux)], 
            y=coates_flux,
            mode='markers',
            name='Coates Estimation',
            marker=dict(color='green', size=5)
        ))
        
        # 设置图表布局
        fig.update_layout(
            title=f'Normalized Flux Comparison (MLE Success: {mle_params["optimization_success"]})',
            xaxis_title='Time Bin',
            yaxis_title='Normalized Photon Flux',
            legend_title='Flux Source',
            font=dict(
                family="Arial, sans-serif",
                size=14,
                color="Black"
            )
        )
    
        # 添加参数信息到图表，显示归一化前的原始参数
        parameter_text = (
            f"Original: pos={self._pulse_pos:.2f}, width={self._pulse_width:.2f}, "
            f"strength={self._signal_strength:.2f}, bg={self._bg_strength:.2f}<br>"
            f"MLE: pos={mle_params['pulse_pos']:.2f}, width={mle_params['pulse_width']:.2f}, "
            f"strength={mle_params['signal_strength']:.2f}, bg={mle_params['bg_strength']:.2f}"
        )
        
        fig.add_annotation(
            xref="paper", yref="paper",
            x=0.01, y=0.98,
            text=parameter_text,
            showarrow=False,
            font=dict(size=12),
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="black",
            borderwidth=1
        )
        
        fig.show()