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
            (0.1, self._num_bins ),    # pulse_width 合理范围
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
        
        # 使用已计算的归一化flux
        original_flux_normalized = self._normalized_flux
        
        # MLE估计的光通量
        mle_pulse = mle_params['signal_strength'] * np.exp(-0.5 * ((x - mle_params['pulse_pos']) / mle_params['pulse_width'])**2)
        mle_flux = mle_pulse + mle_params['bg_strength']
        
        # Coates估计的光通量
        coates_flux = self._coates_estimation / self._cycles
        
        total_flux = np.sum(self._flux)
        signal_strength_normalized= self._signal_strength / total_flux
        bg_strength_normalized = self._bg_strength / total_flux
        
        # 创建图表
        fig = go.Figure()
        
        # 添加归一化原始光通量曲线
        fig.add_trace(go.Scatter(
            x=x, 
            y=original_flux_normalized,
            mode='lines',
            name='Normalized Original Flux',
            line=dict(color='#1f77b4', width=3)
        ))
        
        # 添加MLE估计的光通量曲线
        fig.add_trace(go.Scatter(
            x=x, 
            y=mle_flux,
            mode='lines',
            name='MLE Estimated Flux',
            line=dict(color='#ff7f0e', width=3)
        ))
        
        # 添加Coates估计点
        fig.add_trace(go.Scatter(
            x=x[:len(coates_flux)], 
            y=coates_flux,
            mode='markers',
            name='Coates Estimation',
            marker=dict(color='#2ca02c', size=8)
        ))
        
        # 设置图表布局 - 保持与父类一致的样式
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
            f"Normalized Real Flux: pos={self._pulse_pos:.2f}, width={self._pulse_width:.2f}, "
            f"strength={signal_strength_normalized:.2f}, bg={bg_strength_normalized:.2f}<br>"
            f"MLE Estimated Flux: pos={mle_params['pulse_pos']:.2f}, width={mle_params['pulse_width']:.2f}, "
            f"strength={mle_params['signal_strength']:.2f}, bg={mle_params['bg_strength']:.2f}"
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
        该方法仅适用于实现了estimate_parameters_mle方法的子类
        """
        if not hasattr(self, 'estimate_parameters_mle'):
            print("此对象不支持MLE参数估计")
            return self.plot_hist_plotly()
        
        # 创建图表
        fig = go.Figure()
        
        # --- 添加直方图部分 ---
        # 理想泊松直方图
        fig.add_trace(go.Bar(
            x=list(range(len(self._ideal_histogram))),
            y=self._ideal_histogram,
            name='Ideal Poisson Histogram',
            marker_color='rgba(31, 119, 180, 1.0)',  # 移除透明度
            opacity=1.0  # 设置不透明度为1
        ))

        # Coates估计直方图
        fig.add_trace(go.Bar(
            x=list(range(len(self._coates_estimation))),
            y=self._coates_estimation,
            name='Coates Estimation Histogram',
            marker_color='rgba(44, 160, 44, 1.0)',  # 移除透明度
            opacity=1.0  # 设置不透明度为1
        ))

        # 同步SPAD检测直方图
        fig.add_trace(go.Bar(
            x=list(range(len(self._simulated_histogram)-1)),
            y=self._simulated_histogram[:-1],
            name='SPAD Detection Histogram',
            marker_color='rgba(255, 127, 14, 1.0)',  # 移除透明度
            opacity=1.0  # 设置不透明度为1
        ))
        
        # --- 添加MLE曲线部分 ---
        # 获取MLE估计参数
        mle_params = self.estimate_parameters_mle()
        
        # 生成x轴数据
        x = np.arange(self._num_bins)
        
        # 归一化参数计算
        total_flux = np.sum(self._flux)
        if hasattr(self, '_signal_strength') and hasattr(self, '_bg_strength'):
            signal_strength_normalized = self._signal_strength / total_flux * self._cycles
            bg_strength_normalized = self._bg_strength / total_flux * self._cycles
        
            # 原始光通量曲线（与直方图对应刻度）
            if hasattr(self, '_pulse_pos') and hasattr(self, '_pulse_width'):
                original_pulse = signal_strength_normalized * np.exp(-0.5 * ((x - self._pulse_pos) / self._pulse_width)**2)
                original_flux_scaled = original_pulse + bg_strength_normalized
                
                fig.add_trace(go.Scatter(
                    x=x, 
                    y=original_flux_scaled,
                    mode='lines',
                    name='Original Flux Model',
                    line=dict(color='#4C72B0', width=3)  # 科研绘图深蓝色
                ))
        
        # MLE估计的光通量曲线（与直方图对应刻度）
        if 'signal_strength' in mle_params and 'bg_strength' in mle_params:
            # 缩放MLE估计结果到直方图尺度
            mle_signal_scaled = mle_params['signal_strength'] * self._cycles
            mle_bg_scaled = mle_params['bg_strength'] * self._cycles
            
            if 'pulse_pos' in mle_params and 'pulse_width' in mle_params:
                mle_pulse = mle_signal_scaled * np.exp(-0.5 * ((x - mle_params['pulse_pos']) / mle_params['pulse_width'])**2)
                mle_flux_scaled = mle_pulse + mle_bg_scaled
                
                fig.add_trace(go.Scatter(
                    x=x, 
                    y=mle_flux_scaled,
                    mode='lines',
                    name='MLE Estimated Model',
                    line=dict(color='#C44E52', width=3)  # 科研绘图红色
                ))
        
        # 设置图表布局
        fig.update_layout(
            title='Photon Counting Histogram with MLE Model Estimation',
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
        
        # 如果是SingleGaussian类，添加参数信息到图表
        if hasattr(self, '_pulse_pos') and hasattr(self, '_pulse_width'):
            parameter_text = (
                f"Original: pos={self._pulse_pos:.2f}, width={self._pulse_width:.2f}<br>"
                f"MLE Estimated: pos={mle_params['pulse_pos']:.2f}, width={mle_params['pulse_width']:.2f}<br>"
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
