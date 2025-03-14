from .DataLoader import DataLoader
import numpy as np
import plotly.graph_objs as go
from ipywidgets import HBox, VBox
from ..GatingAlgo.SPADSimulateEngine import SPADSimulateEngine

from plotly.subplots import make_subplots
# 创建颜色映射函数 - 使用Viridis配色
import matplotlib.cm as cm
from matplotlib.colors import Normalize
class DataChecker(DataLoader):
    def draw_strength(self, mode='original'):
        """
        绘制强度图和直方图
        
        参数:
            mode: 字符串，指定直方图显示模式
                - 'original': 原始直方图
                - 'coates': Coates估计后的直方图
                - 'compare': 对比显示原始和Coates估计直方图
        """
        strength_matrix = self._strength_matrix
        fig_strength = go.FigureWidget(
            data=[go.Heatmap(z=strength_matrix, colorscale='Gray', zmin=0, zmax=255)]
        )
        
        delay, width = self._gate_info[0], self._gate_info[1] - self._gate_info[0]
        fig_strength.update_layout(
            title=f"强度图 - 延迟: {delay}, 门宽: {width}",
            xaxis=dict(scaleanchor='y', scaleratio=1),
            yaxis=dict(autorange='reversed'),
            width=512, height=470,
            autosize=False,
            margin=dict(l=50, r=30, t=50, b=50)
        )

        # 根据模式设置标题
        if mode == 'original':
            title = "光子直方图"
        elif mode == 'coates':
            title = "Coates估计光子直方图"
        else:  # compare
            title = "直方图对比 (原始 vs Coates估计)"
        
        # 创建直方图图表
        if mode == 'compare':
            fig_hist = go.FigureWidget(
                data=[
                    go.Bar(x=[], y=[], name='原始直方图', marker_color='#ff7f0e'),
                    go.Bar(x=[], y=[], name='Coates估计', marker_color='#2ca02c')
                ]
            )
        else:
            fig_hist = go.FigureWidget(
                data=[go.Bar(x=[], y=[], hoverinfo='x+y')]
            )
        
        fig_hist.update_layout(
            title=title,
            xaxis=dict(title="Value"),
            yaxis=dict(title="Frequency"),
            width=1024, height=470,
            margin=dict(l=60, r=30, t=50, b=50),
            barmode='group' if mode == 'compare' else 'relative'
        )

        @fig_strength.data[0].on_click
        def update_hist(trace, points, state):
            if points.xs and points.ys:
                x_idx, y_idx = int(points.xs[0]), int(points.ys[0])
                if mode == 'original':
                    self.update_histogram(fig_hist, x_idx, y_idx)
                elif mode == 'coates':
                    self.update_histogram_coates(fig_hist, x_idx, y_idx)
                else:  # compare
                    self.update_histogram_compare(fig_hist, x_idx, y_idx)
        
        return HBox([fig_strength, fig_hist])
    
    
    def update_histogram(self, fig_hist, x_idx, y_idx):
        """更新直方图显示"""
    # 调用父类方法获取数据
        _, hist_counts, flux_value = self.get_channel_data(x_idx=x_idx, y_idx=y_idx)        
        # 构建bins数组用于绘图
        low, high = self._gate_info
        bins = np.arange(low, high + 1)
        
        with fig_hist.batch_update():
            fig_hist.data[0].x = bins[:-1]
            fig_hist.data[0].y = hist_counts
            fig_hist.update_layout(
                title=f"像素({x_idx}, {y_idx})的直方图<br>光通量: {flux_value:.2f}%",
                xaxis=dict(range=[low, high])
            )
            
    def update_histogram_coates(self, fig_hist, x_idx, y_idx):
        """更新直方图显示，使用Coates估计器处理"""
        index = y_idx * 64 + x_idx
        low, high = self._gate_info
        data = self._matrix_data[:self._exposure, index]
        mask = (data >= low) & (data < high)
        filtered = data[mask]
        
        bins = np.arange(low, high + 1)
        hist_counts, _ = np.histogram(filtered, bins=bins)
        
        # 创建带溢出仓的直方图数组
        hist_with_overflow = np.zeros(len(hist_counts) + 1)
        hist_with_overflow[:-1] = hist_counts
        # 溢出仓计数 = 总曝光次数 - 所有直方图计数总和
        hist_with_overflow[-1] = self._exposure - np.sum(hist_counts)
        
        # 使用Coates估计器处理直方图
        
        coates_hist = SPADSimulateEngine.coates_estimator(hist_with_overflow)*self._exposure
        print(sum(coates_hist))
        
        # 计算光通量
        flux_value = len(filtered) / self._exposure * 100
        
        with fig_hist.batch_update():
            fig_hist.data[0].x = bins[:-1][:len(coates_hist)]  # 确保长度一致
            fig_hist.data[0].y = coates_hist
            fig_hist.update_layout(
                title=f"像素({x_idx}, {y_idx})的Coates估计直方图<br>光通量: {flux_value:.2f}%",
                xaxis=dict(range=[low, high])
            )
    
    def update_histogram_compare(self, fig_hist, x_idx, y_idx):
        """对比显示原始直方图和Coates估计直方图"""
        index = y_idx * 64 + x_idx
        low, high = self._gate_info
        data = self._matrix_data[:self._exposure, index]
        mask = (data >= low) & (data < high)
        filtered = data[mask]
        
        # 计算原始直方图
        bins = np.arange(low, high + 1)
        hist_counts, _ = np.histogram(filtered, bins=bins)
        
        # 创建带溢出仓的直方图数组
        hist_with_overflow = np.zeros(len(hist_counts) + 1)
        hist_with_overflow[:-1] = hist_counts
        # 溢出仓计数 = 总曝光次数 - 所有直方图计数总和
        hist_with_overflow[-1] = self._exposure - np.sum(hist_counts)
        
        # 使用Coates估计器处理直方图
        coates_hist = SPADSimulateEngine.coates_estimator(hist_with_overflow)*self._exposure
        
        # 计算光通量
        flux_value = len(filtered) / self._exposure * 100
        
        with fig_hist.batch_update():
            # 更新原始直方图
            fig_hist.data[0].x = bins[:-1]
            fig_hist.data[0].y = hist_counts
            fig_hist.data[0].marker.color = '#1f77b4'  # 蓝色
            
            # 更新Coates估计直方图
            x_vals = bins[:-1][:len(coates_hist)]
            fig_hist.data[1].x = x_vals
            fig_hist.data[1].y = coates_hist
            fig_hist.data[1].marker.color = '#ff7f0e'  # 橙色
            
            # 更新布局和标题
            fig_hist.update_layout(
                title=f"像素({x_idx}, {y_idx})的直方图对比<br>光通量: {flux_value:.2f}%",
                xaxis=dict(
                    title="时间箱",
                    range=[low, high]
                ),
                yaxis=dict(
                    title="光子计数"
                ),
                barmode='group',  # 确保两种直方图并排显示
                legend=dict(
                    x=0.99,
                    y=0.99,
                    xanchor='right',
                    yanchor='top',
                    bgcolor='rgba(255,255,255,0.5)',
                    bordercolor='rgba(0,0,0,0.5)',
                    borderwidth=1,
                    font=dict(
                        size=12  # 调整图例的字号
                    )
                )
            )

    def draw_point_cloud(self):
        """
        绘制所有像素的三维点云，忽略强度为0的点
        - 透明度与强度值非线性关联，低强度点几乎不可见，高强度点更明显
        - 所有点使用蓝色
        - Y轴整体反转，使Y轴的远点和X轴的远点在同一方向
        - Z轴整体反转，使Z轴的原点和X、Y轴的原点在同一侧
        - X和Y轴范围扩展至128
        
        返回:
            go.FigureWidget: 三维点云图
        """
        # 创建图表
        fig = go.FigureWidget()
        
        # 获取栅格参数
        low, high = self._gate_info
        gate_length = high - low
        
        # 蓝色 RGB 值
        blue_r, blue_g, blue_b = 200, 0, 0
        
        # 非线性映射指数
        nonlinear_power = 1.5
        
        # 计算所有强度值的最大值，用于归一化透明度
        max_intensity = np.max(self._tensor_data)
        if max_intensity <= 0:
            max_intensity = 1
        
        # 使用numpy索引操作一次性获取所有非零点的坐标和强度
        x, y, z, intensities = [], [], [], []
        
        for py in range(64):
            for px in range(64):
                hist_data = self._tensor_data[py, px, :]
                
                for t_idx in range(gate_length):
                    if hist_data[t_idx] > 0:
                        x.append(px)
                        # 使用原始y坐标，不进行反转
                        y.append(py)
                        # 直接使用t_idx作为z坐标，不进行反转
                        z.append(t_idx)
                        intensities.append(hist_data[t_idx])
        
        # 转换为NumPy数组以便进行向量化操作
        intensities = np.array(intensities)
        
        # 计算透明度 - 使用非线性映射
        norm_intensities = intensities / max_intensity
        alphas = np.minimum(0.001 + (norm_intensities**nonlinear_power) * 0.999, 1.0)*0.6
        
        # 创建RGBA颜色数组
        rgba_colors = [f'rgba({blue_r},{blue_g},{blue_b},{alpha})' for alpha in alphas]
        
        # 添加散点图
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z,
            mode='markers',
            marker=dict(
                size=2,
                color=rgba_colors,
                colorscale=None,
            ),
            name='点云数据'
        ))
        
        fig.update_layout(
            title="光子点云（强度>0的点）- 蓝色点，透明度表示强度",
            scene=dict(
                xaxis=dict(
                    title='像素 X',
                    autorange='reversed'
                ),
                yaxis=dict(
                    title='像素 Y',
                    autorange='reversed'  
                ),
                zaxis=dict(
                    title='时间 (ns)',
                    autorange='reversed'
                ),
                aspectmode='data'  
            ),
            width=900,  # 减小宽度以适应notebook
            height=600, # 适当减小高度
            scene_camera=dict(
                eye=dict(x=1.5, y=0.8, z=1.8),
                # center=dict(x=0, y=0.2, z=4),
                up=dict(x=0, y=1, z=0)
            ),
            # margin=dict(l=40, r=20, b=60, t=40),  # 增加边距以提高可读性
            template="plotly_white",  # 使用白色背景模板更适合notebook
            autosize=True,  
        )
        
        return fig