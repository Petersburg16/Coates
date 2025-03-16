from .DataLoader import DataLoader
import numpy as np
import plotly.graph_objs as go
from ipywidgets import HBox, VBox
from ..GatingAlgo.SPADSimulateEngine import SPADSimulateEngine

# 创建Z轴范围滑块
from ipywidgets import IntRangeSlider, VBox, Label, HBox
            
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
            - 鼠标悬停显示点的坐标和强度信息
            - 支持交互式选择Z轴（时间）的显示范围
            
            返回:
                VBox: 包含点云图和Z轴范围控制滑块的垂直布局
            """
            # 创建图表
            fig = go.FigureWidget()
            
            # 获取栅格参数
            low, high = self._gate_info
            gate_length = high - low
            
            # 使用numpy的where找出所有非零强度的点的索引
            # tensor_data形状为 [y, x, time]
            non_zero_indices = np.where(self._tensor_data > 0)
            
            # 提取坐标和强度值
            y_coords = non_zero_indices[0]
            x_coords = non_zero_indices[1]
            time_idx = non_zero_indices[2]
            intensities = self._tensor_data[non_zero_indices]
            
            # 获取时间轴的最小值和最大值
            z_min = int(np.min(time_idx)) if len(time_idx) > 0 else low
            z_max = int(np.max(time_idx)) if len(time_idx) > 0 else high
            
            # 设置滑块的默认范围为整个时间范围的中间50%
            default_z_min = z_min 
            default_z_max = z_max
            

            z_range_slider = IntRangeSlider(
                value=[default_z_min, default_z_max],
                min=low,
                max=high,
                step=1,
                description='Z轴范围:',
                disabled=False,
                continuous_update=False,
                orientation='horizontal',
                readout=True,
                readout_format='d',
                layout={'width': '80%'}
            )
            
            z_range_label = Label(value=f'当前显示范围: [{default_z_min}, {default_z_max}]')
            
            # 创建更新点云的函数
            def update_point_cloud(*args):
                # 获取选定的Z范围
                z_min_val, z_max_val = z_range_slider.value
                
                # 更新标签
                z_range_label.value = f'当前显示范围: [{z_min_val}, {z_max_val}]'
                
                # 筛选Z范围内的点
                mask = (time_idx >= z_min_val) & (time_idx <= z_max_val)
                filtered_x = x_coords[mask]
                filtered_y = y_coords[mask]
                filtered_z = time_idx[mask]
                filtered_intensities = intensities[mask]
                
                # 计算透明度 - 使用非线性映射
                max_intensity = np.max(filtered_intensities) if filtered_intensities.size > 0 else 1
                norm_intensities = filtered_intensities / max_intensity
                alphas = np.minimum(0.001 + (norm_intensities**1.5) * 0.999, 1.0) * 0.6
                
                # 创建RGBA颜色数组 - 蓝色 RGB: 0, 0, 200
                rgba_colors = [f'rgba(0,0,200,{alpha})' for alpha in alphas]
                
                # 更新图表数据
                with fig.batch_update():
                    if not fig.data:
                        # 如果图表没有数据，添加新的散点图
                        fig.add_trace(go.Scatter3d(
                            x=filtered_x, 
                            y=filtered_y, 
                            z=filtered_z,
                            mode='markers',
                            marker=dict(
                                size=2,
                                color=rgba_colors,
                                colorscale=None,
                            ),
                            hoverinfo='text',
                            hovertemplate='像素: (%{x}, %{y})<br>时间: %{z}<br>强度: %{customdata}<extra></extra>',
                            customdata=filtered_intensities,
                            name='点云数据'
                        ))
                    else:
                        # 更新现有散点图
                        fig.data[0].x = filtered_x
                        fig.data[0].y = filtered_y
                        fig.data[0].z = filtered_z
                        fig.data[0].marker.color = rgba_colors
                        fig.data[0].customdata = filtered_intensities
                    
                    # 更新标题和Z轴范围
                    fig.update_layout(
                        title=f"光子点云 (Z轴范围: {z_min_val}-{z_max_val})",
                        scene=dict(
                            zaxis=dict(
                                range=[z_min_val, z_max_val]
                            )
                        )
                    )
            
            # 将滑块与更新函数关联
            z_range_slider.observe(update_point_cloud, names='value')
            
            # 初始化点云图
            update_point_cloud()
            
            # 设置图表布局
            fig.update_layout(
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
                width=900,
                height=500,
                scene_camera=dict(
                    eye=dict(x=1.5, y=0.8, z=1.8),
                    up=dict(x=0, y=1, z=0)
                ),
                template="plotly_white",
                autosize=True,
                hoverlabel=dict(
                    bgcolor="white",
                    font_size=12,
                    font_family="Arial"
                ),
                margin=dict(l=10, r=10, t=40, b=20)  # 添加margin设置
            )
            
            # 返回包含图表和控件的布局
            controls = VBox([HBox([z_range_slider, z_range_label])])
            return VBox([fig, controls])