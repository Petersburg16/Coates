from .DataLoader import DataLoader
from dash import Dash, html
import dash_vtk
from dash_vtk.utils import to_volume_state

import numpy as np
import vtk
from vtkmodules.util import numpy_support



class PointCloudVisualizer(DataLoader):
    def _to_vtk_volume(self):
        """
        To ensure _tensor_data can be rendered by Dash VTK, we need to convert it to a VTK volume state.

        Process: 
        normalize the tensor data -> convert it to a VTK image data -> convert it to a volume state.
        """
        normalized_tensor = self._normalize_tensor()
        vtk_image_data = self._numpy_to_vtk_image_data(normalized_tensor)
        volume_state = to_volume_state(vtk_image_data)
        return volume_state
    def _normalize_tensor(self):
        tensor = self._tensor_data
        tensor_min, tensor_max = tensor.min(), tensor.max()
        normalized_tensor = ((tensor - tensor_min) / (tensor_max - tensor_min) * 255).astype(np.uint8)
        return normalized_tensor

    def _numpy_to_vtk_image_data(self, numpy_array):
        depth, height, width = numpy_array.shape
        vtk_data = numpy_support.numpy_to_vtk(num_array=numpy_array.ravel(), deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
        image_data = vtk.vtkImageData()
        image_data.SetDimensions(width, height, depth)
        image_data.GetPointData().SetScalars(vtk_data)
        return image_data

    def render(self):
        """
        Create a Dash application to display the 3D volume rendering of the photon cloud.
        """
        volume_state = self._to_vtk_volume()
        app = Dash(__name__)
        app.layout = html.Div(
            style={"width": "100%", "height": "600px"},
            children=[
                dash_vtk.View(
                    [
                        dash_vtk.VolumeRepresentation([
                            dash_vtk.VolumeController(),  # 控制面板
                            dash_vtk.Volume(state=volume_state),  # 体渲染
                        ]),
                    ],
                    background=[1, 1, 1]  # 设置背景为白色 (RGB: 1, 1, 1)
                )
            ]
        )
        return app

    def show(self):
        app = self.render()
        app.run(mode="inline", debug=True,port=8090)


    def draw_point_cloud_plotly(self):
            """
            绘制所有像素的三维点云，忽略强度为0的点
            - 透明度与强度值非线性关联，低强度点几乎不可见，高强度点更明显
            - 所有点使用蓝色
            - 鼠标悬停显示点的坐标和强度信息
            - 支持交互式选择Z轴（时间）的显示范围
            
            返回:
                VBox: 包含点云图和Z轴范围控制滑块的垂直布局
                
            """
            from ipywidgets import IntRangeSlider, VBox, Label, HBox
            import plotly.graph_objs as go
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

