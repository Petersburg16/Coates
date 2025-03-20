from dash import Dash, html
import dash_vtk
from dash_vtk.utils import to_volume_state
import numpy as np
import vtk
from vtkmodules.util import numpy_support

class PointCloudVisualizer:
    def __init__(self, tensor_data):
        """
        初始化 PointCloudVisualizer 类

        参数:
            tensor_data: numpy.ndarray, 形状为 (64, 64, depth)，表示点云张量数据
        """
        self.tensor_data = tensor_data

    def normalize_tensor(self):
        """
        将张量数据归一化到 [0, 255] 范围，并生成透明度数组

        Returns:
            numpy.ndarray: 归一化后的张量
        """
        tensor = self.tensor_data
        tensor_min, tensor_max = tensor.min(), tensor.max()
        normalized_tensor = ((tensor - tensor_min) / (tensor_max - tensor_min) * 255).astype(np.uint8)
        return normalized_tensor

    def numpy_to_vtk_image_data(self, numpy_array):
        """
        将 NumPy 数组转换为 VTK 的 vtkImageData 对象

        参数:
            numpy_array: numpy.ndarray, 形状为 (64, 64, depth)

        Returns:
            vtk.vtkImageData: VTK 图像数据对象
        """
        depth, height, width = numpy_array.shape
        vtk_data = numpy_support.numpy_to_vtk(num_array=numpy_array.ravel(), deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)

        image_data = vtk.vtkImageData()
        image_data.SetDimensions(width, height, depth)
        image_data.GetPointData().SetScalars(vtk_data)

        return image_data

    def to_vtk_volume(self):
        """
        将张量数据转换为适合 dash_vtk 的体渲染格式

        Returns:
            dict: dash_vtk.utils.to_volume_state 返回的体渲染状态
        """
        normalized_tensor = self.normalize_tensor()
        vtk_image_data = self.numpy_to_vtk_image_data(normalized_tensor)
        volume_state = to_volume_state(vtk_image_data)
        return volume_state

    def render(self):
        """
        使用 Dash 和 dash_vtk 渲染点云强度图

        Returns:
            Dash: Dash 应用实例
        """
        volume_state = self.to_vtk_volume()

        # 创建 Dash 应用
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
        """
        在 Jupyter Notebook 中显示 Dash 应用
        """
        from jupyter_dash import JupyterDash
        app = self.render()
        app.run(mode="inline", debug=True)
