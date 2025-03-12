import numpy as np
import plotly.graph_objs as go
from ipywidgets import HBox, VBox
from IPython.display import display

class DataLoader:
    def __init__(self,file_path,exposure=500,gate_info=(0,200)):
        self._file_path=file_path
        self._exposure=exposure
        self._gate_info=(gate_info[0] + 15, gate_info[1] - 10)
        

        self.load_data()
        self.get_strength_matrix()
        self.draw_strength()
        
    def load_data(self):
        try:
            with open(self._file_path, 'rb') as f:
                raw_data = np.frombuffer(f.read(), dtype=np.uint16)
            if len(raw_data) % 4096 != 0:
                raise ValueError("数据长度无法整除 4096，无法构建矩阵。")
            num_lines = len(raw_data) // 4096
            self._matrix_data = raw_data.reshape(num_lines, 4096).astype(float)
        except FileNotFoundError:
            raise FileNotFoundError(f"无法打开文件: {self._file_path}，请检查文件路径是否正确。")
        except PermissionError:
            raise PermissionError(f"没有权限读取文件: {self._file_path}，请检查文件权限。")
        except IOError as e:
            raise IOError(f"读取文件时发生错误: {self._file_path}，错误信息: {str(e)}")
        
    def get_strength_matrix(self):
        """
        计算并返回强度图矩阵
        后续需要检查这里的返回的矩阵是否需要翻转、转置等操作，不再修改绘图函数
        Returns:
            numpy.ndarray: 64x64的强度图矩阵
        """
        if self._exposure > self._matrix_data.shape[0]:
            raise ValueError("曝光时间超过文件大小")

        selected_data = self._matrix_data[:self._exposure, :]
        mask = (selected_data >= self._gate_info[0]) & (selected_data < self._gate_info[1])
        count_under = mask.sum(axis=0) / self._exposure
        gray_values = np.round(count_under * 255).astype(int)

        # 将强度值重塑为64x64矩阵
        self._strength_matrix = gray_values.reshape(64, 64)
        return self._strength_matrix
    
    def draw_strength(self):
        strength_matrix = self._strength_matrix
        fig_strength = go.FigureWidget(
            data=[go.Heatmap(z=strength_matrix, colorscale='Gray', zmin=0, zmax=255)]
        )
        fig_strength.update_layout(
            title=f"强度图 - 延迟: {self._gate_info[0]}, 门宽: {self._gate_info[1] - self._gate_info[0]}",
            xaxis=dict(scaleanchor='y', scaleratio=1),
            yaxis=dict(autorange='reversed'),
            width=512,
            height=470,
            autosize=False,
            margin=dict(l=50, r=30, t=50, b=50)
            )

        fig_hist = go.FigureWidget(data=[go.Bar(x=[], y=[])])
        fig_hist.update_layout(
            title="光子直方图",
            xaxis=dict(title="Value"),
            yaxis=dict(title="Frequency"),
            width=1024,
            height=470,
            margin=dict(l=60, r=30, t=50, b=50)
        )

        @fig_strength.data[0].on_click
        def update_hist(trace, points, state):
            if points.xs and points.ys:
                x_idx = points.xs[0]
                y_idx = points.ys[0]
                index = y_idx * 64 + x_idx
                data = self._matrix_data[:self._exposure, index]
                low, high = self._gate_info
                filtered = data[(data >= low) & (data < high)]
                hist_counts, bin_edges = np.histogram(filtered, bins=range(low, high + 1))
                fig_hist.data[0].x = bin_edges[:-1]
                fig_hist.data[0].y = hist_counts

                flux_value = (len(filtered) / self._exposure) * 100
                fig_hist.update_layout(
                    title=f"像素({x_idx}, {y_idx})的直方图\n光通量: {flux_value:.2f}%",
                    xaxis=dict(range=[low, high])
                )

        # 回调函数定义结束后（缩进减少）
        layout = HBox([fig_strength, fig_hist])
        return layout    
        
# 使用Plotly的show方法（适用于大多数环境）
import plotly.io as pio
pio.renderers.default = "browser"  # 在浏览器中打开

# 修改您的主函数部分
if __name__ == "__main__":
    file_path = '/Users/ming/Documents/PythonCode/Coates/src/DataChecker/2025-01-17_15-09-54_Delay-0_Width-200.raw'
    data_loader = DataLoader(file_path, exposure=500, gate_info=(0, 200))
    layout = data_loader.draw_strength()
    
    # 创建一个单独的图形对象并显示
    fig = go.Figure()
    fig = go.Figure(layout)
    fig.show()
