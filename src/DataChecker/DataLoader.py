import numpy as np
import plotly.graph_objs as go
from ipywidgets import HBox, VBox


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

        # 使用切片视图而非复制数据
        selected_data = self._matrix_data[:self._exposure]
        
        # 使用布尔索引一次性计算
        low, high = self._gate_info
        count_under = np.sum((selected_data >= low) & (selected_data < high), axis=0) / self._exposure
        
        # 一次性操作
        self._strength_matrix = np.round(count_under * 255).astype(int).reshape(64, 64)
        return self._strength_matrix
    
    def draw_strength(self):
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

        # 预创建直方图
        fig_hist = go.FigureWidget(
            data=[go.Bar(x=[], y=[], hoverinfo='x+y')],  # 添加hoverinfo提高交互性能
        )
        fig_hist.update_layout(
            title="光子直方图",
            xaxis=dict(title="Value"),
            yaxis=dict(title="Frequency"),
            width=1024, height=470,
            margin=dict(l=60, r=30, t=50, b=50)
        )


        # 优化点击事件处理
        @fig_strength.data[0].on_click
        def update_hist(trace, points, state):
            if points.xs and points.ys:
                self.update_histogram(fig_hist, int(points.xs[0]), int(points.ys[0]))

        
        return HBox([fig_strength, fig_hist])
    
    
    def update_histogram(self, fig_hist, x_idx, y_idx):
        """更新直方图显示"""
        # 计算一维索引
        index = y_idx * 64 + x_idx
        
        # 使用预先计算好的数据
        low, high = self._gate_info
        
        # 避免不必要的中间变量和数据复制
        data = self._matrix_data[:self._exposure, index]
        mask = (data >= low) & (data < high)
        filtered = data[mask]
        
        # 优化直方图计算，使用面向数组的操作
        bins = np.arange(low, high + 1)
        hist_counts, _ = np.histogram(filtered, bins=bins)
        
        # 更新图形，减少不必要的计算
        flux_value = len(filtered) / self._exposure * 100
        
        # 批量更新图形数据
        with fig_hist.batch_update():
            fig_hist.data[0].x = bins[:-1]
            fig_hist.data[0].y = hist_counts
            fig_hist.update_layout(
                title=f"像素({x_idx}, {y_idx})的直方图<br>光通量: {flux_value:.2f}%",
                xaxis=dict(range=[low, high])
            )
        
        
    def get_channel_data(self, x_idx, y_idx):
        """
        获取指定像素的通道数据
        
        Args:
            x_idx: X坐标
            y_idx: Y坐标
        
        Returns:
            tuple: (过滤后的数据, 直方图计数, 光通量值)
        """
        index = y_idx * 64 + x_idx
        data = self._matrix_data[:self._exposure, index]
        low, high = self._gate_info
        filtered = data[(data >= low) & (data < high)]
        hist_counts, bin_edges = np.histogram(filtered, bins=range(low, high + 1))
        flux_value = (len(filtered) / self._exposure) * 100
        
        return filtered, hist_counts, flux_value
    


def test():
    file_path = '/Users/ming/Documents/PythonCode/Coates/src/DataChecker/2025-01-17_15-09-54_Delay-0_Width-200.raw'  # 修改为你的数据文件路径
    exposure = 500
    gate_info = (0, 200)
    
    data_loader = DataLoader(file_path, exposure, gate_info)
    
    # 显示绘图结果
    fig_strength, fig_hist = data_loader.draw_strength().children
    fig_strength.show()
    fig_hist.show()
    
# 修改您的主函数部分
if __name__ == "__main__":
    test()
