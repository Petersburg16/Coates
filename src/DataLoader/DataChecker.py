import DataLoader
import numpy as np
import plotly.graph_objs as go
from ipywidgets import HBox, VBox
class DataChecker(DataLoader):
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

        @fig_strength.data[0].on_click
        def update_hist(trace, points, state):
            if points.xs and points.ys:
                self.update_histogram(fig_hist, int(points.xs[0]), int(points.ys[0]))

        
        return HBox([fig_strength, fig_hist])
    
    
    def update_histogram(self, fig_hist, x_idx, y_idx):
        """更新直方图显示"""
        index = y_idx * 64 + x_idx
        low, high = self._gate_info
        data = self._matrix_data[:self._exposure, index]
        mask = (data >= low) & (data < high)
        filtered = data[mask]
        

        bins = np.arange(low, high + 1)
        hist_counts, _ = np.histogram(filtered, bins=bins)

        flux_value = len(filtered) / self._exposure * 100
        with fig_hist.batch_update():
            fig_hist.data[0].x = bins[:-1]
            fig_hist.data[0].y = hist_counts
            fig_hist.update_layout(
                title=f"像素({x_idx}, {y_idx})的直方图<br>光通量: {flux_value:.2f}%",
                xaxis=dict(range=[low, high])
            )
            

def test():
    file_path = '/Users/ming/Documents/PythonCode/Coates/src/DataChecker/2025-01-17_15-09-54_Delay-0_Width-200.raw'  # 修改为你的数据文件路径
    exposure = 500
    gate_info = (0, 200)
    
    data_loader = DataChecker(file_path, exposure, gate_info)
    
    # 显示绘图结果
    fig_strength, fig_hist = data_loader.draw_strength().children
    fig_strength.show()
    fig_hist.show()
    
# 修改您的主函数部分
if __name__ == "__main__":
    test()
