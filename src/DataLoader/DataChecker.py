from .DataLoader import DataLoader
import numpy as np
from ..GatingSimulator.SPADSimulateEngine import SPADSimulateEngine
import dash
from dash import dcc, html, Input, Output
import plotly.graph_objs as go


from typing import Optional
class DataChecker(DataLoader):
    def draw_strength(self, mode='original'):
        """
        绘制强度图和直方图，并启动交互式Dash应用
        """
        # 初始化数据
        strength_matrix = self._strength_matrix
        exposure = self._exposure
        gate_info = self._gate_info
        matrix_data = self._matrix_data
        
        app = self._create_dash_app(strength_matrix, gate_info)
        self._setup_callbacks(app, gate_info, matrix_data, exposure)
        app.run(jupyter_mode='inline', port=8050)

    def _create_dash_app(self, strength_matrix, gate_info):
        """
        创建并配置Dash应用
        """
        app = dash.Dash(__name__)
        
        # 构建热图
        heatmap_fig = self._create_heatmap(strength_matrix, gate_info)
        
        # 应用布局
        app.layout = self.create_app_layout(heatmap_fig)
        
        return app

    def _create_heatmap(self, strength_matrix, gate_info):
        """
        创建热图
        """
        heatmap_fig = go.Figure(
            data=[go.Heatmap(
                z=strength_matrix,
                colorscale='Gray',
                zmin=0,
                zmax=255,
                hoverinfo='none'
            )]
        )
        
        heatmap_fig.update_layout(
            title=f"强度图 - 延迟: {gate_info[0]}，门宽: {gate_info[1]-gate_info[0]}",
            xaxis=dict(scaleanchor='y', scaleratio=1),
            yaxis=dict(autorange='reversed'),
            width=512,
            height=470,
            margin=dict(l=50, r=30, t=50, b=50)
        )
        
        return heatmap_fig

    def _setup_callbacks(self, app, gate_info, matrix_data, exposure):
        """
        设置应用回调函数
        """
        @app.callback(
            Output('histogram', 'figure'),
            [Input('heatmap', 'clickData'),
            Input('mode-selector', 'value')],
            prevent_initial_call=True
        )
        def update_histogram(click_data, mode):
            if not click_data:
                return dash.no_update
            
            data_info = self.extract_pixel_data(click_data, gate_info, matrix_data, exposure)
            fig = self.create_histogram(data_info, mode)
            self.apply_layout(fig, data_info, mode, gate_info)
            
            return fig

    def extract_pixel_data(self,click_data, gate_info, matrix_data, exposure):
        """
        解包回调函数中的点击数据，提取像素数据
        """
        x_idx = int(click_data['points'][0]['x'])
        y_idx = int(click_data['points'][0]['y'])
        index = y_idx * 64 + x_idx
        low, high = gate_info
        data = matrix_data[:exposure, index]
        mask = (data >= low) & (data < high)
        filtered = data[mask]
        flux_value = len(filtered) / exposure * 100
        
        bins = np.arange(low, high)
        hist_counts, _ = np.histogram(filtered, bins=bins)
        hist_with_overflow = np.concatenate([hist_counts, [exposure - hist_counts.sum()]])
        coates_hist = SPADSimulateEngine.coates_estimator(hist_with_overflow) * exposure
        
        return {
            'x_idx': x_idx,
            'y_idx': y_idx,
            'flux_value': flux_value,
            'bins': bins,
            'hist_counts': hist_counts,
            'coates_hist': coates_hist
        }

    def create_histogram(self,data_info, mode) -> Optional[go.Figure]:
        """
        根据选择的模式交互创建直方图
        """
        bins = data_info['bins']
        hist_counts = data_info['hist_counts']
        coates_hist = data_info['coates_hist']

        # 定义模式对应的处理逻辑
        mode_handlers = {
            'original': lambda: go.Figure([go.Bar(
                x=bins,
                y=hist_counts,
                marker_color='#ff7f0e'
            )]),
            'coates': lambda: go.Figure([go.Bar(
                x=bins[:len(coates_hist)],
                y=coates_hist,
                marker_color='#2ca02c'
            )]),
            'compare': lambda: go.Figure([
                go.Bar(
                    name='原始',
                    x=bins,
                    y=hist_counts,
                    marker_color='#1f77b4'
                ),
                go.Bar(
                    name='Coates',
                    x=bins[:len(coates_hist)],
                    y=coates_hist,
                    marker_color='#ff7f0e'
                )
            ]).update_layout(barmode='group')
        }

        # 根据模式调用对应的处理逻辑
        return mode_handlers.get(mode, lambda: go.Figure())()

    def apply_layout(self,fig, data_info, mode, gate_info):
        """
        更新布局
        """
        x_idx = data_info['x_idx']
        y_idx = data_info['y_idx']
        flux_value = data_info['flux_value']
        low, high = gate_info
        
        # 根据模式设置标题
        if mode == 'original':
            title = f"像素({x_idx}, {y_idx})的直方图<br>光通量: {flux_value:.2f}%"
        elif mode == 'coates':
            title = f"Coates估计直方图<br>光通量: {flux_value:.2f}%"
        elif mode == 'compare':
            title = f"直方图对比<br>光通量: {flux_value:.2f}%"
        
        # 更新布局
        fig.update_layout(
            title=title,
            xaxis=dict(range=[low, high]),
            yaxis=dict(title='光子计数'),
            width=1024,
            height=470,
            margin=dict(l=60, r=30, t=80, b=50),
            showlegend=mode == 'compare',
            legend=dict(
                x=0.02,
                y=0.98,
                bgcolor='rgba(255, 255, 255, 0.5)',
                bordercolor='black',
                borderwidth=1
            )
        )

           
    def create_app_layout(self,heatmap_fig):
        """创建应用布局"""
        graph_style = {'flex': '1', 'margin': '0'}
        dropdown_style = {
            'width': '150px'
        }
        dropdown_container_style = {
            'position': 'absolute',
            'top': '10px',
            'right': '10px',
            'zIndex': '1000'
        }
        layout_style = {
            'display': 'flex',
            'flexDirection': 'column',
            'alignItems': 'center',
            'margin': '0',
            'padding': '0'
        }
        row_style = {
            'display': 'flex',
            'width': '100%',
            'margin': '0',
            'gap': '0'
        }
        return html.Div([
            html.Div([
                dcc.Graph(id='heatmap', figure=heatmap_fig, style=graph_style),
                html.Div([
                    dcc.Graph(id='histogram', figure=go.Figure(), style=graph_style),
                    html.Div([
                        dcc.Dropdown(
                            id='mode-selector',
                            options=[
                                {'label': '原始模式', 'value': 'original'},
                                {'label': 'Coates模式', 'value': 'coates'},
                                {'label': '对比模式', 'value': 'compare'}
                            ],
                            value='original',  # 默认值
                            clearable=False,
                            style=dropdown_style
                        )
                    ], style=dropdown_container_style)
                ], style={'position': 'relative', 'flex': '1', 'margin': '0'})
            ], style=row_style)
        ], style=layout_style)
