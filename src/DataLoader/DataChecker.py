from .DataLoader import DataLoader
import numpy as np
from ..GatingSimulator.SPADSimulateEngine import SPADSimulateEngine
import dash
from dash import dcc, html, Input, Output
import plotly.graph_objs as go


class DataChecker(DataLoader):
    def draw_strength(self, mode='original'):
        strength_matrix = self._strength_matrix
        exposure = self._exposure
        gate_info = self._gate_info
        matrix_data = self._matrix_data
        # 创建Dash应用
        app = dash.Dash(__name__)
        # 构建热图
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
        # 应用布局
        app.layout = html.Div([
            html.Div([
                dcc.Graph(id='heatmap', figure=heatmap_fig, style={'flex': '1', 'margin': '0'}),
                html.Div([
                    dcc.Graph(id='histogram', figure=go.Figure(), style={'flex': '1', 'margin': '0'}),
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
                            style={'width': '150px'}
                        )
                    ], style={
                        'position': 'absolute',
                        'top': '10px',
                        'right': '10px',
                        'zIndex': '1000'
                    })
                ], style={'position': 'relative', 'flex': '1', 'margin': '0'})
            ], style={'display': 'flex', 'width': '100%', 'margin': '0', 'gap': '0'})
        ], style={'display': 'flex', 'flexDirection': 'column', 'alignItems': 'center', 'margin': '0', 'padding': '0'})

        # 回调函数
        @app.callback(
            Output('histogram', 'figure'),
            [Input('heatmap', 'clickData'),
             Input('mode-selector', 'value')],  # 添加模式选择器的输入
            prevent_initial_call=True
        )
        def update_histogram(click_data, mode):  # 添加 mode 参数
            if not click_data:
                return dash.no_update
            # 解析点击坐标
            x_idx = int(click_data['points'][0]['x'])
            y_idx = int(click_data['points'][0]['y'])
            index = y_idx * 64 + x_idx
            # 获取基本数据
            low, high = gate_info
            data = matrix_data[:exposure, index]
            mask = (data >= low) & (data < high)
            filtered = data[mask]
            flux_value = len(filtered) / exposure * 100
            # 生成基础直方图
            bins = np.arange(low, high)
            hist_counts, _ = np.histogram(filtered, bins=bins)
            # 根据模式生成不同图形
            if mode == 'original':
                fig = go.Figure([go.Bar(
                    x=bins,
                    y=hist_counts,
                    marker_color='#ff7f0e'
                )])
                title = f"像素({x_idx}, {y_idx})的直方图<br>光通量: {flux_value:.2f}%"
                
            elif mode == 'coates':
                # Coates估计处理
                hist_with_overflow = np.concatenate([hist_counts, [exposure - hist_counts.sum()]])
                coates_hist = SPADSimulateEngine.coates_estimator(hist_with_overflow) * exposure
                
                fig = go.Figure([go.Bar(
                    x=bins[:len(coates_hist)],
                    y=coates_hist,
                    marker_color='#2ca02c'
                )])
                title = f"Coates估计直方图<br>光通量: {flux_value:.2f}%"
                
            elif mode == 'compare':
                # 对比模式
                hist_with_overflow = np.concatenate([hist_counts, [exposure - hist_counts.sum()]])
                coates_hist = SPADSimulateEngine.coates_estimator(hist_with_overflow) * exposure
                
                fig = go.Figure([
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
                ])
                title = f"直方图对比<br>光通量: {flux_value:.2f}%"
                fig.update_layout(barmode='group')
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
            return fig
        # 在Jupyter中运行
        app.run(jupyter_mode='inline', port=8050)