import dash
from dash import html, dcc
import plotly.graph_objects as go
import numpy as np
from .SPADSimulateEngine import SPADDataContainer

class DualGaussianViewer:
    """
    处理SPAD模拟器的Dash可视化界面
    """
    
    @staticmethod
    def create_dash_app(data_container, simulator_instance, title="SPAD模拟器"):
        """
        创建一个完整的Dash应用
        
        Args:
            data_container: 包含SPAD数据的容器
            simulator_instance: 模拟器实例，用于回调更新
            title: 应用标题
            
        Returns:
            app: 配置好的Dash应用
        """
        app = dash.Dash(__name__)
        fig = DualGaussianViewer.create_fig(data_container)
        app = DualGaussianViewer.create_app_layout(app, fig, title)
        
        # 添加交互回调
        DualGaussianViewer.add_callbacks(app, simulator_instance)
        
        return app
    
    @staticmethod
    def create_fig(data: SPADDataContainer):
        """创建绘图对象"""
        fig = go.Figure()
        

        fig.add_trace(go.Scatter(
            x=np.linspace(0, data.gate_length-1, data.gate_length*data.resolution_factor),
            y=data.gated_smooth_flux*data.exposure, 
            mode='lines', 
            name='光通量', 

        ))
    
        
        # 添加三种直方图
        fig.add_trace(go.Bar(x=np.arange(data.gate_length), y=data.ideal_histogram, name='Ideal Histogram'))
        fig.add_trace(go.Bar(x=np.arange(data.gate_length), y=data.spad_histogram_without_overflow, name='Simulated Histogram'))
        fig.add_trace(go.Bar(x=np.arange(data.gate_length), y=data.coates_histogram, name='Coates Histogram'))
        
        # 设置图表布局
        fig.update_layout(
            title='SPAD模拟及Coates估计结果',
            xaxis=dict(title='时间 (ns)'),
            yaxis=dict(title='光子计数'),

            bargap=0.1,
            legend=dict(x=0.01, y=0.99, bgcolor='rgba(255, 255, 255, 0.5)'),
            template='plotly_white'
        )
        
        return fig

    @staticmethod
    def create_app_layout(app, fig, title):
        """创建应用界面布局"""
        app.layout = html.Div([
            html.H1(title),
            
            # 控制面板和图表的外层容器
            html.Div([
                # 控制面板容器
                html.Div([
                    # 使用一个水平布局的容器来放置三列滑块
                    html.Div([
                        # 第一列：第一峰值参数
                        html.Div([
                            html.H4("第一峰值参数", style={'text-align': 'center', 'margin-bottom': '15px'}),
                            
                            html.Label('位置'),
                            dcc.Slider(
                                id='pulse-pos1-slider',
                                min=10, max=80, value=40, step=1,
                                marks={i: str(i) for i in range(10, 81, 10)},
                                tooltip={"placement": "bottom", "always_visible": True},
                                updatemode='drag'  # 添加这一行，实现拖动更新
                            ),
                            
                            html.Label('宽度'),
                            dcc.Slider(
                                id='pulse-width1-slider',
                                min=1, max=20, value=5, step=0.5,
                                marks={i: str(i) for i in range(1, 21, 2)},
                                tooltip={"placement": "bottom", "always_visible": True},
                                updatemode='drag'  # 添加这一行
                            ),
                            
                            html.Label('强度'),
                            dcc.Slider(
                                id='strength1-slider',
                                min=0, max=1, value=0.5, step=0.05,
                                marks={i/10: str(i/10) for i in range(0, 11, 2)},
                                tooltip={"placement": "bottom", "always_visible": True},
                                updatemode='drag'  # 添加这一行
                            ),
                        ], style={'width': '32%', 'display': 'inline-block', 'vertical-align': 'top',
                                'padding': '10px', 'border': '1px solid #ddd', 'border-radius': '5px',
                                'box-shadow': '2px 2px 2px lightgrey', 'margin-right': '1%'}),
                        
                        # 第二列：第二峰值参数
                        html.Div([
                            html.H4("第二峰值参数", style={'text-align': 'center', 'margin-bottom': '15px'}),
                            
                            html.Label('位置'),
                            dcc.Slider(
                                id='pulse-pos2-slider',
                                min=20, max=180, value=70, step=1,
                                marks={i: str(i) for i in range(20, 181, 20)},
                                tooltip={"placement": "bottom", "always_visible": True},
                                updatemode='drag'  # 添加这一行
                            ),
                            
                            html.Label('宽度'),
                            dcc.Slider(
                                id='pulse-width2-slider',
                                min=1, max=20, value=5, step=0.5,
                                marks={i: str(i) for i in range(1, 21, 2)},
                                tooltip={"placement": "bottom", "always_visible": True},
                                updatemode='drag'  # 添加这一行
                            ),
                            
                            html.Label('强度'),
                            dcc.Slider(
                                id='strength2-slider',
                                min=0, max=1, value=0.5, step=0.05,
                                marks={i/10: str(i/10) for i in range(0, 11, 2)},
                                tooltip={"placement": "bottom", "always_visible": True},
                                updatemode='drag'  # 添加这一行
                            ),
                        ], style={'width': '32%', 'display': 'inline-block', 'vertical-align': 'top', 
                                'padding': '10px', 'border': '1px solid #ddd', 'border-radius': '5px',
                                'box-shadow': '2px 2px 2px lightgrey', 'margin-right': '1%'}),
                        
                        # 第三列：背景和总强度
                        html.Div([
                            html.H4("全局参数", style={'text-align': 'center', 'margin-bottom': '15px'}),
                            
                            html.Label('背景强度'),
                            dcc.Slider(
                                id='bg-strength-slider',
                                min=0, max=0.2, value=0.01, step=0.01,
                                marks={i/100: str(i/100) for i in range(0, 21, 5)},
                                tooltip={"placement": "bottom", "always_visible": True},
                                updatemode='drag'  # 添加这一行
                            ),
                            
                            html.Label('总强度'),
                            dcc.Slider(
                                id='total-strength-slider',
                                min=0.1, max=10, value=1, step=0.1,
                                marks={i: str(i) for i in range(0, 11, 1)},
                                tooltip={"placement": "bottom", "always_visible": True},
                                updatemode='drag'  # 添加这一行
                            ),
                            
                            # 显示光通量信息
                            html.Div(id='photon-flux-info', 
                                    style={'margin-top': '40px', 'font-weight': 'bold', 'color': '#2c3e50',
                                        'text-align': 'center', 'font-size': '18px'})
                        ], style={'width': '32%', 'display': 'inline-block', 'vertical-align': 'top',
                                'padding': '10px', 'border': '1px solid #ddd', 'border-radius': '5px',
                                'box-shadow': '2px 2px 2px lightgrey'}),
                    ], style={'margin-bottom': '20px', 'display': 'flex', 'justify-content': 'space-between'}),
                    
                    # 图表
                    html.Div([
                        dcc.Graph(id='simulation-graph', figure=fig)
                    ], style={'padding': '10px', 'border': '1px solid #ddd', 'border-radius': '5px',
                            'box-shadow': '2px 2px 2px lightgrey'})
                ], style={'width': '95%', 'margin': '0 auto', 'padding': '20px'})
            ])
        ], style={'font-family': 'Arial, sans-serif', 'background-color': '#f9f9f9', 'min-height': '100vh'})
        
        return app

    @staticmethod
    def add_callbacks(app, simulator_instance):
        """添加交互回调"""
        @app.callback(
            dash.Output('simulation-graph', 'figure'),
            dash.Output('photon-flux-info', 'children'),
            [dash.Input('pulse-pos1-slider', 'value'),
             dash.Input('pulse-pos2-slider', 'value'),
             dash.Input('pulse-width1-slider', 'value'),
             dash.Input('pulse-width2-slider', 'value'),
             dash.Input('strength1-slider', 'value'),
             dash.Input('strength2-slider', 'value'),
             dash.Input('bg-strength-slider', 'value'),
             dash.Input('total-strength-slider', 'value')],
            prevent_initial_call=False,  # 允许初始加载时调用
            throttle=True,               # 启用节流
            throttle_time=5             # 每50毫秒最多更新一次
        )
        def update_simulation(pos1, pos2, width1, width2, str1, str2, bg, total):
            # 创建一个参数字典
            new_params = {
                "pulse_pos1": pos1,
                "pulse_pos2": pos2,
                "pulse_width1": width1,
                "pulse_width2": width2,
                "signal_strength1": str1,
                "signal_strength2": str2,
                "bg_strength": bg,
                "total_strength": total
            }
            
            # 调用模拟器实例的更新方法
            simulator_instance.update_simulation_params(new_params)
            
            # 创建新图形
            new_fig = DualGaussianViewer.create_fig(simulator_instance.data)
            
            # 返回更新后的图形和信息
            flux_info = f"光子通量水平: {simulator_instance.data.flux_level:.2f}%"
            return new_fig, flux_info