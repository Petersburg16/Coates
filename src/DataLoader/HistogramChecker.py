from .DataLoader import DataLoader
import numpy as np
from ..GatingSimulator.SPADSimulateEngine import SPADSimulateEngine
import dash
from dash import dcc, html, Input, Output
import plotly.graph_objs as go


from typing import Optional
class HistogramChecker(DataLoader):
    def draw_strength(self, mode='original'):
        """
        Create a Dash application to display the intensity image and histogram of the selected pixel.
        The histogram can be displayed in three modes: original, Coates, and comparison.
        """
        strength_matrix = self._strength_matrix
        exposure = self._exposure
        gate_info = self._gate_info
        matrix_data = self._matrix_data
        
        app = self._create_dash_app(strength_matrix, gate_info)
        self._setup_callbacks(app, gate_info, matrix_data, exposure)
        app.run(mode="inline",port=8060)

    def _create_dash_app(self, strength_matrix, gate_info):
        app = dash.Dash(__name__)
        heatmap_fig = self._create_heatmap(strength_matrix, gate_info)
        app.layout = self.create_app_layout(heatmap_fig)
        
        return app

    def _create_heatmap(self, strength_matrix, gate_info):
        """
        Create a heatmap figure(gray) to display the intensity image.
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
            title=f"Intensity Image Gate Delay{gate_info[0]}，Gate Width: {gate_info[1]-gate_info[0]}",
            xaxis=dict(scaleanchor='y', scaleratio=1),
            yaxis=dict(autorange='reversed'),
            width=512,
            height=470,
            margin=dict(l=50, r=30, t=50, b=50)
        )
        return heatmap_fig

    def _setup_callbacks(self, app, gate_info, matrix_data, exposure):
        """
        Set up callbacks for the Dash application.
        When the heatmap is clicked, the histogram will be updated.
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
            
            data_info = self._extract_pixel_data(click_data, gate_info, matrix_data, exposure)
            fig = self._create_histogram(data_info, mode)
            self._apply_layout(fig, data_info, mode, gate_info)
            
            return fig

    def _extract_pixel_data(self,click_data, gate_info, matrix_data, exposure):
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

    def _create_histogram(self,data_info, mode) -> Optional[go.Figure]:
        """
        Create a histogram figure based on the selected mode.
        """
        bins = data_info['bins']
        hist_counts = data_info['hist_counts']
        coates_hist = data_info['coates_hist']

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

        return mode_handlers.get(mode, lambda: go.Figure())()

    def _apply_layout(self,fig, data_info, mode, gate_info):
        x_idx = data_info['x_idx']
        y_idx = data_info['y_idx']
        flux_value = data_info['flux_value']
        low, high = gate_info
        
        if mode == 'original':
            title = f"像素({x_idx}, {y_idx})的直方图<br>光通量: {flux_value:.2f}%"
        elif mode == 'coates':
            title = f"Coates估计直方图<br>光通量: {flux_value:.2f}%"
        elif mode == 'compare':
            title = f"直方图对比<br>光通量: {flux_value:.2f}%"

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
                            value='original',  # 默认选中原始模式
                            clearable=False,
                            style=dropdown_style
                        )
                    ], style=dropdown_container_style)
                ], style={'position': 'relative', 'flex': '1', 'margin': '0'})
            ], style=row_style)
        ], style=layout_style)
