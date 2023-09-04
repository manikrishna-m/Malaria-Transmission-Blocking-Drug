import json
import pickle
from argparse import ArgumentParser
from functools import lru_cache
from pathlib import Path
from typing import List, Tuple

import dash
import dash_core_components as dcc
import dash_html_components as html
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dash.dependencies import Input, Output
from loguru import logger

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)


def load_figure(file: Path) -> go.Figure:
    data = json.loads(file.read_text())
    fig = go.Figure(data=data)
    fig.update_traces(
        marker_size=1,
        selector=dict(mode='markers')
    )
    return fig

@lru_cache
def load_file_contents(file: Path) -> List[Path]:
    data = pickle.loads(file.read_bytes())
    files, points = zip(*list(data.items()))
    return files

def prepare_dash(fig: go.FigureWidget) -> dash.Dash:
    app.layout = html.Div([
        html.Pre(id="selection-data", style={"fontSize": "22px"}),
        dcc.Graph(figure=fig, id="scatter-plot", style={"width": "100vw", "height": "100vh"}),
    ])
    return app

@app.callback(
    Output(component_id='selection-data', component_property='children'),
    Input(component_id='scatter-plot', component_property='clickData')
)
def handle_selection(data):
    index = data['points'][0]["pointIndex"]
    files = load_file_contents(Path("data/embeddings.pkl"))
    
    return json.dumps(
        {
            "cell": files[index],
            "parent_image": Path(files[index]).parent.name
        }, indent=4
    )

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("data", type=Path)
    args = parser.parse_args()
    figure = load_figure(file=args.data)
    dash = prepare_dash(fig=figure)
    dash.run_server(debug=False, use_reloader=False)
