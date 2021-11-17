import base64
import datetime
import io

import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import dash_table
import pandas as pd
import plotly.graph_objects as go

global_data: pd.DataFrame

# Function definitions
def field_list(data=None):
    columns = [] if data is None else data.columns
    long = [x for x in columns if 'Long' in x]
    lat = [x for x in columns if 'Lat' in x]
    return [
        dcc.Markdown('**Plot Fields**'),
        html.Label('Identifier'),
        dcc.Dropdown(
            id='table-col-idx',
            options=[{'label': i, 'value': i} for i in columns],
            value= columns[0] if len(columns) > 0 else None,
        ),
        html.Label('Longitude'),
        dcc.Dropdown(
            id='table-col-long',
            options=[{'label': i, 'value': i} for i in columns],
            value=long[0] if len(long) > 0 else None
        ),
        html.Label('Latitude'),
        dcc.Dropdown(
            id='table-col-lat',
            options=[{'label': i, 'value': i} for i in columns],
            value=lat[0] if len(lat) > 0 else None
        )]

def parse_contents(contents, filename, delim):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    try:
        # Assume this is a CSV file
        df = pd.read_csv(
            io.StringIO(decoded.decode('utf-8')),
            delimiter=delim
        )
    except Exception as e:
        print(e)
        return html.Div([
            'Error processing the file'
        ])

    return html.Div([
        dash_table.DataTable(
            id='table-md-table',
            data=df.to_dict('records'),
            columns=[{'name': i, 'id': i, 'deletable': True} for i in df.columns],
            page_action='none',
            style_table=dict(height='60vh',overflowY='auto'),
            filter_action='native',
            sort_action='native',
            sort_mode='multi',
            editable=True,

        )
    ]), field_list(df)

def make_map(data, lon, lat):
    fig = go.Figure(data=go.Scattergeo(
        lon = data[lon],
        lat = data[lat],
        # text = [row_to_text(x) for x in md.iterrows()],
        mode = 'markers'
    ))
    fig.update_layout(
        # title=f'Selected Samples ({len(data)})',
        geo = dict(
            showland = True,
            showocean=True, oceancolor='LightBlue',
            projection_type='natural earth'
        ),
        # width=400,
        margin=dict(r=10, t=0, l=10, b=0)
    )
    return dcc.Graph(
        id='map',
        figure=fig
    )

# Make an interface for loading data, metadata, and allowing filtering
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

load_panel = html.Div([
    dcc.Markdown('''
    This page allows you to select a subset of samples. Please upload a CSV with metadata for samples, which should 
    include at minimum the following columns:
    * A unique identifier for each sample
    * Latitude
    * Longitude
    '''),
    html.Label('Delimiter'),
    dcc.Dropdown(
        id='upload-md-delim',
        options=[
            {'label': 'Comma', 'value': ','},
            {'label': 'Tab', 'value': '\t'},
            {'label': 'Pipe', 'value': '|'}
        ],
        value = ','
    ),
    dcc.Upload(
        id='upload-md-file',
        children=html.Button('Select CSV')
    ),
    html.Div(
        id='upload-md-fname'
    )
])

map_panel = html.Div([], id='map-div')

field_panel = html.Div([

], id='')

panel_layout = html.Div([
    html.Div([load_panel], id='load-panel', style=dict(flex=1)),
    html.Div(field_list(None), id='field-panel', style=dict(flex=1)),
    html.Div([map_panel], id='map-panel', style=dict(flex=2))
], id='panel-container', style=dict(display='flex', height='30vh')
)

app.layout = html.Div([
    html.H2('Sample Selection'),
    panel_layout,
    html.Div(id='table-md'),
])


# CALLBACKS FOR DATA SELECTION
@app.callback(
    Output('table-md', 'children'),
    Output('field-panel', 'children'),
    Input('upload-md-file', 'contents'),
    State('upload-md-file', 'filename'),
    State('upload-md-delim', 'value')
)
def update_output(contents, filename, delim):
    if not contents is None:
        return parse_contents(contents, filename, delim)


@app.callback(
    Output('upload-md-fname', 'children'),
    Input('upload-md-file', 'filename')
)
def update_output_fname(filename):
    if not filename is None:
        return html.Div(f'Opened {filename}')


# CALLBACKS FOR DATA TABLE
@app.callback(
    Output('map-div', 'children'),
    Input('table-md-table', 'derived_virtual_data'),
    State('table-col-idx', 'value'),
    State('table-col-long', 'value'),
    State('table-col-lat', 'value')
)
def update_map(data, index, long, lat):
    # If no data passed, send empty data frame (drawing no points)
    if data is None:
        data = pd.DataFrame()
    else:
        data = pd.DataFrame(data)
        if index in data.columns:
            data = data.set_index(index)
    return make_map(data, long, lat)

if __name__ == '__main__':
    app.run_server(debug=True)
