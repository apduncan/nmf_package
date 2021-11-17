import base64
from io import BytesIO

import dash
import dash_core_components as dcc
import dash_html_components as html
import matplotlib.pyplot as plt
from pickle import load
from dash.dependencies import Output, Input

from mg_nmf.nmf import NMFGeneSetEnrichment
from mg_nmf import nmf as visualise, nmf as selection

"""
Attempt at a multipage app to
* View heatmaps summarising the model
* View a PCA-RGB map of the model
* Enrichment analysis and gene breakdown
"""

# Load some results to look at
with open('/home/hal/Dropbox/PHD/FunctionalAbundance/nmf/data/metatranscriptome_k6.enrich', 'rb') as f:
    analysis: NMFGeneSetEnrichment = load(f)
    
MODEL_RES = '/home/hal/Dropbox/PHD/FunctionalAbundance/nmf/data/metatranscriptome_k6.model'
res = selection.NMFModelSelectionResults.load_model(MODEL_RES)

# Methods to generate some plots
def heatmap_image(results):
    fig = visualise.heatmap_plot(res, axes=[0], return_fig=True, log=False)
    pic_bytes = BytesIO()
    plt.savefig(pic_bytes, format='png')
    pic_bytes.seek(0)
    pic_hash = base64.b64encode(pic_bytes.read()).decode("ascii").replace("\n", "")
    img = html.Img(src="data:image/png;base64,{}".format(pic_hash))
    return img

def enrichment_plot(enrichment):
    return dcc.Graph(
        id='enrichment-heatmap',
        figure=enrichment.plot_enrichment(enrichment.results(significance=0.05), group='namespace', label='name',
                                          width=None, height=None),
        style=dict(height='100%'),
        config=dict(displayModeBar=False)
    )

app = dash.Dash(__name__)

heatmap_page = html.Div([
    html.H1('Hi'),
    heatmap_image(res)
])

enrichment_page = html.Div([
        html.Div([
            enrichment_plot(enrichment=analysis),
        ], style=dict(flex=1)),
        html.Div([], id='enrichment-term-plots', style=dict(flex=1, overflowY='auto'))
    ], style=dict(display='flex', height='100%'))

navigation_buttons = html.Div([
        html.Button('Model Heatmap', id='button-heatmap'),
        html.Button('Enrichment', id='button-enrichment'),
        html.Button('Map', id='button-map')
    ], style=dict(display='flex'),
)

layout_index = html.Div([
    navigation_buttons,
    html.Div([], id='content-div', style=dict(height='90vh'))
])

app.layout = layout_index

# Button callbacks
@app.callback(
    Output('content-div', 'children'),
    Input('button-heatmap', 'n_clicks'),
    Input('button-enrichment', 'n_clicks'),
    Input('button-map', 'n_clicks')
)
def button_event(btn1, btn2, btn3):
    ctx = dash.callback_context
    if ctx is None:
        return heatmap_image(res)
    # Find which button triggered the callback
    btn_trigger = ctx.triggered[0]['prop_id'].split('.')[0]
    if btn_trigger == 'button-heatmap':
        return heatmap_image(res)
    if btn_trigger == 'button-enrichment':
        return enrichment_page
    return []

# Heatmap page callbacks

# Enrichment page callbacks
@app.callback(
    Output('enrichment-term-plots', 'children'),
    Input('enrichment-heatmap', 'clickData'))
def display_click_data(clickData):
    # Extract the GO Term from this
    if clickData is None:
        return html.Div()
    go = clickData['points'][0]['y'].split("GO:")[1][:-1]
    go = 'GO:' + go
    component = clickData['points'][0]['x']
    return dcc.Graph(
        id='enrichment-correlation',
        figure=analysis.plot_geneset_correlation(component=component, gene_set_id=go, cols=3, vspace=0.03)
    )

if __name__ == '__main__':
    app.run_server(debug=True)