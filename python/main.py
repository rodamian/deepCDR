
import pandas as pd
import os
import re
import plotly.express as px  # (version 4.7.0)
import plotly.graph_objects as go

import dash  # (version 1.12.0) pip install dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

app = dash.Dash(__name__)

# ------------------------------------------------------------------------------
# Import and clean data (importing csv into pandas)
dir_names = [directory for directory in os.listdir("Immunizations") if "mouse" in directory]
df_list = [pd.read_csv("Immunizations/" + directory + "/all_contig_annotations.csv") for directory in dir_names]
data = pd.concat(df_list, keys=dir_names, names=["mouse_id", "id"])

length_cdr3 = data.groupby("mouse_id")["cdr3"].apply(lambda x: [len(y) for y in x])

# ------------------------------------------------------------------------------
# App layout
app.layout = html.Div([
    html.H1("Clonotype data visualization", style={'text-align': 'center'}),
    dcc.Dropdown(id="slct_clonotype",
                 options=[
                     {"label": "mouse1", "value": "mouse1"},
                     {"label": "mouse2", "value": "mouse2"},
                     {"label": "mouse3", "value": "mouse3"},
                     {"label": "mouse4", "value": "mouse4"},
                     {"label": "mouse4", "value": "mouse4"}],
                 multi=False,
                 value="mouse1",
                 style={'width': "40%"}
                 ),

    html.Div(id='output_container', children=[]),
    html.Br(),

    dcc.Graph(id='len_hist', figure={})

])


# ------------------------------------------------------------------------------
# Connect the Plotly graphs with Dash Components
@app.callback(
    [Output(component_id='output_container', component_property='children'),
     Output(component_id='len_hist', component_property='figure')],
    [Input(component_id='slct_clonotype', component_property='value')]
)
def update_graph(option_slctd):
    print(option_slctd)
    print(type(option_slctd))

    container = "The mouse chosen by user was: {}".format(option_slctd)

    dff = length_cdr3.copy()
    dff = dff.loc[option_slctd]

    # Plotly Express
    fig = px.histogram(dff, template='plotly_dark',
                       labels={'length CDR3 region': 'frequency'},
                       hover_data=[],
    )
    # Plotly Graph Objects (GO)
    # fig = go.Figure(
    #     data=[go.Choropleth(
    #         locationmode='USA-states',
    #         locations=dff['state_code'],
    #         z=dff["Pct of Colonies Impacted"].astype(float),
    #         colorscale='Reds',
    #     )]
    # )
    #
    # fig.update_layout(
    #     title_text="Bees Affected by Mites in the USA",
    #     title_xanchor="center",
    #     title_font=dict(size=24),
    #     title_x=0.5,
    #     geo=dict(scope='usa'),
    # )

    return container, fig

# ------------------------------------------------------------------------------
if __name__ == '__main__':
    app.run_server(debug=True)
