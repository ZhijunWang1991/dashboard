import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from dash import Dash, dcc, html, Input, Output, State
import plotly.express as px
import base64
import io

# Initialize the Dash app
app = Dash(__name__)

# Layout of the dashboard
app.layout = html.Div([
    html.H1("ANOVA Analysis Dashboard"),
    
    # File upload component
    dcc.Upload(
        id='upload-data',
        children=html.Button('Upload CSV File'),
        multiple=False
    ),
    
    # Dropdowns for selecting X and Y axes for scatter plot
    html.Div([
        html.Label("Select X-axis:"),
        dcc.Dropdown(id='x-axis', clearable=False),
        
        html.Label("Select Y-axis:"),
        dcc.Dropdown(id='y-axis', clearable=False),
        
        html.Label("Select Categorical Variable for ANOVA:"),
        dcc.Dropdown(id='anova-cat', clearable=False),
    ], style={'display': 'flex', 'flex-direction': 'column', 'width': '20%'}),
    
    # Graph to display scatter plot
    dcc.Graph(id='scatter-plot'),
    
    # Display ANOVA results
    html.Div(id='anova-results')
])

# Callback to parse uploaded CSV and update dropdown options
@app.callback(
    [Output('x-axis', 'options'),
     Output('y-axis', 'options'),
     Output('anova-cat', 'options')],
    Input('upload-data', 'contents')
)
def parse_data(contents):
    if contents is None:
        return [], [], []
    
    # Parse contents
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
    
    # Generate column options
    options = [{'label': col, 'value': col} for col in df.columns]
    return options, options, options

# Callback to update scatter plot and perform ANOVA
@app.callback(
    [Output('scatter-plot', 'figure'), Output('anova-results', 'children')],
    [Input('x-axis', 'value'),
     Input('y-axis', 'value'),
     Input('anova-cat', 'value')],
    State('upload-data', 'contents')
)
def update_graph(x_axis, y_axis, anova_cat, contents):
    if contents is None or x_axis is None or y_axis is None or anova_cat is None:
        return {}, "Upload a dataset and select variables."

    # Parse contents
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))

    # Generate scatter plot
    fig = px.scatter(df, x=x_axis, y=y_axis, color=anova_cat, title="Scatter Plot")

    # Perform ANOVA
    model = ols(f'{y_axis} ~ C({anova_cat})', data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)

    # Convert ANOVA results to HTML
    results = f"<h3>ANOVA Results:</h3>{anova_table.to_html()}"
    return fig, results

# Run the app locally
if __name__ == "__main__":
    app.run_server(debug=False)
