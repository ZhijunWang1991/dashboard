import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from dash import Dash, dcc, html, Input, Output
import plotly.express as px

# Sample data generation (replace with your actual data)
data = {
    "Treatment": ["A", "B", "C", "A", "B", "C", "A", "B", "C"],
    "Score": [23, 45, 33, 20, 44, 34, 25, 46, 35],
}
df = pd.DataFrame(data)

# Initialize the Dash app
app = Dash(__name__)

# Layout of the dashboard
app.layout = html.Div([
    html.H1("ANOVA Analysis Dashboard"),
    dcc.Graph(id='anova-boxplot'),
    html.Div(id='anova-results')
])

# Callbacks for plot and ANOVA results
@app.callback(
    [Output('anova-boxplot', 'figure'), Output('anova-results', 'children')],
    Input('anova-boxplot', 'id')
)
def update_graph(_):
    # Plot boxplot
    fig = px.box(df, x="Treatment", y="Score", title="Treatment vs. Score")

    # Perform ANOVA
    model = ols('Score ~ C(Treatment)', data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    
    # Convert ANOVA results to HTML
    results = f"<h3>ANOVA Results:</h3>{anova_table.to_html()}"

    return fig, results

# Run the app locally
if __name__ == "__main__":
    app.run_server(debug=False)
