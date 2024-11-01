from dash import Dash, dcc, html, dash_table, Input, Output, State
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests
import base64
import io

app = Dash(__name__)

app.layout = html.Div([
    html.H1("Advanced Significance Comparison Dashboard"),
    dcc.Upload(
        id='upload-data',
        children=html.Div(['Drag and Drop or ', html.A('Select Files')]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        multiple=False
    ),
    dcc.Graph(id="bar-plot"),
    html.H2("Mean, Standard Deviation, and Significance Results by Group"),
    dash_table.DataTable(id="summary-table"),
    html.H2("Pairwise Comparisons within Groups"),
    dash_table.DataTable(id="comparison-table")
])

def parse_data(contents):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
    return df

def analyze_data(df):
    # 计算每个 group 和 subgroup 的均值和标准差
    summary_df = df.groupby(['group', 'subgroup'])['value'].agg(['mean', 'std']).reset_index()

    # 对每个主分组（group）内的子组（subgroup）进行配对比较
    comparison_results = []
    unique_groups = df['group'].unique()

    for group in unique_groups:
        subgroups = df[df['group'] == group]['subgroup'].unique()
        p_values = []
        comparisons = []

        # 配对t检验并收集 p 值
        for i in range(len(subgroups)):
            for j in range(i + 1, len(subgroups)):
                data1 = df[(df['group'] == group) & (df['subgroup'] == subgroups[i])]['value']
                data2 = df[(df['group'] == group) & (df['subgroup'] == subgroups[j])]['value']
                stat, p_val = ttest_ind(data1, data2)
                comparisons.append((group, subgroups[i], subgroups[j]))
                p_values.append(p_val)

        # 校正 p 值
        if p_values:
            reject, pvals_corrected, _, _ = multipletests(p_values, alpha=0.05, method='bonferroni')
            for (grp, sg1, sg2), p_val_corr, sig in zip(comparisons, pvals_corrected, reject):
                comparison_results.append({
                    'group': grp,
                    'subgroup1': sg1,
                    'subgroup2': sg2,
                    'p_value_corrected': p_val_corr,
                    'significance': '*' if sig else 'ns'
                })

    comparison_df = pd.DataFrame(comparison_results)
    return summary_df, comparison_df

@app.callback(
    [Output('bar-plot', 'figure'),
     Output('summary-table', 'data'),
     Output('summary-table', 'columns'),
     Output('comparison-table', 'data'),
     Output('comparison-table', 'columns')],
    Input('upload-data', 'contents')
)
def update_output(contents):
    if contents is None:
        return {}, [], [], [], []

    # 解析数据
    df = parse_data(contents)

    # 进行数据分析
    summary_df, comparison_df = analyze_data(df)

    # 绘制柱状图
    fig = go.Figure()
    for _, row in summary_df.iterrows():
        fig.add_trace(go.Bar(
            x=[f"{row['group']} - {row['subgroup']}"],
            y=[row['mean']],
            name=f"{row['group']} - {row['subgroup']}",
            error_y=dict(type='data', array=[row['std']])
        ))

    fig.update_layout(
        yaxis_title="Mean Value",
        xaxis_title="Groups and Subgroups",
        title="Mean Comparison with Significance Annotations"
    )

    # 生成数据表列
    summary_columns = [{"name": i, "id": i} for i in summary_df.columns]
    comparison_columns = [
        {"name": "Group", "id": "group"},
        {"name": "Subgroup 1", "id": "subgroup1"},
        {"name": "Subgroup 2", "id": "subgroup2"},
        {"name": "Corrected P-value", "id": "p_value_corrected"},
        {"name": "Significance", "id": "significance"}
    ]

    return fig, summary_df.to_dict('records'), summary_columns, comparison_df.to_dict('records'), comparison_columns

if __name__ == '__main__':
    app.run_server(debug=True)
