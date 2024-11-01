from flask import Flask, render_template, request, jsonify
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests
import base64
import io
from itertools import cycle, product
import json
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)  # Set logging level

# Parse uploaded data
def parse_data(contents):
    try:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        
        # Check if the required columns are present
        if not {'group', 'subgroup', 'value'}.issubset(df.columns):
            raise ValueError("Uploaded CSV must contain 'group', 'subgroup', and 'value' columns.")
        
        return df
    except Exception as e:
        logging.error(f"Data parsing failed: {e}")
        return None

# Generate alphabetical significance labels
def generate_labels():
    single_letter_labels = [chr(i) for i in range(ord('a'), ord('z') + 1)]
    double_letter_labels = [''.join(pair) for pair in product(single_letter_labels, repeat=2)]
    return cycle(single_letter_labels + double_letter_labels)

# Analyze data and compute group comparisons
def analyze_data(df):
    try:
        summary_df = df.groupby(['group', 'subgroup'])['value'].agg(['mean', 'std']).reset_index()
        comparison_results = []
        unique_groups = df['group'].unique()
        significance_labels = generate_labels()

        for group in unique_groups:
            subgroups = df[df['group'] == group]['subgroup'].unique()
            p_values = []
            comparisons = []

            for i in range(len(subgroups)):
                for j in range(i + 1, len(subgroups)):
                    data1 = df[(df['group'] == group) & (df['subgroup'] == subgroups[i])]['value']
                    data2 = df[(df['group'] == group) & (df['subgroup'] == subgroups[j])]['value']
                    stat, p_val = ttest_ind(data1, data2)
                    comparisons.append((group, subgroups[i], subgroups[j]))
                    p_values.append(p_val)

            if p_values:
                reject, pvals_corrected, _, _ = multipletests(p_values, alpha=0.05, method='bonferroni')
                for (grp, sg1, sg2), p_val_corr, sig in zip(comparisons, pvals_corrected, reject):
                    comp_label = next(significance_labels) if sig else 'ns'
                    comparison_results.append({
                        'group': grp,
                        'subgroup1': sg1,
                        'subgroup2': sg2,
                        'p_value_corrected': p_val_corr,
                        'significance': comp_label
                    })

        comparison_df = pd.DataFrame(comparison_results)
        return summary_df, comparison_df
    except Exception as e:
        logging.error(f"Data analysis failed: {e}")
        return None, None

# Generate bar plot
def generate_bar_plot(summary_df):
    try:
        fig = go.Figure()
        unique_groups = summary_df['group'].unique()
        bar_colors = ['#4d4d4d', '#7f7f7f', '#a6a6a6', '#d9d9d9']

        for i, group in enumerate(unique_groups):
            group_data = summary_df[summary_df['group'] == group]
            for j, (_, row) in enumerate(group_data.iterrows()):
                fig.add_trace(go.Bar(
                    x=[f"{group} - {row['subgroup']}"],
                    y=[row['mean']],
                    name=f"{group} - {row['subgroup']}",
                    marker_color=bar_colors[j % len(bar_colors)],
                    error_y=dict(type='data', array=[row['std']])
                ))

        fig.update_layout(
            title="Group and Subgroup Mean Values with Significance Labels",
            yaxis_title="Mean Value",
            xaxis_title="Groups and Subgroups",
            xaxis=dict(type='category', categoryorder='category ascending'),
            legend_title="Groups and Subgroups",
            barmode='group',
            bargap=0.15,
            bargroupgap=0.1
        )
        return fig.to_json()
    except Exception as e:
        logging.error(f"Plot generation failed: {e}")
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_data():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    contents = base64.b64encode(file.read()).decode('utf-8')
    df = parse_data("data:text/csv;base64," + contents)

    if df is None:
        return jsonify({"error": "Failed to parse CSV. Ensure it has 'group', 'subgroup', and 'value' columns."}), 400

    # Data analysis
    summary_df, comparison_df = analyze_data(df)
    if summary_df is None or comparison_df is None:
        return jsonify({"error": "Data analysis failed. Check the logs for details."}), 500
    
    # Plot
    plot_json = generate_bar_plot(summary_df)
    if plot_json is None:
        return jsonify({"error": "Failed to generate plot. Check the logs for details."}), 500

    # Table data
    summary_table_data = summary_df.to_dict('records')
    summary_columns = [{"name": col, "id": col} for col in summary_df.columns]
    
    comparison_table_data = comparison_df.to_dict('records')
    comparison_columns = [
        {"name": "Group", "id": "group"},
        {"name": "Subgroup 1", "id": "subgroup1"},
        {"name": "Subgroup 2", "id": "subgroup2"},
        {"name": "Corrected P-value", "id": "p_value_corrected"},
        {"name": "Significance", "id": "significance"}
    ]

    return jsonify({
        "plot": plot_json,
        "summary_table": {
            "data": summary_table_data,
            "columns": summary_columns
        },
        "comparison_table": {
            "data": comparison_table_data,
            "columns": comparison_columns
        }
    })

if __name__ == '__main__':
    app.run(debug=False)  # Enable debug for development; turn off in production
