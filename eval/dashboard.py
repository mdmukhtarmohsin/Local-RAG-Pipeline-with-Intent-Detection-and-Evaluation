"""
Dashboard for visualizing RAG evaluation results
"""

import gradio as gr
import pandas as pd
import plotly.express as px
import os

EVAL_DIR = "eval"
SUMMARY_FILE = os.path.join(EVAL_DIR, "evaluation_summary.csv")
DETAILED_FILE = os.path.join(EVAL_DIR, "evaluation_results_detailed.csv")

def load_data():
    """Load evaluation data."""
    summary_df = pd.DataFrame()
    detailed_df = pd.DataFrame()

    if os.path.exists(SUMMARY_FILE):
        summary_df = pd.read_csv(SUMMARY_FILE)
    
    if os.path.exists(DETAILED_FILE):
        detailed_df = pd.read_csv(DETAILED_FILE)
        
    return summary_df, detailed_df

def create_summary_plot(summary_df):
    """Create a bar plot comparing model performance."""
    if summary_df.empty:
        return None
        
    fig = px.bar(
        summary_df,
        x='model_provider',
        y=['intent_accuracy', 'relevance_score', 'context_utilization'],
        barmode='group',
        title="Model Performance Comparison",
        labels={'value': 'Score', 'variable': 'Metric', 'model_provider': 'Model Provider'}
    )
    return fig

def create_latency_plot(summary_df):
    """Create a bar plot for latency."""
    if summary_df.empty:
        return None
        
    fig = px.bar(
        summary_df,
        x='model_provider',
        y='latency',
        title="Average Latency per Model",
        labels={'latency': 'Latency (s)', 'model_provider': 'Model Provider'}
    )
    return fig

def create_cost_plot(summary_df):
    """Create a bar plot for token cost."""
    if summary_df.empty:
        return None
        
    fig = px.bar(
        summary_df,
        x='model_provider',
        y='token_cost',
        title="Total Token Cost per Model",
        labels={'token_cost': 'Total Tokens', 'model_provider': 'Model Provider'}
    )
    return fig

def build_dashboard():
    """Builds the Gradio dashboard UI."""
    summary_df, detailed_df = load_data()

    with gr.Blocks(theme="soft", title="Evaluation Dashboard") as demo:
        gr.Markdown("# 📊 RAG Pipeline Evaluation Dashboard")

        if summary_df.empty or detailed_df.empty:
            gr.Markdown("## No evaluation data found.")
            gr.Markdown(f"Please run `python eval/evaluator.py` first to generate the results.")
            return demo

        with gr.Tab("Summary & Charts"):
            gr.Markdown("## - Evaluation Summary")
            gr.DataFrame(summary_df, interactive=False)
            
            with gr.Row():
                gr.Plot(create_summary_plot(summary_df))
                gr.Plot(create_latency_plot(summary_df))
            
            with gr.Row():
                gr.Plot(create_cost_plot(summary_df))

        with gr.Tab("Detailed Results"):
            gr.Markdown("## - Detailed Evaluation Results")
            gr.DataFrame(detailed_df, interactive=True)

    return demo

if __name__ == "__main__":
    dashboard = build_dashboard()
    dashboard.launch(server_name="0.0.0.0", server_port=7861) 