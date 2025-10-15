import streamlit as st
import pandas as pd
import math
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from PIL import Image
import io

def plot_interaction_diagram(P_range, moments):
    """Plot P-M interaction diagram."""
    fig = go.Figure()
    
    if len(P_range) > 0 and len(moments) > 0:
        fig.add_trace(go.Scatter(
            x=moments,
            y=P_range,
            mode='lines+markers',
            name='P-M Interaction',
            line=dict(color='darkblue', width=3),
            marker=dict(size=4)
        ))
        
        # Add balanced point indicator
        max_moment_idx = np.argmax(moments)
        fig.add_trace(go.Scatter(
            x=[moments[max_moment_idx]],
            y=[P_range[max_moment_idx]],
            mode='markers',
            name='Balanced Point',
            marker=dict(color='red', size=10, symbol='star')
        ))
    
    fig.update_layout(
        title='Axial Load - Moment Interaction Diagram',
        xaxis=dict(
            title='Moment (kg-m)',
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray'
        ),
        yaxis=dict(
            title='Axial Load (tons)',
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray'
        ),
        width=600,
        height=500
    )
    
    return fig