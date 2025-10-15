import streamlit as st
import pandas as pd
import math
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from PIL import Image
import io

def hognestad_stress_strain(Ec, fc):
    """Generate Hognestad stress-strain relationship for concrete."""
    # Calculate key parameters
    e0 = 2 * fc / Ec  # Strain at peak stress
    ec = np.linspace(0, 0.005, 100)  # Strain range up to 0.5%
    
    fci_list = []
    for strain in ec:
        if strain <= e0:
            # Ascending branch
            fci = fc * (2 * strain / e0 - (strain / e0) ** 2)
        else:
            # Descending branch
            fci = fc * (1 - 0.15 * ((strain - e0) / (0.0038 - e0)))
            if fci < 0:
                fci = 0
        fci_list.append(fci)
    
    # Create plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ec * 1000,  # Convert to microstrain
        y=fci_list,
        mode='lines',
        name='Hognestad Model',
        line=dict(color='blue', width=3)
    ))
    
    fig.update_layout(
        title='Concrete Stress-Strain Relationship (Hognestad Model)',
        xaxis=dict(
            title='Strain (×10⁻³)',
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray'
        ),
        yaxis=dict(
            title='Stress (kg/cm²)',
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray'
        ),
        width=600,
        height=400
    )
    
    return ec, fci_list, e0, fig
