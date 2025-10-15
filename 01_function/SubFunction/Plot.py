import streamlit as st
import pandas as pd
import math
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from PIL import Image
import io



def plot_rc_beam_section(b, H, top_steel_list, bottom_steel_list, top_diameters, bottom_diameters, ds, covering, top_row_spacing, bottom_row_spacing):
    """Plot RC beam section with reinforcement."""
    bottom_steel_y = [
        covering + ds + bottom_diameters[i] / 2 + i * (bottom_diameters[i] + bottom_row_spacing)
        for i in range(len(bottom_steel_list))
    ]
    top_steel_y = [
        H - (covering + ds + top_diameters[i] / 2 + i * (top_diameters[i] + top_row_spacing))
        for i in range(len(top_steel_list))
    ]

    # Stirrup coordinates
    stirrup_outer_x = [covering, b - covering, b - covering, covering, covering]
    stirrup_outer_y = [covering, covering, H - covering, H - covering, covering]
    stirrup_inner_x = [covering + ds, b - covering - ds, b - covering - ds, covering + ds, covering + ds]
    stirrup_inner_y = [covering + ds, covering + ds, H - covering - ds, H - covering - ds, covering + ds]

    fig = go.Figure()

    # Beam section
    fig.add_shape(
        type="rect",
        x0=0, y0=0, x1=b, y1=H,
        line=dict(color="black", width=2),
        fillcolor="lightgray",
        opacity=0.5
    )

    # Stirrup lines
    fig.add_trace(go.Scatter(
        x=stirrup_outer_x, y=stirrup_outer_y, 
        mode="lines", 
        line=dict(color="blue", width=2), 
        showlegend=False,
        name="Stirrup"
    ))
    fig.add_trace(go.Scatter(
        x=stirrup_inner_x, y=stirrup_inner_y, 
        mode="lines", 
        line=dict(color="blue", width=2), 
        showlegend=False
    ))

    def distribute_steel_bars(steel_list, y_positions, diameters, color="red"):
        for i, y_position in enumerate(y_positions):
            num_bars = steel_list[i]
            diameter = diameters[i]
            
            if num_bars == 1:
                x_positions = [b / 2]
            elif num_bars == 2:
                x_positions = [covering + ds + diameter / 2, b - covering - ds - diameter / 2]
            else:
                spacing = (b - 2 * (covering + ds) - diameter) / (num_bars - 1)
                x_positions = [covering + ds + diameter / 2 + j * spacing for j in range(num_bars)]
            
            for x in x_positions:
                fig.add_shape(
                    type="circle",
                    x0=x - diameter / 2, y0=y_position - diameter / 2,
                    x1=x + diameter / 2, y1=y_position + diameter / 2,
                    line=dict(color=color, width=2), 
                    fillcolor=color
                )

    # Add steel bars
    if top_steel_list:
        distribute_steel_bars(top_steel_list, top_steel_y, top_diameters, "darkred")
    distribute_steel_bars(bottom_steel_list, bottom_steel_y, bottom_diameters, "red")

    # Add dimensions
    fig.add_annotation(
        x=b/2, y=-H*0.1,
        text=f"b = {b} cm",
        showarrow=False,
        font=dict(size=12, color="black")
    )
    fig.add_annotation(
        x=-b*0.15, y=H/2,
        text=f"H = {H} cm",
        showarrow=False,
        font=dict(size=12, color="black"),
        textangle=90
    )

    fig.update_layout(
        title="RC Beam Cross-Section",
        xaxis=dict(
            showgrid=True, gridwidth=0.5, gridcolor="lightgray",
            scaleanchor="y",
            tickmode="linear", dtick=5,
            range=[-b*0.2, b*1.2],
            title="Width (cm)"
        ),
        yaxis=dict(
            showgrid=True, gridwidth=0.5, gridcolor="lightgray",
            tickmode="linear", dtick=5,
            range=[-H*0.2, H*1.2],
            title="Height (cm)"
        ),
        width=600,
        height=400,
        margin=dict(l=50, r=50, t=50, b=50)
    )

    return fig



def plot_rc_beam_section(b, H, top_steel_list, bottom_steel_list, top_diameters, bottom_diameters, ds, covering, top_row_spacing, bottom_row_spacing):
    """Plot RC beam section with reinforcement."""
    bottom_steel_y = [
        covering + ds + bottom_diameters[i] / 2 + i * (bottom_diameters[i] + bottom_row_spacing)
        for i in range(len(bottom_steel_list))
    ]
    top_steel_y = [
        H - (covering + ds + top_diameters[i] / 2 + i * (top_diameters[i] + top_row_spacing))
        for i in range(len(top_steel_list))
    ]

    # Stirrup coordinates
    stirrup_outer_x = [covering, b - covering, b - covering, covering, covering]
    stirrup_outer_y = [covering, covering, H - covering, H - covering, covering]
    stirrup_inner_x = [covering + ds, b - covering - ds, b - covering - ds, covering + ds, covering + ds]
    stirrup_inner_y = [covering + ds, covering + ds, H - covering - ds, H - covering - ds, covering + ds]

    fig = go.Figure()

    # Beam section
    fig.add_shape(
        type="rect",
        x0=0, y0=0, x1=b, y1=H,
        line=dict(color="black", width=2),
        fillcolor="lightgray",
        opacity=0.5
    )

    # Stirrup lines
    fig.add_trace(go.Scatter(
        x=stirrup_outer_x, y=stirrup_outer_y, 
        mode="lines", 
        line=dict(color="blue", width=2), 
        showlegend=False,
        name="Stirrup"
    ))
    fig.add_trace(go.Scatter(
        x=stirrup_inner_x, y=stirrup_inner_y, 
        mode="lines", 
        line=dict(color="blue", width=2), 
        showlegend=False
    ))
