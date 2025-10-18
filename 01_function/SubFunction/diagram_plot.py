"""
Diagram Plotting Module
Handles all visualization functions using Plotly
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np


def plot_section_and_strain(b_mm, h_mm, cover_mm, bar_data, strain_profile):
    """
    Create combined plot showing beam section and strain diagram side by side
    Both diagrams share the same vertical (height) scale for visual alignment
    
    Parameters:
    -----------
    b_mm : float
        Beam width in mm
    h_mm : float
        Beam height in mm
    cover_mm : float
        Concrete cover in mm
    bar_data : list
        List of dictionaries containing bar information
    strain_profile : dict
        Strain distribution data
    
    Returns:
    --------
    plotly.graph_objects.Figure
    """
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Beam Cross-Section', 'Strain Distribution'),
        horizontal_spacing=0.12,
        specs=[[{"type": "xy"}, {"type": "xy"}]]
    )
    
    # ===========================
    # SUBPLOT 1: BEAM SECTION
    # ===========================
    
    # Draw beam outline
    fig.add_shape(
        type="rect",
        x0=0, y0=0, x1=b_mm, y1=h_mm,
        line=dict(color="black", width=2),
        fillcolor="rgba(220, 220, 220, 0.4)",
        row=1, col=1
    )
    
    if strain_profile:
        # Draw compression block
        a_mm = strain_profile['a_mm']
        fig.add_shape(
            type="rect",
            x0=0, y0=h_mm - a_mm, x1=b_mm, y1=h_mm,
            fillcolor="rgba(255, 99, 71, 0.25)",
            line=dict(color="red", width=1, dash="dash"),
            row=1, col=1
        )
        
        # Draw neutral axis line
        c_mm = strain_profile['c_mm']
        fig.add_shape(
            type="line",
            x0=0, y0=h_mm - c_mm, x1=b_mm, y1=h_mm - c_mm,
            line=dict(color="green", width=2, dash="dash"),
            row=1, col=1
        )
        
        # Neutral axis label
        fig.add_annotation(
            x=b_mm * 1.08, y=h_mm - c_mm,
            text="N.A.",
            showarrow=False,
            font=dict(size=10, color="green", family="Arial Black"),
            row=1, col=1
        )
    
    # Draw reinforcement bars
    if bar_data:
        for bar in bar_data:
            y = bar['y_mm']
            dia = bar['dia_mm']
            num = bar['num_bars']
            
            # Distribute bars across width
            if num == 1:
                x_positions = [b_mm / 2]
            elif num == 2:
                x_positions = [cover_mm + dia / 2, 
                              b_mm - cover_mm - dia / 2]
            else:
                # Evenly distribute bars
                spacing = (b_mm - 2 * cover_mm - dia) / (num - 1)
                x_positions = [cover_mm + dia / 2 + i * spacing 
                              for i in range(num)]
            
            # Draw each bar
            for x in x_positions:
                fig.add_shape(
                    type="circle",
                    x0=x - dia / 2, y0=y - dia / 2,
                    x1=x + dia / 2, y1=y + dia / 2,
                    fillcolor="darkred",
                    line=dict(color="black", width=1),
                    row=1, col=1
                )
    
    # Add dimension annotations
    fig.add_annotation(
        x=b_mm / 2, y=-h_mm * 0.08,
        text=f"b = {b_mm:.0f} mm",
        showarrow=False,
        font=dict(size=11, color="black"),
        row=1, col=1
    )
    
    fig.add_annotation(
        x=-b_mm * 0.12, y=h_mm / 2,
        text=f"h = {h_mm:.0f} mm",
        showarrow=False,
        font=dict(size=11, color="black"),
        textangle=-90,
        row=1, col=1
    )
    
    # ===========================
    # SUBPLOT 2: STRAIN DIAGRAM
    # ===========================
    
    if strain_profile:
        epsilon_c = strain_profile['epsilon_c']
        c_mm = strain_profile['c_mm']
        d_mm = strain_profile['d_mm']
        epsilon_t = strain_profile['epsilon_t']
        
        # Strain at bottom fiber
        epsilon_bottom = epsilon_c * (c_mm - h_mm) / c_mm
        
        # Strain profile line (from top to bottom)
        strain_x = [epsilon_c, epsilon_bottom]
        strain_y = [h_mm, 0]
        
        fig.add_trace(
            go.Scatter(
                x=strain_x, y=strain_y,
                mode='lines',
                line=dict(color='blue', width=3),
                name='Strain Profile',
                showlegend=False
            ),
            row=1, col=2
        )
        
        # Compression zone shading
        fig.add_trace(
            go.Scatter(
                x=[0, epsilon_c, epsilon_c, 0, 0],
                y=[h_mm - c_mm, h_mm, h_mm, h_mm - c_mm, h_mm - c_mm],
                fill='toself',
                fillcolor='rgba(255, 99, 71, 0.2)',
                line=dict(width=0),
                showlegend=False,
                hoverinfo='skip'
            ),
            row=1, col=2
        )
        
        # Tension zone shading
        fig.add_trace(
            go.Scatter(
                x=[0, epsilon_bottom, epsilon_bottom, 0, 0],
                y=[h_mm - c_mm, 0, 0, h_mm - c_mm, h_mm - c_mm],
                fill='toself',
                fillcolor='rgba(30, 144, 255, 0.2)',
                line=dict(width=0),
                showlegend=False,
                hoverinfo='skip'
            ),
            row=1, col=2
        )
        
        # Neutral axis line
        fig.add_shape(
            type="line",
            x0=min(epsilon_bottom, 0) * 1.1, y0=h_mm - c_mm,
            x1=epsilon_c * 1.1, y1=h_mm - c_mm,
            line=dict(color="green", width=2, dash="dash"),
            row=1, col=2
        )
        
        # Mark strains at bar locations
        for y_pos, eps in strain_profile['bar_strains']:
            fig.add_trace(
                go.Scatter(
                    x=[eps], y=[y_pos],
                    mode='markers',
                    marker=dict(size=8, color='red', symbol='circle'),
                    showlegend=False,
                    hovertemplate=f'y={y_pos:.0f}mm<br>ε={eps:.5f}<extra></extra>'
                ),
                row=1, col=2
            )
        
        # Strain annotations
        fig.add_annotation(
            x=epsilon_c, y=h_mm,
            text=f"εc = {epsilon_c:.4f}",
            showarrow=True,
            arrowhead=2,
            ax=-40, ay=-25,
            font=dict(size=10, color="red"),
            row=1, col=2
        )
        
        fig.add_annotation(
            x=epsilon_bottom, y=0,
            text=f"ε = {epsilon_bottom:.4f}",
            showarrow=True,
            arrowhead=2,
            ax=40, ay=25,
            font=dict(size=10, color="blue"),
            row=1, col=2
        )
        
        # ACI tension limit line
        fig.add_shape(
            type="line",
            x0=-0.005, y0=0, x1=-0.005, y1=h_mm,
            line=dict(color="purple", width=1, dash="dot"),
            row=1, col=2
        )
        
        fig.add_annotation(
            x=-0.005, y=h_mm * 0.85,
            text="εt = 0.005<br>(Tension limit)",
            showarrow=False,
            font=dict(size=9, color="purple"),
            row=1, col=2
        )
    
    # Update axes
    fig.update_xaxes(
        title_text="Width (mm)", 
        row=1, col=1, 
        range=[-b_mm * 0.2, b_mm * 1.25],
        showgrid=True,
        gridcolor='lightgray'
    )
    
    fig.update_xaxes(
        title_text="Strain (ε)", 
        row=1, col=2,
        showgrid=True,
        gridcolor='lightgray',
        zeroline=True,
        zerolinecolor='black',
        zerolinewidth=2
    )
    
    fig.update_yaxes(
        title_text="Height (mm)", 
        row=1, col=1, 
        range=[-h_mm * 0.15, h_mm * 1.1],
        showgrid=True,
        gridcolor='lightgray'
    )
    
    fig.update_yaxes(
        title_text="Height (mm)", 
        row=1, col=2, 
        range=[0, h_mm * 1.05],
        showgrid=True,
        gridcolor='lightgray'
    )
    
    fig.update_layout(
        height=500,
        showlegend=False,
        hovermode='closest',
        plot_bgcolor='white',
        font=dict(family="Arial", size=11)
    )
    
    return fig


def plot_three_section_comparison(results_data):
    """
    Create bar charts comparing capacities across three sections
    
    Parameters:
    -----------
    results_data : list
        List of dictionaries with section results
    
    Returns:
    --------
    plotly.graph_objects.Figure
    """
    
    sections = [r['Section'] for r in results_data]
    moments = [float(r['φMn (tonf·m)']) if r['φMn (tonf·m)'] != 'Error' else 0 
               for r in results_data]
    shears = [float(r['φVn (tonf)']) if r['φVn (tonf)'] != 'Error' else 0 
              for r in results_data]
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Bending Capacity Comparison', 'Shear Capacity Comparison')
    )
    
    # Bending capacity bars
    fig.add_trace(
        go.Bar(
            x=sections, 
            y=moments, 
            name='φMn',
            marker_color='#667eea',
            text=[f'{m:.2f}' for m in moments],
            textposition='outside',
            textfont=dict(size=12, color='black')
        ),
        row=1, col=1
    )
    
    # Shear capacity bars
    fig.add_trace(
        go.Bar(
            x=sections, 
            y=shears, 
            name='φVn',
            marker_color='#764ba2',
            text=[f'{s:.2f}' for s in shears],
            textposition='outside',
            textfont=dict(size=12, color='black')
        ),
        row=1, col=2
    )
    
    fig.update_xaxes(title_text="Section", row=1, col=1)
    fig.update_xaxes(title_text="Section", row=1, col=2)
    fig.update_yaxes(title_text="Moment (tonf·m)", row=1, col=1)
    fig.update_yaxes(title_text="Shear (tonf)", row=1, col=2)
    
    fig.update_layout(
        height=400,
        showlegend=False,
        hovermode='x unified',
        plot_bgcolor='white',
        font=dict(family="Arial", size=11)
    )
    
    return fig
