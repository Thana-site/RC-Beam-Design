"""
Visualization Module
All plotting functions using Plotly
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np


def draw_section_and_strain(b_mm, h_mm, cover_mm, bar_data, strain_profile):
    """
    Create unified figure with section and strain diagrams side-by-side
    Both share the same vertical scale for visual alignment
    
    Args:
        b_mm: Beam width (mm)
        h_mm: Beam height (mm)
        cover_mm: Cover (mm)
        bar_data: List of bar dictionaries
        strain_profile: Strain distribution data
        
    Returns:
        plotly Figure object
    """
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('<b>Beam Cross-Section</b>', '<b>Strain Distribution</b>'),
        horizontal_spacing=0.15,
        specs=[[{"type": "xy"}, {"type": "xy"}]]
    )
    
    # ==========================================
    # LEFT: BEAM SECTION
    # ==========================================
    
    # Beam outline
    fig.add_shape(
        type="rect",
        x0=0, y0=0, x1=b_mm, y1=h_mm,
        line=dict(color="#2c3e50", width=3),
        fillcolor="rgba(189, 195, 199, 0.3)",
        row=1, col=1
    )
    
    # Compression block shading
    if strain_profile:
        a_mm = strain_profile['a_mm']
        fig.add_shape(
            type="rect",
            x0=0, y0=h_mm - a_mm, x1=b_mm, y1=h_mm,
            fillcolor="rgba(231, 76, 60, 0.2)",
            line=dict(color="#e74c3c", width=2, dash="dash"),
            row=1, col=1
        )
        
        # Neutral axis line
        c_mm = strain_profile['c_mm']
        fig.add_shape(
            type="line",
            x0=-b_mm*0.05, y0=h_mm - c_mm,
            x1=b_mm*1.05, y1=h_mm - c_mm,
            line=dict(color="#27ae60", width=3, dash="dash"),
            row=1, col=1
        )
        
        fig.add_annotation(
            x=b_mm * 1.12, y=h_mm - c_mm,
            text="<b>N.A.</b>",
            showarrow=False,
            font=dict(size=11, color="#27ae60", family="Arial Black"),
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
                x_positions = [cover_mm + dia/2, b_mm - cover_mm - dia/2]
            else:
                spacing = (b_mm - 2*cover_mm - dia) / (num - 1)
                x_positions = [cover_mm + dia/2 + i*spacing for i in range(num)]
            
            for x in x_positions:
                fig.add_shape(
                    type="circle",
                    x0=x - dia/2, y0=y - dia/2,
                    x1=x + dia/2, y1=y + dia/2,
                    fillcolor="#8e44ad",
                    line=dict(color="#2c3e50", width=2),
                    row=1, col=1
                )
    
    # Dimension annotations
    fig.add_annotation(
        x=b_mm/2, y=-h_mm*0.08,
        text=f"<b>b = {b_mm:.0f} mm</b>",
        showarrow=False,
        font=dict(size=12, color="#34495e"),
        row=1, col=1
    )
    
    fig.add_annotation(
        x=-b_mm*0.15, y=h_mm/2,
        text=f"<b>h = {h_mm:.0f} mm</b>",
        showarrow=False,
        font=dict(size=12, color="#34495e"),
        textangle=-90,
        row=1, col=1
    )
    
    # ==========================================
    # RIGHT: STRAIN DIAGRAM
    # ==========================================
    
    if strain_profile:
        epsilon_c = strain_profile['epsilon_c']
        c_mm = strain_profile['c_mm']
        epsilon_t = strain_profile['epsilon_t']
        
        # Strain at bottom
        epsilon_bottom = epsilon_c * (c_mm - h_mm) / c_mm
        
        # Main strain profile line
        fig.add_trace(
            go.Scatter(
                x=[epsilon_c, epsilon_bottom],
                y=[h_mm, 0],
                mode='lines',
                line=dict(color='#3498db', width=4),
                name='Strain',
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
                fillcolor='rgba(231, 76, 60, 0.15)',
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
                fillcolor='rgba(52, 152, 219, 0.15)',
                line=dict(width=0),
                showlegend=False,
                hoverinfo='skip'
            ),
            row=1, col=2
        )
        
        # Neutral axis
        x_min = min(epsilon_bottom, 0) * 1.15
        x_max = epsilon_c * 1.15
        fig.add_shape(
            type="line",
            x0=x_min, y0=h_mm - c_mm,
            x1=x_max, y1=h_mm - c_mm,
            line=dict(color="#27ae60", width=3, dash="dash"),
            row=1, col=2
        )
        
        # Strain at bar locations
        for y_pos, eps in strain_profile['bar_strains']:
            fig.add_trace(
                go.Scatter(
                    x=[eps], y=[y_pos],
                    mode='markers',
                    marker=dict(size=10, color='#8e44ad', symbol='circle', line=dict(color='white', width=2)),
                    showlegend=False,
                    hovertemplate=f'<b>Bar Level</b><br>y = {y_pos:.0f} mm<br>ε = {eps:.5f}<extra></extra>'
                ),
                row=1, col=2
            )
        
        # Strain value annotations
        fig.add_annotation(
            x=epsilon_c, y=h_mm,
            text=f"<b>εc = {epsilon_c:.4f}</b>",
            showarrow=True,
            arrowhead=2,
            arrowcolor="#e74c3c",
            ax=-50, ay=-30,
            font=dict(size=11, color="#e74c3c"),
            row=1, col=2
        )
        
        fig.add_annotation(
            x=epsilon_bottom, y=0,
            text=f"<b>ε = {epsilon_bottom:.4f}</b>",
            showarrow=True,
            arrowhead=2,
            arrowcolor="#3498db",
            ax=50, ay=30,
            font=dict(size=11, color="#3498db"),
            row=1, col=2
        )
        
        # Tension limit reference line
        fig.add_shape(
            type="line",
            x0=-0.005, y0=0, x1=-0.005, y1=h_mm,
            line=dict(color="#9b59b6", width=2, dash="dot"),
            row=1, col=2
        )
        
        fig.add_annotation(
            x=-0.005, y=h_mm*0.92,
            text="<b>εt = 0.005</b><br>(Tension limit)",
            showarrow=False,
            font=dict(size=10, color="#9b59b6"),
            row=1, col=2
        )
    
    # Axis configuration
    fig.update_xaxes(
        title_text="<b>Width (mm)</b>",
        row=1, col=1,
        range=[-b_mm*0.25, b_mm*1.3],
        showgrid=True,
        gridcolor='rgba(0,0,0,0.1)',
        gridwidth=1
    )
    
    fig.update_xaxes(
        title_text="<b>Strain (ε)</b>",
        row=1, col=2,
        showgrid=True,
        gridcolor='rgba(0,0,0,0.1)',
        zeroline=True,
        zerolinecolor='black',
        zerolinewidth=2
    )
    
    fig.update_yaxes(
        title_text="<b>Height (mm)</b>",
        row=1, col=1,
        range=[-h_mm*0.15, h_mm*1.12],
        showgrid=True,
        gridcolor='rgba(0,0,0,0.1)'
    )
    
    fig.update_yaxes(
        title_text="<b>Height (mm)</b>",
        row=1, col=2,
        range=[0, h_mm*1.05],
        showgrid=True,
        gridcolor='rgba(0,0,0,0.1)'
    )
    
    fig.update_layout(
        height=550,
        showlegend=False,
        hovermode='closest',
        plot_bgcolor='white',
        font=dict(family="Arial", size=11),
        margin=dict(l=60, r=60, t=60, b=60)
    )
    
    return fig


def plot_section_comparison(results_list):
    """
    Create bar chart comparing three sections
    
    Args:
        results_list: List of result dictionaries
        
    Returns:
        plotly Figure object
    """
    
    sections = [r['Section'] for r in results_list]
    moments = []
    shears = []
    
    for r in results_list:
        try:
            moments.append(float(r['φMn (tonf·m)']))
        except:
            moments.append(0)
        
        try:
            shears.append(float(r['φVn (tonf)']))
        except:
            shears.append(0)
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('<b>Bending Capacity</b>', '<b>Shear Capacity</b>'),
        horizontal_spacing=0.15
    )
    
    # Moment bars
    fig.add_trace(
        go.Bar(
            x=sections,
            y=moments,
            name='φMn',
            marker=dict(
                color='#3498db',
                line=dict(color='#2c3e50', width=2)
            ),
            text=[f'{m:.2f}' for m in moments],
            textposition='outside',
            textfont=dict(size=13, color='#2c3e50', family='Arial Black')
        ),
        row=1, col=1
    )
    
    # Shear bars
    fig.add_trace(
        go.Bar(
            x=sections,
            y=shears,
            name='φVn',
            marker=dict(
                color='#e74c3c',
                line=dict(color='#2c3e50', width=2)
            ),
            text=[f'{s:.2f}' for s in shears],
            textposition='outside',
            textfont=dict(size=13, color='#2c3e50', family='Arial Black')
        ),
        row=1, col=2
    )
    
    fig.update_xaxes(title_text="<b>Section</b>", row=1, col=1)
    fig.update_xaxes(title_text="<b>Section</b>", row=1, col=2)
    fig.update_yaxes(title_text="<b>Moment (tonf·m)</b>", row=1, col=1)
    fig.update_yaxes(title_text="<b>Shear (tonf)</b>", row=1, col=2)
    
    fig.update_layout(
        height=450,
        showlegend=False,
        plot_bgcolor='white',
        font=dict(family="Arial", size=11),
        hovermode='x unified'
    )
    
    return fig
