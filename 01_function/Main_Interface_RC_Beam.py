import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle, Circle, FancyBboxPatch
from datetime import datetime
import io
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY

# ===========================
# PAGE CONFIGURATION
# ===========================
st.set_page_config(
    page_title="RC Beam Designer & Analyzer - ACI 318",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===========================
# CUSTOM CSS STYLING
# ===========================
st.markdown("""
    <style>
    /* Main container styling */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    /* Header styling */
    .main-header {
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        box-shadow: 0 20px 40px rgba(0,0,0,0.15);
    }
    
    .main-header h1 {
        font-size: 2.8em;
        margin: 0;
        font-weight: 800;
        text-shadow: 3px 3px 6px rgba(0,0,0,0.3);
    }
    
    .main-header p {
        font-size: 1.2em;
        margin-top: 0.5rem;
        opacity: 0.95;
        font-weight: 300;
    }
    
    /* Mode selector card */
    .mode-selector {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        margin-bottom: 2rem;
        border: 2px solid #e2e8f0;
    }
    
    /* Result cards */
    .result-card {
        background: white;
        padding: 1.8rem;
        border-radius: 15px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        margin-bottom: 1.5rem;
        border-left: 5px solid #667eea;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .result-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 35px rgba(0,0,0,0.15);
    }
    
    /* Status boxes */
    .status-adequate {
        background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
        color: #0d5f3a;
        padding: 1.2rem;
        border-radius: 12px;
        text-align: center;
        font-weight: 700;
        font-size: 1.2em;
        margin: 1rem 0;
        box-shadow: 0 6px 20px rgba(132,250,176,0.4);
        border: 2px solid #5dd39e;
    }
    
    .status-inadequate {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        color: #8b0000;
        padding: 1.2rem;
        border-radius: 12px;
        text-align: center;
        font-weight: 700;
        font-size: 1.2em;
        margin: 1rem 0;
        box-shadow: 0 6px 20px rgba(250,112,154,0.4);
        border: 2px solid #ff6b9d;
    }
    
    /* Metric displays */
    .metric-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.2rem;
        border-radius: 12px;
        text-align: center;
        margin: 0.8rem 0;
        color: white;
        box-shadow: 0 6px 20px rgba(102,126,234,0.3);
        transition: transform 0.3s ease;
    }
    
    .metric-box:hover {
        transform: scale(1.05);
    }
    
    .metric-label {
        color: rgba(255,255,255,0.9);
        font-size: 0.95em;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .metric-value {
        color: white;
        font-size: 2.2em;
        font-weight: 800;
        margin: 0.3rem 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    .metric-unit {
        color: rgba(255,255,255,0.85);
        font-size: 1em;
        font-weight: 400;
    }
    
    /* Control type indicator */
    .control-type-tension {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 12px;
        text-align: center;
        font-weight: 700;
        margin: 1rem 0;
        box-shadow: 0 6px 20px rgba(102,126,234,0.3);
    }
    
    .control-type-transition {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1rem;
        border-radius: 12px;
        text-align: center;
        font-weight: 700;
        margin: 1rem 0;
        box-shadow: 0 6px 20px rgba(240,147,251,0.3);
    }
    
    .control-type-compression {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        color: #7a0025;
        padding: 1rem;
        border-radius: 12px;
        text-align: center;
        font-weight: 700;
        margin: 1rem 0;
        box-shadow: 0 6px 20px rgba(250,112,154,0.3);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: white;
        padding: 0.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        font-weight: 600;
        padding: 0.5rem 1.5rem;
        background-color: #f8fafc;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    /* Equation display */
    .equation-box {
        background: #f8fafc;
        border: 2px solid #e2e8f0;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        font-family: 'Courier New', monospace;
        font-size: 1.1em;
    }
    
    /* Info boxes */
    .info-box {
        background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%);
        border-left: 4px solid #667eea;
        padding: 1.2rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .warning-box {
        background: #fef3c7;
        border-left: 4px solid #f59e0b;
        padding: 1.2rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background: white;
        padding: 1.2rem;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.8rem 2rem;
        font-size: 1.1em;
        font-weight: 700;
        border-radius: 12px;
        box-shadow: 0 6px 20px rgba(102,126,234,0.3);
        transition: all 0.3s ease;
        width: 100%;
        margin-top: 1rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(102,126,234,0.4);
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        border-radius: 10px;
        font-weight: 600;
    }
    </style>
    """, unsafe_allow_html=True)

# ===========================
# HELPER FUNCTIONS - ACI 318 CALCULATIONS
# ===========================

def calculate_beta1(fc_ksc):
    """Calculate β₁ factor based on concrete strength per ACI 318"""
    fc_mpa = fc_ksc * 0.0980665  # Convert ksc to MPa
    if fc_mpa <= 28:
        return 0.85
    elif fc_mpa >= 55:
        return 0.65
    else:
        return 0.85 - 0.05 * (fc_mpa - 28) / 7

def calculate_phi_factor(epsilon_t):
    """Calculate strength reduction factor φ based on strain per ACI 318"""
    epsilon_ty = 0.002  # Yield strain for Grade 420 steel (typical)
    
    if epsilon_t >= 0.005:
        # Tension-controlled
        return 0.90, "Tension-Controlled"
    elif epsilon_t <= epsilon_ty:
        # Compression-controlled
        return 0.65, "Compression-Controlled"
    else:
        # Transition zone
        phi = 0.65 + (epsilon_t - epsilon_ty) * (0.25 / (0.005 - epsilon_ty))
        return round(phi, 3), "Transition Zone"

def strength_analysis(fc_ksc, fy_ksc, b_mm, h_mm, cover_mm, As_bot_mm2, As_top_mm2=0):
    """
    Perform strength analysis of existing RC beam section
    Returns nominal moment capacity and strain analysis
    """
    # Calculate effective depth
    d_mm = h_mm - cover_mm - 20  # Assume 20mm to centroid of bottom steel
    d_prime_mm = cover_mm + 20  # Distance to top steel centroid
    
    # Unit conversions
    fc_N_mm2 = fc_ksc * 0.0980665  # ksc to N/mm² (MPa)
    fy_N_mm2 = fy_ksc * 0.0980665  # ksc to N/mm² (MPa)
    
    # ACI 318 parameters
    beta1 = calculate_beta1(fc_ksc)
    
    # For singly reinforced (ignore top steel for now)
    if As_top_mm2 == 0:
        # Neutral axis depth from force equilibrium
        c_mm = As_bot_mm2 * fy_N_mm2 / (0.85 * fc_N_mm2 * beta1 * b_mm)
        
        # Strain analysis
        epsilon_c = 0.003  # Concrete crushing strain
        epsilon_t = epsilon_c * (d_mm - c_mm) / c_mm
        
        # Get phi factor and control type
        phi, control_type = calculate_phi_factor(epsilon_t)
        
        # Compression block depth
        a_mm = beta1 * c_mm
        
        # Nominal moment capacity
        Mn_Nmm = As_bot_mm2 * fy_N_mm2 * (d_mm - a_mm/2)
        Mn_tonfm = Mn_Nmm / 9.80665e6  # Convert to tonf·m
        
    else:
        # Doubly reinforced beam
        # Assume compression steel yields (simplified)
        c_mm = (As_bot_mm2 - As_top_mm2) * fy_N_mm2 / (0.85 * fc_N_mm2 * beta1 * b_mm)
        
        # Check if compression steel yields
        epsilon_c = 0.003
        epsilon_s_prime = epsilon_c * (c_mm - d_prime_mm) / c_mm
        
        if epsilon_s_prime * 200000 > fy_N_mm2:  # Es = 200 GPa
            fs_prime = fy_N_mm2
        else:
            fs_prime = epsilon_s_prime * 200000
        
        # Strain in tension steel
        epsilon_t = epsilon_c * (d_mm - c_mm) / c_mm
        
        # Get phi factor and control type
        phi, control_type = calculate_phi_factor(epsilon_t)
        
        # Compression block depth
        a_mm = beta1 * c_mm
        
        # Nominal moment capacity
        Mn1_Nmm = (As_bot_mm2 - As_top_mm2) * fy_N_mm2 * (d_mm - a_mm/2)
        Mn2_Nmm = As_top_mm2 * (fs_prime - 0.85 * fc_N_mm2) * (d_mm - d_prime_mm)
        Mn_Nmm = Mn1_Nmm + Mn2_Nmm
        Mn_tonfm = Mn_Nmm / 9.80665e6
    
    return {
        'Mn_tonfm': Mn_tonfm,
        'phi': phi,
        'phi_Mn_tonfm': phi * Mn_tonfm,
        'c_mm': c_mm,
        'a_mm': a_mm,
        'epsilon_c': epsilon_c,
        'epsilon_t': epsilon_t,
        'control_type': control_type,
        'd_mm': d_mm,
        'd_prime_mm': d_prime_mm if As_top_mm2 > 0 else None
    }

def design_beam(fc_ksc, fy_ksc, b_mm, h_mm, cover_mm, Mu_tonfm):
    """
    Design RC beam for given moment
    Returns required steel area and checks
    """
    # Calculate effective depth
    d_mm = h_mm - cover_mm - 25  # Assume 25mm to centroid
    
    # Unit conversions
    fc_N_mm2 = fc_ksc * 0.0980665
    fy_N_mm2 = fy_ksc * 0.0980665
    Mu_Nmm = Mu_tonfm * 9.80665e6
    
    # ACI parameters
    phi = 0.9  # Initial assumption for design
    beta1 = calculate_beta1(fc_ksc)
    
    # Required nominal moment
    Mn_req_Nmm = Mu_Nmm / phi
    
    # Quadratic equation for As
    A = fy_N_mm2**2 / (1.7 * fc_N_mm2 * b_mm)
    B = -fy_N_mm2 * d_mm
    C = Mn_req_Nmm
    
    discriminant = B**2 - 4*A*C
    
    if discriminant < 0:
        return {
            'success': False,
            'message': 'Section inadequate - increase dimensions',
            'As_req_mm2': None
        }
    
    # Solve for As
    As_mm2 = (-B - np.sqrt(discriminant)) / (2*A)
    
    # Check minimum reinforcement
    As_min_mm2 = max(
        3 * np.sqrt(fc_N_mm2) / fy_N_mm2 * b_mm * d_mm,
        200 / fy_N_mm2 * b_mm * d_mm
    )
    
    As_req_mm2 = max(As_mm2, As_min_mm2)
    
    # Check maximum reinforcement (0.75 * balanced)
    rho_b = 0.85 * beta1 * fc_N_mm2 / fy_N_mm2 * 600 / (600 + fy_N_mm2)
    rho_max = 0.75 * rho_b
    As_max_mm2 = rho_max * b_mm * d_mm
    
    if As_req_mm2 > As_max_mm2:
        return {
            'success': False,
            'message': 'Over-reinforced - increase section or use compression steel',
            'As_req_mm2': As_req_mm2,
            'As_max_mm2': As_max_mm2
        }
    
    # Calculate actual values for verification
    a_mm = As_req_mm2 * fy_N_mm2 / (0.85 * fc_N_mm2 * b_mm)
    c_mm = a_mm / beta1
    epsilon_t = 0.003 * (d_mm - c_mm) / c_mm
    
    phi_actual, control_type = calculate_phi_factor(epsilon_t)
    
    # Recalculate moment capacity with actual phi
    Mn_Nmm = As_req_mm2 * fy_N_mm2 * (d_mm - a_mm/2)
    Mn_tonfm = Mn_Nmm / 9.80665e6
    phi_Mn_tonfm = phi_actual * Mn_tonfm
    
    return {
        'success': True,
        'As_req_mm2': As_req_mm2,
        'As_min_mm2': As_min_mm2,
        'As_max_mm2': As_max_mm2,
        'd_mm': d_mm,
        'a_mm': a_mm,
        'c_mm': c_mm,
        'epsilon_t': epsilon_t,
        'control_type': control_type,
        'Mn_tonfm': Mn_tonfm,
        'phi': phi_actual,
        'phi_Mn_tonfm': phi_Mn_tonfm,
        'rho': As_req_mm2 / (b_mm * d_mm),
        'rho_b': rho_b,
        'rho_max': rho_max,
        'beta1': beta1
    }

def calculate_bar_arrangement(As_required_mm2, bar_dia_mm):
    """Calculate number of bars needed"""
    bar_area_mm2 = np.pi * bar_dia_mm**2 / 4
    num_bars = int(np.ceil(As_required_mm2 / bar_area_mm2))
    As_provided_mm2 = num_bars * bar_area_mm2
    
    return {
        'num_bars': num_bars,
        'As_provided_mm2': As_provided_mm2,
        'bar_area_mm2': bar_area_mm2,
        'excess_percent': (As_provided_mm2 - As_required_mm2) / As_required_mm2 * 100
    }

# ===========================
# VISUALIZATION FUNCTIONS
# ===========================

def create_strain_diagram_plotly(results, h_mm):
    """Create interactive strain diagram using Plotly"""
    epsilon_c = results.get('epsilon_c', 0.003)
    epsilon_t = results.get('epsilon_t', 0)
    c_mm = results.get('c_mm', 0)
    d_mm = results.get('d_mm', h_mm * 0.9)
    control_type = results.get('control_type', 'Unknown')
    phi = results.get('phi', 0.9)
    
    fig = go.Figure()
    
    # Define strain profile points
    strain_values = [epsilon_c, 0, -epsilon_t]
    height_values = [h_mm, h_mm - c_mm, 0]
    
    # Add compression zone (red area)
    fig.add_trace(go.Scatter(
        x=[0, epsilon_c, epsilon_c, 0, 0],
        y=[h_mm - c_mm, h_mm, h_mm, h_mm - c_mm, h_mm - c_mm],
        fill='toself',
        fillcolor='rgba(255, 99, 71, 0.3)',
        line=dict(color='rgba(255, 99, 71, 0.5)', width=2),
        name='Compression Zone',
        showlegend=True,
        hovertemplate='Compression Zone<br>εc = 0.003'
    ))
    
    # Add tension zone (blue area)
    fig.add_trace(go.Scatter(
        x=[0, -epsilon_t, -epsilon_t, 0, 0],
        y=[h_mm - c_mm, 0, 0, h_mm - c_mm, h_mm - c_mm],
        fill='toself',
        fillcolor='rgba(30, 144, 255, 0.3)',
        line=dict(color='rgba(30, 144, 255, 0.5)', width=2),
        name='Tension Zone',
        showlegend=True,
        hovertemplate=f'Tension Zone<br>εt = {epsilon_t:.5f}'
    ))
    
    # Add strain profile line
    fig.add_trace(go.Scatter(
        x=[epsilon_c, -epsilon_t],
        y=[h_mm, 0],
        mode='lines',
        line=dict(color='black', width=3),
        name='Strain Profile',
        showlegend=True
    ))
    
    # Add neutral axis
    fig.add_trace(go.Scatter(
        x=[-epsilon_t * 0.2, epsilon_c * 0.2],
        y=[h_mm - c_mm, h_mm - c_mm],
        mode='lines',
        line=dict(color='green', width=3, dash='dash'),
        name='Neutral Axis',
        showlegend=True,
        hovertemplate=f'N.A. at {c_mm:.1f} mm from top'
    ))
    
    # Add critical points
    fig.add_trace(go.Scatter(
        x=[epsilon_c, 0, -epsilon_t],
        y=[h_mm, h_mm - c_mm, 0],
        mode='markers+text',
        marker=dict(size=10, color=['red', 'green', 'blue']),
        text=[f'εc = {epsilon_c:.3f}', 'N.A.', f'εt = {epsilon_t:.5f}'],
        textposition=['top right', 'middle right', 'bottom right'],
        textfont=dict(size=12, color='black'),
        showlegend=False
    ))
    
    # Add effective depth marker
    fig.add_trace(go.Scatter(
        x=[-epsilon_t * 0.95],
        y=[h_mm - d_mm],
        mode='markers+text',
        marker=dict(size=8, color='purple', symbol='diamond'),
        text=[f'd = {d_mm:.0f} mm'],
        textposition='middle left',
        showlegend=False
    ))
    
    # Color-code background based on control type
    bg_color = {
        'Tension-Controlled': 'rgba(102, 126, 234, 0.05)',
        'Transition Zone': 'rgba(240, 147, 251, 0.05)',
        'Compression-Controlled': 'rgba(250, 112, 154, 0.05)'
    }.get(control_type, 'rgba(200, 200, 200, 0.05)')
    
    # Update layout
    fig.update_layout(
        title={
            'text': f'Strain Distribution Diagram<br><sub>{control_type} Section | φ = {phi:.3f}</sub>',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'color': '#1e293b'}
        },
        xaxis=dict(
            title='Strain (ε)',
            showgrid=True,
            gridcolor='lightgray',
            zeroline=True,
            zerolinecolor='gray',
            zerolinewidth=2,
            range=[-max(epsilon_t, 0.006) * 1.1, epsilon_c * 1.2]
        ),
        yaxis=dict(
            title='Height (mm)',
            showgrid=True,
            gridcolor='lightgray',
            range=[-h_mm * 0.05, h_mm * 1.05]
        ),
        hovermode='closest',
        plot_bgcolor=bg_color,
        paper_bgcolor='white',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5
        ),
        height=500
    )
    
    # Add annotations for ACI limits
    fig.add_annotation(
        x=-0.005, y=h_mm * 0.9,
        text="εt = 0.005<br>(Tension limit)",
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=1,
        arrowcolor="blue",
        ax=-40, ay=-30,
        bordercolor="blue",
        borderwidth=1,
        bgcolor="rgba(255,255,255,0.9)",
        font=dict(size=10)
    )
    
    return fig

def draw_beam_section(b_mm, h_mm, cover_mm, As_bot_mm2, As_top_mm2, bar_dia_mm):
    """Create beam cross-section diagram"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Set up the plot
    ax.set_xlim(-b_mm*0.2, b_mm*1.2)
    ax.set_ylim(-h_mm*0.2, h_mm*1.2)
    ax.set_aspect('equal')
    
    # Draw beam outline with gradient effect
    beam = FancyBboxPatch((0, 0), b_mm, h_mm,
                          boxstyle="round,pad=5",
                          linewidth=3, edgecolor='#1e293b',
                          facecolor='#e5e7eb')
    ax.add_patch(beam)
    
    # Draw compression zone if available
    if 'a_mm' in st.session_state and st.session_state.a_mm > 0:
        comp_zone = Rectangle((0, h_mm - st.session_state.a_mm), b_mm, st.session_state.a_mm,
                             linewidth=0, facecolor='#ff6b6b', alpha=0.3)
        ax.add_patch(comp_zone)
    
    # Calculate bar positions
    stirrup_dia = 10  # Assume 10mm stirrups
    
    # Bottom reinforcement
    if As_bot_mm2 > 0:
        bar_info = calculate_bar_arrangement(As_bot_mm2, bar_dia_mm)
        num_bars = bar_info['num_bars']
        
        if num_bars > 1:
            spacing = (b_mm - 2*cover_mm - 2*stirrup_dia - num_bars*bar_dia_mm) / (num_bars - 1)
            bar_positions = [cover_mm + stirrup_dia + bar_dia_mm/2 + i*(bar_dia_mm + spacing) 
                           for i in range(num_bars)]
        else:
            bar_positions = [b_mm/2]
        
        bar_y = cover_mm + stirrup_dia + bar_dia_mm/2
        
        for x in bar_positions:
            circle = Circle((x, bar_y), bar_dia_mm/2,
                          color='#dc2626', linewidth=2,
                          edgecolor='#7f1d1d', zorder=5)
            ax.add_patch(circle)
        
        # Add label for bottom steel
        ax.text(b_mm/2, bar_y - bar_dia_mm - 10,
               f'{num_bars}Ø{bar_dia_mm}\nAs = {As_bot_mm2:.0f} mm²',
               ha='center', va='top', fontsize=11, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='#fee2e2',
                        edgecolor='#dc2626', linewidth=2))
    
    # Top reinforcement (if present)
    if As_top_mm2 > 0:
        bar_info_top = calculate_bar_arrangement(As_top_mm2, bar_dia_mm)
        num_bars_top = bar_info_top['num_bars']
        
        if num_bars_top > 1:
            spacing = (b_mm - 2*cover_mm - 2*stirrup_dia - num_bars_top*bar_dia_mm) / (num_bars_top - 1)
            bar_positions = [cover_mm + stirrup_dia + bar_dia_mm/2 + i*(bar_dia_mm + spacing) 
                           for i in range(num_bars_top)]
        else:
            bar_positions = [b_mm/2]
        
        bar_y = h_mm - cover_mm - stirrup_dia - bar_dia_mm/2
        
        for x in bar_positions:
            circle = Circle((x, bar_y), bar_dia_mm/2,
                          color='#2563eb', linewidth=2,
                          edgecolor='#1e3a8a', zorder=5)
            ax.add_patch(circle)
        
        # Add label for top steel
        ax.text(b_mm/2, bar_y + bar_dia_mm + 10,
               f"{num_bars_top}Ø{bar_dia_mm}\nA's = {As_top_mm2:.0f} mm²",
               ha='center', va='bottom', fontsize=11, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='#dbeafe',
                        edgecolor='#2563eb', linewidth=2))
    
    # Draw stirrups
    stirrup_x = [cover_mm + stirrup_dia/2,
                b_mm - cover_mm - stirrup_dia/2,
                b_mm - cover_mm - stirrup_dia/2,
                cover_mm + stirrup_dia/2,
                cover_mm + stirrup_dia/2]
    stirrup_y = [cover_mm + stirrup_dia/2,
                cover_mm + stirrup_dia/2,
                h_mm - cover_mm - stirrup_dia/2,
                h_mm - cover_mm - stirrup_dia/2,
                cover_mm + stirrup_dia/2]
    
    ax.plot(stirrup_x, stirrup_y, color='#059669', linewidth=2.5, linestyle='-')
    
    # Add dimensions
    # Width dimension
    ax.annotate('', xy=(0, -h_mm*0.1), xytext=(b_mm, -h_mm*0.1),
               arrowprops=dict(arrowstyle='<->', color='#64748b', lw=2))
    ax.text(b_mm/2, -h_mm*0.15, f'{b_mm} mm',
           ha='center', va='top', fontsize=12, color='#334155', weight='bold')
    
    # Height dimension
    ax.annotate('', xy=(b_mm*1.1, 0), xytext=(b_mm*1.1, h_mm),
               arrowprops=dict(arrowstyle='<->', color='#64748b', lw=2))
    ax.text(b_mm*1.15, h_mm/2, f'{h_mm} mm',
           ha='left', va='center', fontsize=12, color='#334155', rotation=90, weight='bold')
    
    # Add title
    ax.text(b_mm/2, h_mm*1.15, 'Beam Cross-Section',
           ha='center', fontsize=18, fontweight='bold', color='#1e293b')
    
    ax.axis('off')
    plt.tight_layout()
    
    return fig

def generate_pdf_report(mode, inputs, results):
    """Generate comprehensive PDF report"""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=0.5*inch, bottomMargin=0.5*inch)
    story = []
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=26,
        textColor=colors.HexColor('#1e293b'),
        spaceAfter=30,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.HexColor('#667eea'),
        spaceBefore=20,
        spaceAfter=15,
        fontName='Helvetica-Bold'
    )
    
    # Title
    story.append(Paragraph(f"RC Beam {mode} Report", title_style))
    story.append(Paragraph(f"ACI 318 Compliant Design", styles['Heading3']))
    story.append(Spacer(1, 0.3*inch))
    
    # Date and project info
    story.append(Paragraph(f"<b>Report Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles['Normal']))
    story.append(Paragraph(f"<b>Analysis Mode:</b> {mode}", styles['Normal']))
    story.append(Paragraph(f"<b>Design Code:</b> ACI 318-19", styles['Normal']))
    story.append(Spacer(1, 0.3*inch))
    
    # Input parameters
    story.append(Paragraph("1. INPUT PARAMETERS", heading_style))
    
    input_data = [
        ['Parameter', 'Value', 'Unit'],
        ['Concrete Strength (fc\')', f"{inputs.get('fc_ksc', 0):.0f}", 'ksc'],
        ['Steel Yield Strength (fy)', f"{inputs.get('fy_ksc', 0):.0f}", 'ksc'],
        ['Beam Width (b)', f"{inputs.get('b_mm', 0):.0f}", 'mm'],
        ['Beam Height (h)', f"{inputs.get('h_mm', 0):.0f}", 'mm'],
        ['Concrete Cover', f"{inputs.get('cover_mm', 0):.0f}", 'mm']
    ]
    
    if mode == "Design Mode":
        input_data.append(['Applied Moment (Mu)', f"{inputs.get('Mu_tonfm', 0):.2f}", 'tonf·m'])
    else:
        input_data.append(['Bottom Steel Area (As)', f"{inputs.get('As_bot_mm2', 0):.0f}", 'mm²'])
        if inputs.get('As_top_mm2', 0) > 0:
            input_data.append(['Top Steel Area (A\'s)', f"{inputs.get('As_top_mm2', 0):.0f}", 'mm²'])
    
    t = Table(input_data, colWidths=[2.5*inch, 1.5*inch, 1*inch])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#667eea')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(t)
    story.append(Spacer(1, 0.3*inch))
    
    # Analysis results
    story.append(Paragraph("2. ANALYSIS RESULTS", heading_style))
    
    results_data = [
        ['Parameter', 'Value', 'Unit']
    ]
    
    if mode == "Design Mode":
        results_data.extend([
            ['Effective Depth (d)', f"{results.get('d_mm', 0):.0f}", 'mm'],
            ['Required Steel Area (As,req)', f"{results.get('As_req_mm2', 0):.0f}", 'mm²'],
            ['Minimum Steel Area (As,min)', f"{results.get('As_min_mm2', 0):.0f}", 'mm²'],
            ['Maximum Steel Area (As,max)', f"{results.get('As_max_mm2', 0):.0f}", 'mm²'],
            ['Neutral Axis Depth (c)', f"{results.get('c_mm', 0):.1f}", 'mm'],
            ['Compression Block (a)', f"{results.get('a_mm', 0):.1f}", 'mm'],
            ['Control Type', results.get('control_type', 'N/A'), '-'],
            ['Tensile Strain (εt)', f"{results.get('epsilon_t', 0):.5f}", '-'],
            ['Phi Factor (φ)', f"{results.get('phi', 0):.3f}", '-'],
            ['Nominal Moment (Mn)', f"{results.get('Mn_tonfm', 0):.2f}", 'tonf·m'],
            ['Design Moment (φMn)', f"{results.get('phi_Mn_tonfm', 0):.2f}", 'tonf·m']
        ])
    else:
        results_data.extend([
            ['Effective Depth (d)', f"{results.get('d_mm', 0):.0f}", 'mm'],
            ['Neutral Axis Depth (c)', f"{results.get('c_mm', 0):.1f}", 'mm'],
            ['Compression Block (a)', f"{results.get('a_mm', 0):.1f}", 'mm'],
            ['Concrete Strain (εc)', f"{results.get('epsilon_c', 0):.3f}", '-'],
            ['Tensile Strain (εt)', f"{results.get('epsilon_t', 0):.5f}", '-'],
            ['Control Type', results.get('control_type', 'N/A'), '-'],
            ['Phi Factor (φ)', f"{results.get('phi', 0):.3f}", '-'],
            ['Nominal Moment (Mn)', f"{results.get('Mn_tonfm', 0):.2f}", 'tonf·m'],
            ['Design Strength (φMn)', f"{results.get('phi_Mn_tonfm', 0):.2f}", 'tonf·m']
        ])
    
    t2 = Table(results_data, colWidths=[2.5*inch, 1.5*inch, 1*inch])
    t2.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#667eea')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(t2)
    story.append(Spacer(1, 0.3*inch))
    
    # Key equations
    story.append(Paragraph("3. KEY EQUATIONS USED", heading_style))
    
    equations = [
        "Nominal Moment: Mn = As × fy × (d - a/2)",
        "Compression Block: a = As × fy / (0.85 × fc' × b)",
        "Neutral Axis: c = a / β₁",
        "Tensile Strain: εt = 0.003 × (d - c) / c",
        "Phi Factor: φ = 0.65 to 0.90 (based on εt per ACI 318)"
    ]
    
    for eq in equations:
        story.append(Paragraph(f"• {eq}", styles['Normal']))
    
    story.append(Spacer(1, 0.3*inch))
    
    # Design status/recommendations
    story.append(Paragraph("4. CONCLUSIONS", heading_style))
    
    control_type = results.get('control_type', 'Unknown')
    if control_type == 'Tension-Controlled':
        status_text = "✓ Section is TENSION-CONTROLLED (εt ≥ 0.005)"
        color = colors.HexColor('#059669')
    elif control_type == 'Transition Zone':
        status_text = "⚠ Section is in TRANSITION ZONE (0.002 < εt < 0.005)"
        color = colors.HexColor('#f59e0b')
    else:
        status_text = "✗ Section is COMPRESSION-CONTROLLED (εt ≤ 0.002)"
        color = colors.HexColor('#dc2626')
    
    status_style = ParagraphStyle(
        'StatusStyle',
        parent=styles['Normal'],
        fontSize=14,
        textColor=color,
        alignment=TA_CENTER,
        spaceAfter=20,
        spaceBefore=20,
        fontName='Helvetica-Bold'
    )
    
    story.append(Paragraph(status_text, status_style))
    
    # Build PDF
    doc.build(story)
    buffer.seek(0)
    
    return buffer

# ===========================
# SIDEBAR - INPUT PARAMETERS
# ===========================

st.sidebar.markdown("## 📋 Input Parameters")

# Mode Selection
mode = st.sidebar.radio(
    "🎯 **Select Analysis Mode**",
    ["Design Mode", "Strength Analysis Mode"],
    help="Design Mode: Calculate required steel for given moment\nStrength Analysis: Calculate capacity of existing section"
)

st.sidebar.markdown("---")

# Material Properties
st.sidebar.markdown("### 🏗️ Material Properties")
col1, col2 = st.sidebar.columns(2)
with col1:
    fc_ksc = st.number_input(
        "fc' (ksc)",
        min_value=200.0,
        max_value=800.0,
        value=280.0,
        step=10.0,
        help="Concrete compressive strength in kg/cm²"
    )
with col2:
    fy_ksc = st.number_input(
        "fy (ksc)",
        min_value=2000.0,
        max_value=6000.0,
        value=4200.0,
        step=100.0,
        help="Steel yield strength in kg/cm²"
    )

# Beam Geometry
st.sidebar.markdown("### 📐 Beam Geometry")
col3, col4 = st.sidebar.columns(2)
with col3:
    b_mm = st.number_input(
        "Width b (mm)",
        min_value=150.0,
        max_value=1000.0,
        value=300.0,
        step=50.0
    )
with col4:
    h_mm = st.number_input(
        "Height h (mm)",
        min_value=200.0,
        max_value=1500.0,
        value=600.0,
        step=50.0
    )

cover_mm = st.sidebar.number_input(
    "Concrete Cover (mm)",
    min_value=20.0,
    max_value=75.0,
    value=40.0,
    step=5.0,
    help="Clear cover to reinforcement"
)

st.sidebar.markdown("---")

# Mode-specific inputs
if mode == "Design Mode":
    st.sidebar.markdown("### ⚡ Loading")
    Mu_tonfm = st.sidebar.number_input(
        "Factored Moment Mu (tonf·m)",
        min_value=0.0,
        max_value=500.0,
        value=50.0,
        step=5.0,
        help="Ultimate design moment"
    )
    
    st.sidebar.markdown("### 🔧 Reinforcement Details")
    bar_dia_mm = st.sidebar.selectbox(
        "Bar Diameter (mm)",
        options=[10, 12, 16, 20, 25, 28, 32],
        index=3,
        help="Select main reinforcement bar diameter"
    )
    
else:  # Strength Analysis Mode
    st.sidebar.markdown("### 🔧 Existing Reinforcement")
    As_bot_mm2 = st.sidebar.number_input(
        "Bottom Steel Area As (mm²)",
        min_value=0.0,
        max_value=10000.0,
        value=1000.0,
        step=100.0,
        help="Total area of bottom (tension) reinforcement"
    )
    
    use_top_steel = st.sidebar.checkbox("Include Top Steel (Doubly Reinforced)")
    
    if use_top_steel:
        As_top_mm2 = st.sidebar.number_input(
            "Top Steel Area A's (mm²)",
            min_value=0.0,
            max_value=10000.0,
            value=500.0,
            step=100.0,
            help="Total area of top (compression) reinforcement"
        )
    else:
        As_top_mm2 = 0
    
    bar_dia_mm = st.sidebar.selectbox(
        "Bar Diameter (mm)",
        options=[10, 12, 16, 20, 25, 28, 32],
        index=3,
        help="For visualization purposes"
    )

# Analysis Button
analyze_button = st.sidebar.button(
    f"🚀 {'DESIGN BEAM' if mode == 'Design Mode' else 'ANALYZE STRENGTH'}",
    use_container_width=True
)

# ===========================
# MAIN CONTENT - HEADER
# ===========================

st.markdown("""
    <div class="main-header">
        <h1>🏗️ RC Beam Designer & Analyzer</h1>
        <p>Advanced Reinforced Concrete Analysis per ACI 318-19</p>
    </div>
    """, unsafe_allow_html=True)

# Mode indicator
mode_color = "#667eea" if mode == "Design Mode" else "#f59e0b"
st.markdown(f"""
    <div class="mode-selector">
        <h2 style="color: {mode_color}; text-align: center; margin: 0;">
            {'📐 DESIGN MODE' if mode == 'Design Mode' else '💪 STRENGTH ANALYSIS MODE'}
        </h2>
    </div>
    """, unsafe_allow_html=True)

# ===========================
# MAIN CONTENT - ANALYSIS
# ===========================

if analyze_button:
    # Store inputs in session state
    if mode == "Design Mode":
        inputs = {
            'fc_ksc': fc_ksc,
            'fy_ksc': fy_ksc,
            'b_mm': b_mm,
            'h_mm': h_mm,
            'cover_mm': cover_mm,
            'Mu_tonfm': Mu_tonfm,
            'bar_dia_mm': bar_dia_mm
        }
        
        # Perform design
        with st.spinner('Calculating required reinforcement...'):
            results = design_beam(fc_ksc, fy_ksc, b_mm, h_mm, cover_mm, Mu_tonfm)
        
    else:  # Strength Analysis Mode
        inputs = {
            'fc_ksc': fc_ksc,
            'fy_ksc': fy_ksc,
            'b_mm': b_mm,
            'h_mm': h_mm,
            'cover_mm': cover_mm,
            'As_bot_mm2': As_bot_mm2,
            'As_top_mm2': As_top_mm2,
            'bar_dia_mm': bar_dia_mm
        }
        
        # Perform strength analysis
        with st.spinner('Analyzing section capacity...'):
            results = strength_analysis(fc_ksc, fy_ksc, b_mm, h_mm, cover_mm, As_bot_mm2, As_top_mm2)
    
    # Store results in session state
    for key, value in results.items():
        st.session_state[key] = value
    
    # Create tabs for organized display
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Results", "📈 Strain Diagram", "🏗️ Section View", "📄 Export"])
    
    # ===========================
    # TAB 1: RESULTS
    # ===========================
    with tab1:
        if mode == "Design Mode" and results.get('success'):
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("### 📐 Design Results")
                
                # Design adequacy status
                if results['phi_Mn_tonfm'] >= Mu_tonfm:
                    st.markdown("""<div class="status-adequate">
                        ✅ DESIGN IS ADEQUATE<br>
                        φMn ≥ Mu ✓
                    </div>""", unsafe_allow_html=True)
                else:
                    st.markdown("""<div class="status-inadequate">
                        ❌ DESIGN NEEDS REVISION<br>
                        φMn < Mu - Check inputs
                    </div>""", unsafe_allow_html=True)
                
                # Key metrics
                st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-label">Required Steel Area</div>
                    <div class="metric-value">{results['As_req_mm2']:.0f}</div>
                    <div class="metric-unit">mm²</div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-label">Design Strength</div>
                    <div class="metric-value">{results['phi_Mn_tonfm']:.2f}</div>
                    <div class="metric-unit">tonf·m</div>
                </div>
                """, unsafe_allow_html=True)
                
                # Control type indicator
                control_class = {
                    'Tension-Controlled': 'control-type-tension',
                    'Transition Zone': 'control-type-transition',
                    'Compression-Controlled': 'control-type-compression'
                }.get(results['control_type'], 'control-type-tension')
                
                st.markdown(f"""<div class="{control_class}">
                    {results['control_type']}<br>
                    φ = {results['phi']:.3f}
                </div>""", unsafe_allow_html=True)
                
            with col2:
                st.markdown("### 📋 Detailed Parameters")
                
                # Create detailed results table
                detailed_df = pd.DataFrame([
                    ["Effective depth (d)", f"{results['d_mm']:.0f} mm"],
                    ["Neutral axis (c)", f"{results['c_mm']:.1f} mm"],
                    ["Compression block (a)", f"{results['a_mm']:.1f} mm"],
                    ["Tensile strain (εt)", f"{results['epsilon_t']:.5f}"],
                    ["Reinforcement ratio (ρ)", f"{results['rho']:.4f}"],
                    ["Balanced ratio (ρb)", f"{results['rho_b']:.4f}"],
                    ["Maximum ratio (ρmax)", f"{results['rho_max']:.4f}"],
                    ["Min. steel (As,min)", f"{results['As_min_mm2']:.0f} mm²"],
                    ["Max. steel (As,max)", f"{results['As_max_mm2']:.0f} mm²"],
                    ["β₁ factor", f"{results['beta1']:.3f}"],
                    ["Nominal moment (Mn)", f"{results['Mn_tonfm']:.2f} tonf·m"]
                ], columns=["Parameter", "Value"])
                
                st.dataframe(detailed_df, hide_index=True, use_container_width=True)
                
                # Bar arrangement suggestion
                st.markdown("### 🔩 Reinforcement Arrangement")
                bar_info = calculate_bar_arrangement(results['As_req_mm2'], bar_dia_mm)
                
                st.info(f"""
                **Suggested Arrangement:**
                - {bar_info['num_bars']} Ø{bar_dia_mm} bars
                - Provided Area: {bar_info['As_provided_mm2']:.0f} mm²
                - Excess: {bar_info['excess_percent']:.1f}%
                """)
                
        elif mode == "Strength Analysis Mode":
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("### 💪 Strength Analysis Results")
                
                # Capacity metrics
                st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-label">Nominal Moment Capacity</div>
                    <div class="metric-value">{results['Mn_tonfm']:.2f}</div>
                    <div class="metric-unit">tonf·m</div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-label">Design Strength</div>
                    <div class="metric-value">{results['phi_Mn_tonfm']:.2f}</div>
                    <div class="metric-unit">tonf·m (φ = {results['phi']:.3f})</div>
                </div>
                """, unsafe_allow_html=True)
                
                # Control type with color coding
                control_class = {
                    'Tension-Controlled': 'control-type-tension',
                    'Transition Zone': 'control-type-transition',
                    'Compression-Controlled': 'control-type-compression'
                }.get(results['control_type'], 'control-type-tension')
                
                st.markdown(f"""<div class="{control_class}">
                    Section Type: {results['control_type']}<br>
                    εt = {results['epsilon_t']:.5f} | φ = {results['phi']:.3f}
                </div>""", unsafe_allow_html=True)
                
            with col2:
                st.markdown("### 📊 Strain Analysis")
                
                # Strain values table
                strain_df = pd.DataFrame([
                    ["Concrete strain (εc)", f"{results['epsilon_c']:.3f}"],
                    ["Steel strain (εt)", f"{results['epsilon_t']:.5f}"],
                    ["Yield strain (εy)", f"{fy_ksc/2.04e6:.3f}"],
                    ["Neutral axis (c)", f"{results['c_mm']:.1f} mm"],
                    ["Compression block (a)", f"{results['a_mm']:.1f} mm"],
                    ["Effective depth (d)", f"{results['d_mm']:.0f} mm"]
                ], columns=["Parameter", "Value"])
                
                st.dataframe(strain_df, hide_index=True, use_container_width=True)
                
                # ACI compliance check
                st.markdown("### ✅ ACI 318 Compliance")
                
                if results['control_type'] == 'Tension-Controlled':
                    st.success("✓ Section satisfies ACI 318 ductility requirements (εt ≥ 0.005)")
                elif results['control_type'] == 'Transition Zone':
                    st.warning("⚠ Section in transition zone - Consider increasing tension steel")
                else:
                    st.error("✗ Compression-controlled - Not recommended for flexural members")
        
        elif mode == "Design Mode" and not results.get('success'):
            st.error(f"❌ Design Failed: {results.get('message', 'Unknown error')}")
            if results.get('As_req_mm2'):
                st.warning(f"Required As = {results['As_req_mm2']:.0f} mm² exceeds maximum allowed")
    
    # ===========================
    # TAB 2: STRAIN DIAGRAM
    # ===========================
    with tab2:
        st.markdown("### 📈 Strain Distribution Analysis")
        
        # Create interactive Plotly strain diagram
        strain_fig = create_strain_diagram_plotly(results, h_mm)
        st.plotly_chart(strain_fig, use_container_width=True)
        
        # Show key equations
        with st.expander("📚 Key Equations Used", expanded=True):
            st.markdown("""
            <div class="equation-box">
            <b>Strain Compatibility:</b><br>
            εt = εc × (d - c) / c = 0.003 × (d - c) / c<br><br>
            
            <b>Neutral Axis Location:</b><br>
            c = a / β₁<br>
            where a = As × fy / (0.85 × fc' × b)<br><br>
            
            <b>Strength Reduction Factor (φ):</b><br>
            • Tension-controlled (εt ≥ 0.005): φ = 0.90<br>
            • Compression-controlled (εt ≤ εty): φ = 0.65<br>
            • Transition: φ = 0.65 + (εt - εty) × 0.25 / (0.005 - εty)<br><br>
            
            <b>Nominal Moment Capacity:</b><br>
            Mn = As × fy × (d - a/2)
            </div>
            """, unsafe_allow_html=True)
        
        # Strain limits info
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.info("""
            **Tension-Controlled**
            - εt ≥ 0.005
            - φ = 0.90
            - Ductile failure
            """)
        
        with col2:
            st.warning("""
            **Transition Zone**
            - 0.002 < εt < 0.005
            - 0.65 < φ < 0.90
            - Mixed behavior
            """)
        
        with col3:
            st.error("""
            **Compression-Controlled**
            - εt ≤ 0.002
            - φ = 0.65
            - Brittle failure
            """)
    
    # ===========================
    # TAB 3: SECTION VIEW
    # ===========================
    with tab3:
        st.markdown("### 🏗️ Beam Cross-Section")
        
        if mode == "Design Mode":
            As_bot_mm2 = results.get('As_req_mm2', 0)
            As_top_mm2 = 0
        else:
            As_bot_mm2 = inputs.get('As_bot_mm2', 0)
            As_top_mm2 = inputs.get('As_top_mm2', 0)
        
        # Draw section
        section_fig = draw_beam_section(b_mm, h_mm, cover_mm, As_bot_mm2, As_top_mm2, bar_dia_mm)
        st.pyplot(section_fig)
        
        # Section properties
        with st.expander("📏 Section Properties", expanded=True):
            props_df = pd.DataFrame([
                ["Beam width (b)", f"{b_mm:.0f} mm"],
                ["Beam height (h)", f"{h_mm:.0f} mm"],
                ["Concrete cover", f"{cover_mm:.0f} mm"],
                ["Effective depth (d)", f"{results.get('d_mm', h_mm-cover_mm-25):.0f} mm"],
                ["Bottom steel (As)", f"{As_bot_mm2:.0f} mm²"],
                ["Top steel (A's)", f"{As_top_mm2:.0f} mm²"]
            ], columns=["Property", "Value"])
            
            st.dataframe(props_df, hide_index=True, use_container_width=True)
    
    # ===========================
    # TAB 4: EXPORT
    # ===========================
    with tab4:
        st.markdown("### 💾 Export Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # PDF Report
            pdf_buffer = generate_pdf_report(mode, inputs, results)
            st.download_button(
                label="📄 Download PDF Report",
                data=pdf_buffer,
                file_name=f"RC_Beam_{mode.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf",
                use_container_width=True
            )
        
        with col2:
            # CSV Export
            export_data = []
            export_data.append(["Analysis Date", datetime.now().strftime('%Y-%m-%d %H:%M:%S')])
            export_data.append(["Analysis Mode", mode])
            export_data.append([""])
            export_data.append(["INPUT PARAMETERS"])
            for key, value in inputs.items():
                export_data.append([key, value])
            export_data.append([""])
            export_data.append(["RESULTS"])
            for key, value in results.items():
                if value is not None:
                    export_data.append([key, value])
            
            export_df = pd.DataFrame(export_data, columns=["Parameter", "Value"])
            csv = export_df.to_csv(index=False)
            
            st.download_button(
                label="📊 Download CSV Data",
                data=csv,
                file_name=f"RC_Beam_{mode.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col3:
            # Text Summary
            summary = f"""RC BEAM ANALYSIS SUMMARY
{'='*50}
Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}
Mode: {mode}
Code: ACI 318-19

INPUT PARAMETERS:
- Concrete: fc' = {inputs.get('fc_ksc', 0)} ksc
- Steel: fy = {inputs.get('fy_ksc', 0)} ksc
- Section: {inputs.get('b_mm', 0)} × {inputs.get('h_mm', 0)} mm
- Cover: {inputs.get('cover_mm', 0)} mm
"""
            
            if mode == "Design Mode":
                summary += f"""- Applied Moment: {inputs.get('Mu_tonfm', 0)} tonf·m

DESIGN RESULTS:
- Required As: {results.get('As_req_mm2', 0):.0f} mm²
- Control Type: {results.get('control_type', 'N/A')}
- φ Factor: {results.get('phi', 0):.3f}
- Design Strength: {results.get('phi_Mn_tonfm', 0):.2f} tonf·m
- Status: {'ADEQUATE' if results.get('success') else 'NEEDS REVISION'}
"""
            else:
                summary += f"""- Bottom Steel: {inputs.get('As_bot_mm2', 0)} mm²
- Top Steel: {inputs.get('As_top_mm2', 0)} mm²

STRENGTH ANALYSIS:
- Nominal Moment: {results.get('Mn_tonfm', 0):.2f} tonf·m
- Design Strength: {results.get('phi_Mn_tonfm', 0):.2f} tonf·m
- Control Type: {results.get('control_type', 'N/A')}
- φ Factor: {results.get('phi', 0):.3f}
- Tensile Strain: {results.get('epsilon_t', 0):.5f}
"""
            
            st.download_button(
                label="📝 Download Text Summary",
                data=summary,
                file_name=f"RC_Beam_Summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                use_container_width=True
            )
        
        # Show summary on screen
        st.markdown("### 📋 Analysis Summary")
        st.code(summary, language='text')

else:
    # Welcome screen
    st.markdown("""
    <div class="result-card">
    <h2>Welcome to the Enhanced RC Beam Designer & Analyzer! 👋</h2>
    
    <p>This advanced application provides comprehensive reinforced concrete beam analysis
    according to <strong>ACI 318-19</strong> building code requirements.</p>
    
    <h3>🎯 Available Modes:</h3>
    
    <div style="display: flex; gap: 20px; margin: 20px 0;">
        <div style="flex: 1; padding: 20px; background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%); border-radius: 10px;">
            <h4>📐 Design Mode</h4>
            <ul>
                <li>Calculate required reinforcement</li>
                <li>Check design adequacy</li>
                <li>Optimize steel arrangement</li>
                <li>Verify ductility requirements</li>
            </ul>
        </div>
        
        <div style="flex: 1; padding: 20px; background: linear-gradient(135deg, #f5930015 0%, #fee14015 100%); border-radius: 10px;">
            <h4>💪 Strength Analysis Mode</h4>
            <ul>
                <li>Analyze existing sections</li>
                <li>Calculate moment capacity</li>
                <li>Determine control type</li>
                <li>Evaluate strain distribution</li>
            </ul>
        </div>
    </div>
    
    <h3>✨ Key Features:</h3>
    <ul>
        <li>✅ Full ACI 318-19 compliance</li>
        <li>📈 Interactive strain diagrams with Plotly</li>
        <li>🎨 Professional visualizations</li>
        <li>📊 Detailed analysis reports</li>
        <li>💾 Multiple export formats (PDF, CSV, TXT)</li>
        <li>🔍 Real-time design verification</li>
        <li>📐 Support for singly and doubly reinforced sections</li>
    </ul>
    
    <div class="info-box">
    <strong>📌 Unit Convention:</strong><br>
    • Stresses: ksc (kg/cm²)<br>
    • Dimensions: mm<br>
    • Moments: tonf·m<br>
    • Areas: mm²
    </div>
    
    <div class="warning-box">
    <strong>⚠️ Important Note:</strong><br>
    This tool is for educational and preliminary design purposes only.
    Professional engineering review is required for construction projects.
    </div>
    
    <p style="text-align: center; margin-top: 30px; font-size: 1.1em;">
    <em>👈 Start by selecting your analysis mode and entering parameters in the sidebar</em>
    </p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #64748b; padding: 1rem;'>
    <p>🏗️ RC Beam Designer & Analyzer v3.0 | ACI 318-19 Compliant</p>
    <p style='font-size: 0.9em;'>
    <em>Developed for structural engineers and students | Professional review required for construction</em>
    </p>
</div>
""", unsafe_allow_html=True)
