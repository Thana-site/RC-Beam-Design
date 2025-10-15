"""
RC Beam Designer - ACI 318-19
Professional web application for designing reinforced concrete beams

Author: Civil Engineering Design Tool
Version: 2.0.0
License: Educational Use
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from io import BytesIO
import math

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="RC Beam Designer - ACI 318",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS STYLING
# ============================================================================

st.markdown("""
<style>
    /* Main header styling */
    .main-header {
        font-size: 2.8rem;
        font-weight: 700;
        color: #1e3a8a;
        text-align: center;
        padding: 1.5rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Section headers */
    .section-header {
        font-size: 1.8rem;
        font-weight: 600;
        color: #2563eb;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #3b82f6;
    }
    
    /* Info boxes */
    .info-box {
        background: linear-gradient(135deg, #e0f2fe 0%, #bfdbfe 100%);
        padding: 1.2rem;
        border-radius: 10px;
        border-left: 5px solid #3b82f6;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    .success-box {
        background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
        padding: 1.2rem;
        border-radius: 10px;
        border-left: 5px solid #10b981;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    .warning-box {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        padding: 1.2rem;
        border-radius: 10px;
        border-left: 5px solid #f59e0b;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    .error-box {
        background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
        padding: 1.2rem;
        border-radius: 10px;
        border-left: 5px solid #ef4444;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    /* Metric cards */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border: 1px solid #e5e7eb;
        transition: transform 0.2s;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #f8fafc;
    }
    
    /* Button styling */
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
        border: none;
        padding: 0.75rem 1.5rem;
        border-radius: 8px;
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.2);
    }
    
    /* Table styling */
    .dataframe {
        border-radius: 8px;
        overflow: hidden;
    }
    
    /* Status badges */
    .status-pass {
        background-color: #10b981;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.875rem;
    }
    
    .status-fail {
        background-color: #ef4444;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.875rem;
    }
    
    .status-warning {
        background-color: #f59e0b;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.875rem;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# DESIGN CALCULATION FUNCTIONS (ACI 318-19)
# ============================================================================

def calculate_beta1(fc_mpa):
    """
    Calculate β1 factor according to ACI 318-19 (Section 22.2.2.4.3)
    
    Args:
        fc_mpa (float): Concrete compressive strength in MPa
        
    Returns:
        float: β1 factor for rectangular stress block
    """
    if fc_mpa <= 28:
        return 0.85
    elif fc_mpa <= 55:
        return max(0.85 - 0.05 * (fc_mpa - 28) / 7, 0.65)
    else:
        return 0.65

def calculate_rho_balanced(fc_mpa, fy_mpa, beta1):
    """
    Calculate balanced reinforcement ratio
    
    Args:
        fc_mpa (float): Concrete strength in MPa
        fy_mpa (float): Steel yield strength in MPa
        beta1 (float): β1 factor
        
    Returns:
        float: Balanced reinforcement ratio ρ_b
    """
    return 0.85 * beta1 * fc_mpa / fy_mpa * (600 / (600 + fy_mpa))

def calculate_rho_min(fc_mpa, fy_mpa):
    """
    Calculate minimum reinforcement ratio (ACI 318-19, Section 9.6.1.2)
    
    Args:
        fc_mpa (float): Concrete strength in MPa
        fy_mpa (float): Steel yield strength in MPa
        
    Returns:
        float: Minimum reinforcement ratio ρ_min
    """
    return max(0.25 * np.sqrt(fc_mpa) / fy_mpa, 1.4 / fy_mpa)

def calculate_required_steel(Mu_kNm, b_mm, d_mm, fc_mpa, fy_mpa, phi=0.9):
    """
    Calculate required steel area using ACI rectangular stress block method
    
    Args:
        Mu_kNm (float): Factored moment in kN·m
        b_mm (float): Beam width in mm
        d_mm (float): Effective depth in mm
        fc_mpa (float): Concrete strength in MPa
        fy_mpa (float): Steel yield strength in MPa
        phi (float): Strength reduction factor (initial assumption)
        
    Returns:
        tuple: (As_mm2, a_mm, c_mm, rho, success, message, detailed_results)
    """
    # Convert units
    Mu_Nmm = Mu_kNm * 1e6  # kN·m to N·mm
    
    # Calculate required moment coefficient Rn
    Rn = Mu_Nmm / (phi * b_mm * d_mm**2)
    
    # Solve for reinforcement ratio using quadratic formula
    # From: Mn = As*fy*(d - a/2) and a = As*fy/(0.85*fc*b)
    # Rearranged: 0.59*fy²/fc' * ρ² - fy*ρ + Rn = 0
    
    a_coef = 0.59 * fy_mpa**2 / fc_mpa
    b_coef = -fy_mpa
    c_coef = Rn
    
    discriminant = b_coef**2 - 4*a_coef*c_coef
    
    if discriminant < 0:
        return None, None, None, None, False, "❌ Moment too large - section cannot be designed as singly reinforced", {}
    
    # Two roots from quadratic formula
    rho1 = (-b_coef - np.sqrt(discriminant)) / (2*a_coef)
    rho2 = (-b_coef + np.sqrt(discriminant)) / (2*a_coef)
    
    # Take the smaller positive root (more economical)
    rho = min(rho1, rho2) if min(rho1, rho2) > 0 else max(rho1, rho2)
    
    # Calculate limits
    beta1 = calculate_beta1(fc_mpa)
    rho_min = calculate_rho_min(fc_mpa, fy_mpa)
    rho_balanced = calculate_rho_balanced(fc_mpa, fy_mpa, beta1)
    rho_max = 0.75 * rho_balanced  # ACI limit for ductility
    
    # Check reinforcement ratio limits
    message_parts = []
    
    if rho < rho_min:
        rho = rho_min
        message_parts.append(f"⚠️ Using minimum reinforcement ratio (ρ_min = {rho:.4f})")
    
    if rho > rho_max:
        return None, None, None, rho, False, \
            f"❌ Required ρ = {rho:.4f} exceeds maximum ρ_max = {rho_max:.4f}. Use doubly reinforced section or increase dimensions.", \
            {'rho_min': rho_min, 'rho_max': rho_max, 'rho_balanced': rho_balanced}
    
    if not message_parts:
        message_parts.append("✅ Design within acceptable limits")
    
    # Calculate steel area
    As_mm2 = rho * b_mm * d_mm
    
    # Calculate neutral axis depth and compression block depth
    c_mm = As_mm2 * fy_mpa / (0.85 * fc_mpa * beta1 * b_mm)
    a_mm = beta1 * c_mm
    
    # Detailed results for verification
    detailed_results = {
        'rho_min': rho_min,
        'rho_max': rho_max,
        'rho_balanced': rho_balanced,
        'beta1': beta1,
        'Rn': Rn
    }
    
    return As_mm2, a_mm, c_mm, rho, True, " | ".join(message_parts), detailed_results

def calculate_moment_capacity(As_mm2, b_mm, d_mm, fc_mpa, fy_mpa):
    """
    Calculate nominal and design moment capacity with strain analysis
    
    Args:
        As_mm2 (float): Steel area in mm²
        b_mm (float): Beam width in mm
        d_mm (float): Effective depth in mm
        fc_mpa (float): Concrete strength in MPa
        fy_mpa (float): Steel yield strength in MPa
        
    Returns:
        tuple: (Mn_kNm, phi_Mn_kNm, phi, c_mm, et, strain_type)
    """
    beta1 = calculate_beta1(fc_mpa)
    Es = 200000  # Steel modulus of elasticity in MPa
    
    # Calculate neutral axis depth
    c_mm = As_mm2 * fy_mpa / (0.85 * fc_mpa * beta1 * b_mm)
    a_mm = beta1 * c_mm
    
    # Calculate tensile strain in steel (strain compatibility)
    et = 0.003 * (d_mm - c_mm) / c_mm
    
    # Calculate yield strain
    ey = fy_mpa / Es
    
    # Determine φ factor based on strain limits (ACI 318-19, Section 21.2.2)
    if et < ey:
        # Compression-controlled
        phi = 0.65
        strain_type = "Compression-Controlled"
    elif et >= 0.005:
        # Tension-controlled
        phi = 0.90
        strain_type = "Tension-Controlled"
    else:
        # Transition zone
        phi = 0.65 + (et - ey) / (0.005 - ey) * (0.90 - 0.65)
        strain_type = "Transition Zone"
    
    # Calculate nominal moment capacity
    Mn_Nmm = As_mm2 * fy_mpa * (d_mm - a_mm/2)
    Mn_kNm = Mn_Nmm / 1e6  # Convert to kN·m
    phi_Mn_kNm = phi * Mn_kNm
    
    return Mn_kNm, phi_Mn_kNm, phi, c_mm, et, strain_type

def suggest_bar_arrangements(As_required_mm2, b_mm, d_mm, cover_mm, stirrup_dia):
    """
    Suggest practical reinforcement bar arrangements
    
    Args:
        As_required_mm2 (float): Required steel area in mm²
        b_mm (float): Beam width in mm
        d_mm (float): Effective depth in mm
        cover_mm (float): Concrete cover in mm
        stirrup_dia (float): Stirrup diameter in mm
        
    Returns:
        pandas.DataFrame: Suggested bar arrangements
    """
    # Standard bar sizes in mm with their areas
    bar_data = {
        10: 78.5,
        12: 113.1,
        16: 201.1,
        20: 314.2,
        25: 490.9,
        28: 615.8,
        32: 804.2
    }
    
    suggestions = []
    
    # Calculate available width for bars
    clear_width = b_mm - 2 * (cover_mm + stirrup_dia + 10)  # 10mm additional clearance
    
    for diameter, area_per_bar in bar_data.items():
        # Calculate number of bars needed
        num_bars = math.ceil(As_required_mm2 / area_per_bar)
        
        # Practical limits
        if num_bars < 2 or num_bars > 8:
            continue
        
        # Calculate spacing between bars
        if num_bars == 2:
            spacing = clear_width
        else:
            spacing = clear_width / (num_bars - 1)
        
        # Check minimum spacing (ACI: 25mm or bar diameter, whichever is larger)
        min_spacing = max(25, diameter)
        
        if spacing < min_spacing:
            continue
        
        # Calculate actual provided area
        total_area = num_bars * area_per_bar
        excess_pct = (total_area - As_required_mm2) / As_required_mm2 * 100
        
        # Add to suggestions
        suggestions.append({
            'Bar Size': f'ø{diameter}mm',
            'Number of Bars': num_bars,
            'Area per Bar (mm²)': round(area_per_bar, 1),
            'Total Area (mm²)': round(total_area, 1),
            'Excess (%)': round(excess_pct, 1),
            'Bar Spacing (mm)': round(spacing, 0),
            'Status': '✅' if spacing >= min_spacing and excess_pct < 50 else '⚠️'
        })
    
    # Create DataFrame and sort by excess percentage
    df = pd.DataFrame(suggestions)
    if len(df) > 0:
        df = df.sort_values('Excess (%)')
    
    return df

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_beam_cross_section(b_mm, h_mm, cover_mm, bar_diameter, num_bars, stirrup_dia):
    """
    Create detailed beam cross-section visualization
    
    Args:
        b_mm (float): Beam width in mm
        h_mm (float): Beam height in mm
        cover_mm (float): Concrete cover in mm
        bar_diameter (float): Main bar diameter in mm
        num_bars (int): Number of main bars
        stirrup_dia (float): Stirrup diameter in mm
        
    Returns:
        matplotlib.figure.Figure: Cross-section diagram
    """
    fig, ax = plt.subplots(figsize=(10, 12))
    
    # Draw concrete section
    concrete = patches.Rectangle(
        (0, 0), b_mm, h_mm,
        linewidth=2.5,
        edgecolor='#2c3e50',
        facecolor='#ecf0f1',
        alpha=0.4
    )
    ax.add_patch(concrete)
    
    # Draw stirrups
    stirrup_outer = patches.Rectangle(
        (cover_mm, cover_mm),
        b_mm - 2*cover_mm,
        h_mm - 2*cover_mm,
        linewidth=2,
        edgecolor='#3498db',
        facecolor='none'
    )
    ax.add_patch(stirrup_outer)
    
    stirrup_inner = patches.Rectangle(
        (cover_mm + stirrup_dia, cover_mm + stirrup_dia),
        b_mm - 2*(cover_mm + stirrup_dia),
        h_mm - 2*(cover_mm + stirrup_dia),
        linewidth=2,
        edgecolor='#3498db',
        facecolor='none',
        linestyle='--'
    )
    ax.add_patch(stirrup_inner)
    
    # Calculate bar positions
    y_position = cover_mm + stirrup_dia + bar_diameter/2
    clear_width = b_mm - 2*(cover_mm + stirrup_dia + bar_diameter/2)
    
    if num_bars == 1:
        x_positions = [b_mm/2]
    elif num_bars == 2:
        x_positions = [
            cover_mm + stirrup_dia + bar_diameter/2,
            b_mm - (cover_mm + stirrup_dia + bar_diameter/2)
        ]
    else:
        spacing = clear_width / (num_bars - 1)
        x_positions = [
            cover_mm + stirrup_dia + bar_diameter/2 + i*spacing
            for i in range(num_bars)
        ]
    
    # Draw reinforcement bars
    for i, x in enumerate(x_positions):
        bar = patches.Circle(
            (x, y_position),
            bar_diameter/2,
            edgecolor='#8b0000',
            facecolor='#c0392b',
            linewidth=2,
            zorder=5
        )
        ax.add_patch(bar)
        
        # Add bar number label
        ax.text(x, y_position, str(i+1),
                ha='center', va='center',
                fontsize=8, fontweight='bold',
                color='white', zorder=6)
    
    # Add dimension lines and labels
    # Width dimension
    ax.annotate('', xy=(b_mm, -h_mm*0.08), xytext=(0, -h_mm*0.08),
                arrowprops=dict(arrowstyle='<->', color='black', lw=2))
    ax.text(b_mm/2, -h_mm*0.13, f'b = {b_mm:.0f} mm',
            ha='center', va='top',
            fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='black'))
    
    # Height dimension
    ax.annotate('', xy=(-b_mm*0.08, h_mm), xytext=(-b_mm*0.08, 0),
                arrowprops=dict(arrowstyle='<->', color='black', lw=2))
    ax.text(-b_mm*0.13, h_mm/2, f'h = {h_mm:.0f} mm',
            ha='center', va='center', rotation=90,
            fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='black'))
    
    # Effective depth indicator
    d_mm = h_mm - (cover_mm + stirrup_dia + bar_diameter/2)
    ax.plot([b_mm + b_mm*0.06, b_mm + b_mm*0.06], [0, d_mm],
            'g-', linewidth=3, label='Effective Depth (d)')
    ax.plot([b_mm + b_mm*0.04, b_mm + b_mm*0.08], [d_mm, d_mm],
            'g-', linewidth=3)
    ax.text(b_mm + b_mm*0.14, d_mm/2, f'd = {d_mm:.0f} mm',
            ha='left', va='center', rotation=90,
            fontsize=11, color='green', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#d4edda', edgecolor='green'))
    
    # Cover dimension
    ax.plot([0, cover_mm], [h_mm + h_mm*0.02, h_mm + h_mm*0.02],
            'k--', linewidth=1, alpha=0.6)
    ax.plot([cover_mm, cover_mm], [h_mm, h_mm + h_mm*0.04],
            'k-', linewidth=1)
    ax.text(cover_mm/2, h_mm + h_mm*0.06, f'cover = {cover_mm:.0f} mm',
            ha='center', fontsize=9, style='italic',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow'))
    
    # Legend
    legend_elements = [
        patches.Patch(facecolor='#ecf0f1', edgecolor='#2c3e50', label='Concrete Section'),
        patches.Patch(facecolor='none', edgecolor='#3498db', label='Stirrups'),
        patches.Circle((0, 0), 1, facecolor='#c0392b', edgecolor='#8b0000', label='Main Bars'),
        patches.Patch(facecolor='none', edgecolor='green', label='Effective Depth')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10, framealpha=0.9)
    
    # Axes settings
    ax.set_xlim(-b_mm*0.2, b_mm*1.25)
    ax.set_ylim(-h_mm*0.2, h_mm*1.15)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Reinforced Concrete Beam Cross-Section',
                fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    return fig

def plot_strain_distribution(c_mm, d_mm, h_mm, et, ey):
    """
    Create strain distribution diagram
    
    Args:
        c_mm (float): Neutral axis depth in mm
        d_mm (float): Effective depth in mm
        h_mm (float): Total beam height in mm
        et (float): Tensile strain in steel
        ey (float): Yield strain
        
    Returns:
        matplotlib.figure.Figure: Strain diagram
    """
    fig, ax = plt.subplots(figsize=(8, 10))
    
    # Maximum compressive strain in concrete
    ec_max = 0.003
    
    # Strain profile
    depths = [0, c_mm, d_mm]
    strains = [ec_max, 0, -et]
    
    # Plot strain diagram
    ax.plot(strains, depths, 'b-', linewidth=3, label='Strain Distribution')
    ax.fill_betweenx(depths, 0, strains, alpha=0.3, color='blue')
    
    # Neutral axis
    ax.axhline(y=c_mm, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Neutral Axis')
    
    # Zero strain line
    ax.axvline(x=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    
    # Yield strain indicators
    if et > ey:
        ey_depth = c_mm + (d_mm - c_mm) * (ey / et)
        ax.axhline(y=ey_depth, color='orange', linestyle=':', linewidth=2, alpha=0.7)
        ax.text(-ey*1.1, ey_depth, f'εy = {ey:.5f}',
                fontsize=10, color='orange', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='#fff3cd'))
    
    # Strain labels
    ax.text(ec_max*0.6, 10, f'εc = {ec_max:.4f}',
            fontsize=11, ha='center', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#cfe2ff'))
    
    ax.text(-et*0.6, d_mm-10, f'εt = {et:.5f}',
            fontsize=11, ha='center', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#f8d7da'))
    
    ax.text(0.0001, c_mm, f'c = {c_mm:.1f} mm',
            fontsize=10, color='red', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white'))
    
    # Axes labels and formatting
    ax.set_xlabel('Strain', fontsize=13, fontweight='bold')
    ax.set_ylabel('Depth from Top (mm)', fontsize=13, fontweight='bold')
    ax.set_title('Strain Distribution at Ultimate\n(Plane Sections Remain Plane)',
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.invert_yaxis()
    ax.legend(loc='best', fontsize=10)
    
    # Add compression/tension zones
    ax.text(ec_max/2, c_mm/2, 'Compression\nZone',
            ha='center', va='center', fontsize=10,
            style='italic', alpha=0.7)
    ax.text(-et/2, (c_mm + d_mm)/2, 'Tension\nZone',
            ha='center', va='center', fontsize=10,
            style='italic', alpha=0.7)
    
    plt.tight_layout()
    return fig

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main application function"""
    
    # Application Header
    st.markdown("""
        <div class="main-header">
            🏗️ RC BEAM DESIGNER<br>
            <span style="font-size: 1.2rem; font-weight: 400;">ACI 318-19 Building Code</span>
        </div>
    """, unsafe_allow_html=True)
    
    # Introduction
    st.markdown("""
        <div class="info-box">
            <h3 style="margin-top: 0;">📖 Professional Reinforced Concrete Beam Design Tool</h3>
            <p style="margin-bottom: 0;">
                Design rectangular RC beams for flexure using the <b>ACI 318-19</b> rectangular stress block method.
                This application performs complete flexural design including:
            </p>
            <ul>
                <li>✅ Required steel area calculation</li>
                <li>✅ Moment capacity verification</li>
                <li>✅ Reinforcement ratio checks (ρ_min, ρ_max)</li>
                <li>✅ Strain compatibility analysis</li>
                <li>✅ Practical bar arrangement suggestions</li>
                <li>✅ Professional visualization and export options</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)
    
    # ========================================================================
    # SIDEBAR - INPUT PARAMETERS
    # ========================================================================
    
    st.sidebar.image("https://via.placeholder.com/300x100/667eea/ffffff?text=RC+BEAM+DESIGNER", use_container_width=True)
    st.sidebar.markdown("---")
    
    st.sidebar.header("📋 Design Input Parameters")
    
    # Material Properties
    with st.sidebar.expander("🔬 Material Properties", expanded=True):
        fc_mpa = st.number_input(
            "Concrete Strength, f'c (MPa)",
            min_value=20.0,
            max_value=80.0,
            value=25.0,
            step=1.0,
            help="Specified 28-day compressive strength of concrete"
        )
        
        fy_mpa = st.number_input(
            "Steel Yield Strength, fy (MPa)",
            min_value=300.0,
            max_value=600.0,
            value=420.0,
            step=10.0,
            help="Specified yield strength of reinforcing steel"
        )
    
    # Beam Geometry
    with st.sidebar.expander("📐 Beam Geometry", expanded=True):
        b_mm = st.number_input(
            "Beam Width, b (mm)",
            min_value=200.0,
            max_value=1000.0,
            value=300.0,
            step=10.0,
            help="Width of the rectangular beam section"
        )
        
        h_mm = st.number_input(
            "Beam Height, h (mm)",
            min_value=300.0,
            max_value=1500.0,
            value=600.0,
            step=10.0,
            help="Total height of the beam section"
        )
        
        cover_mm = st.number_input(
            "Concrete Cover (mm)",
            min_value=20.0,
            max_value=75.0,
            value=40.0,
            step=5.0,
            help="Clear concrete cover to stirrups (ACI: minimum 40mm for beams)"
        )
    
    # Design Loads
    with st.sidebar.expander("⚡ Design Loads", expanded=True):
        Mu_kNm = st.number_input(
            "Factored Moment, Mu (kN·m)",
            min_value=1.0,
            max_value=2000.0,
            value=150.0,
            step=5.0,
            help="Factored design moment from load combinations"
        )
    
    # Reinforcement Details
    with st.sidebar.expander("🔩 Reinforcement Details", expanded=True):
        bar_diameter = st.selectbox(
            "Main Bar Diameter (mm)",
            options=[10, 12, 16, 20, 25, 28, 32],
            index=3,
            help="Diameter of longitudinal tension reinforcement"
        )
        
        stirrup_dia = st.selectbox(
            "Stirrup Diameter (mm)",
            options=[8, 10, 12],
            index=1,
            help="Diameter of transverse reinforcement (stirrups)"
        )
    
    st.sidebar.markdown("---")
    
    # Design Button
    design_button = st.sidebar.button(
        "🔍 PERFORM DESIGN ANALYSIS",
        type="primary",
        use_container_width=True
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
        **Design Standard:** ACI 318-19  
        **Method:** Rectangular Stress Block  
        **Version:** 2.0.0
        
        ---
        
        ⚠️ **Disclaimer:** This tool is for educational and preliminary design purposes.
        All designs should be verified by a licensed professional engineer.
    """)
    
    # ========================================================================
    # MAIN PANEL - DESIGN RESULTS
    # ========================================================================
    
    if design_button:
        # Calculate effective depth
        d_mm = h_mm - (cover_mm + stirrup_dia + bar_diameter/2)
        
        # Display calculation progress
        with st.spinner('🔄 Performing design calculations...'):
            
            # Section 1: Material and Section Properties
            st.markdown('<div class="section-header">📊 Material & Section Properties</div>', 
                       unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Concrete Strength", f"{fc_mpa} MPa", help="f'c")
                beta1 = calculate_beta1(fc_mpa)
                st.caption(f"β₁ = {beta1:.3f}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Steel Strength", f"{fy_mpa} MPa", help="fy")
                Es = 200000
                st.caption(f"Es = {Es:,} MPa")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col3:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Effective Depth", f"{d_mm:.1f} mm", help="d")
                st.caption(f"d/h = {d_mm/h_mm:.3f}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col4:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Factored Moment", f"{Mu_kNm:.2f} kN·m", help="Mu")
                st.caption(f"b×h = {b_mm}×{h_mm} mm")
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Section 2: Design Calculations
            st.markdown('<div class="section-header">🔬 Flexural Design Analysis</div>', 
                       unsafe_allow_html=True)
            
            # Perform design
            result = calculate_required_steel(Mu_kNm, b_mm, d_mm, fc_mpa, fy_mpa)
            As_required, a_mm, c_mm, rho, success, message, details = result
            
            if success:
                # Calculate moment capacity
                Mn, phi_Mn, phi, c_final, et, strain_type = calculate_moment_capacity(
                    As_required, b_mm, d_mm, fc_mpa, fy_mpa
                )
                
                # Success message
                st.markdown(f"""
                    <div class="success-box">
                        <h3 style="margin-top: 0;">✅ Design Successful!</h3>
                        <p style="margin-bottom: 0;">{message}</p>
                    </div>
                """, unsafe_allow_html=True)
                
                # Detailed Results Table
                st.markdown("#### 📋 Detailed Design Results")
                
                rho_min = details['rho_min']
                rho_max = details['rho_max']
                rho_balanced = details['rho_balanced']
                ey = fy_mpa / Es
                
                results_data = {
                    'Design Parameter': [
                        'Required Steel Area (As)',
                        'Reinforcement Ratio (ρ)',
                        'Min. Reinforcement Ratio (ρ_min)',
                        'Max. Reinforcement Ratio (ρ_max)',
                        'Balanced Ratio (ρ_balanced)',
                        'Neutral Axis Depth (c)',
                        'Compression Block Depth (a)',
                        'Steel Tensile Strain (εt)',
                        'Steel Yield Strain (εy)',
                        'Strain Control Type',
                        'Strength Reduction Factor (φ)',
                        'Nominal Moment Capacity (Mn)',
                        'Design Moment Capacity (φMn)',
                        'Required Factored Moment (Mu)',
                        'Capacity Ratio (φMn/Mu)'
                    ],
                    'Value': [
                        f"{As_required:.0f} mm²",
                        f"{rho:.4f}",
                        f"{rho_min:.4f}",
                        f"{rho_max:.4f}",
                        f"{rho_balanced:.4f}",
                        f"{c_final:.1f} mm",
                        f"{a_mm:.1f} mm",
                        f"{et:.5f}",
                        f"{ey:.5f}",
                        strain_type,
                        f"{phi:.3f}",
                        f"{Mn:.2f} kN·m",
                        f"{phi_Mn:.2f} kN·m",
                        f"{Mu_kNm:.2f} kN·m",
                        f"{phi_Mn/Mu_kNm:.3f}"
                    ],
                    'Check': [
                        '✅',
                        '✅' if rho_min <= rho <= rho_max else '❌',
                        '✅',
                        '✅' if rho <= rho_max else '❌',
                        'ℹ️',
                        '✅',
                        '✅',
                        '✅' if et >= ey else '⚠️',
                        'ℹ️',
                        '✅' if strain_type == "Tension-Controlled" else ('⚠️' if strain_type == "Transition Zone" else '❌'),
                        '✅',
                        '✅',
                        '✅',
                        '✅',
                        '✅' if phi_Mn >= Mu_kNm else '❌'
                    ]
                }
                
                results_df = pd.DataFrame(results_data)
                st.dataframe(
                    results_df,
                    use_container_width=True,
                    hide_index=True,
                    height=600
                )
                
                # Adequacy Check
                st.markdown("#### ✔️ Design Adequacy")
                
                if phi_Mn >= Mu_kNm:
                    capacity_excess = (phi_Mn - Mu_kNm) / Mu_kNm * 100
                    st.markdown(f"""
                        <div class="success-box">
                            <h4 style="margin-top: 0;">✅ DESIGN IS ADEQUATE</h4>
                            <p><b>φMn = {phi_Mn:.2f} kN·m ≥ Mu = {Mu_kNm:.2f} kN·m</b></p>
                            <p>Excess capacity: <b>{capacity_excess:.1f}%</b></p>
                            <p style="margin-bottom: 0;">The section has sufficient moment capacity to resist the factored loads.</p>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    capacity_deficit = (Mu_kNm - phi_Mn) / Mu_kNm * 100
                    st.markdown(f"""
                        <div class="error-box">
                            <h4 style="margin-top: 0;">❌ DESIGN IS INADEQUATE</h4>
                            <p><b>φMn = {phi_Mn:.2f} kN·m < Mu = {Mu_kNm:.2f} kN·m</b></p>
                            <p>Capacity deficit: <b>{capacity_deficit:.1f}%</b></p>
                            <p style="margin-bottom: 0;">⚠️ The section does not have sufficient capacity. Increase dimensions or use doubly reinforced section.</p>
                        </div>
                    """, unsafe_allow_html=True)
                
                # Bar Arrangements
                st.markdown('<div class="section-header">🔩 Recommended Bar Arrangements</div>', 
                           unsafe_allow_html=True)
                
                suggestions_df = suggest_bar_arrangements(
                    As_required, b_mm, d_mm, cover_mm, stirrup_dia
                )
                
                if len(suggestions_df) > 0:
                    st.markdown("""
                        <div class="info-box">
                            <p style="margin: 0;"><b>💡 Selection Guide:</b> Choose an arrangement with ✅ status, 
                            reasonable excess percentage (&lt;30%), and adequate bar spacing for concrete placement.</p>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    st.dataframe(
                        suggestions_df,
                        use_container_width=True,
                        hide_index=True,
                        height=300
                    )
                    
                    # Select arrangement for visualization
                    st.markdown("#### 🎨 Select Arrangement for Visualization")
                    
                    selected_idx = st.selectbox(
                        "Choose bar arrangement:",
                        range(len(suggestions_df)),
                        format_func=lambda i: f"{suggestions_df.iloc[i]['Bar Size']} - {suggestions_df.iloc[i]['Number of Bars']} bars ({suggestions_df.iloc[i]['Status']})"
                    )
                    
                    selected = suggestions_df.iloc[selected_idx]
                    selected_dia = int(selected['Bar Size'].replace('ø', '').replace('mm', ''))
                    selected_num = int(selected['Number of Bars'])
                    
                    st.info(f"**Selected:** {selected_num} bars of {selected['Bar Size']} providing {selected['Total Area (mm²)']} mm² (excess: {selected['Excess (%)']}%)")
                    
                else:
                    st.warning("⚠️ No suitable bar arrangements found. Consider increasing beam dimensions.")
                    selected_dia = bar_diameter
                    selected_num = max(2, int(As_required / (math.pi * (bar_diameter/2)**2)))
                
                # Visualizations
                st.markdown('<div class="section-header">📈 Design Visualizations</div>', 
                           unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### 🏗️ Beam Cross-Section")
                    fig_section = plot_beam_cross_section(
                        b_mm, h_mm, cover_mm, selected_dia, selected_num, stirrup_dia
                    )
                    st.pyplot(fig_section)
                    plt.close()
                
                with col2:
                    st.markdown("#### 📊 Strain Distribution")
                    fig_strain = plot_strain_distribution(
                        c_final, d_mm, h_mm, et, ey
                    )
                    st.pyplot(fig_strain)
                    plt.close()
                
                # Design Summary
                st.markdown('<div class="section-header">📄 Design Summary Report</div>', 
                           unsafe_allow_html=True)
                
                summary_text = f"""
═══════════════════════════════════════════════════════════════
        RC BEAM DESIGN SUMMARY - ACI 318-19
═══════════════════════════════════════════════════════════════

PROJECT INFORMATION
─────────────────────────────────────────────────────────────
Design Code:          ACI 318-19
Design Method:        Rectangular Stress Block
Analysis Date:        {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

MATERIAL PROPERTIES
─────────────────────────────────────────────────────────────
Concrete Strength:    f'c = {fc_mpa} MPa
Steel Yield Strength: fy = {fy_mpa} MPa
Steel Modulus:        Es = {Es:,} MPa
Beta Factor:          β₁ = {beta1:.3f}

BEAM GEOMETRY
─────────────────────────────────────────────────────────────
Beam Width:           b = {b_mm} mm
Beam Height:          h = {h_mm} mm
Effective Depth:      d = {d_mm:.1f} mm
Concrete Cover:       {cover_mm} mm (to stirrups)
Stirrup Diameter:     {stirrup_dia} mm

DESIGN LOADS
─────────────────────────────────────────────────────────────
Factored Moment:      Mu = {Mu_kNm:.2f} kN·m

REINFORCEMENT LIMITS
─────────────────────────────────────────────────────────────
Minimum Ratio:        ρ_min = {rho_min:.4f}
Maximum Ratio:        ρ_max = {rho_max:.4f} (0.75 × ρ_balanced)
Balanced Ratio:       ρ_balanced = {rho_balanced:.4f}

DESIGN RESULTS
─────────────────────────────────────────────────────────────
Required Steel Area:  As = {As_required:.0f} mm²
Actual Ratio:         ρ = {rho:.4f}
Neutral Axis Depth:   c = {c_final:.1f} mm
Compression Block:    a = {a_mm:.1f} mm

STRAIN ANALYSIS
─────────────────────────────────────────────────────────────
Tension Strain:       εt = {et:.5f}
Yield Strain:         εy = {ey:.5f}
Strain Control:       {strain_type}
Reduction Factor:     φ = {phi:.3f}

MOMENT CAPACITY
─────────────────────────────────────────────────────────────
Nominal Capacity:     Mn = {Mn:.2f} kN·m
Design Capacity:      φMn = {phi_Mn:.2f} kN·m
Required Moment:      Mu = {Mu_kNm:.2f} kN·m
Capacity Ratio:       φMn/Mu = {phi_Mn/Mu_kNm:.3f}

RECOMMENDED REINFORCEMENT
─────────────────────────────────────────────────────────────
Bar Size:             {selected['Bar Size']}
Number of Bars:       {selected_num} bars
Provided Area:        {selected['Total Area (mm²)']} mm²
Bar Spacing:          {selected['Bar Spacing (mm)']} mm

DESIGN STATUS
─────────────────────────────────────────────────────────────
{'✅ DESIGN ADEQUATE - Section capacity exceeds required moment' if phi_Mn >= Mu_kNm else '❌ DESIGN INADEQUATE - Increase section or use doubly reinforced'}
{'✅ Strain limits satisfied - ' + strain_type if et >= ey else '⚠️ Check strain limits'}
{'✅ Reinforcement within code limits' if rho_min <= rho <= rho_max else '❌ Reinforcement ratio out of limits'}

NOTES
─────────────────────────────────────────────────────────────
1. Design based on ACI 318-19 rectangular stress block method
2. Shear design and detailing requirements not included
3. Check development length and splice requirements separately
4. Verify deflection and crack width if required
5. All designs must be reviewed by licensed professional engineer

═══════════════════════════════════════════════════════════════
                END OF DESIGN SUMMARY
═══════════════════════════════════════════════════════════════
"""
                
                st.code(summary_text, language=None)
                
                # Export Options
                st.markdown('<div class="section-header">💾 Export Design Results</div>', 
                           unsafe_allow_html=True)
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    csv_data = results_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results (CSV)",
                        data=csv_data,
                        file_name=f"rc_beam_design_results_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                
                with col2:
                    st.download_button(
                        label="📄 Download Summary (TXT)",
                        data=summary_text,
                        file_name=f"rc_beam_design_summary_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain",
                        use_container_width=True
                    )
                
                with col3:
                    # Export bar arrangements
                    bars_csv = suggestions_df.to_csv(index=False)
                    st.download_button(
                        label="🔩 Download Bar Options (CSV)",
                        data=bars_csv,
                        file_name=f"rc_beam_bar_arrangements_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
            
            else:
                # Design Failed
                st.markdown(f"""
                    <div class="error-box">
                        <h3 style="margin-top: 0;">❌ Design Failed</h3>
                        <p><b>{message}</b></p>
                        <br>
                        <h4>📌 Recommendations:</h4>
                        <ul>
                            <li>✓ Increase beam width (b) or height (h)</li>
                            <li>✓ Use higher strength concrete (f'c)</li>
                            <li>✓ Consider doubly reinforced section (compression steel)</li>
                            <li>✓ Reduce applied factored moment</li>
                            <li>✓ Use T-beam or L-beam section if applicable</li>
                        </ul>
                    </div>
                """, unsafe_allow_html=True)
                
                # Show limits
                beta1 = calculate_beta1(fc_mpa)
                rho_max = 0.75 * calculate_rho_balanced(fc_mpa, fy_mpa, beta1)
                As_max = rho_max * b_mm * d_mm
                
                if As_max > 0:
                    Mn_max, phi_Mn_max, _, _, _, _ = calculate_moment_capacity(
                        As_max, b_mm, d_mm, fc_mpa, fy_mpa
                    )
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        st.metric(
                            "Maximum Capacity",
                            f"{phi_Mn_max:.2f} kN·m",
                            help="For singly reinforced section"
                        )
                        st.caption(f"ρ_max = {rho_max:.4f}")
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        st.metric(
                            "Required Moment",
                            f"{Mu_kNm:.2f} kN·m",
                            delta=f"{((Mu_kNm - phi_Mn_max)/phi_Mn_max*100):.1f}%",
                            delta_color="inverse"
                        )
                        st.caption("Exceeds section capacity")
                        st.markdown('</div>', unsafe_allow_html=True)
    
    else:
        # Initial State - Instructions
        st.markdown('<div class="section-header">📖 How to Use This Application</div>', 
                   unsafe_allow_html=True)
        
        st.markdown("""
            <div class="info-box">
                <h3 style="margin-top: 0;">Step-by-Step Design Process</h3>
                <ol style="font-size: 1.05rem; line-height: 1.8;">
                    <li><b>Material Properties:</b> Enter concrete strength (f'c) and steel yield strength (fy)</li>
                    <li><b>Beam Geometry:</b> Specify beam width (b), height (h), and concrete cover</li>
                    <li><b>Design Loads:</b> Input the factored design moment (Mu) from your structural analysis</li>
                    <li><b>Reinforcement Details:</b> Select main bar diameter and stirrup size</li>
                    <li><b>Run Analysis:</b> Click "PERFORM DESIGN ANALYSIS" button</li>
                    <li><b>Review Results:</b> Examine design calculations, adequacy checks, and visualizations</li>
                    <li><b>Select Bars:</b> Choose appropriate bar arrangement from suggestions</li>
                    <li><b>Export:</b> Download design summary and results for documentation</li>
                </ol>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown('<div class="section-header">📚 Design Methodology</div>', 
                   unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
                <div class="info-box">
                    <h4>🔬 ACI 318-19 Provisions</h4>
                    <p><b>Design Code Sections:</b></p>
                    <ul>
                        <li><b>Section 22.2:</b> Flexural strength requirements</li>
                        <li><b>Section 9.6:</b> Minimum reinforcement</li>
                        <li><b>Section 21.2:</b> Strength reduction factors (φ)</li>
                        <li><b>Section 22.2.2.4.3:</b> β₁ factor calculation</li>
                    </ul>
                    
                    <p><b>Key Equations:</b></p>
                    <ul>
                        <li>As = ρ × b × d</li>
                        <li>Mn = As × fy × (d - a/2)</li>
                        <li>a = As × fy / (0.85 × f'c × b)</li>
                        <li>φMn ≥ Mu</li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
                <div class="warning-box">
                    <h4>⚠️ Design Limitations</h4>
                    <ul>
                        <li>Flexural design only (no shear design)</li>
                        <li>Singly reinforced rectangular sections</li>
                        <li>Serviceability not checked (deflection, cracks)</li>
                        <li>Development length verification required</li>
                        <li>Constructability and detailing not included</li>
                    </ul>
                    
                    <h4 style="margin-top: 1rem;">✅ Design Checks Performed</h4>
                    <ul>
                        <li>✓ Minimum reinforcement ratio (ρ_min)</li>
                        <li>✓ Maximum reinforcement ratio (ρ_max)</li>
                        <li>✓ Strain compatibility</li>
                        <li>✓ Moment capacity adequacy</li>
                        <li>✓ Bar spacing requirements</li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown('<div class="section-header">💡 Best Practices</div>', 
                   unsafe_allow_html=True)
        
        st.markdown("""
            <div class="success-box">
                <h4 style="margin-top: 0;">Professional Engineering Practice</h4>
                <ul style="font-size: 1.05rem; line-height: 1.8;">
                    <li>✅ <b>Verify Results:</b> Always check calculations with hand calculations or alternative software</li>
                    <li>✅ <b>Local Codes:</b> Ensure compliance with local building codes and amendments</li>
                    <li>✅ <b>Constructability:</b> Consider practical aspects like bar spacing, congestion, and placement</li>
                    <li>✅ <b>Development Length:</b> Verify adequate development and splice lengths separately</li>
                    <li>✅ <b>Shear Design:</b> Perform complete shear and torsion design as required</li>
                    <li>✅ <b>Serviceability:</b> Check deflection limits and crack width control</li>
                    <li>✅ <b>Detailing:</b> Prepare complete reinforcement details per ACI 315</li>
                    <li>✅ <b>Professional Review:</b> Have designs reviewed by licensed structural engineer</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
        
        # Sample Design Example
        st.markdown('<div class="section-header">📋 Example Design Problem</div>', 
                   unsafe_allow_html=True)
        
        st.markdown("""
            <div class="info-box">
                <h4 style="margin-top: 0;">Sample Input Values</h4>
                <p>Try these values to see a typical design:</p>
                <ul>
                    <li><b>Concrete Strength:</b> f'c = 25 MPa (3625 psi)</li>
                    <li><b>Steel Strength:</b> fy = 420 MPa (Grade 60)</li>
                    <li><b>Beam Width:</b> b = 300 mm (12 inches)</li>
                    <li><b>Beam Height:</b> h = 600 mm (24 inches)</li>
                    <li><b>Cover:</b> 40 mm (1.6 inches)</li>
                    <li><b>Factored Moment:</b> Mu = 150 kN·m</li>
                    <li><b>Main Bars:</b> 20 mm diameter</li>
                    <li><b>Stirrups:</b> 10 mm diameter</li>
                </ul>
                <p style="margin-bottom: 0;"><b>Expected Result:</b> As ≈ 1100-1200 mm² (3-4 bars of 20mm)</p>
            </div>
        """, unsafe_allow_html=True)

# ============================================================================
# RUN APPLICATION
# ============================================================================

if __name__ == "__main__":
    main()
