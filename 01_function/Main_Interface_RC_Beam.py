import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle, Circle
import pandas as pd
from datetime import datetime
import io
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

# ===========================
# PAGE CONFIGURATION
# ===========================
st.set_page_config(
    page_title="RC Beam Designer Pro - ACI 318",
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
        background-color: #f5f7fa;
    }
    
    /* Header styling */
    .main-header {
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    
    .main-header h1 {
        font-size: 2.5em;
        margin: 0;
        font-weight: 700;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    .main-header p {
        font-size: 1.1em;
        margin-top: 0.5rem;
        opacity: 0.95;
    }
    
    /* Card containers */
    .result-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        margin-bottom: 1.5rem;
        border-left: 4px solid #667eea;
    }
    
    /* Status boxes */
    .status-adequate {
        background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
        color: #1a5e3a;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-weight: 600;
        font-size: 1.1em;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(132,250,176,0.3);
    }
    
    .status-inadequate {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        color: #8b0000;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-weight: 600;
        font-size: 1.1em;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(250,112,154,0.3);
    }
    
    /* Value displays */
    .metric-box {
        background: linear-gradient(135deg, #e0e5ec 0%, #f5f7fa 100%);
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        color: #64748b;
        font-size: 0.9em;
        font-weight: 500;
    }
    
    .metric-value {
        color: #1e293b;
        font-size: 1.8em;
        font-weight: 700;
        margin: 0.2rem 0;
    }
    
    .metric-unit {
        color: #64748b;
        font-size: 0.9em;
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background-color: white;
        padding: 1rem;
        border-radius: 10px;
    }
    
    /* Input styling */
    .stNumberInput > div > div > input {
        border-radius: 8px;
        border: 2px solid #e2e8f0;
        padding: 0.5rem;
        font-size: 1rem;
    }
    
    .stNumberInput > div > div > input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 2px rgba(102,126,234,0.1);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        font-size: 1.1em;
        font-weight: 600;
        border-radius: 10px;
        box-shadow: 0 4px 15px rgba(102,126,234,0.3);
        transition: all 0.3s ease;
        width: 100%;
        margin-top: 1rem;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102,126,234,0.4);
    }
    
    /* Tables */
    .dataframe {
        font-size: 0.95em;
    }
    
    /* Section headers */
    h2 {
        color: #1e293b;
        font-weight: 700;
        margin-top: 2rem;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #667eea;
    }
    
    h3 {
        color: #334155;
        font-weight: 600;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    
    /* Info boxes */
    .info-box {
        background: #eff6ff;
        border-left: 4px solid #3b82f6;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .warning-box {
        background: #fef3c7;
        border-left: 4px solid #f59e0b;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

# ===========================
# HEADER
# ===========================
st.markdown("""
    <div class="main-header">
        <h1>🏗️ RC Beam Designer Pro</h1>
        <p>Advanced Reinforced Concrete Beam Design per ACI 318</p>
    </div>
    """, unsafe_allow_html=True)

# ===========================
# HELPER FUNCTIONS
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

def design_beam(fc_ksc, fy_ksc, b_mm, h_mm, cover_mm, Mu_tonfm, bar_dia_mm=20):
    """
    Design RC beam per ACI 318 using consistent units
    
    Units:
    - fc_ksc, fy_ksc: Strength in ksc (kg/cm²)
    - b_mm, h_mm, cover_mm: Dimensions in mm
    - Mu_tonfm: Moment in tonf·m
    - bar_dia_mm: Bar diameter in mm
    
    Returns: Dictionary with design results
    """
    
    # Calculate effective depth
    d_mm = h_mm - cover_mm - bar_dia_mm/2  # Simplified assumption
    
    # Unit conversions for internal calculations
    # Convert to consistent N and mm units
    fc_N_mm2 = fc_ksc * 0.0980665  # ksc to N/mm² (MPa)
    fy_N_mm2 = fy_ksc * 0.0980665  # ksc to N/mm² (MPa)
    Mu_Nmm = Mu_tonfm * 9.80665e6  # tonf·m to N·mm
    
    # ACI 318 parameters
    phi = 0.9  # Strength reduction factor for flexure
    beta1 = calculate_beta1(fc_ksc)
    
    # Calculate required steel area using quadratic equation
    # Mu = φ * As * fy * (d - a/2)
    # where a = As * fy / (0.85 * fc * b)
    
    # Rearranging into quadratic: As² * A + As * B + C = 0
    A = fy_N_mm2**2 / (1.7 * fc_N_mm2 * b_mm)
    B = -fy_N_mm2 * d_mm
    C = Mu_Nmm / phi
    
    discriminant = B**2 - 4*A*C
    
    if discriminant < 0:
        return {
            'success': False,
            'message': 'Section inadequate - increase dimensions or strength',
            'As_mm2': None
        }
    
    # Solve for As (take smaller positive root)
    As_mm2 = (-B - np.sqrt(discriminant)) / (2*A)
    
    # Check minimum reinforcement per ACI 318
    As_min_mm2 = max(
        3 * np.sqrt(fc_N_mm2) / fy_N_mm2 * b_mm * d_mm,
        200 / fy_N_mm2 * b_mm * d_mm  # 200 psi = 1.38 MPa
    )
    
    As_required_mm2 = max(As_mm2, As_min_mm2)
    
    # Calculate actual values
    a_mm = As_required_mm2 * fy_N_mm2 / (0.85 * fc_N_mm2 * b_mm)
    c_mm = a_mm / beta1
    
    # Check if tension-controlled (εt ≥ 0.005)
    epsilon_t = 0.003 * (d_mm - c_mm) / c_mm
    is_tension_controlled = epsilon_t >= 0.005
    
    # Calculate moment capacity
    Mn_Nmm = As_required_mm2 * fy_N_mm2 * (d_mm - a_mm/2)
    Mn_tonfm = Mn_Nmm / 9.80665e6  # Convert back to tonf·m
    phi_Mn_tonfm = phi * Mn_tonfm
    
    # Check adequacy
    is_adequate = phi_Mn_tonfm >= Mu_tonfm
    
    # Calculate reinforcement ratio
    rho = As_required_mm2 / (b_mm * d_mm)
    rho_balanced = 0.85 * beta1 * fc_N_mm2 / fy_N_mm2 * 600 / (600 + fy_N_mm2)
    
    return {
        'success': True,
        'As_mm2': As_required_mm2,
        'As_min_mm2': As_min_mm2,
        'd_mm': d_mm,
        'a_mm': a_mm,
        'c_mm': c_mm,
        'epsilon_t': epsilon_t,
        'is_tension_controlled': is_tension_controlled,
        'Mn_tonfm': Mn_tonfm,
        'phi_Mn_tonfm': phi_Mn_tonfm,
        'is_adequate': is_adequate,
        'rho': rho,
        'rho_balanced': rho_balanced,
        'beta1': beta1
    }

def calculate_bar_arrangement(As_required_mm2, bar_dia_mm):
    """Calculate number of bars and provided area"""
    bar_area_mm2 = np.pi * bar_dia_mm**2 / 4
    num_bars = int(np.ceil(As_required_mm2 / bar_area_mm2))
    As_provided_mm2 = num_bars * bar_area_mm2
    
    return {
        'num_bars': num_bars,
        'As_provided_mm2': As_provided_mm2,
        'bar_area_mm2': bar_area_mm2,
        'excess_percent': (As_provided_mm2 - As_required_mm2) / As_required_mm2 * 100
    }

def draw_beam_section(b_mm, h_mm, cover_mm, bar_dia_mm, num_bars, results):
    """Create beam cross-section diagram"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Set up the plot
    ax.set_xlim(-b_mm*0.2, b_mm*1.2)
    ax.set_ylim(-h_mm*0.2, h_mm*1.2)
    ax.set_aspect('equal')
    
    # Draw beam outline
    beam = Rectangle((0, 0), b_mm, h_mm, 
                    linewidth=3, edgecolor='#1e293b', 
                    facecolor='#e5e7eb')
    ax.add_patch(beam)
    
    # Draw compression zone
    if results.get('a_mm'):
        comp_zone = Rectangle((0, h_mm - results['a_mm']), b_mm, results['a_mm'],
                            linewidth=0, facecolor='#93c5fd', alpha=0.5)
        ax.add_patch(comp_zone)
        
        # Add compression zone label
        ax.text(b_mm/2, h_mm - results['a_mm']/2, 
               f"a = {results['a_mm']:.1f} mm",
               ha='center', va='center', fontsize=10,
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    # Draw neutral axis
    if results.get('c_mm'):
        ax.axhline(y=h_mm - results['c_mm'], color='#ef4444', 
                  linestyle='--', linewidth=2, alpha=0.7)
        ax.text(-b_mm*0.15, h_mm - results['c_mm'], 'N.A.', 
               ha='center', va='center', fontsize=10, color='#ef4444')
    
    # Calculate bar positions
    stirrup_dia = 10  # Assume 10mm stirrups
    clear_space = b_mm - 2*cover_mm - 2*stirrup_dia - num_bars*bar_dia_mm
    
    if num_bars > 1:
        spacing = clear_space / (num_bars - 1)
        bar_positions = [cover_mm + stirrup_dia + bar_dia_mm/2 + i*(bar_dia_mm + spacing) 
                        for i in range(num_bars)]
    else:
        bar_positions = [b_mm/2]
    
    bar_y = cover_mm + stirrup_dia + bar_dia_mm/2
    
    # Draw reinforcement bars
    for x in bar_positions:
        circle = Circle((x, bar_y), bar_dia_mm/2, 
                       color='#dc2626', linewidth=2, 
                       edgecolor='#7f1d1d', zorder=5)
        ax.add_patch(circle)
    
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
    
    ax.plot(stirrup_x, stirrup_y, color='#059669', linewidth=2.5)
    
    # Add dimensions
    # Width dimension
    ax.annotate('', xy=(0, -h_mm*0.1), xytext=(b_mm, -h_mm*0.1),
               arrowprops=dict(arrowstyle='<->', color='#64748b', lw=1.5))
    ax.text(b_mm/2, -h_mm*0.15, f'{b_mm} mm', 
           ha='center', va='top', fontsize=11, color='#334155')
    
    # Height dimension
    ax.annotate('', xy=(b_mm*1.1, 0), xytext=(b_mm*1.1, h_mm),
               arrowprops=dict(arrowstyle='<->', color='#64748b', lw=1.5))
    ax.text(b_mm*1.15, h_mm/2, f'{h_mm} mm', 
           ha='left', va='center', fontsize=11, color='#334155', rotation=90)
    
    # Effective depth dimension
    ax.annotate('', xy=(-b_mm*0.1, 0), xytext=(-b_mm*0.1, h_mm - results['d_mm']),
               arrowprops=dict(arrowstyle='<->', color='#ef4444', lw=2))
    ax.text(-b_mm*0.15, (h_mm - results['d_mm'])/2, f"d = {results['d_mm']:.0f} mm", 
           ha='right', va='center', fontsize=10, color='#ef4444', rotation=90)
    
    # Add reinforcement label
    ax.text(b_mm/2, bar_y + bar_dia_mm + 10, 
           f'{num_bars}Ø{bar_dia_mm}',
           ha='center', va='bottom', fontsize=12, fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='#fee2e2', 
                    edgecolor='#dc2626', linewidth=2))
    
    # Add title
    ax.text(b_mm/2, h_mm*1.15, 'Beam Cross-Section', 
           ha='center', fontsize=16, fontweight='bold', color='#1e293b')
    
    ax.axis('off')
    plt.tight_layout()
    
    return fig

def draw_strain_diagram(results, h_mm):
    """Create strain diagram"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Concrete crushing strain
    epsilon_c = 0.003
    epsilon_t = results['epsilon_t']
    c_mm = results['c_mm']
    d_mm = results['d_mm']
    
    # Set up plot
    ax.set_xlim(-0.001, max(epsilon_t, 0.006))
    ax.set_ylim(-50, h_mm + 50)
    
    # Draw beam outline (simplified)
    beam_x = [-0.0005, -0.0005]
    beam_y = [0, h_mm]
    ax.plot(beam_x, beam_y, 'k-', linewidth=20, alpha=0.3)
    
    # Draw strain profile
    strain_x = [0, epsilon_c, -epsilon_t, 0]
    strain_y = [h_mm - c_mm, h_mm, h_mm - d_mm, h_mm - c_mm]
    
    # Fill compression zone
    comp_x = [0, epsilon_c, 0, 0]
    comp_y = [h_mm - c_mm, h_mm, h_mm, h_mm - c_mm]
    ax.fill(comp_x, comp_y, color='#3b82f6', alpha=0.3, label='Compression')
    
    # Fill tension zone
    tens_x = [0, -epsilon_t, 0, 0]
    tens_y = [h_mm - c_mm, h_mm - d_mm, h_mm - d_mm, h_mm - c_mm]
    ax.fill(tens_x, tens_y, color='#ef4444', alpha=0.3, label='Tension')
    
    # Draw strain profile line
    ax.plot([epsilon_c, -epsilon_t], [h_mm, h_mm - d_mm], 
           'k-', linewidth=3, label='Strain Profile')
    
    # Draw neutral axis
    ax.axhline(y=h_mm - c_mm, color='#10b981', linestyle='--', 
              linewidth=2, label='Neutral Axis')
    ax.plot(0, h_mm - c_mm, 'ko', markersize=8)
    
    # Add strain values
    ax.text(epsilon_c + 0.0002, h_mm, f'εc = {epsilon_c:.3f}', 
           fontsize=11, ha='left', va='center',
           bbox=dict(boxstyle='round,pad=0.3', facecolor='white'))
    
    ax.text(-epsilon_t - 0.0002, h_mm - d_mm, f'εt = {epsilon_t:.4f}', 
           fontsize=11, ha='right', va='center',
           bbox=dict(boxstyle='round,pad=0.3', facecolor='white'))
    
    # Labels
    ax.set_xlabel('Strain', fontsize=12)
    ax.set_ylabel('Height (mm)', fontsize=12)
    ax.set_title('Strain Distribution', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    
    # Add vertical line at zero strain
    ax.axvline(x=0, color='k', linestyle='-', alpha=0.5)
    
    plt.tight_layout()
    return fig

def generate_pdf_report(inputs, results, bar_info):
    """Generate PDF design report"""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    story = []
    styles = getSampleStyleSheet()
    
    # Title
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#1e293b'),
        spaceAfter=30,
        alignment=1  # Center
    )
    
    story.append(Paragraph("RC Beam Design Report", title_style))
    story.append(Spacer(1, 0.25*inch))
    
    # Project info
    story.append(Paragraph(f"Design Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles['Normal']))
    story.append(Paragraph("Design Code: ACI 318", styles['Normal']))
    story.append(Spacer(1, 0.5*inch))
    
    # Input parameters section
    story.append(Paragraph("1. Input Parameters", styles['Heading2']))
    
    input_data = [
        ['Parameter', 'Value', 'Unit'],
        ['Concrete Strength (fc\')', f"{inputs['fc_ksc']:.0f}", 'ksc'],
        ['Steel Yield Strength (fy)', f"{inputs['fy_ksc']:.0f}", 'ksc'],
        ['Beam Width (b)', f"{inputs['b_mm']:.0f}", 'mm'],
        ['Beam Height (h)', f"{inputs['h_mm']:.0f}", 'mm'],
        ['Concrete Cover', f"{inputs['cover_mm']:.0f}", 'mm'],
        ['Factored Moment (Mu)', f"{inputs['Mu_tonfm']:.2f}", 'tonf·m'],
        ['Bar Diameter', f"{inputs['bar_dia_mm']:.0f}", 'mm']
    ]
    
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
    story.append(Spacer(1, 0.5*inch))
    
    # Design results section
    story.append(Paragraph("2. Design Results", styles['Heading2']))
    
    results_data = [
        ['Parameter', 'Value', 'Unit'],
        ['Effective Depth (d)', f"{results['d_mm']:.0f}", 'mm'],
        ['Required Steel Area (As)', f"{results['As_mm2']:.0f}", 'mm²'],
        ['Minimum Steel Area (As,min)', f"{results['As_min_mm2']:.0f}", 'mm²'],
        ['Compression Block Depth (a)', f"{results['a_mm']:.1f}", 'mm'],
        ['Neutral Axis Depth (c)', f"{results['c_mm']:.1f}", 'mm'],
        ['Tensile Strain (εt)', f"{results['epsilon_t']:.5f}", '-'],
        ['Reinforcement Ratio (ρ)', f"{results['rho']:.4f}", '-'],
        ['Balanced Ratio (ρb)', f"{results['rho_balanced']:.4f}", '-'],
        ['β₁ Factor', f"{results['beta1']:.3f}", '-']
    ]
    
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
    
    # Moment capacity
    story.append(Paragraph("3. Moment Capacity", styles['Heading2']))
    
    moment_data = [
        ['Description', 'Value', 'Unit'],
        ['Nominal Moment (Mn)', f"{results['Mn_tonfm']:.2f}", 'tonf·m'],
        ['Design Moment (φMn)', f"{results['phi_Mn_tonfm']:.2f}", 'tonf·m'],
        ['Applied Moment (Mu)', f"{inputs['Mu_tonfm']:.2f}", 'tonf·m'],
        ['Capacity Ratio (φMn/Mu)', f"{results['phi_Mn_tonfm']/inputs['Mu_tonfm']:.2f}", '-']
    ]
    
    t3 = Table(moment_data, colWidths=[2.5*inch, 1.5*inch, 1*inch])
    t3.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#667eea')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    story.append(t3)
    story.append(Spacer(1, 0.3*inch))
    
    # Design status
    if results['is_adequate']:
        status_color = colors.HexColor('#059669')
        status_text = "✓ DESIGN IS ADEQUATE"
    else:
        status_color = colors.HexColor('#dc2626')
        status_text = "✗ DESIGN IS INADEQUATE"
    
    status_style = ParagraphStyle(
        'StatusStyle',
        parent=styles['Normal'],
        fontSize=16,
        textColor=status_color,
        alignment=1,
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
# SIDEBAR INPUTS
# ===========================

st.sidebar.header("📋 Input Parameters")

# Material Properties
st.sidebar.subheader("🏗️ Material Properties")
col1, col2 = st.sidebar.columns(2)
with col1:
    fc_ksc = st.number_input(
        "fc' (ksc)", 
        min_value=200.0, 
        max_value=800.0, 
        value=280.0,
        step=10.0,
        help="Concrete compressive strength in ksc"
    )
with col2:
    fy_ksc = st.number_input(
        "fy (ksc)", 
        min_value=2000.0, 
        max_value=6000.0, 
        value=4200.0,
        step=100.0,
        help="Steel yield strength in ksc"
    )

# Beam Geometry
st.sidebar.subheader("📐 Beam Geometry")
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

# Loading
st.sidebar.subheader("⚡ Loading")
Mu_tonfm = st.sidebar.number_input(
    "Factored Moment Mu (tonf·m)", 
    min_value=0.0, 
    max_value=500.0, 
    value=50.0,
    step=5.0,
    help="Ultimate design moment"
)

# Reinforcement Details
st.sidebar.subheader("🔧 Reinforcement")
bar_dia_mm = st.sidebar.selectbox(
    "Bar Diameter (mm)",
    options=[10, 12, 16, 20, 25, 28, 32],
    index=3,
    help="Select main reinforcement bar diameter"
)

# Design Button
design_button = st.sidebar.button("🚀 Design Beam", use_container_width=True)

# ===========================
# MAIN CONTENT
# ===========================

if design_button:
    # Store inputs
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
    results = design_beam(fc_ksc, fy_ksc, b_mm, h_mm, cover_mm, Mu_tonfm, bar_dia_mm)
    
    if results['success']:
        # Calculate bar arrangement
        bar_info = calculate_bar_arrangement(results['As_mm2'], bar_dia_mm)
        
        # Layout columns
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("## 📊 Design Results")
            
            # Design status
            if results['is_adequate']:
                st.markdown("""<div class="status-adequate">✅ DESIGN IS ADEQUATE<br>
                φMn ≥ Mu ✓</div>""", unsafe_allow_html=True)
            else:
                st.markdown("""<div class="status-inadequate">❌ DESIGN IS INADEQUATE<br>
                φMn < Mu - Increase section or reinforcement</div>""", unsafe_allow_html=True)
            
            # Key metrics in styled boxes
            st.markdown("### Key Parameters")
            
            metric_col1, metric_col2 = st.columns(2)
            with metric_col1:
                st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-label">Effective Depth</div>
                    <div class="metric-value">{results['d_mm']:.0f}</div>
                    <div class="metric-unit">mm</div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-label">Nominal Moment</div>
                    <div class="metric-value">{results['Mn_tonfm']:.2f}</div>
                    <div class="metric-unit">tonf·m</div>
                </div>
                """, unsafe_allow_html=True)
                
            with metric_col2:
                st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-label">Required As</div>
                    <div class="metric-value">{results['As_mm2']:.0f}</div>
                    <div class="metric-unit">mm²</div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-label">Design Moment</div>
                    <div class="metric-value">{results['phi_Mn_tonfm']:.2f}</div>
                    <div class="metric-unit">tonf·m</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Detailed results table
            st.markdown("### Detailed Analysis")
            
            # Create results dataframe
            detailed_results = pd.DataFrame([
                ["Compression block depth (a)", f"{results['a_mm']:.1f} mm"],
                ["Neutral axis depth (c)", f"{results['c_mm']:.1f} mm"],
                ["Tensile strain (εt)", f"{results['epsilon_t']:.5f}"],
                ["Section type", "Tension-controlled" if results['is_tension_controlled'] else "Transition zone"],
                ["Reinforcement ratio (ρ)", f"{results['rho']:.4f}"],
                ["Balanced ratio (ρb)", f"{results['rho_balanced']:.4f}"],
                ["Min. reinforcement (As,min)", f"{results['As_min_mm2']:.0f} mm²"],
                ["β₁ factor", f"{results['beta1']:.3f}"],
                ["Capacity ratio (φMn/Mu)", f"{results['phi_Mn_tonfm']/Mu_tonfm:.2f}"]
            ], columns=["Parameter", "Value"])
            
            st.dataframe(detailed_results, hide_index=True, use_container_width=True)
            
            # Bar arrangement
            st.markdown("### 🔩 Reinforcement Details")
            st.info(f"""
            **Required**: {bar_info['num_bars']} Ø{bar_dia_mm} bars  
            **Provided Area**: {bar_info['As_provided_mm2']:.0f} mm²  
            **Excess**: {bar_info['excess_percent']:.1f}%
            """)
            
            # Check warnings
            if results['rho'] > 0.75 * results['rho_balanced']:
                st.warning("⚠️ Reinforcement ratio exceeds 75% of balanced ratio - consider increasing section")
            
            if bar_info['num_bars'] > 5:
                st.warning("⚠️ Large number of bars - check spacing and consider bundling")
        
        with col2:
            st.markdown("## 📐 Visual Analysis")
            
            # Beam cross-section
            fig1 = draw_beam_section(b_mm, h_mm, cover_mm, bar_dia_mm, 
                                   bar_info['num_bars'], results)
            st.pyplot(fig1)
            
            # Strain diagram
            if st.checkbox("Show Strain Diagram", value=True):
                fig2 = draw_strain_diagram(results, h_mm)
                st.pyplot(fig2)
        
        # Export options
        st.markdown("---")
        st.markdown("## 💾 Export Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Generate PDF report
            pdf_buffer = generate_pdf_report(inputs, results, bar_info)
            st.download_button(
                label="📄 Download PDF Report",
                data=pdf_buffer,
                file_name=f"RC_Beam_Design_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf"
            )
        
        with col2:
            # Export to CSV
            export_data = pd.DataFrame([
                ["Design Date", datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
                ["Concrete Strength (ksc)", fc_ksc],
                ["Steel Yield (ksc)", fy_ksc],
                ["Beam Width (mm)", b_mm],
                ["Beam Height (mm)", h_mm],
                ["Cover (mm)", cover_mm],
                ["Applied Moment (tonf·m)", Mu_tonfm],
                ["Effective Depth (mm)", results['d_mm']],
                ["Required As (mm²)", results['As_mm2']],
                ["Number of bars", bar_info['num_bars']],
                ["Bar diameter (mm)", bar_dia_mm],
                ["Provided As (mm²)", bar_info['As_provided_mm2']],
                ["Nominal Moment (tonf·m)", results['Mn_tonfm']],
                ["Design Moment (tonf·m)", results['phi_Mn_tonfm']],
                ["Design Status", "ADEQUATE" if results['is_adequate'] else "INADEQUATE"]
            ], columns=["Parameter", "Value"])
            
            csv = export_data.to_csv(index=False)
            st.download_button(
                label="📊 Download CSV",
                data=csv,
                file_name=f"RC_Beam_Design_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        with col3:
            # Text summary
            summary = f"""RC BEAM DESIGN SUMMARY
=====================
Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}
Code: ACI 318

INPUTS:
- Concrete: fc' = {fc_ksc} ksc
- Steel: fy = {fy_ksc} ksc  
- Section: {b_mm} x {h_mm} mm
- Cover: {cover_mm} mm
- Moment: {Mu_tonfm} tonf·m

RESULTS:
- Effective depth: {results['d_mm']:.0f} mm
- Required As: {results['As_mm2']:.0f} mm²
- Reinforcement: {bar_info['num_bars']} Ø{bar_dia_mm}
- Provided As: {bar_info['As_provided_mm2']:.0f} mm²
- φMn: {results['phi_Mn_tonfm']:.2f} tonf·m

STATUS: {'ADEQUATE' if results['is_adequate'] else 'INADEQUATE'}
"""
            st.download_button(
                label="📝 Download Summary",
                data=summary,
                file_name=f"RC_Beam_Summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )
    
    else:
        st.error(f"❌ Design Failed: {results['message']}")
        st.warning("💡 Try increasing beam dimensions or using higher strength materials")

else:
    # Welcome screen
    st.markdown("""
    <div class="result-card">
    <h2>Welcome to RC Beam Designer Pro! 👋</h2>
    
    <p>This advanced application designs reinforced concrete beams according to <strong>ACI 318</strong> 
    building code requirements.</p>
    
    <h3>🎯 Features:</h3>
    <ul>
        <li>✅ Complete flexural design per ACI 318</li>
        <li>✅ Interactive visualizations and diagrams</li>
        <li>✅ Automatic code compliance checks</li>
        <li>✅ Professional PDF reports</li>
        <li>✅ Export results in multiple formats</li>
    </ul>
    
    <h3>🚀 Quick Start:</h3>
    <ol>
        <li>Enter material properties (fc', fy) in the sidebar</li>
        <li>Define beam geometry (b, h, cover)</li>
        <li>Input the factored moment (Mu)</li>
        <li>Select bar diameter</li>
        <li>Click "Design Beam" to see results!</li>
    </ol>
    
    <div class="info-box">
    <strong>📌 Unit Convention:</strong><br>
    • Stresses: ksc (kg/cm²)<br>
    • Dimensions: mm<br>
    • Moments: tonf·m
    </div>
    
    <p><em>Start by entering your design parameters in the sidebar →</em></p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #64748b; padding: 1rem;'>
    <p>🏗️ RC Beam Designer Pro v2.0 | ACI 318 Compliant</p>
    <p style='font-size: 0.9em;'><em>For educational and preliminary design purposes. 
    Professional engineering review required for construction.</em></p>
</div>
""", unsafe_allow_html=True)
