import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
from io import BytesIO

# ===========================
# PAGE CONFIGURATION
# ===========================
st.set_page_config(
    page_title="RC Beam Designer - ACI 318",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===========================
# CUSTOM CSS STYLING
# ===========================
st.markdown("""
    <style>
    .main-header {
        text-align: center;
        color: #1f77b4;
        font-size: 2.5em;
        font-weight: bold;
        padding: 20px;
        background: linear-gradient(90deg, #e3f2fd 0%, #bbdefb 100%);
        border-radius: 10px;
        margin-bottom: 30px;
    }
    .success-box {
        padding: 15px;
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        border-radius: 5px;
        margin: 10px 0;
    }
    .error-box {
        padding: 15px;
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
        border-radius: 5px;
        margin: 10px 0;
    }
    .warning-box {
        padding: 15px;
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
        border-radius: 5px;
        margin: 10px 0;
    }
    .info-box {
        padding: 15px;
        background-color: #d1ecf1;
        border-left: 5px solid #17a2b8;
        border-radius: 5px;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

# ===========================
# HEADER
# ===========================
st.markdown('<div class="main-header">🏗️ Reinforced Concrete Beam Designer<br><small style="font-size:0.5em;">ACI 318 Standard</small></div>', unsafe_allow_html=True)

# ===========================
# UNIT CONVERSION FUNCTIONS
# ===========================
def mpa_to_ksc(mpa):
    """Convert MPa to kg/cm² (ksc)"""
    return mpa * 10.197162

def ksc_to_mpa(ksc):
    """Convert kg/cm² to MPa"""
    return ksc / 10.197162

def knm_to_tonfm(knm):
    """Convert kN·m to tonf·m"""
    return knm / 9.80665

def tonfm_to_knm(tonfm):
    """Convert tonf·m to kN·m"""
    return tonfm * 9.80665

# ===========================
# SIDEBAR INPUTS
# ===========================
st.sidebar.header("📊 Input Parameters")

st.sidebar.subheader("Material Properties")
fc_mpa = st.sidebar.number_input(
    "Concrete Strength fc' (MPa)", 
    min_value=15.0, 
    max_value=100.0, 
    value=25.0, 
    step=5.0,
    help="Concrete compressive strength"
)
fc_ksc = mpa_to_ksc(fc_mpa)

fy_mpa = st.sidebar.number_input(
    "Steel Yield Strength fy (MPa)", 
    min_value=200.0, 
    max_value=600.0, 
    value=420.0, 
    step=20.0,
    help="Reinforcement steel yield strength"
)
fy_ksc = mpa_to_ksc(fy_mpa)

st.sidebar.subheader("Beam Dimensions")
b = st.sidebar.number_input(
    "Beam Width b (mm)", 
    min_value=100.0, 
    max_value=1000.0, 
    value=300.0, 
    step=50.0
)

h = st.sidebar.number_input(
    "Beam Height h (mm)", 
    min_value=200.0, 
    max_value=2000.0, 
    value=500.0, 
    step=50.0
)

cover = st.sidebar.number_input(
    "Concrete Cover (mm)", 
    min_value=20.0, 
    max_value=100.0, 
    value=40.0, 
    step=5.0,
    help="Clear cover to stirrups"
)

st.sidebar.subheader("Loading")
Mu_knm = st.sidebar.number_input(
    "Factored Moment Mu (kN·m)", 
    min_value=0.0, 
    max_value=1000.0, 
    value=150.0, 
    step=10.0,
    help="Ultimate design moment"
)
Mu_tonfm = knm_to_tonfm(Mu_knm)

st.sidebar.subheader("Reinforcement Details")
show_advanced = st.sidebar.checkbox("Show Advanced Options", value=True)

if show_advanced:
    bar_diameter = st.sidebar.selectbox(
        "Bar Diameter (mm)",
        options=[10, 12, 16, 20, 25, 28, 32],
        index=3,
        help="Select reinforcement bar diameter"
    )
    
    stirrup_dia = st.sidebar.selectbox(
        "Stirrup Diameter (mm)",
        options=[6, 8, 10, 12],
        index=1,
        help="Select stirrup bar diameter"
    )
else:
    bar_diameter = 20
    stirrup_dia = 8

# ===========================
# DESIGN CALCULATIONS
# ===========================

def calculate_effective_depth(h, cover, stirrup_dia, bar_dia):
    """Calculate effective depth"""
    d = h - cover - stirrup_dia - bar_dia/2
    return d

def calculate_min_reinforcement_ratio(fc_ksc, fy_ksc):
    """Calculate minimum reinforcement ratio per ACI 318"""
    rho_min_1 = np.sqrt(fc_ksc) / (4 * fy_ksc)
    rho_min_2 = 1.4 / fy_ksc
    rho_min = max(rho_min_1, rho_min_2)
    return rho_min

def calculate_max_reinforcement_ratio(fc_ksc, fy_ksc, beta1):
    """Calculate maximum reinforcement ratio for tension-controlled section"""
    # Tension-controlled limit: εt = 0.005
    # c/d = 0.003/(0.003 + εt) = 0.375
    c_d_max = 0.375
    rho_max = 0.85 * beta1 * fc_ksc / fy_ksc * c_d_max
    return rho_max

def get_beta1(fc_ksc):
    """Calculate beta1 factor per ACI 318"""
    fc_mpa = ksc_to_mpa(fc_ksc)
    if fc_mpa <= 28:
        beta1 = 0.85
    elif fc_mpa >= 56:
        beta1 = 0.65
    else:
        beta1 = 0.85 - 0.05 * (fc_mpa - 28) / 7
    return beta1

def design_beam(fc_ksc, fy_ksc, b, d, Mu_tonfm):
    """
    Design reinforced concrete beam using ACI 318
    
    Parameters:
    - fc_ksc: Concrete strength in kg/cm²
    - fy_ksc: Steel yield strength in kg/cm²
    - b: Beam width in mm
    - d: Effective depth in mm
    - Mu_tonfm: Factored moment in tonf·m
    
    Returns:
    - Dictionary with design results
    """
    
    # Convert units for calculation
    b_cm = b / 10  # mm to cm
    d_cm = d / 10  # mm to cm
    Mu_kgcm = Mu_tonfm * 1000 * 100  # tonf·m to kg·cm
    
    # ACI 318 parameters
    phi = 0.9  # Strength reduction factor for tension-controlled
    beta1 = get_beta1(fc_ksc)
    
    # Calculate reinforcement ratios
    rho_min = calculate_min_reinforcement_ratio(fc_ksc, fy_ksc)
    rho_max = calculate_max_reinforcement_ratio(fc_ksc, fy_ksc, beta1)
    
    # Calculate required reinforcement using quadratic equation
    # Mu = φ * As * fy * (d - a/2)
    # a = As * fy / (0.85 * fc' * b)
    # Solving: Mu = φ * As * fy * (d - As*fy/(1.7*fc'*b))
    
    # Rearranging: As²*fy/(1.7*fc'*b) - As*d + Mu/φ = 0
    # Coefficient form: A*As² + B*As + C = 0
    
    A = fy_ksc / (1.7 * fc_ksc * b_cm)
    B = -d_cm
    C = Mu_kgcm / phi
    
    # Solve quadratic equation
    discriminant = B**2 - 4*A*C
    
    if discriminant < 0:
        return {
            'success': False,
            'message': 'Section is inadequate - moment too large for given dimensions',
            'As_required': None
        }
    
    As_required_cm2 = (-B - np.sqrt(discriminant)) / (2*A)
    
    # Check minimum reinforcement
    As_min_cm2 = rho_min * b_cm * d_cm
    As_required_cm2 = max(As_required_cm2, As_min_cm2)
    
    # Check maximum reinforcement
    As_max_cm2 = rho_max * b_cm * d_cm
    
    if As_required_cm2 > As_max_cm2:
        return {
            'success': False,
            'message': 'Required reinforcement exceeds maximum - use compression steel or increase section',
            'As_required': As_required_cm2,
            'As_max': As_max_cm2
        }
    
    # Calculate actual values
    rho_actual = As_required_cm2 / (b_cm * d_cm)
    a_cm = As_required_cm2 * fy_ksc / (0.85 * fc_ksc * b_cm)
    c_cm = a_cm / beta1
    
    # Check if tension-controlled
    epsilon_t = 0.003 * (d_cm - c_cm) / c_cm
    
    if epsilon_t >= 0.005:
        section_type = "Tension-controlled (φ = 0.9)"
        phi_actual = 0.9
    elif epsilon_t >= 0.004:
        section_type = "Transition zone"
        phi_actual = 0.65 + (epsilon_t - 0.002) * (200/3)
    else:
        section_type = "Compression-controlled"
        phi_actual = 0.65
    
    # Calculate nominal and design moment capacity
    Mn_kgcm = As_required_cm2 * fy_ksc * (d_cm - a_cm/2)
    Mn_tonfm = Mn_kgcm / (1000 * 100)
    phi_Mn_tonfm = phi_actual * Mn_tonfm
    
    # Design adequacy
    adequacy_ratio = phi_Mn_tonfm / Mu_tonfm
    is_adequate = adequacy_ratio >= 1.0
    
    return {
        'success': True,
        'is_adequate': is_adequate,
        'As_required': As_required_cm2,
        'As_min': As_min_cm2,
        'As_max': As_max_cm2,
        'rho_actual': rho_actual,
        'rho_min': rho_min,
        'rho_max': rho_max,
        'a': a_cm,
        'c': c_cm,
        'epsilon_t': epsilon_t,
        'section_type': section_type,
        'phi_actual': phi_actual,
        'Mn': Mn_tonfm,
        'phi_Mn': phi_Mn_tonfm,
        'adequacy_ratio': adequacy_ratio,
        'beta1': beta1
    }

def calculate_number_of_bars(As_required_cm2, bar_dia_mm):
    """Calculate number of bars required"""
    bar_area_cm2 = np.pi * (bar_dia_mm/10)**2 / 4
    num_bars = np.ceil(As_required_cm2 / bar_area_cm2)
    As_provided_cm2 = num_bars * bar_area_cm2
    return int(num_bars), As_provided_cm2, bar_area_cm2

# ===========================
# PERFORM DESIGN
# ===========================

# Calculate effective depth
d = calculate_effective_depth(h, cover, stirrup_dia, bar_diameter)
d_cm = d / 10

# Perform design
results = design_beam(fc_ksc, fy_ksc, b, d, Mu_tonfm)

# Calculate bar details if design is successful
if results['success'] and results.get('As_required'):
    num_bars, As_provided_cm2, bar_area_cm2 = calculate_number_of_bars(
        results['As_required'], 
        bar_diameter
    )
else:
    num_bars = 0
    As_provided_cm2 = 0
    bar_area_cm2 = 0

# ===========================
# DISPLAY RESULTS
# ===========================

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📋 Design Summary")
    
    # Display status message
    if not results['success']:
        st.markdown(f'<div class="error-box">❌ <strong>Design Failed:</strong> {results["message"]}</div>', 
                   unsafe_allow_html=True)
    elif results['is_adequate']:
        st.markdown('<div class="success-box">✅ <strong>Design is ADEQUATE</strong> - Section capacity exceeds demand</div>', 
                   unsafe_allow_html=True)
    else:
        st.markdown('<div class="error-box">❌ <strong>Design is INADEQUATE</strong> - Increase section size or concrete strength</div>', 
                   unsafe_allow_html=True)
    
    # Create results dataframe
    if results['success'] and results.get('As_required'):
        
        # Input parameters table
        st.markdown("**Input Parameters:**")
        input_data = {
            'Parameter': [
                'Concrete Strength fc\'',
                'Steel Yield Strength fy',
                'Beam Width b',
                'Beam Height h',
                'Effective Depth d',
                'Concrete Cover',
                'Factored Moment Mu'
            ],
            'Value': [
                f'{fc_mpa:.1f} MPa ({fc_ksc:.1f} ksc)',
                f'{fy_mpa:.0f} MPa ({fy_ksc:.1f} ksc)',
                f'{b:.0f} mm',
                f'{h:.0f} mm',
                f'{d:.1f} mm ({d_cm:.2f} cm)',
                f'{cover:.0f} mm',
                f'{Mu_knm:.2f} kN·m ({Mu_tonfm:.3f} tonf·m)'
            ]
        }
        st.dataframe(pd.DataFrame(input_data), hide_index=True, use_container_width=True)
        
        st.markdown("---")
        
        # Design results table
        st.markdown("**Design Results:**")
        results_data = {
            'Parameter': [
                'Required Steel Area As',
                'Minimum Steel Area As,min',
                'Maximum Steel Area As,max',
                'Reinforcement Ratio ρ',
                'Section Type',
                'Strength Reduction φ',
                'Neutral Axis Depth c',
                'Compression Block Depth a',
                'Tensile Strain εt',
                'Nominal Moment Mn',
                'Design Moment φMn',
                'Demand/Capacity Ratio'
            ],
            'Value': [
                f'{results["As_required"]:.2f} cm²',
                f'{results["As_min"]:.2f} cm²',
                f'{results["As_max"]:.2f} cm²',
                f'{results["rho_actual"]:.4f}',
                results['section_type'],
                f'{results["phi_actual"]:.3f}',
                f'{results["c"]:.2f} cm',
                f'{results["a"]:.2f} cm',
                f'{results["epsilon_t"]:.5f}',
                f'{results["Mn"]:.3f} tonf·m',
                f'{results["phi_Mn"]:.3f} tonf·m',
                f'{results["adequacy_ratio"]:.2f}'
            ]
        }
        st.dataframe(pd.DataFrame(results_data), hide_index=True, use_container_width=True)
        
        # Check reinforcement ratio limits
        if results['rho_actual'] < results['rho_min']:
            st.markdown('<div class="warning-box">⚠️ Using minimum reinforcement</div>', 
                       unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Bar arrangement
        st.markdown("**Reinforcement Details:**")
        bar_data = {
            'Parameter': [
                'Bar Diameter',
                'Bar Area (each)',
                'Number of Bars',
                'Total Provided Area As,prov',
                'Stirrup Diameter'
            ],
            'Value': [
                f'{bar_diameter} mm',
                f'{bar_area_cm2:.2f} cm²',
                f'{num_bars} bars',
                f'{As_provided_cm2:.2f} cm²',
                f'{stirrup_dia} mm'
            ]
        }
        st.dataframe(pd.DataFrame(bar_data), hide_index=True, use_container_width=True)
        
        # Check bar spacing
        clear_spacing = (b - 2*cover - 2*stirrup_dia - num_bars*bar_diameter) / (num_bars - 1) if num_bars > 1 else 0
        
        if clear_spacing < bar_diameter:
            st.markdown('<div class="warning-box">⚠️ <strong>Warning:</strong> Bar spacing is tight. Consider increasing beam width or using smaller bars.</div>', 
                       unsafe_allow_html=True)
        elif clear_spacing < 25:
            st.markdown('<div class="warning-box">⚠️ <strong>Warning:</strong> Clear spacing less than 25mm (ACI minimum for aggregate placement).</div>', 
                       unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="info-box">ℹ️ Clear spacing between bars: {clear_spacing:.1f} mm</div>', 
                       unsafe_allow_html=True)

with col2:
    st.subheader("📐 Beam Cross-Section")
    
    if results['success'] and results.get('As_required'):
        # Create beam cross-section diagram
        fig, ax = plt.subplots(figsize=(8, 10))
        
        # Draw beam outline
        beam_rect = patches.Rectangle((0, 0), b, h, 
                                      linewidth=2, 
                                      edgecolor='black', 
                                      facecolor='lightgray',
                                      alpha=0.3)
        ax.add_patch(beam_rect)
        
        # Draw compression zone
        a_mm = results['a'] * 10
        comp_zone = patches.Rectangle((0, h - a_mm), b, a_mm,
                                      linewidth=1,
                                      edgecolor='blue',
                                      facecolor='lightblue',
                                      alpha=0.5,
                                      linestyle='--',
                                      label='Compression zone (a)')
        ax.add_patch(comp_zone)
        
        # Draw neutral axis
        c_mm = results['c'] * 10
        ax.plot([0, b], [h - c_mm, h - c_mm], 
               'b--', linewidth=1.5, alpha=0.7, label='Neutral axis (c)')
        
        # Draw effective depth
        ax.plot([0, b], [h - d, h - d], 
               'r--', linewidth=1, alpha=0.5, label='Effective depth (d)')
        
        # Draw cover lines
        ax.plot([cover, b - cover], [cover, cover], 'g--', linewidth=1, alpha=0.5)
        
        # Draw reinforcement bars
        bar_spacing = (b - 2*cover - 2*stirrup_dia - num_bars*bar_diameter) / (num_bars - 1) if num_bars > 1 else 0
        bar_y = cover + stirrup_dia + bar_diameter/2
        
        for i in range(num_bars):
            bar_x = cover + stirrup_dia + bar_diameter/2 + i * (bar_diameter + bar_spacing)
            circle = patches.Circle((bar_x, bar_y), 
                                   bar_diameter/2,
                                   linewidth=2,
                                   edgecolor='darkred',
                                   facecolor='red',
                                   alpha=0.8)
            ax.add_patch(circle)
        
        # Draw stirrups
        stirrup_outline = patches.Rectangle(
            (cover + stirrup_dia/2, cover + stirrup_dia/2),
            b - 2*cover - stirrup_dia,
            h - 2*cover - stirrup_dia,
            linewidth=2,
            edgecolor='green',
            facecolor='none',
            linestyle='-',
            label='Stirrups'
        )
        ax.add_patch(stirrup_outline)
        
        # Add dimensions
        ax.annotate('', xy=(b + 30, 0), xytext=(b + 30, h),
                   arrowprops=dict(arrowstyle='<->', color='black', lw=1.5))
        ax.text(b + 60, h/2, f'h = {h:.0f} mm', rotation=90, va='center', fontsize=10, fontweight='bold')
        
        ax.annotate('', xy=(0, -30), xytext=(b, -30),
                   arrowprops=dict(arrowstyle='<->', color='black', lw=1.5))
        ax.text(b/2, -60, f'b = {b:.0f} mm', ha='center', fontsize=10, fontweight='bold')
        
        # Add cover dimension
        ax.annotate('', xy=(-20, 0), xytext=(-20, cover),
                   arrowprops=dict(arrowstyle='<->', color='green', lw=1))
        ax.text(-40, cover/2, f'{cover:.0f}', rotation=90, va='center', fontsize=8, color='green')
        
        # Add effective depth dimension
        ax.annotate('', xy=(b + 80, h), xytext=(b + 80, h - d),
                   arrowprops=dict(arrowstyle='<->', color='red', lw=1))
        ax.text(b + 110, h - d/2, f'd = {d:.1f} mm', rotation=90, va='center', fontsize=9, color='red')
        
        # Add compression zone dimension
        ax.text(b/2, h - a_mm/2, f'a = {a_mm:.1f} mm', 
               ha='center', va='center', fontsize=9, 
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        # Add reinforcement label
        ax.text(b/2, bar_y - bar_diameter - 15, 
               f'{num_bars}φ{bar_diameter}', 
               ha='center', fontsize=10, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='white', edgecolor='red', linewidth=2))
        
        # Set axis properties
        ax.set_xlim(-100, b + 150)
        ax.set_ylim(-100, h + 50)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.legend(loc='upper right', fontsize=9)
        
        st.pyplot(fig)
        plt.close()
        
        # Add strain diagram
        st.markdown("---")
        st.markdown("**Strain Diagram:**")
        
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        
        # Strain profile
        strain_top = 0.003  # Concrete crushing strain
        strain_bottom = results['epsilon_t']
        
        # Draw section representation
        ax2.fill_between([0, 2], [0, 0], [h, h], color='lightgray', alpha=0.3, label='Section')
        ax2.plot([0, 2], [h - c_mm, h - c_mm], 'b--', linewidth=2, label='Neutral axis')
        ax2.plot([0, 2], [h - d, h - d], 'r--', linewidth=1, label='Steel level')
        
        # Draw strain diagram
        ax2.plot([3, 3 + strain_top*10000], [h, h], 'b-', linewidth=2)
        ax2.plot([3, 3 - strain_bottom*10000], [h - d, h - d], 'r-', linewidth=2)
        ax2.plot([3 + strain_top*10000, 3 - strain_bottom*10000], [h, h - d], 'k-', linewidth=2)
        
        # Fill strain diagram
        strain_x = [3, 3 + strain_top*10000, 3 - strain_bottom*10000, 3]
        strain_y = [h - c_mm, h, h - d, h - c_mm]
        ax2.fill(strain_x, strain_y, color='yellow', alpha=0.3)
        
        # Add labels
        ax2.text(3 + strain_top*10000 + 0.5, h, f'εc = {strain_top:.4f}', fontsize=10, va='bottom')
        ax2.text(3 - strain_bottom*10000 - 0.5, h - d, f'εt = {strain_bottom:.4f}', fontsize=10, va='top', ha='right')
        ax2.text(1, h + 20, 'Section', ha='center', fontsize=11, fontweight='bold')
        ax2.text(3 + strain_top*5000, h + 20, 'Strain', ha='center', fontsize=11, fontweight='bold')
        
        ax2.set_xlim(-1, 3 + max(strain_top, abs(strain_bottom))*10000 + 2)
        ax2.set_ylim(-50, h + 50)
        ax2.set_aspect('equal')
        ax2.axis('off')
        ax2.legend(loc='lower right', fontsize=9)
        
        st.pyplot(fig2)
        plt.close()

# ===========================
# ADDITIONAL INFORMATION
# ===========================
st.markdown("---")

with st.expander("ℹ️ Design Information & Assumptions"):
    st.markdown("""
    ### ACI 318 Design Methodology
    
    This application designs reinforced concrete beams following ACI 318 provisions:
    
    **Key Design Equations:**
    - Effective depth: `d = h - cover - stirrup_dia - bar_dia/2`
    - Neutral axis depth: `c = a / β₁`
    - Compression block depth: `a = As·fy / (0.85·fc'·b)`
    - Nominal moment capacity: `Mn = As·fy·(d - a/2)`
    - Design moment capacity: `φMn` where φ = 0.9 for tension-controlled sections
    
    **Reinforcement Limits:**
    - Minimum: `ρmin = max(√fc'/4fy, 1.4/fy)`
    - Maximum: Based on tension-controlled limit (εt ≥ 0.005)
    
    **Section Classification:**
    - Tension-controlled: εt ≥ 0.005 (φ = 0.9)
    - Transition zone: 0.004 ≤ εt < 0.005
    - Compression-controlled: εt < 0.004 (φ = 0.65)
    
    **Design Check:**
    - Section is adequate if: `φMn ≥ Mu`
    
    **Assumptions:**
    - Singly reinforced rectangular section
    - Concrete cover measured to stirrups
    - Normal weight concrete
    - Grade 60 steel or equivalent
    - Single layer of tension reinforcement
    
    **Units:**
    - Section dimensions: mm
    - Area: cm²
    - Stress: kg/cm² (ksc) and MPa
    - Force: kg and kN
    - Moment: tonf·m and kN·m
    """)

with st.expander("📖 How to Use This App"):
    st.markdown("""
    1. **Enter Material Properties:** Input concrete and steel strengths
    2. **Define Beam Geometry:** Specify width, height, and cover
    3. **Input Loading:** Enter the factored moment (Mu)
    4. **Review Results:** Check if design is adequate
    5. **Adjust Parameters:** Modify dimensions if needed to achieve adequate design
    
    **Tips:**
    - Increase beam depth for higher moment capacity
    - Ensure clear spacing between bars ≥ 25mm
    - For inadequate sections, try:
        - Increasing beam height (h)
        - Increasing beam width (b)
        - Using higher strength concrete
    """)

# ===========================
# FOOTER
# ===========================
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: gray; padding: 20px;'>
        <p>🏗️ <strong>RC Beam Designer v1.0</strong></p>
        <p>Designed according to ACI 318 Building Code Requirements for Structural Concrete</p>
        <p><em>⚠️ For educational and preliminary design purposes only. Professional engineer review required for actual construction.</em></p>
    </div>
    """, unsafe_allow_html=True)
