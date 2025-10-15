import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from io import BytesIO
import math

# ============================================================================
# CONFIGURATION & STYLING
# ============================================================================

st.set_page_config(
    page_title="RC Beam Designer - ACI 318",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
        border-bottom: 3px solid #1f77b4;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #2c3e50;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #dc3545;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# HELPER FUNCTIONS - ACI 318 CALCULATIONS
# ============================================================================

def calculate_beta1(fc_mpa):
    """
    Calculate β1 factor according to ACI 318-19 (Section 22.2.2.4.3)
    
    Parameters:
    -----------
    fc_mpa : float
        Concrete compressive strength in MPa
    
    Returns:
    --------
    float : β1 factor
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
    
    Parameters:
    -----------
    fc_mpa : float
        Concrete strength in MPa
    fy_mpa : float
        Steel yield strength in MPa
    beta1 : float
        β1 factor
    
    Returns:
    --------
    float : balanced reinforcement ratio
    """
    return 0.85 * beta1 * fc_mpa / fy_mpa * (600 / (600 + fy_mpa))

def calculate_rho_min(fy_mpa):
    """
    Calculate minimum reinforcement ratio (ACI 318-19, Section 9.6.1.2)
    
    Parameters:
    -----------
    fy_mpa : float
        Steel yield strength in MPa
    
    Returns:
    --------
    float : minimum reinforcement ratio
    """
    return max(0.25 * np.sqrt(25) / fy_mpa, 1.4 / fy_mpa)  # Using fc'=25 MPa as reference

def calculate_required_steel(Mu_kNm, b_mm, d_mm, fc_mpa, fy_mpa, phi=0.9):
    """
    Calculate required steel area using ACI rectangular stress block
    
    Parameters:
    -----------
    Mu_kNm : float
        Factored moment in kN·m
    b_mm : float
        Beam width in mm
    d_mm : float
        Effective depth in mm
    fc_mpa : float
        Concrete strength in MPa
    fy_mpa : float
        Steel yield strength in MPa
    phi : float
        Strength reduction factor
    
    Returns:
    --------
    tuple : (As_mm2, a_mm, c_mm, rho, success, message)
    """
    # Convert units
    Mu_Nmm = Mu_kNm * 1e6  # kN·m to N·mm
    
    # Calculate Rn
    Rn = Mu_Nmm / (phi * b_mm * d_mm**2)
    
    # Calculate required reinforcement ratio using quadratic formula
    # Rn = ρ*fy*(1 - 0.59*ρ*fy/fc')
    # Rearranging: 0.59*fy²/fc' * ρ² - fy*ρ + Rn = 0
    
    a_coef = 0.59 * fy_mpa**2 / fc_mpa
    b_coef = -fy_mpa
    c_coef = Rn
    
    discriminant = b_coef**2 - 4*a_coef*c_coef
    
    if discriminant < 0:
        return None, None, None, None, False, "Moment too large - section cannot be designed as singly reinforced"
    
    rho1 = (-b_coef - np.sqrt(discriminant)) / (2*a_coef)
    rho2 = (-b_coef + np.sqrt(discriminant)) / (2*a_coef)
    
    # Take the smaller positive root
    rho = min(rho1, rho2) if min(rho1, rho2) > 0 else max(rho1, rho2)
    
    # Check limits
    beta1 = calculate_beta1(fc_mpa)
    rho_min = calculate_rho_min(fy_mpa)
    rho_max = 0.75 * calculate_rho_balanced(fc_mpa, fy_mpa, beta1)
    
    if rho < rho_min:
        rho = rho_min
        message = f"Using minimum reinforcement ratio (ρ = {rho:.4f})"
    elif rho > rho_max:
        return None, None, None, rho, False, f"Required ρ = {rho:.4f} exceeds maximum ρ_max = {rho_max:.4f}. Use doubly reinforced section."
    else:
        message = "Design satisfactory"
    
    # Calculate steel area
    As_mm2 = rho * b_mm * d_mm
    
    # Calculate neutral axis depth
    c_mm = As_mm2 * fy_mpa / (0.85 * fc_mpa * beta1 * b_mm)
    a_mm = beta1 * c_mm
    
    return As_mm2, a_mm, c_mm, rho, True, message

def calculate_moment_capacity(As_mm2, b_mm, d_mm, fc_mpa, fy_mpa):
    """
    Calculate nominal and design moment capacity
    
    Parameters:
    -----------
    As_mm2 : float
        Steel area in mm²
    b_mm : float
        Beam width in mm
    d_mm : float
        Effective depth in mm
    fc_mpa : float
        Concrete strength in MPa
    fy_mpa : float
        Steel yield strength in MPa
    
    Returns:
    --------
    tuple : (Mn_kNm, phi_Mn_kNm, phi, c_mm, et)
    """
    beta1 = calculate_beta1(fc_mpa)
    
    # Calculate neutral axis depth
    c_mm = As_mm2 * fy_mpa / (0.85 * fc_mpa * beta1 * b_mm)
    a_mm = beta1 * c_mm
    
    # Calculate tensile strain in steel
    et = 0.003 * (d_mm - c_mm) / c_mm
    
    # Determine phi factor based on strain (ACI 318-19, Section 21.2.2)
    Es = 200000  # MPa
    ey = fy_mpa / Es  # Yield strain
    
    if et < ey:
        # Compression-controlled
        phi = 0.65
    elif et >= 0.005:
        # Tension-controlled
        phi = 0.90
    else:
        # Transition zone
        phi = 0.65 + (et - ey) / (0.005 - ey) * (0.90 - 0.65)
    
    # Calculate moment capacity
    Mn_Nmm = As_mm2 * fy_mpa * (d_mm - a_mm/2)
    Mn_kNm = Mn_Nmm / 1e6  # Convert to kN·m
    phi_Mn_kNm = phi * Mn_kNm
    
    return Mn_kNm, phi_Mn_kNm, phi, c_mm, et

def determine_bar_arrangement(As_required_mm2):
    """
    Suggest practical bar arrangements
    
    Parameters:
    -----------
    As_required_mm2 : float
        Required steel area in mm²
    
    Returns:
    --------
    list : List of tuples (bar_size, num_bars, total_area)
    """
    # Common bar sizes (diameter in mm, area in mm²)
    bar_sizes = {
        10: 78.5,
        12: 113,
        16: 201,
        20: 314,
        25: 491,
        28: 616,
        32: 804
    }
    
    suggestions = []
    
    for dia, area in bar_sizes.items():
        num_bars = math.ceil(As_required_mm2 / area)
        if num_bars <= 8:  # Practical limit
            total_area = num_bars * area
            excess_pct = (total_area - As_required_mm2) / As_required_mm2 * 100
            suggestions.append({
                'Bar Size': f'ø{dia}',
                'Number': num_bars,
                'Area (mm²)': round(total_area, 1),
                'Excess (%)': round(excess_pct, 1)
            })
    
    return pd.DataFrame(suggestions).sort_values('Excess (%)')

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_beam_section(b_mm, h_mm, cover_mm, bar_diameter, num_bars, stirrup_dia=10):
    """
    Create beam cross-section diagram with reinforcement
    
    Parameters:
    -----------
    b_mm : float
        Beam width in mm
    h_mm : float
        Beam height in mm
    cover_mm : float
        Concrete cover in mm
    bar_diameter : float
        Main bar diameter in mm
    num_bars : int
        Number of bars
    stirrup_dia : float
        Stirrup diameter in mm
    """
    fig, ax = plt.subplots(figsize=(8, 10))
    
    # Scale factor for display
    scale = max(b_mm, h_mm) / 400
    
    # Draw beam outline
    beam_rect = patches.Rectangle((0, 0), b_mm, h_mm, 
                                  linewidth=2, edgecolor='black', 
                                  facecolor='lightgray', alpha=0.3)
    ax.add_patch(beam_rect)
    
    # Draw stirrups
    stirrup_inner_x = cover_mm + stirrup_dia/2
    stirrup_inner_y = cover_mm + stirrup_dia/2
    stirrup_inner_w = b_mm - 2*(cover_mm + stirrup_dia/2)
    stirrup_inner_h = h_mm - 2*(cover_mm + stirrup_dia/2)
    
    stirrup_rect = patches.Rectangle(
        (stirrup_inner_x, stirrup_inner_y),
        stirrup_inner_w, stirrup_inner_h,
        linewidth=2, edgecolor='blue', facecolor='none'
    )
    ax.add_patch(stirrup_rect)
    
    # Calculate bar positions
    clear_width = b_mm - 2*(cover_mm + stirrup_dia + bar_diameter/2)
    
    if num_bars == 1:
        x_positions = [b_mm/2]
    elif num_bars == 2:
        x_positions = [cover_mm + stirrup_dia + bar_diameter/2,
                      b_mm - (cover_mm + stirrup_dia + bar_diameter/2)]
    else:
        spacing = clear_width / (num_bars - 1)
        x_positions = [cover_mm + stirrup_dia + bar_diameter/2 + i*spacing 
                      for i in range(num_bars)]
    
    y_position = cover_mm + stirrup_dia + bar_diameter/2
    
    # Draw reinforcement bars
    for x in x_positions:
        bar_circle = patches.Circle((x, y_position), bar_diameter/2,
                                   edgecolor='darkred', facecolor='red',
                                   linewidth=2)
        ax.add_patch(bar_circle)
    
    # Add dimensions
    # Width dimension
    ax.annotate('', xy=(b_mm, -h_mm*0.1), xytext=(0, -h_mm*0.1),
                arrowprops=dict(arrowstyle='<->', color='black', lw=1.5))
    ax.text(b_mm/2, -h_mm*0.15, f'b = {b_mm:.0f} mm',
            ha='center', va='top', fontsize=11, fontweight='bold')
    
    # Height dimension
    ax.annotate('', xy=(-b_mm*0.1, h_mm), xytext=(-b_mm*0.1, 0),
                arrowprops=dict(arrowstyle='<->', color='black', lw=1.5))
    ax.text(-b_mm*0.15, h_mm/2, f'h = {h_mm:.0f} mm',
            ha='center', va='center', fontsize=11, fontweight='bold', rotation=90)
    
    # Cover dimension
    ax.plot([0, cover_mm], [h_mm, h_mm], 'k--', linewidth=0.8, alpha=0.5)
    ax.plot([cover_mm, cover_mm], [h_mm, h_mm - h_mm*0.05], 'k-', linewidth=0.8)
    ax.text(cover_mm/2, h_mm + h_mm*0.03, f'cover = {cover_mm:.0f} mm',
            ha='center', fontsize=9, style='italic')
    
    # Effective depth indicator
    d_mm = h_mm - (cover_mm + stirrup_dia + bar_diameter/2)
    ax.plot([b_mm + b_mm*0.05, b_mm + b_mm*0.05], [0, d_mm], 'g-', linewidth=2)
    ax.plot([b_mm + b_mm*0.03, b_mm + b_mm*0.07], [d_mm, d_mm], 'g-', linewidth=2)
    ax.text(b_mm + b_mm*0.12, d_mm/2, f'd = {d_mm:.0f} mm',
            ha='left', va='center', fontsize=10, color='green', fontweight='bold', rotation=90)
    
    # Add legend
    legend_elements = [
        patches.Patch(facecolor='lightgray', edgecolor='black', label='Concrete'),
        patches.Patch(facecolor='none', edgecolor='blue', label='Stirrups'),
        patches.Circle((0, 0), 1, facecolor='red', edgecolor='darkred', label='Main Bars')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9)
    
    # Set axis properties
    ax.set_xlim(-b_mm*0.2, b_mm*1.2)
    ax.set_ylim(-h_mm*0.2, h_mm*1.15)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('RC Beam Cross-Section', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    return fig

def plot_strain_diagram(c_mm, d_mm, et, ey):
    """
    Create strain distribution diagram
    """
    fig, ax = plt.subplots(figsize=(6, 8))
    
    # Strain at top (compression)
    ec = 0.003
    
    # Points for strain diagram
    y_points = [0, c_mm, d_mm]
    strain_points = [ec, 0, -et]
    
    # Draw strain diagram
    ax.plot(strain_points, y_points, 'b-', linewidth=2)
    ax.fill_betweenx(y_points, 0, strain_points, alpha=0.3, color='blue')
    
    # Add reference lines
    ax.axvline(x=0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.axhline(y=c_mm, color='r', linestyle='--', linewidth=1, alpha=0.7)
    
    # Labels
    ax.text(ec*0.5, 5, f'εc = {ec:.4f}', fontsize=10, ha='center')
    ax.text(-et*0.5, d_mm-5, f'εt = {et:.4f}', fontsize=10, ha='center')
    ax.text(0.001, c_mm, f'c = {c_mm:.1f} mm', fontsize=9, color='red')
    
    # Add yield strain line
    ey_y = c_mm + (d_mm - c_mm) * (0 - (-ey)) / (0 - (-et))
    ax.axhline(y=ey_y, color='orange', linestyle=':', linewidth=1.5, alpha=0.7)
    ax.text(-ey*1.2, ey_y, f'εy = {ey:.4f}', fontsize=9, color='orange')
    
    ax.set_xlabel('Strain', fontsize=11, fontweight='bold')
    ax.set_ylabel('Depth from top (mm)', fontsize=11, fontweight='bold')
    ax.set_title('Strain Distribution', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.invert_yaxis()
    
    plt.tight_layout()
    return fig

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    # Header
    st.markdown('<div class="main-header">🏗️ RC Beam Designer - ACI 318</div>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        <b>Professional Reinforced Concrete Beam Design Tool</b><br>
        Design rectangular RC beams according to ACI 318 Building Code Requirements.
        This tool uses the rectangular stress block method for flexural design.
    </div>
    """, unsafe_allow_html=True)
    
    # ========================================================================
    # SIDEBAR - INPUT PARAMETERS
    # ========================================================================
    
    st.sidebar.header("📋 Design Parameters")
    
    st.sidebar.subheader("Material Properties")
    fc_mpa = st.sidebar.number_input(
        "Concrete Strength f'c (MPa)",
        min_value=20.0, max_value=80.0, value=25.0, step=1.0,
        help="Specified compressive strength of concrete"
    )
    
    fy_mpa = st.sidebar.number_input(
        "Steel Yield Strength fy (MPa)",
        min_value=300.0, max_value=600.0, value=420.0, step=10.0,
        help="Specified yield strength of reinforcing steel"
    )
    
    st.sidebar.subheader("Beam Geometry")
    b_mm = st.sidebar.number_input(
        "Beam Width b (mm)",
        min_value=200.0, max_value=1000.0, value=300.0, step=10.0,
        help="Width of the beam section"
    )
    
    h_mm = st.sidebar.number_input(
        "Beam Height h (mm)",
        min_value=300.0, max_value=1500.0, value=600.0, step=10.0,
        help="Total height of the beam section"
    )
    
    cover_mm = st.sidebar.number_input(
        "Concrete Cover (mm)",
        min_value=20.0, max_value=75.0, value=40.0, step=5.0,
        help="Clear concrete cover to stirrups"
    )
    
    st.sidebar.subheader("Design Loads")
    Mu_kNm = st.sidebar.number_input(
        "Factored Moment Mu (kN·m)",
        min_value=1.0, max_value=1000.0, value=150.0, step=5.0,
        help="Factored design moment"
    )
    
    st.sidebar.subheader("Reinforcement Details")
    bar_diameter = st.sidebar.selectbox(
        "Bar Diameter (mm)",
        [10, 12, 16, 20, 25, 28, 32],
        index=3,
        help="Diameter of main reinforcement bars"
    )
    
    stirrup_dia = st.sidebar.selectbox(
        "Stirrup Diameter (mm)",
        [8, 10, 12],
        index=1,
        help="Diameter of stirrup bars"
    )
    
    # Calculate button
    calculate = st.sidebar.button("🔍 Design Beam", type="primary", use_container_width=True)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Design Code:** ACI 318-19")
    st.sidebar.markdown("**Method:** Rectangular Stress Block")
    
    # ========================================================================
    # MAIN PANEL - CALCULATIONS AND RESULTS
    # ========================================================================
    
    if calculate:
        # Calculate effective depth
        d_mm = h_mm - (cover_mm + stirrup_dia + bar_diameter/2)
        
        st.markdown('<div class="sub-header">📊 Design Calculations</div>', 
                   unsafe_allow_html=True)
        
        # Display material properties
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Concrete Strength", f"{fc_mpa} MPa")
            beta1 = calculate_beta1(fc_mpa)
            st.caption(f"β₁ = {beta1:.3f}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Steel Strength", f"{fy_mpa} MPa")
            Es = 200000  # MPa
            st.caption(f"Es = {Es:,.0f} MPa")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Effective Depth", f"{d_mm:.1f} mm")
            st.caption(f"d/h = {d_mm/h_mm:.3f}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Perform design calculations
        st.markdown("### Design Analysis")
        
        with st.spinner("Calculating required reinforcement..."):
            result = calculate_required_steel(Mu_kNm, b_mm, d_mm, fc_mpa, fy_mpa)
            As_required, a_mm, c_mm, rho, success, message = result
        
        if success:
            # Calculate moment capacity with required steel
            Mn, phi_Mn, phi, c_final, et = calculate_moment_capacity(
                As_required, b_mm, d_mm, fc_mpa, fy_mpa
            )
            
            # Success message
            st.markdown(f"""
            <div class="success-box">
                <b>✅ Design Successful!</b><br>
                {message}
            </div>
            """, unsafe_allow_html=True)
            
            # Design results table
            st.markdown("### Design Results")
            
            results_data = {
                'Parameter': [
                    'Required Steel Area (As)',
                    'Reinforcement Ratio (ρ)',
                    'Minimum Ratio (ρ_min)',
                    'Maximum Ratio (ρ_max)',
                    'Neutral Axis Depth (c)',
                    'Compression Block (a)',
                    'Tension Strain (εt)',
                    'Strength Reduction (φ)',
                    'Nominal Moment (Mn)',
                    'Design Moment (φMn)',
                    'Factored Moment (Mu)',
                    'Capacity Ratio (φMn/Mu)'
                ],
                'Value': [
                    f"{As_required:.0f} mm²",
                    f"{rho:.4f}",
                    f"{calculate_rho_min(fy_mpa):.4f}",
                    f"{0.75 * calculate_rho_balanced(fc_mpa, fy_mpa, beta1):.4f}",
                    f"{c_final:.1f} mm",
                    f"{a_mm:.1f} mm",
                    f"{et:.4f}",
                    f"{phi:.3f}",
                    f"{Mn:.2f} kN·m",
                    f"{phi_Mn:.2f} kN·m",
                    f"{Mu_kNm:.2f} kN·m",
                    f"{phi_Mn/Mu_kNm:.2f}"
                ],
                'Status': [
                    '✅',
                    '✅' if rho >= calculate_rho_min(fy_mpa) else '❌',
                    '✅',
                    '✅' if rho <= 0.75 * calculate_rho_balanced(fc_mpa, fy_mpa, beta1) else '❌',
                    '✅',
                    '✅',
                    '✅ Tension' if et >= 0.005 else ('⚠️ Transition' if et >= fy_mpa/Es else '❌ Compression'),
                    '✅',
                    '✅',
                    '✅',
                    '✅',
                    '✅' if phi_Mn >= Mu_kNm else '❌'
                ]
            }
            
            results_df = pd.DataFrame(results_data)
            st.dataframe(results_df, use_container_width=True, hide_index=True)
            
            # Check adequacy
            if phi_Mn >= Mu_kNm:
                capacity_excess = (phi_Mn - Mu_kNm) / Mu_kNm * 100
                st.markdown(f"""
                <div class="success-box">
                    <b>✅ Design Adequate</b><br>
                    φMn = {phi_Mn:.2f} kN·m ≥ Mu = {Mu_kNm:.2f} kN·m<br>
                    Excess capacity: {capacity_excess:.1f}%
                </div>
                """, unsafe_allow_html=True)
            else:
                capacity_deficit = (Mu_kNm - phi_Mn) / Mu_kNm * 100
                st.markdown(f"""
                <div class="error-box">
                    <b>❌ Design Inadequate</b><br>
                    φMn = {phi_Mn:.2f} kN·m < Mu = {Mu_kNm:.2f} kN·m<br>
                    Deficit: {capacity_deficit:.1f}%
                </div>
                """, unsafe_allow_html=True)
            
            # Reinforcement suggestions
            st.markdown("### Recommended Bar Arrangements")
            
            suggestions_df = determine_bar_arrangement(As_required)
            st.dataframe(suggestions_df, use_container_width=True, hide_index=True)
            
            # Let user select arrangement
            selected_idx = st.selectbox(
                "Select bar arrangement for visualization:",
                range(len(suggestions_df)),
                format_func=lambda i: f"{suggestions_df.iloc[i]['Bar Size']} - {suggestions_df.iloc[i]['Number']} bars"
            )
            
            selected_arrangement = suggestions_df.iloc[selected_idx]
            selected_dia = int(selected_arrangement['Bar Size'].replace('ø', ''))
            selected_num = int(selected_arrangement['Number'])
            
            # Visualizations
            st.markdown("### Design Visualizations")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Beam Cross-Section")
                fig_section = plot_beam_section(
                    b_mm, h_mm, cover_mm, selected_dia, selected_num, stirrup_dia
                )
                st.pyplot(fig_section)
                plt.close()
            
            with col2:
                st.markdown("#### Strain Distribution")
                ey = fy_mpa / Es
                fig_strain = plot_strain_diagram(c_final, d_mm, et, ey)
                st.pyplot(fig_strain)
                plt.close()
            
            # Design summary for export
            st.markdown("### Design Summary")
            
            summary_text = f"""
            **RC BEAM DESIGN SUMMARY - ACI 318**
            =====================================
            
            **Materials:**
            - Concrete: f'c = {fc_mpa} MPa (β₁ = {beta1:.3f})
            - Steel: fy = {fy_mpa} MPa
            
            **Geometry:**
            - Width: b = {b_mm} mm
            - Height: h = {h_mm} mm
            - Effective depth: d = {d_mm:.1f} mm
            - Cover: {cover_mm} mm
            
            **Loading:**
            - Factored moment: Mu = {Mu_kNm} kN·m
            
            **Design Results:**
            - Required steel: As = {As_required:.0f} mm²
            - Reinforcement ratio: ρ = {rho:.4f}
            - Design moment capacity: φMn = {phi_Mn:.2f} kN·m
            - Strength reduction factor: φ = {phi:.3f}
            - Neutral axis depth: c = {c_final:.1f} mm
            - Tension strain: εt = {et:.4f}
            
            **Recommended Reinforcement:**
            - {selected_arrangement['Bar Size']} - {selected_arrangement['Number']} bars
            - Provided area: {selected_arrangement['Area (mm²)']} mm²
            
            **Design Status:** {'ADEQUATE ✅' if phi_Mn >= Mu_kNm else 'INADEQUATE ❌'}
            """
            
            st.code(summary_text)
            
            # Export options
            st.markdown("### Export Options")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Create CSV download
                csv_data = results_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results (CSV)",
                    data=csv_data,
                    file_name="rc_beam_design_results.csv",
                    mime="text/csv"
                )
            
            with col2:
                # Create text summary download
                st.download_button(
                    label="📄 Download Summary (TXT)",
                    data=summary_text,
                    file_name="rc_beam_design_summary.txt",
                    mime="text/plain"
                )
        
        else:
            # Design failed
            st.markdown(f"""
            <div class="error-box">
                <b>❌ Design Failed</b><br>
                {message}<br><br>
                <b>Suggestions:</b><br>
                • Increase beam dimensions (b or h)<br>
                • Use higher strength concrete<br>
                • Consider doubly reinforced section<br>
                • Reduce applied moment
            </div>
            """, unsafe_allow_html=True)
            
            # Show what the limits are
            beta1 = calculate_beta1(fc_mpa)
            rho_max = 0.75 * calculate_rho_balanced(fc_mpa, fy_mpa, beta1)
            As_max = rho_max * b_mm * d_mm
            
            Mn_max, phi_Mn_max, _, _, _ = calculate_moment_capacity(
                As_max, b_mm, d_mm, fc_mpa, fy_mpa
            )
            
            st.warning(f"Maximum capacity for singly reinforced section: φMn_max = {phi_Mn_max:.2f} kN·m")
            st.info(f"Required capacity: Mu = {Mu_kNm:.2f} kN·m")
            
    else:
        # Initial state - show instructions
        st.markdown("### 📖 How to Use")
        
        st.markdown("""
        <div class="info-box">
        <b>Step-by-step guide:</b><br><br>
        1. <b>Enter Material Properties:</b> Specify concrete strength (f'c) and steel yield strength (fy)<br>
        2. <b>Define Beam Geometry:</b> Input beam width (b), height (h), and concrete cover<br>
        3. <b>Input Design Loads:</b> Enter the factored design moment (Mu)<br>
        4. <b>Select Reinforcement:</b> Choose bar diameter and stirrup size<br>
        5. <b>Click "Design Beam":</b> The app will calculate required steel and verify adequacy<br>
        6. <b>Review Results:</b> Examine the design results, visualizations, and bar arrangements<br>
        7. <b>Export Results:</b> Download design summary and results for documentation
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 📚 Design Method")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Rectangular Stress Block (ACI 318-19)**
            
            The design follows ACI 318 provisions:
            
            - **Section 22.2:** Flexural strength
            - **Section 9.6:** Minimum reinforcement
            - **Section 21.2:** Strength reduction factors
            
            **Key Equations:**
            - Required steel: As = ρ × b × d
            - Moment capacity: Mn = As × fy × (d - a/2)
            - Compression block: a = As × fy / (0.85 × f'c × b)
            """)
        
        with col2:
            st.markdown("""
            **Strength Reduction Factors (φ):**
            
            - **Tension-controlled** (εt ≥ 0.005): φ = 0.90
            - **Transition zone**: φ varies linearly
            - **Compression-controlled** (εt < εy): φ = 0.65
            
            **Reinforcement Limits:**
            - ρ_min = max(0.25√f'c/fy, 1.4/fy)
            - ρ_max = 0.75 × ρ_balanced
            - ρ_balanced = 0.85β₁(f'c/fy)(600/(600+fy))
            """)
        
        st.markdown("### ⚠️ Important Notes")
        
        st.warning("""
        **Limitations and Assumptions:**
        - This tool designs for flexure only (shear design not included)
        - Assumes singly reinforced rectangular sections
        - Does not include serviceability checks (deflection, crack width)
        - Development length must be verified separately
        - For doubly reinforced sections or T-beams, use advanced analysis
        """)
        
        st.info("""
        **Best Practices:**
        - Verify all calculations with hand calculations or other software
        - Check local building codes for additional requirements
        - Consider constructability and bar spacing requirements
        - Ensure adequate development length for all reinforcement
        - Perform shear and torsion design separately
        """)

# ============================================================================
# RUN APPLICATION
# ============================================================================

if __name__ == "__main__":
    main()
