"""
UI Layout Module
Handles page layout, styling, and display components
"""

import streamlit as st
import pandas as pd


def apply_custom_css():
    """Apply custom CSS styling to the app"""
    
    st.markdown("""
    <style>
        /* Main header styling */
        .main-header {
            text-align: center;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 2rem;
            border-radius: 15px;
            margin-bottom: 2rem;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }
        
        /* Capacity cards */
        .capacity-card {
            background: white;
            padding: 1.5rem;
            border-radius: 12px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            margin: 1rem 0;
            border-left: 5px solid #667eea;
        }
        
        /* Failure mode badges */
        .mode-badge-tension {
            background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
            color: #0d5f3a;
            padding: 0.8rem;
            border-radius: 8px;
            text-align: center;
            font-weight: bold;
            margin: 1rem 0;
        }
        
        .mode-badge-transition {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            color: white;
            padding: 0.8rem;
            border-radius: 8px;
            text-align: center;
            font-weight: bold;
            margin: 1rem 0;
        }
        
        .mode-badge-compression {
            background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
            color: #8b0000;
            padding: 0.8rem;
            border-radius: 8px;
            text-align: center;
            font-weight: bold;
            margin: 1rem 0;
        }
        
        /* Input section styling */
        .input-section {
            background: #f8f9fa;
            padding: 1.5rem;
            border-radius: 10px;
            margin: 1rem 0;
        }
        
        /* Results section */
        .results-container {
            background: linear-gradient(135deg, #e0e7ff 0%, #f3f4f6 100%);
            padding: 2rem;
            border-radius: 12px;
            margin: 2rem 0;
        }
        
        /* Metric styling */
        div[data-testid="stMetricValue"] {
            font-size: 1.8rem;
            color: #667eea;
            font-weight: bold;
        }
    </style>
    """, unsafe_allow_html=True)


def render_header():
    """Render the main application header"""
    
    st.markdown("""
    <div class="main-header">
        <h1>🏗️ RC Beam Design & Analysis</h1>
        <p>ACI 318 Standard | Professional Structural Analysis Tool</p>
        <p style="font-size: 0.9em; margin-top: 0.5rem;">
            Metric Units: mm (geometry) | tonf·m (moment) | ksc (stresses)
        </p>
    </div>
    """, unsafe_allow_html=True)


def render_sidebar_config():
    """
    Render sidebar configuration options
    
    Returns:
    --------
    dict : Configuration parameters
    """
    
    st.sidebar.markdown("## ⚙️ Analysis Configuration")
    
    analysis_mode = st.sidebar.radio(
        "**Analysis Mode**",
        ["Single Section", "3-Section (Start-Mid-End)"],
        help="Choose single section or analyze three sections along beam length"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🏗️ Material Properties")
    
    fc_ksc = st.sidebar.number_input(
        "Concrete Strength f'c (ksc)",
        min_value=200.0,
        max_value=800.0,
        value=280.0,
        step=10.0,
        help="Concrete compressive strength in kg/cm²"
    )
    
    fy_main_ksc = st.sidebar.number_input(
        "Main Steel Grade fy (ksc)",
        min_value=2000.0,
        max_value=6000.0,
        value=4200.0,
        step=100.0,
        help="Yield strength of main reinforcement"
    )
    
    fy_stirrup_ksc = st.sidebar.number_input(
        "Stirrup Steel Grade fy (ksc)",
        min_value=2000.0,
        max_value=5000.0,
        value=2800.0,
        step=100.0,
        help="Yield strength of shear reinforcement"
    )
    
    return {
        'analysis_mode': analysis_mode,
        'fc_ksc': fc_ksc,
        'fy_main_ksc': fy_main_ksc,
        'fy_stirrup_ksc': fy_stirrup_ksc
    }


def render_geometry_inputs():
    """
    Render beam geometry input section
    
    Returns:
    --------
    dict : Geometry parameters
    """
    
    st.markdown("### 📐 Beam Geometry")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        b_mm = st.number_input(
            "Width b (mm)",
            min_value=150.0,
            max_value=1000.0,
            value=300.0,
            step=50.0
        )
    
    with col2:
        h_mm = st.number_input(
            "Height h (mm)",
            min_value=200.0,
            max_value=1500.0,
            value=600.0,
            step=50.0
        )
    
    with col3:
        cover_mm = st.number_input(
            "Concrete Cover (mm)",
            min_value=20.0,
            max_value=75.0,
            value=40.0,
            step=5.0
        )
    
    return {
        'b_mm': b_mm,
        'h_mm': h_mm,
        'cover_mm': cover_mm
    }


def display_failure_mode_badge(control_type, phi, epsilon_t):
    """
    Display styled failure mode badge
    
    Parameters:
    -----------
    control_type : str
        Failure control type
    phi : float
        Strength reduction factor
    epsilon_t : float
        Tension steel strain
    """
    
    if 'Tension' in control_type:
        badge_class = 'mode-badge-tension'
    elif 'Transition' in control_type:
        badge_class = 'mode-badge-transition'
    else:
        badge_class = 'mode-badge-compression'
    
    st.markdown(f"""
    <div class="{badge_class}">
        <strong>Failure Mode:</strong> {control_type}<br>
        φ = {phi:.3f} | εt = {epsilon_t:.5f}
    </div>
    """, unsafe_allow_html=True)


def display_capacity_results(flex_results, shear_results):
    """
    Display capacity results in formatted cards
    
    Parameters:
    -----------
    flex_results : dict
        Flexural analysis results
    shear_results : dict
        Shear analysis results
    """
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="capacity-card">
            <h3>💪 Bending Capacity</h3>
        </div>
        """, unsafe_allow_html=True)
        
        st.metric(
            "Nominal Moment (Mn)",
            f"{flex_results['Mn_tonfm']:.2f} tonf·m"
        )
        st.metric(
            "Design Moment (φMn)",
            f"{flex_results['phi_Mn_tonfm']:.2f} tonf·m",
            help="Factored moment capacity"
        )
        st.info(f"✓ Neutral Axis: c = {flex_results['c_mm']:.1f} mm")
        st.info(f"✓ Effective Depth: d = {flex_results['d_mm']:.1f} mm")
    
    with col2:
        st.markdown("""
        <div class="capacity-card">
            <h3>⚡ Shear Capacity</h3>
        </div>
        """, unsafe_allow_html=True)
        
        st.metric(
            "Nominal Shear (Vn)",
            f"{shear_results['Vn_tonf']:.2f} tonf"
        )
        st.metric(
            "Design Shear (φVn)",
            f"{shear_results['phi_Vn_tonf']:.2f} tonf",
            help="Factored shear capacity"
        )
        st.info(f"✓ Concrete: Vc = {shear_results['Vc_tonf']:.2f} tonf")
        st.info(f"✓ Steel: Vs = {shear_results['Vs_tonf']:.2f} tonf")


def display_detailed_parameters(flex_results):
    """
    Display detailed analysis parameters in expander
    
    Parameters:
    -----------
    flex_results : dict
        Flexural analysis results
    """
    
    with st.expander("📋 Detailed Parameters", expanded=False):
        details_df = pd.DataFrame([
            ["Effective depth (d)", f"{flex_results['d_mm']:.1f} mm"],
            ["Neutral axis (c)", f"{flex_results['c_mm']:.1f} mm"],
            ["Compression block (a)", f"{flex_results['a_mm']:.1f} mm"],
            ["Concrete strain (εc)", f"{flex_results['strain_profile']['epsilon_c']:.4f}"],
            ["Steel strain (εt)", f"{flex_results['epsilon_t']:.5f}"],
            ["β₁ factor", f"{flex_results['beta1']:.3f}"],
            ["φ factor", f"{flex_results['phi']:.3f}"],
            ["Control type", flex_results['control_type']]
        ], columns=["Parameter", "Value"])
        
        st.dataframe(
            details_df,
            hide_index=True,
            use_container_width=True
        )


def render_footer():
    """Render application footer with information"""
    
    st.markdown("---")
    
    with st.expander("ℹ️ About This Application", expanded=False):
        st.markdown("""
        ### RC Beam Design & Analysis Tool
        
        **Design Code:** ACI 318 (American Concrete Institute)
        
        **Units:**
        - Dimensions: millimeters (mm)
        - Moments: tonf·m (metric ton-force × meter)
        - Stresses: ksc (kg/cm²)
        - Shear: tonf (metric ton-force)
        
        **Features:**
        - ✅ Flexural strength analysis per ACI 318
        - ✅ Shear capacity calculation
        - ✅ Strain distribution visualization
        - ✅ Failure mode identification
        - ✅ Single section or three-section analysis
        - ✅ Dynamic reinforcement configuration
        
        **Version:** 1.0.0 (Modular Architecture)  
        **Framework:** Streamlit + Plotly
        
        *For educational and preliminary design purposes*
        """)
