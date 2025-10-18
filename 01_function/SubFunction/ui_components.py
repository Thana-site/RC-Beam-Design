"""
UI Components Module
Layout elements, styling, and display functions
"""

import streamlit as st
import pandas as pd


def apply_custom_styling():
    """Apply professional CSS styling"""
    
    st.markdown("""
    <style>
        /* Main container */
        .main {
            background-color: #f8f9fa;
        }
        
        /* Header banner */
        .header-banner {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 2.5rem;
            border-radius: 15px;
            text-align: center;
            margin-bottom: 2rem;
            box-shadow: 0 10px 40px rgba(102, 126, 234, 0.3);
        }
        
        .header-banner h1 {
            margin: 0;
            font-size: 2.5rem;
            font-weight: 700;
        }
        
        .header-banner p {
            margin: 0.5rem 0 0 0;
            font-size: 1.1rem;
            opacity: 0.95;
        }
        
        /* Input cards */
        .input-card {
            background: white;
            padding: 1.5rem;
            border-radius: 12px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.08);
            margin-bottom: 1.5rem;
        }
        
        /* Results section */
        .results-section {
            background: linear-gradient(135deg, #e3f2fd 0%, #f3e5f5 100%);
            padding: 2rem;
            border-radius: 15px;
            margin: 2rem 0;
            border-left: 6px solid #667eea;
        }
        
        /* Capacity card */
        .capacity-card {
            background: white;
            padding: 1.5rem;
            border-radius: 10px;
            box-shadow: 0 3px 12px rgba(0,0,0,0.1);
            margin: 1rem 0;
        }
        
        /* Failure mode badges */
        .badge-tension {
            background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
            color: #0a5f38;
            padding: 1rem;
            border-radius: 10px;
            text-align: center;
            font-weight: 700;
            font-size: 1.1rem;
            margin: 1rem 0;
            box-shadow: 0 4px 15px rgba(67, 233, 123, 0.3);
        }
        
        .badge-transition {
            background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
            color: #8b2500;
            padding: 1rem;
            border-radius: 10px;
            text-align: center;
            font-weight: 700;
            font-size: 1.1rem;
            margin: 1rem 0;
            box-shadow: 0 4px 15px rgba(250, 112, 154, 0.3);
        }
        
        .badge-compression {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            color: #ffffff;
            padding: 1rem;
            border-radius: 10px;
            text-align: center;
            font-weight: 700;
            font-size: 1.1rem;
            margin: 1rem 0;
            box-shadow: 0 4px 15px rgba(240, 147, 251, 0.3);
        }
        
        /* Metrics */
        div[data-testid="stMetricValue"] {
            font-size: 2rem;
            color: #667eea;
            font-weight: 700;
        }
        
        /* Buttons */
        .stButton>button {
            width: 100%;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            font-weight: 600;
            border: none;
            padding: 0.75rem;
            border-radius: 8px;
            transition: all 0.3s;
        }
        
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 20px rgba(102, 126, 234, 0.4);
        }
        
        /* Tables */
        .stDataFrame {
            border-radius: 8px;
            overflow: hidden;
        }
        
        /* Section divider */
        hr {
            margin: 2rem 0;
            border: none;
            border-top: 2px solid #e0e0e0;
        }
    </style>
    """, unsafe_allow_html=True)


def render_header():
    """Render main application header"""
    
    st.markdown("""
    <div class="header-banner">
        <h1>🏗️ RC Beam Design & Analysis</h1>
        <p><strong>ACI 318-19 Standard</strong> | Professional Structural Engineering Tool</p>
        <p style="font-size: 0.95rem; margin-top: 0.8rem;">
            📏 Metric Units: mm • tonf·m • ksc
        </p>
    </div>
    """, unsafe_allow_html=True)


def render_sidebar_controls():
    """
    Render sidebar configuration options
    
    Returns:
        dict: Configuration parameters
    """
    
    st.sidebar.markdown("## ⚙️ Configuration")
    
    analysis_mode = st.sidebar.radio(
        "**Analysis Mode**",
        ["Single Section", "3-Section Analysis"],
        help="Analyze one section or three sections (Start-Mid-End)"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🏗️ Material Properties")
    
    fc_ksc = st.sidebar.number_input(
        "Concrete f'c (ksc)",
        min_value=200.0,
        max_value=800.0,
        value=280.0,
        step=10.0,
        help="Compressive strength (kg/cm²)"
    )
    
    fy_main = st.sidebar.number_input(
        "Main Steel fy (ksc)",
        min_value=2000.0,
        max_value=6000.0,
        value=4200.0,
        step=100.0,
        help="Yield strength"
    )
    
    fy_stirrup = st.sidebar.number_input(
        "Stirrup fy (ksc)",
        min_value=2000.0,
        max_value=5000.0,
        value=2800.0,
        step=100.0,
        help="Stirrup yield strength"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📐 Beam Geometry")
    
    b_mm = st.sidebar.number_input(
        "Width b (mm)",
        min_value=150.0,
        max_value=1000.0,
        value=300.0,
        step=50.0
    )
    
    h_mm = st.sidebar.number_input(
        "Height h (mm)",
        min_value=200.0,
        max_value=1500.0,
        value=600.0,
        step=50.0
    )
    
    cover_mm = st.sidebar.number_input(
        "Cover (mm)",
        min_value=20.0,
        max_value=75.0,
        value=40.0,
        step=5.0
    )
    
    return {
        'analysis_mode': analysis_mode,
        'fc_ksc': fc_ksc,
        'fy_main': fy_main,
        'fy_stirrup': fy_stirrup,
        'b_mm': b_mm,
        'h_mm': h_mm,
        'cover_mm': cover_mm
    }


def display_failure_mode(control_type, phi, epsilon_t):
    """Display styled failure mode badge"""
    
    if 'Tension' in control_type:
        badge_class = 'badge-tension'
    elif 'Transition' in control_type:
        badge_class = 'badge-transition'
    else:
        badge_class = 'badge-compression'
    
    st.markdown(f"""
    <div class="{badge_class}">
        <strong>Failure Mode:</strong> {control_type}<br>
        φ = {phi:.3f} | εt = {epsilon_t:.5f}
    </div>
    """, unsafe_allow_html=True)


def display_capacity_metrics(flex_results, shear_results):
    """Display capacity results in clean layout"""
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="capacity-card">', unsafe_allow_html=True)
        st.markdown("### 💪 Bending Capacity")
        st.metric(
            "Nominal Moment Mn",
            f"{flex_results['Mn_tonfm']:.2f} tonf·m"
        )
        st.metric(
            "Design Moment φMn",
            f"{flex_results['phi_Mn_tonfm']:.2f} tonf·m"
        )
        st.info(f"✓ Neutral Axis: c = {flex_results['c_mm']:.1f} mm")
        st.info(f"✓ Effective Depth: d = {flex_results['d_mm']:.1f} mm")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="capacity-card">', unsafe_allow_html=True)
        st.markdown("### ⚡ Shear Capacity")
        st.metric(
            "Nominal Shear Vn",
            f"{shear_results['Vn_tonf']:.2f} tonf"
        )
        st.metric(
            "Design Shear φVn",
            f"{shear_results['phi_Vn_tonf']:.2f} tonf"
        )
        st.info(f"✓ Concrete: Vc = {shear_results['Vc_tonf']:.2f} tonf")
        st.info(f"✓ Steel: Vs = {shear_results['Vs_tonf']:.2f} tonf")
        st.markdown('</div>', unsafe_allow_html=True)


def display_detailed_results(results):
    """Show detailed parameters in expander"""
    
    with st.expander("📋 Detailed Analysis Parameters", expanded=False):
        data = [
            ["Effective depth (d)", f"{results['d_mm']:.1f} mm"],
            ["Neutral axis (c)", f"{results['c_mm']:.1f} mm"],
            ["Compression block (a)", f"{results['a_mm']:.1f} mm"],
            ["Concrete strain (εc)", f"{results['strain_profile']['epsilon_c']:.4f}"],
            ["Tension strain (εt)", f"{results['epsilon_t']:.5f}"],
            ["β₁ factor", f"{results['beta1']:.3f}"],
            ["φ factor", f"{results['phi']:.3f}"],
            ["Control mode", results['control_type']]
        ]
        
        df = pd.DataFrame(data, columns=["Parameter", "Value"])
        st.dataframe(df, hide_index=True, use_container_width=True)


def render_footer():
    """Application footer"""
    
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #7f8c8d; padding: 1.5rem;'>
        <p><strong>RC Beam Design & Analysis v2.0</strong></p>
        <p style='font-size: 0.9rem;'>
            ACI 318-19 Compliant | Educational & Preliminary Design Tool
        </p>
        <p style='font-size: 0.85rem; margin-top: 0.5rem;'>
            ⚠️ Always verify results with licensed professional engineer
        </p>
    </div>
    """, unsafe_allow_html=True)
