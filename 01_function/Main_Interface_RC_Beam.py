"""
Simple RC Beam Design & Analysis - ACI 318
No custom HTML/CSS - Pure Streamlit components
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ================================
# MATERIAL CONSTANTS
# ================================
class MaterialConfig:
    ES = 2.04e6  # Steel modulus (ksc)
    EPSILON_C = 0.003  # Ultimate concrete strain
    EPSILON_TY = 0.002  # Yield strain
    EPSILON_T_LIMIT = 0.005  # Tension-controlled limit

def get_bar_area(diameter_mm):
    """Calculate bar area in mm²"""
    return np.pi * (diameter_mm ** 2) / 4

def calculate_beta1(fc_ksc):
    """Calculate β₁ factor per ACI 318"""
    fc_mpa = fc_ksc * 0.0980665
    if fc_mpa <= 28:
        return 0.85
    elif fc_mpa >= 55:
        return 0.65
    else:
        return 0.85 - 0.05 * (fc_mpa - 28) / 7

def calculate_phi_factor(epsilon_t):
    """Calculate φ factor based on tension strain"""
    if epsilon_t >= 0.005:
        return 0.90, "Tension-Controlled"
    elif epsilon_t <= MaterialConfig.EPSILON_TY:
        return 0.65, "Compression-Controlled"
    else:
        phi = 0.65 + (epsilon_t - MaterialConfig.EPSILON_TY) * (0.25 / (0.005 - MaterialConfig.EPSILON_TY))
        return round(phi, 3), "Transition Zone"


# ================================
# ANALYSIS FUNCTIONS
# ================================
def analyze_flexural_capacity(b_mm, h_mm, cover_mm, bars_df, fc_ksc, fy_ksc):
    """Analyze flexural capacity per ACI 318"""
    
    if bars_df.empty or bars_df['Number of Bars'].sum() == 0:
        return None
    
    fc_mpa = fc_ksc * 0.0980665
    fy_mpa = fy_ksc * 0.0980665
    Es = MaterialConfig.ES * 0.0980665
    epsilon_c = MaterialConfig.EPSILON_C
    beta1 = calculate_beta1(fc_ksc)
    
    # Process reinforcement
    bar_data = []
    for idx, row in bars_df.iterrows():
        num_bars = int(row['Number of Bars'])
        if num_bars == 0:
            continue
        
        dia_mm = row['Diameter (mm)']
        As_single = get_bar_area(dia_mm)
        As_total = As_single * num_bars
        
        layer = str(row['Layer Position'])
        if 'Top' in layer or 'top' in layer:
            y_mm = cover_mm + dia_mm / 2
        elif 'Bottom' in layer or 'bottom' in layer:
            y_mm = h_mm - cover_mm - dia_mm / 2
        else:
            y_mm = h_mm / 2
        
        bar_data.append({
            'y_mm': y_mm,
            'd_mm': h_mm - y_mm,
            'As_mm2': As_total,
            'dia_mm': dia_mm,
            'num_bars': num_bars
        })
    
    if not bar_data:
        return None
    
    # Effective depth
    tension_bars = [b for b in bar_data if b['y_mm'] > h_mm / 2]
    d_mm = max([b['d_mm'] for b in tension_bars]) if tension_bars else h_mm - cover_mm - 20
    
    # Iterative solution for neutral axis
    c_mm = d_mm / 2
    tolerance = 0.01
    max_iter = 100
    
    for iteration in range(max_iter):
        a_mm = beta1 * c_mm
        C_concrete = 0.85 * fc_mpa * b_mm * a_mm
        
        T_steel = 0
        C_steel = 0
        
        for bar in bar_data:
            y = bar['y_mm']
            As = bar['As_mm2']
            epsilon_s = epsilon_c * (c_mm - y) / c_mm
            
            if abs(epsilon_s * Es) <= fy_mpa:
                fs = epsilon_s * Es
            else:
                fs = fy_mpa * np.sign(epsilon_s)
            
            F = As * fs
            if y < c_mm:
                C_steel += F
            else:
                T_steel += abs(F)
        
        C_total = C_concrete + C_steel
        force_error = abs(C_total - T_steel)
        
        if force_error < tolerance:
            break
        
        if C_total > T_steel:
            c_mm *= 0.98
        else:
            c_mm *= 1.02
        
        c_mm = max(10, min(c_mm, h_mm - 10))
    
    # Calculate moment capacity
    a_mm = beta1 * c_mm
    C_c = 0.85 * fc_mpa * b_mm * a_mm
    M_concrete = C_c * (d_mm - a_mm / 2)
    
    M_steel = 0
    for bar in bar_data:
        y = bar['y_mm']
        As = bar['As_mm2']
        epsilon_s = epsilon_c * (c_mm - y) / c_mm
        
        if abs(epsilon_s * Es) <= fy_mpa:
            fs = epsilon_s * Es
        else:
            fs = fy_mpa * np.sign(epsilon_s)
        
        F = As * fs
        M_steel += F * (d_mm - (h_mm - y))
    
    Mn_Nmm = abs(M_concrete + M_steel)
    Mn_tonfm = Mn_Nmm / 9.80665e6
    
    epsilon_t = epsilon_c * (d_mm - c_mm) / c_mm
    phi, control_type = calculate_phi_factor(epsilon_t)
    phi_Mn_tonfm = phi * Mn_tonfm
    
    strain_profile = {
        'c_mm': c_mm,
        'a_mm': a_mm,
        'd_mm': d_mm,
        'epsilon_c': epsilon_c,
        'epsilon_t': epsilon_t,
        'bar_strains': [(b['y_mm'], epsilon_c * (c_mm - b['y_mm']) / c_mm) for b in bar_data]
    }
    
    return {
        'Mn_tonfm': round(Mn_tonfm, 2),
        'phi_Mn_tonfm': round(phi_Mn_tonfm, 2),
        'phi': phi,
        'control_type': control_type,
        'strain_profile': strain_profile,
        'bar_data': bar_data,
        'c_mm': round(c_mm, 1),
        'a_mm': round(a_mm, 1),
        'd_mm': round(d_mm, 1),
        'epsilon_t': epsilon_t,
        'beta1': beta1
    }


def analyze_shear_capacity(b_mm, h_mm, d_mm, fc_ksc, stirrup_df):
    """Calculate shear capacity per ACI 318"""
    
    fc_mpa = fc_ksc * 0.0980665
    Vc_N = 0.17 * np.sqrt(fc_mpa) * b_mm * d_mm
    
    Vs_N = 0
    if not stirrup_df.empty and stirrup_df['Number of Legs'].sum() > 0:
        for idx, row in stirrup_df.iterrows():
            dia_mm = row['Diameter (mm)']
            n_legs = int(row['Number of Legs'])
            spacing_mm = row['Spacing (mm)']
            fy_stirrup_ksc = row['Steel Grade (ksc)']
            
            if spacing_mm > 0 and n_legs > 0:
                Av = n_legs * get_bar_area(dia_mm)
                fy_stirrup_mpa = fy_stirrup_ksc * 0.0980665
                Vs_N += Av * fy_stirrup_mpa * d_mm / spacing_mm
    
    Vn_N = Vc_N + Vs_N
    phi_shear = 0.75
    phi_Vn_N = phi_shear * Vn_N
    
    return {
        'Vc_tonf': round(Vc_N / 9806.65, 2),
        'Vs_tonf': round(Vs_N / 9806.65, 2),
        'Vn_tonf': round(Vn_N / 9806.65, 2),
        'phi_Vn_tonf': round(phi_Vn_N / 9806.65, 2),
        'phi': phi_shear
    }


# ================================
# PLOTTING FUNCTIONS
# ================================
def plot_section_and_strain(b_mm, h_mm, cover_mm, bar_data, strain_profile):
    """Create beam section and strain diagram"""
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Beam Cross-Section', 'Strain Distribution'),
        horizontal_spacing=0.15
    )
    
    # Beam section
    fig.add_shape(
        type="rect", x0=0, y0=0, x1=b_mm, y1=h_mm,
        line=dict(color="black", width=2),
        fillcolor="lightgray", opacity=0.3,
        row=1, col=1
    )
    
    if strain_profile:
        # Compression block
        a_mm = strain_profile['a_mm']
        fig.add_shape(
            type="rect", x0=0, y0=h_mm - a_mm, x1=b_mm, y1=h_mm,
            fillcolor="rgba(255, 0, 0, 0.2)",
            line=dict(color="red", width=2, dash="dash"),
            row=1, col=1
        )
        
        # Neutral axis
        c_mm = strain_profile['c_mm']
        fig.add_shape(
            type="line", x0=0, y0=h_mm - c_mm, x1=b_mm, y1=h_mm - c_mm,
            line=dict(color="green", width=2, dash="dash"),
            row=1, col=1
        )
    
    # Draw bars
    if bar_data:
        for bar in bar_data:
            y = bar['y_mm']
            dia = bar['dia_mm']
            num = bar['num_bars']
            
            if num == 1:
                x_positions = [b_mm / 2]
            elif num == 2:
                x_positions = [cover_mm + dia / 2, b_mm - cover_mm - dia / 2]
            else:
                spacing = (b_mm - 2 * cover_mm - dia) / (num - 1)
                x_positions = [cover_mm + dia / 2 + i * spacing for i in range(num)]
            
            for x in x_positions:
                fig.add_shape(
                    type="circle",
                    x0=x - dia / 2, y0=y - dia / 2,
                    x1=x + dia / 2, y1=y + dia / 2,
                    fillcolor="purple", line=dict(color="black", width=2),
                    row=1, col=1
                )
    
    # Strain diagram
    if strain_profile:
        epsilon_c = strain_profile['epsilon_c']
        c_mm = strain_profile['c_mm']
        epsilon_bottom = epsilon_c * (c_mm - h_mm) / c_mm
        
        fig.add_trace(
            go.Scatter(
                x=[epsilon_c, epsilon_bottom], y=[h_mm, 0],
                mode='lines', line=dict(color='blue', width=3),
                name='Strain', showlegend=False
            ),
            row=1, col=2
        )
        
        # Mark bar strains
        for y_pos, eps in strain_profile['bar_strains']:
            fig.add_trace(
                go.Scatter(
                    x=[eps], y=[y_pos],
                    mode='markers', marker=dict(size=10, color='purple'),
                    showlegend=False,
                    hovertemplate=f'y={y_pos:.0f}mm<br>ε={eps:.5f}'
                ),
                row=1, col=2
            )
    
    fig.update_xaxes(title_text="Width (mm)", row=1, col=1)
    fig.update_xaxes(title_text="Strain (ε)", row=1, col=2)
    fig.update_yaxes(title_text="Height (mm)", row=1, col=1)
    fig.update_yaxes(title_text="Height (mm)", row=1, col=2)
    
    fig.update_layout(height=500, showlegend=False)
    return fig


# ================================
# STREAMLIT APP
# ================================
st.set_page_config(page_title="RC Beam Analysis", page_icon="🏗️", layout="wide")

st.title("🏗️ RC Beam Design & Analysis")
st.subheader("ACI 318-19 Standard | Metric Units")

# Sidebar
st.sidebar.header("⚙️ Configuration")

analysis_mode = st.sidebar.radio(
    "Analysis Mode",
    ["Single Section", "3-Section Analysis"]
)

st.sidebar.subheader("Material Properties")
fc_ksc = st.sidebar.number_input("Concrete Strength f'c (ksc)", 200.0, 800.0, 280.0, 10.0)
fy_main_ksc = st.sidebar.number_input("Main Steel fy (ksc)", 2000.0, 6000.0, 4200.0, 100.0)
fy_stirrup_ksc = st.sidebar.number_input("Stirrup Steel fy (ksc)", 2000.0, 5000.0, 2800.0, 100.0)

# Geometry
st.header("📐 Beam Geometry")
col1, col2, col3 = st.columns(3)
with col1:
    b_mm = st.number_input("Width b (mm)", 150.0, 1000.0, 300.0, 50.0)
with col2:
    h_mm = st.number_input("Height h (mm)", 200.0, 1500.0, 600.0, 50.0)
with col3:
    cover_mm = st.number_input("Cover (mm)", 20.0, 75.0, 40.0, 5.0)

st.divider()

# ================================
# SINGLE SECTION MODE
# ================================
if analysis_mode == "Single Section":
    
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.subheader("🔩 Main Reinforcement")
        if 'bars_df' not in st.session_state:
            st.session_state.bars_df = pd.DataFrame({
                'Diameter (mm)': [20, 20],
                'Number of Bars': [3, 2],
                'Layer Position': ['Bottom', 'Top']
            })
        
        bars_df = st.data_editor(
            st.session_state.bars_df,
            num_rows="dynamic",
            use_container_width=True,
            key="bars_editor"
        )
        st.session_state.bars_df = bars_df
    
    with col_right:
        st.subheader("⚡ Stirrups")
        if 'stirrup_df' not in st.session_state:
            st.session_state.stirrup_df = pd.DataFrame({
                'Diameter (mm)': [10],
                'Number of Legs': [2],
                'Spacing (mm)': [150],
                'Steel Grade (ksc)': [fy_stirrup_ksc]
            })
        
        stirrup_df = st.data_editor(
            st.session_state.stirrup_df,
            num_rows="dynamic",
            use_container_width=True,
            key="stirrup_editor"
        )
        st.session_state.stirrup_df = stirrup_df
    
    st.divider()
    
    if st.button("🚀 Analyze Section", type="primary", use_container_width=True):
        
        with st.spinner("Analyzing..."):
            flex_results = analyze_flexural_capacity(
                b_mm, h_mm, cover_mm, bars_df, fc_ksc, fy_main_ksc
            )
            
            if flex_results is None:
                st.error("Unable to analyze. Check reinforcement input.")
            else:
                st.session_state.flex_results = flex_results
                
                shear_results = analyze_shear_capacity(
                    b_mm, h_mm, flex_results['d_mm'], fc_ksc, stirrup_df
                )
                st.session_state.shear_results = shear_results
                
                st.success("✅ Analysis completed!")
    
    # Display results
    if 'flex_results' in st.session_state and st.session_state.flex_results:
        
        flex_results = st.session_state.flex_results
        
        st.header("📊 Analysis Results")
        
        # Visualization
        fig = plot_section_and_strain(
            b_mm, h_mm, cover_mm,
            flex_results['bar_data'],
            flex_results['strain_profile']
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Failure mode
        st.subheader("Failure Mode")
        control_type = flex_results['control_type']
        if 'Tension' in control_type:
            st.success(f"✓ {control_type} - φ = {flex_results['phi']:.3f} | εt = {flex_results['epsilon_t']:.5f}")
        elif 'Transition' in control_type:
            st.warning(f"⚡ {control_type} - φ = {flex_results['phi']:.3f} | εt = {flex_results['epsilon_t']:.5f}")
        else:
            st.error(f"⚠ {control_type} - φ = {flex_results['phi']:.3f} | εt = {flex_results['epsilon_t']:.5f}")
        
        # Capacities
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("💪 Bending Capacity")
            st.metric("Nominal Moment Mn", f"{flex_results['Mn_tonfm']:.2f} tonf·m")
            st.metric("Design Moment φMn", f"{flex_results['phi_Mn_tonfm']:.2f} tonf·m")
            st.info(f"Neutral Axis: c = {flex_results['c_mm']:.1f} mm")
            st.info(f"Effective Depth: d = {flex_results['d_mm']:.1f} mm")
        
        with col2:
            if 'shear_results' in st.session_state:
                shear = st.session_state.shear_results
                st.subheader("⚡ Shear Capacity")
                st.metric("Nominal Shear Vn", f"{shear['Vn_tonf']:.2f} tonf")
                st.metric("Design Shear φVn", f"{shear['phi_Vn_tonf']:.2f} tonf")
                st.info(f"Concrete: Vc = {shear['Vc_tonf']:.2f} tonf")
                st.info(f"Steel: Vs = {shear['Vs_tonf']:.2f} tonf")
        
        # Detailed parameters
        with st.expander("📋 Detailed Parameters"):
            details = pd.DataFrame([
                ["Effective depth (d)", f"{flex_results['d_mm']:.1f} mm"],
                ["Neutral axis (c)", f"{flex_results['c_mm']:.1f} mm"],
                ["Compression block (a)", f"{flex_results['a_mm']:.1f} mm"],
                ["Concrete strain (εc)", f"{flex_results['strain_profile']['epsilon_c']:.4f}"],
                ["Steel strain (εt)", f"{flex_results['epsilon_t']:.5f}"],
                ["β₁ factor", f"{flex_results['beta1']:.3f}"],
                ["φ factor", f"{flex_results['phi']:.3f}"],
                ["Control type", flex_results['control_type']]
            ], columns=["Parameter", "Value"])
            
            st.dataframe(details, hide_index=True, use_container_width=True)


# ================================
# THREE-SECTION MODE
# ================================
else:
    st.info("📌 Define reinforcement for Start, Mid, and End sections")
    
    tab1, tab2, tab3 = st.tabs(["📍 Start", "📍 Mid", "📍 End"])
    
    for idx, (tab, section) in enumerate(zip([tab1, tab2, tab3], ["start", "mid", "end"])):
        with tab:
            st.subheader(f"{section.title()} Section")
            
            col_left, col_right = st.columns(2)
            
            with col_left:
                st.write("Main Reinforcement")
                bars_key = f'bars_df_{section}'
                if bars_key not in st.session_state:
                    st.session_state[bars_key] = pd.DataFrame({
                        'Diameter (mm)': [20, 20],
                        'Number of Bars': [3, 2],
                        'Layer Position': ['Bottom', 'Top']
                    })
                
                st.session_state[bars_key] = st.data_editor(
                    st.session_state[bars_key],
                    num_rows="dynamic",
                    key=f"bars_{section}"
                )
            
            with col_right:
                st.write("Stirrups")
                stirrup_key = f'stirrup_df_{section}'
                if stirrup_key not in st.session_state:
                    st.session_state[stirrup_key] = pd.DataFrame({
                        'Diameter (mm)': [10],
                        'Number of Legs': [2],
                        'Spacing (mm)': [150],
                        'Steel Grade (ksc)': [fy_stirrup_ksc]
                    })
                
                st.session_state[stirrup_key] = st.data_editor(
                    st.session_state[stirrup_key],
                    num_rows="dynamic",
                    key=f"stirrup_{section}"
                )
    
    st.divider()
    
    if st.button("🚀 Analyze All Sections", type="primary", use_container_width=True):
        
        with st.spinner("Analyzing all sections..."):
            results_summary = []
            
            for section in ["start", "mid", "end"]:
                bars_key = f'bars_df_{section}'
                stirrup_key = f'stirrup_df_{section}'
                
                flex_results = analyze_flexural_capacity(
                    b_mm, h_mm, cover_mm,
                    st.session_state[bars_key],
                    fc_ksc, fy_main_ksc
                )
                
                if flex_results:
                    shear_results = analyze_shear_capacity(
                        b_mm, h_mm, flex_results['d_mm'],
                        fc_ksc, st.session_state[stirrup_key]
                    )
                    
                    results_summary.append({
                        'Section': section.title(),
                        'φMn (tonf·m)': f"{flex_results['phi_Mn_tonfm']:.2f}",
                        'φVn (tonf)': f"{shear_results['phi_Vn_tonf']:.2f}",
                        'Control Type': flex_results['control_type'],
                        'φ': f"{flex_results['phi']:.3f}",
                        'εt': f"{flex_results['epsilon_t']:.5f}"
                    })
                else:
                    results_summary.append({
                        'Section': section.title(),
                        'φMn (tonf·m)': 'Error',
                        'φVn (tonf)': 'Error',
                        'Control Type': 'N/A',
                        'φ': 'N/A',
                        'εt': 'N/A'
                    })
            
            st.session_state.three_section_results = results_summary
            st.success("✅ Analysis completed!")
    
    # Display results
    if 'three_section_results' in st.session_state:
        st.header("📊 Three-Section Results")
        
        results_df = pd.DataFrame(st.session_state.three_section_results)
        st.dataframe(results_df, hide_index=True, use_container_width=True)
        
        # Export
        col1, col2 = st.columns(2)
        with col1:
            csv = results_df.to_csv(index=False)
            st.download_button(
                "📥 Download CSV",
                csv,
                "rc_beam_results.csv",
                "text/csv",
                use_container_width=True
            )

st.divider()
st.caption("RC Beam Design & Analysis v2.0 | ACI 318-19 | For educational purposes")
