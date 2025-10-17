import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import math

# ================================
# PAGE CONFIGURATION
# ================================
st.set_page_config(
    page_title="RC Beam Design & Analysis - ACI",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================================
# CUSTOM CSS STYLING
# ================================
st.markdown("""
<style>
    .main-header {
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    .capacity-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin: 1rem 0;
        border-left: 5px solid #667eea;
    }
    .mode-badge-tension {
        background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
        color: #0d5f3a;
        padding: 0.8rem;
        border-radius: 8px;
        text-align: center;
        font-weight: bold;
        margin: 1rem 0;
    }
    .mode-badge-balanced {
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
</style>
""", unsafe_allow_html=True)

# ================================
# HELPER FUNCTIONS - ACI 318
# ================================

def calculate_beta1(fc_ksc):
    """Calculate β₁ factor based on concrete strength per ACI 318"""
    fc_mpa = fc_ksc * 0.0980665  # Convert ksc to MPa
    if fc_mpa <= 28:
        return 0.85
    elif fc_mpa >= 55:
        return 0.65
    else:
        return 0.85 - 0.05 * (fc_mpa - 28) / 7

def calculate_phi_factor(epsilon_t, epsilon_ty=0.002):
    """Calculate strength reduction factor φ based on strain per ACI 318"""
    if epsilon_t >= 0.005:
        return 0.90, "Tension-Controlled"
    elif epsilon_t <= epsilon_ty:
        return 0.65, "Compression-Controlled"
    else:
        phi = 0.65 + (epsilon_t - epsilon_ty) * (0.25 / (0.005 - epsilon_ty))
        return round(phi, 3), "Balanced/Transition"

def analyze_rc_section(b_mm, h_mm, cover_mm, bars_df, fc_ksc, fy_ksc):
    """
    Analyze RC beam section for flexural capacity
    Returns moment capacity, strain profile, and failure mode
    """
    if bars_df.empty or bars_df['Number of Bars'].sum() == 0:
        return None
    
    # Constants
    fc_N_mm2 = fc_ksc * 0.0980665  # ksc to MPa
    fy_N_mm2 = fy_ksc * 0.0980665
    Es = 200000  # MPa (steel modulus)
    epsilon_c = 0.003  # Concrete crushing strain
    beta1 = calculate_beta1(fc_ksc)
    
    # Calculate bar positions and areas
    bar_data = []
    for idx, row in bars_df.iterrows():
        num_bars = int(row['Number of Bars'])
        if num_bars == 0:
            continue
            
        dia_mm = row['Diameter (mm)']
        As_single = np.pi * dia_mm**2 / 4
        As_total = As_single * num_bars
        
        # Determine y-position from top
        layer = row['Layer Position']
        if 'Top' in layer:
            y_mm = cover_mm + dia_mm/2
        elif 'Bottom' in layer:
            y_mm = h_mm - cover_mm - dia_mm/2
        elif 'Middle' in layer or 'Mid' in layer:
            y_mm = h_mm / 2
        else:
            # Custom y-distance from top
            try:
                y_mm = float(layer)
            except:
                y_mm = h_mm - cover_mm - dia_mm/2  # Default to bottom
        
        bar_data.append({
            'y_mm': y_mm,
            'd_mm': h_mm - y_mm,  # Distance from top fiber
            'As_mm2': As_total,
            'dia_mm': dia_mm,
            'num_bars': num_bars
        })
    
    if not bar_data:
        return None
    
    # Effective depth (to centroid of tension steel)
    tension_bars = [b for b in bar_data if b['y_mm'] > h_mm/2]
    if tension_bars:
        d_mm = max([b['d_mm'] for b in tension_bars])
    else:
        d_mm = h_mm - cover_mm - 20
    
    # Iterative solution for neutral axis depth
    c_mm = d_mm / 2  # Initial guess
    tolerance = 0.01
    max_iter = 100
    
    for iteration in range(max_iter):
        # Calculate forces in each bar layer
        C_concrete = 0
        T_steel = 0
        M_internal = 0
        
        a_mm = beta1 * c_mm
        C_concrete = 0.85 * fc_N_mm2 * b_mm * a_mm
        M_concrete = C_concrete * (h_mm - a_mm/2)
        
        for bar in bar_data:
            y = bar['y_mm']
            As = bar['As_mm2']
            
            # Calculate strain at bar location
            epsilon_s = epsilon_c * (c_mm - y) / c_mm
            
            # Calculate stress
            if abs(epsilon_s) * Es <= fy_N_mm2:
                fs = epsilon_s * Es
            else:
                fs = fy_N_mm2 * np.sign(epsilon_s)
            
            # Force
            F = As * fs
            
            if y < c_mm:  # Compression zone
                C_concrete += F
            else:  # Tension zone
                T_steel += F
            
            # Moment contribution
            M_internal += F * (h_mm - y)
        
        # Check force equilibrium
        force_error = abs(C_concrete - T_steel)
        
        if force_error < tolerance:
            break
        
        # Update neutral axis depth
        if C_concrete > T_steel:
            c_mm *= 0.95  # Reduce c
        else:
            c_mm *= 1.05  # Increase c
        
        # Bounds check
        c_mm = max(10, min(c_mm, h_mm - 10))
    
    # Calculate final moment capacity
    a_mm = beta1 * c_mm
    Mn_Nmm = 0
    
    # Concrete compression force
    C_c = 0.85 * fc_N_mm2 * b_mm * a_mm
    Mn_Nmm += C_c * (d_mm - a_mm/2)
    
    # Steel forces
    for bar in bar_data:
        y = bar['y_mm']
        As = bar['As_mm2']
        epsilon_s = epsilon_c * (c_mm - y) / c_mm
        
        if abs(epsilon_s) * Es <= fy_N_mm2:
            fs = epsilon_s * Es
        else:
            fs = fy_N_mm2 * np.sign(epsilon_s)
        
        if y > c_mm:  # Tension steel
            Mn_Nmm += As * fs * (d_mm - (h_mm - y))
    
    Mn_tonfm = abs(Mn_Nmm) / 9.80665e6
    
    # Calculate tension steel strain
    epsilon_t = epsilon_c * (d_mm - c_mm) / c_mm
    
    # Get phi factor and control type
    phi, control_type = calculate_phi_factor(epsilon_t)
    
    # Design moment capacity
    phi_Mn_tonfm = phi * Mn_tonfm
    
    # Prepare strain profile for plotting
    strain_profile = {
        'c_mm': c_mm,
        'a_mm': a_mm,
        'd_mm': d_mm,
        'epsilon_c': epsilon_c,
        'epsilon_t': epsilon_t,
        'bar_strains': [(b['y_mm'], epsilon_c * (c_mm - b['y_mm']) / c_mm) for b in bar_data]
    }
    
    return {
        'Mn_tonfm': Mn_tonfm,
        'phi_Mn_tonfm': phi_Mn_tonfm,
        'phi': phi,
        'control_type': control_type,
        'strain_profile': strain_profile,
        'bar_data': bar_data,
        'c_mm': c_mm,
        'a_mm': a_mm,
        'd_mm': d_mm,
        'epsilon_t': epsilon_t
    }

def calculate_shear_capacity(b_mm, h_mm, d_mm, fc_ksc, stirrup_df):
    """
    Calculate shear capacity per ACI 318
    """
    fc_N_mm2 = fc_ksc * 0.0980665
    
    # Concrete shear capacity
    Vc_N = 0.17 * np.sqrt(fc_N_mm2) * b_mm * d_mm
    
    # Steel shear capacity (if stirrups provided)
    Vs_N = 0
    if not stirrup_df.empty and stirrup_df['Number of Legs'].sum() > 0:
        for idx, row in stirrup_df.iterrows():
            dia_mm = row['Diameter (mm)']
            n_legs = int(row['Number of Legs'])
            spacing_mm = row['Spacing (mm)']
            fy_stirrup_ksc = row['Steel Grade (ksc)']
            
            if spacing_mm > 0 and n_legs > 0:
                Av = n_legs * np.pi * dia_mm**2 / 4
                fy_stirrup_N_mm2 = fy_stirrup_ksc * 0.0980665
                Vs_N += Av * fy_stirrup_N_mm2 * d_mm / spacing_mm
    
    # Total shear capacity
    Vn_N = Vc_N + Vs_N
    phi_shear = 0.75  # ACI 318 shear reduction factor
    phi_Vn_N = phi_shear * Vn_N
    
    # Convert to tonf
    Vn_tonf = Vn_N / 9806.65
    phi_Vn_tonf = phi_Vn_N / 9806.65
    
    return {
        'Vc_tonf': Vc_N / 9806.65,
        'Vs_tonf': Vs_N / 9806.65,
        'Vn_tonf': Vn_tonf,
        'phi_Vn_tonf': phi_Vn_tonf,
        'phi': phi_shear
    }

def plot_section_and_strain(b_mm, h_mm, cover_mm, bar_data, strain_profile):
    """
    Create combined plot showing beam section and strain diagram side by side
    Both share the same vertical (height) scale
    """
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Beam Cross-Section', 'Strain Distribution'),
        horizontal_spacing=0.15,
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
        fillcolor="lightgray",
        opacity=0.3,
        row=1, col=1
    )
    
    # Draw compression block
    if strain_profile:
        a_mm = strain_profile['a_mm']
        fig.add_shape(
            type="rect",
            x0=0, y0=h_mm-a_mm, x1=b_mm, y1=h_mm,
            fillcolor="rgba(255,99,71,0.3)",
            line=dict(color="red", width=1, dash="dash"),
            row=1, col=1
        )
        
        # Neutral axis line
        c_mm = strain_profile['c_mm']
        fig.add_shape(
            type="line",
            x0=0, y0=h_mm-c_mm, x1=b_mm, y1=h_mm-c_mm,
            line=dict(color="green", width=2, dash="dash"),
            row=1, col=1
        )
        
        fig.add_annotation(
            x=b_mm*1.05, y=h_mm-c_mm,
            text="N.A.",
            showarrow=False,
            font=dict(size=10, color="green"),
            row=1, col=1
        )
    
    # Draw reinforcement bars
    for bar in bar_data:
        y = bar['y_mm']
        dia = bar['dia_mm']
        num = bar['num_bars']
        
        # Distribute bars across width
        if num == 1:
            x_positions = [b_mm/2]
        elif num == 2:
            x_positions = [cover_mm + dia/2, b_mm - cover_mm - dia/2]
        else:
            spacing = (b_mm - 2*cover_mm - dia) / (num - 1)
            x_positions = [cover_mm + dia/2 + i*spacing for i in range(num)]
        
        for x in x_positions:
            fig.add_shape(
                type="circle",
                x0=x-dia/2, y0=y-dia/2, x1=x+dia/2, y1=y+dia/2,
                fillcolor="darkred",
                line=dict(color="black", width=1),
                row=1, col=1
            )
    
    # Add dimension annotations
    fig.add_annotation(
        x=b_mm/2, y=-h_mm*0.1,
        text=f"b = {b_mm:.0f} mm",
        showarrow=False,
        font=dict(size=11),
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
        
        # Strain at bottom
        epsilon_bottom = epsilon_c * (c_mm - h_mm) / c_mm
        
        # Strain profile line
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
                y=[h_mm-c_mm, h_mm, h_mm, h_mm-c_mm, h_mm-c_mm],
                fill='toself',
                fillcolor='rgba(255,99,71,0.2)',
                line=dict(width=0),
                showlegend=False
            ),
            row=1, col=2
        )
        
        # Tension zone shading
        fig.add_trace(
            go.Scatter(
                x=[0, epsilon_bottom, epsilon_bottom, 0, 0],
                y=[h_mm-c_mm, 0, 0, h_mm-c_mm, h_mm-c_mm],
                fill='toself',
                fillcolor='rgba(30,144,255,0.2)',
                line=dict(width=0),
                showlegend=False
            ),
            row=1, col=2
        )
        
        # Mark neutral axis
        fig.add_shape(
            type="line",
            x0=min(epsilon_bottom, 0), y0=h_mm-c_mm,
            x1=epsilon_c*1.1, y1=h_mm-c_mm,
            line=dict(color="green", width=2, dash="dash"),
            row=1, col=2
        )
        
        # Mark strain at bar locations
        for y_pos, eps in strain_profile['bar_strains']:
            fig.add_trace(
                go.Scatter(
                    x=[eps], y=[y_pos],
                    mode='markers',
                    marker=dict(size=8, color='red', symbol='circle'),
                    showlegend=False
                ),
                row=1, col=2
            )
        
        # Add strain annotations
        fig.add_annotation(
            x=epsilon_c, y=h_mm,
            text=f"εc = {epsilon_c:.4f}",
            showarrow=True,
            arrowhead=2,
            ax=-30, ay=-20,
            font=dict(size=10),
            row=1, col=2
        )
        
        fig.add_annotation(
            x=epsilon_bottom, y=0,
            text=f"ε = {epsilon_bottom:.4f}",
            showarrow=True,
            arrowhead=2,
            ax=30, ay=20,
            font=dict(size=10),
            row=1, col=2
        )
        
        # ACI limit lines
        fig.add_shape(
            type="line",
            x0=-0.005, y0=0, x1=-0.005, y1=h_mm,
            line=dict(color="blue", width=1, dash="dot"),
            row=1, col=2
        )
        
        fig.add_annotation(
            x=-0.005, y=h_mm*0.9,
            text="εt = 0.005<br>(Tension limit)",
            showarrow=False,
            font=dict(size=9, color="blue"),
            row=1, col=2
        )
    
    # Update axes
    fig.update_xaxes(title_text="Width (mm)", row=1, col=1, range=[-b_mm*0.2, b_mm*1.3])
    fig.update_xaxes(title_text="Strain (ε)", row=1, col=2)
    fig.update_yaxes(title_text="Height (mm)", row=1, col=1, range=[-h_mm*0.2, h_mm*1.1])
    fig.update_yaxes(title_text="Height (mm)", row=1, col=2, range=[0, h_mm*1.05])
    
    fig.update_layout(
        height=500,
        showlegend=False,
        hovermode='closest'
    )
    
    return fig

# ================================
# SIDEBAR - ANALYSIS MODE & MATERIALS
# ================================

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
    help="Concrete compressive strength"
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

# ================================
# MAIN PAGE - HEADER
# ================================

st.markdown("""
<div class="main-header">
    <h1>🏗️ RC Beam Design & Analysis</h1>
    <p>ACI 318 Standard | Professional Structural Analysis Tool</p>
</div>
""", unsafe_allow_html=True)

# ================================
# BEAM GEOMETRY INPUT
# ================================

st.markdown("### 📐 Beam Geometry")

col1, col2, col3 = st.columns(3)

with col1:
    b_mm = st.number_input("Width b (mm)", 
                           min_value=150.0, max_value=1000.0, 
                           value=300.0, step=50.0)

with col2:
    h_mm = st.number_input("Height h (mm)", 
                           min_value=200.0, max_value=1500.0, 
                           value=600.0, step=50.0)

with col3:
    cover_mm = st.number_input("Concrete Cover (mm)", 
                               min_value=20.0, max_value=75.0, 
                               value=40.0, step=5.0)

st.markdown("---")

# ================================
# REINFORCEMENT INPUT TABLES
# ================================

if analysis_mode == "Single Section":
    st.markdown("### 🔩 Main Reinforcement")
    
    # Initialize default reinforcement data
    if 'bars_df' not in st.session_state:
        st.session_state.bars_df = pd.DataFrame({
            'Diameter (mm)': [20, 20],
            'Number of Bars': [3, 3],
            'Layer Position': ['Bottom', 'Top']
        })
    
    st.session_state.bars_df = st.data_editor(
        st.session_state.bars_df,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "Diameter (mm)": st.column_config.NumberColumn(
                "Diameter (mm)",
                help="Bar diameter in mm",
                min_value=10,
                max_value=40,
                step=2,
                format="%.0f"
            ),
            "Number of Bars": st.column_config.NumberColumn(
                "Number of Bars",
                help="Number of bars in this layer",
                min_value=0,
                max_value=20,
                step=1,
                format="%.0f"
            ),
            "Layer Position": st.column_config.TextColumn(
                "Layer Position",
                help="Enter: Top, Bottom, Middle, or custom y-distance from top (mm)",
                default="Bottom"
            )
        }
    )
    
    st.markdown("---")
    st.markdown("### ⚡ Shear Reinforcement (Stirrups)")
    
    # Initialize default stirrup data
    if 'stirrup_df' not in st.session_state:
        st.session_state.stirrup_df = pd.DataFrame({
            'Diameter (mm)': [10],
            'Number of Legs': [2],
            'Spacing (mm)': [150],
            'Steel Grade (ksc)': [fy_stirrup_ksc]
        })
    
    st.session_state.stirrup_df = st.data_editor(
        st.session_state.stirrup_df,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "Diameter (mm)": st.column_config.NumberColumn(
                "Diameter (mm)",
                min_value=8,
                max_value=16,
                step=2,
                format="%.0f"
            ),
            "Number of Legs": st.column_config.NumberColumn(
                "Number of Legs",
                min_value=2,
                max_value=6,
                step=1,
                format="%.0f"
            ),
            "Spacing (mm)": st.column_config.NumberColumn(
                "Spacing (mm)",
                min_value=50,
                max_value=500,
                step=10,
                format="%.0f"
            ),
            "Steel Grade (ksc)": st.column_config.NumberColumn(
                "Steel Grade (ksc)",
                format="%.0f"
            )
        }
    )
    
    st.markdown("---")
    
    # ================================
    # ANALYSIS & RESULTS
    # ================================
    
    if st.button("🚀 Analyze Section", use_container_width=True, type="primary"):
        with st.spinner("Analyzing beam section..."):
            # Perform flexural analysis
            flex_results = analyze_rc_section(
                b_mm, h_mm, cover_mm, 
                st.session_state.bars_df, 
                fc_ksc, fy_main_ksc
            )
            
            if flex_results is None:
                st.error("⚠️ Unable to analyze section. Please check reinforcement input.")
            else:
                # Store results in session state
                st.session_state.flex_results = flex_results
                
                # Perform shear analysis
                shear_results = calculate_shear_capacity(
                    b_mm, h_mm, 
                    flex_results['d_mm'],
                    fc_ksc, 
                    st.session_state.stirrup_df
                )
                st.session_state.shear_results = shear_results
    
    # Display results if available
    if 'flex_results' in st.session_state and st.session_state.flex_results:
        flex_results = st.session_state.flex_results
        
        st.markdown("### 📊 Analysis Results")
        
        # Create visualization
        fig = plot_section_and_strain(
            b_mm, h_mm, cover_mm,
            flex_results['bar_data'],
            flex_results['strain_profile']
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Display failure mode
        control_type = flex_results['control_type']
        if 'Tension' in control_type:
            badge_class = 'mode-badge-tension'
        elif 'Balanced' in control_type or 'Transition' in control_type:
            badge_class = 'mode-badge-balanced'
        else:
            badge_class = 'mode-badge-compression'
        
        st.markdown(f"""
        <div class="{badge_class}">
            Failure Mode: {control_type}<br>
            φ = {flex_results['phi']:.3f} | εt = {flex_results['epsilon_t']:.5f}
        </div>
        """, unsafe_allow_html=True)
        
        # Capacity cards
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="capacity-card">
                <h3>💪 Bending Capacity</h3>
            </div>
            """, unsafe_allow_html=True)
            
            st.metric("Nominal Moment (Mn)", f"{flex_results['Mn_tonfm']:.2f} tonf·m")
            st.metric("Design Moment (φMn)", f"{flex_results['phi_Mn_tonfm']:.2f} tonf·m",
                     help="Factored moment capacity")
            st.info(f"Neutral Axis: c = {flex_results['c_mm']:.1f} mm")
        
        with col2:
            if 'shear_results' in st.session_state:
                shear_results = st.session_state.shear_results
                
                st.markdown("""
                <div class="capacity-card">
                    <h3>⚡ Shear Capacity</h3>
                </div>
                """, unsafe_allow_html=True)
                
                st.metric("Nominal Shear (Vn)", f"{shear_results['Vn_tonf']:.2f} tonf")
                st.metric("Design Shear (φVn)", f"{shear_results['phi_Vn_tonf']:.2f} tonf",
                         help="Factored shear capacity")
                st.info(f"Vc = {shear_results['Vc_tonf']:.2f} tonf | Vs = {shear_results['Vs_tonf']:.2f} tonf")
        
        # Detailed parameters
        with st.expander("📋 Detailed Parameters", expanded=False):
            details_df = pd.DataFrame([
                ["Effective depth (d)", f"{flex_results['d_mm']:.0f} mm"],
                ["Neutral axis (c)", f"{flex_results['c_mm']:.1f} mm"],
                ["Compression block (a)", f"{flex_results['a_mm']:.1f} mm"],
                ["Concrete strain (εc)", f"{flex_results['strain_profile']['epsilon_c']:.4f}"],
                ["Steel strain (εt)", f"{flex_results['epsilon_t']:.5f}"],
                ["β₁ factor", f"{calculate_beta1(fc_ksc):.3f}"],
                ["φ factor", f"{flex_results['phi']:.3f}"],
                ["Control type", flex_results['control_type']]
            ], columns=["Parameter", "Value"])
            
            st.dataframe(details_df, hide_index=True, use_container_width=True)

else:
    # ================================
    # 3-SECTION MODE
    # ================================
    
    st.markdown("### 🔀 Three-Section Analysis (Start - Mid - End)")
    
    st.info("📌 In 3-section mode, define reinforcement for each section separately.")
    
    # Create tabs for each section
    tab1, tab2, tab3 = st.tabs(["📍 Start Section", "📍 Mid Section", "📍 End Section"])
    
    sections_data = {}
    
    for idx, (tab, section_name) in enumerate(zip([tab1, tab2, tab3], 
                                                   ["Start", "Mid", "End"])):
        with tab:
            st.markdown(f"#### Reinforcement for {section_name} Section")
            
            # Main reinforcement
            bars_key = f'bars_df_{section_name.lower()}'
            if bars_key not in st.session_state:
                st.session_state[bars_key] = pd.DataFrame({
                    'Diameter (mm)': [20, 20],
                    'Number of Bars': [3, 2],
                    'Layer Position': ['Bottom', 'Top']
                })
            
            st.session_state[bars_key] = st.data_editor(
                st.session_state[bars_key],
                num_rows="dynamic",
                key=f"bars_editor_{section_name}",
                column_config={
                    "Diameter (mm)": st.column_config.NumberColumn(
                        "Diameter (mm)",
                        min_value=10,
                        max_value=40,
                        step=2,
                        format="%.0f"
                    ),
                    "Number of Bars": st.column_config.NumberColumn(
                        "Number of Bars",
                        min_value=0,
                        max_value=20,
                        step=1,
                        format="%.0f"
                    ),
                    "Layer Position": st.column_config.TextColumn(
                        "Layer Position",
                        default="Bottom"
                    )
                }
            )
            
            st.markdown("**Stirrups:**")
            
            stirrup_key = f'stirrup_df_{section_name.lower()}'
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
                key=f"stirrup_editor_{section_name}",
                column_config={
                    "Diameter (mm)": st.column_config.NumberColumn(
                        "Diameter (mm)",
                        min_value=8,
                        max_value=16,
                        step=2,
                        format="%.0f"
                    ),
                    "Number of Legs": st.column_config.NumberColumn(
                        "Number of Legs",
                        min_value=2,
                        max_value=6,
                        step=1,
                        format="%.0f"
                    ),
                    "Spacing (mm)": st.column_config.NumberColumn(
                        "Spacing (mm)",
                        min_value=50,
                        max_value=500,
                        step=10,
                        format="%.0f"
                    ),
                    "Steel Grade (ksc)": st.column_config.NumberColumn(
                        "Steel Grade (ksc)",
                        format="%.0f"
                    )
                }
            )
    
    st.markdown("---")
    
    if st.button("🚀 Analyze All Sections", use_container_width=True, type="primary"):
        with st.spinner("Analyzing all three sections..."):
            results_summary = []
            
            for section_name in ["Start", "Mid", "End"]:
                bars_key = f'bars_df_{section_name.lower()}'
                stirrup_key = f'stirrup_df_{section_name.lower()}'
                
                # Flexural analysis
                flex_results = analyze_rc_section(
                    b_mm, h_mm, cover_mm,
                    st.session_state[bars_key],
                    fc_ksc, fy_main_ksc
                )
                
                if flex_results:
                    # Shear analysis
                    shear_results = calculate_shear_capacity(
                        b_mm, h_mm,
                        flex_results['d_mm'],
                        fc_ksc,
                        st.session_state[stirrup_key]
                    )
                    
                    results_summary.append({
                        'Section': section_name,
                        'φMn (tonf·m)': f"{flex_results['phi_Mn_tonfm']:.2f}",
                        'φVn (tonf)': f"{shear_results['phi_Vn_tonf']:.2f}",
                        'Control Type': flex_results['control_type'],
                        'φ': f"{flex_results['phi']:.3f}",
                        'εt': f"{flex_results['epsilon_t']:.5f}"
                    })
                else:
                    results_summary.append({
                        'Section': section_name,
                        'φMn (tonf·m)': 'Error',
                        'φVn (tonf)': 'Error',
                        'Control Type': 'N/A',
                        'φ': 'N/A',
                        'εt': 'N/A'
                    })
            
            st.session_state.three_section_results = results_summary
    
    # Display 3-section results
    if 'three_section_results' in st.session_state:
        st.markdown("### 📊 Three-Section Analysis Summary")
        
        results_df = pd.DataFrame(st.session_state.three_section_results)
        
        st.dataframe(
            results_df,
            hide_index=True,
            use_container_width=True,
            column_config={
                "Section": st.column_config.TextColumn("Section", width="small"),
                "φMn (tonf·m)": st.column_config.TextColumn("Bending Capacity φMn", width="medium"),
                "φVn (tonf)": st.column_config.TextColumn("Shear Capacity φVn", width="medium"),
                "Control Type": st.column_config.TextColumn("Failure Mode", width="medium"),
                "φ": st.column_config.TextColumn("φ Factor", width="small"),
                "εt": st.column_config.TextColumn("Tensile Strain", width="small")
            }
        )
        
        # Visual comparison
        st.markdown("#### 📈 Capacity Comparison")
        
        sections = [r['Section'] for r in st.session_state.three_section_results]
        moments = [float(r['φMn (tonf·m)']) if r['φMn (tonf·m)'] != 'Error' else 0 
                   for r in st.session_state.three_section_results]
        shears = [float(r['φVn (tonf)']) if r['φVn (tonf)'] != 'Error' else 0 
                  for r in st.session_state.three_section_results]
        
        fig_compare = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Bending Capacity', 'Shear Capacity')
        )
        
        fig_compare.add_trace(
            go.Bar(x=sections, y=moments, name='φMn', 
                   marker_color='#667eea', text=moments, textposition='auto'),
            row=1, col=1
        )
        
        fig_compare.add_trace(
            go.Bar(x=sections, y=shears, name='φVn', 
                   marker_color='#764ba2', text=shears, textposition='auto'),
            row=1, col=2
        )
        
        fig_compare.update_xaxes(title_text="Section", row=1, col=1)
        fig_compare.update_xaxes(title_text="Section", row=1, col=2)
        fig_compare.update_yaxes(title_text="Moment (tonf·m)", row=1, col=1)
        fig_compare.update_yaxes(title_text="Shear (tonf)", row=1, col=2)
        
        fig_compare.update_layout(
            height=400,
            showlegend=False,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig_compare, use_container_width=True)
        
        # Export results
        col1, col2 = st.columns(2)
        
        with col1:
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Results (CSV)",
                data=csv,
                file_name="rc_beam_3section_analysis.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            # Text summary
            summary = f"""RC BEAM THREE-SECTION ANALYSIS
{'='*50}
Beam Geometry: {b_mm} × {h_mm} mm
Concrete: f'c = {fc_ksc} ksc
Steel: fy = {fy_main_ksc} ksc

RESULTS SUMMARY:
{'='*50}
"""
            for r in st.session_state.three_section_results:
                summary += f"\n{r['Section']} Section:\n"
                summary += f"  Bending: φMn = {r['φMn (tonf·m)']} tonf·m\n"
                summary += f"  Shear: φVn = {r['φVn (tonf)']} tonf\n"
                summary += f"  Mode: {r['Control Type']}\n"
            
            st.download_button(
                label="📄 Download Summary (TXT)",
                data=summary,
                file_name="rc_beam_3section_summary.txt",
                mime="text/plain",
                use_container_width=True
            )

# ================================
# FOOTER & INFORMATION
# ================================

st.markdown("---")

with st.expander("ℹ️ About This Application", expanded=False):
    st.markdown("""
    ### RC Beam Design & Analysis Tool
    
    **Design Code:** ACI 318 (American Concrete Institute)
    
    **Units:**
    - Dimensions: millimeters (mm)
    - Moments: tonf·m (metric ton-force × meter)
    - Stresses: ksc (kg/cm² or kg force per square centimeter)
    - Shear: tonf (metric ton-force)
    
    **Features:**
    - ✅ Flexural strength analysis per ACI 318
    - ✅ Shear capacity calculation
    - ✅ Strain distribution visualization
    - ✅ Failure mode identification (Tension/Balanced/Compression-controlled)
    - ✅ Single section or three-section analysis
    - ✅ Dynamic reinforcement input with flexible bar placement
    
    **Key Equations:**
    
    **Flexural Capacity:**
    ```
    Mn = As × fy × (d - a/2)
    a = β₁ × c
    φMn = φ × Mn
    ```
    
    **Strain Limits:**
    - Tension-controlled: εt ≥ 0.005 → φ = 0.90
    - Transition zone: 0.002 < εt < 0.005 → φ = 0.65 to 0.90
    - Compression-controlled: εt ≤ 0.002 → φ = 0.65
    
    **Shear Capacity:**
    ```
    Vc = 0.17√f'c × b × d
    Vs = Av × fy × d / s
    Vn = Vc + Vs
    φVn = 0.75 × Vn
    ```
    
    **Notes:**
    - This tool is for educational and preliminary design purposes
    - Professional engineering review required for construction
    - Always verify calculations with hand calculations
    - Consider additional code requirements (deflection, crack control, etc.)
    
    **Version:** 1.0.0  
    **Last Updated:** 2025
    """)

with st.expander("📚 How to Use", expanded=False):
    st.markdown("""
    ### Step-by-Step Guide
    
    **1. Select Analysis Mode (Sidebar)**
    - **Single Section:** Analyze one beam section with detailed strain diagram
    - **3-Section:** Analyze start, mid, and end sections (simplified results)
    
    **2. Set Material Properties (Sidebar)**
    - Enter concrete strength (f'c)
    - Enter main reinforcement grade (fy)
    - Enter stirrup steel grade (fy_stirrup)
    
    **3. Define Beam Geometry**
    - Width (b): Cross-section width in mm
    - Height (h): Total beam depth in mm
    - Cover: Concrete cover to reinforcement in mm
    
    **4. Input Main Reinforcement**
    - Add rows for different bar layers
    - For each layer, specify:
      - Bar diameter (mm)
      - Number of bars
      - Layer position: "Top", "Bottom", "Middle", or custom y-distance from top (mm)
    
    **5. Input Stirrup Details**
    - Diameter: Stirrup bar size
    - Number of legs: Typically 2 for rectangular beams
    - Spacing: Distance between stirrups (mm)
    - Steel grade: Usually lower than main reinforcement
    
    **6. Analyze**
    - Click "Analyze Section" or "Analyze All Sections"
    - Review results, strain diagram, and failure mode
    - Export results if needed
    
    **Tips:**
    - Use "Bottom" for tension reinforcement (positive moment)
    - Use "Top" for compression reinforcement or negative moment regions
    - Custom y-positions allow precise control over bar placement
    - In 3-section mode, vary reinforcement to match moment diagram
    """)

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #64748b; padding: 1rem;'>
    <p>🏗️ RC Beam Design & Analysis v1.0 | ACI 318 Compliant</p>
    <p style='font-size: 0.9em;'>
        <em>Professional structural analysis tool for educational and preliminary design purposes</em>
    </p>
</div>
""", unsafe_allow_html=True)
