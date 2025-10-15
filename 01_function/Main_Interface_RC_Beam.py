import streamlit as st
import pandas as pd
import math
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from PIL import Image
import io

# Set page config
st.set_page_config(
    page_title="Advanced RC Beam Design - ACI Standard", 
    page_icon="🏗️",
    layout="wide"
)

# Helper Functions
def beta1(fc):
    """Calculates β1 based on concrete compressive strength (fc in kg/cm²)."""
    if fc <= 280:
        return 0.85
    elif 280 < fc <= 550:
        return 0.85 - ((fc - 280) / 70) * 0.05
    else:
        return 0.65

def calculate_bottom_centroid(steel_list, diameters, H, covering, ds, bottom_row_spacing):
    """Calculate centroid for the bottom reinforcement bars."""
    total_area = 0
    total_moment = 0
    y_list = []

    for i, (num_bars, diameter) in enumerate(zip(steel_list, diameters)):
        As = math.pi * (0.5 * diameter) ** 2
        ys = covering + ds + (diameter / 2) + i * (diameter + bottom_row_spacing)
        moment = As * num_bars * ys

        total_area += As * num_bars
        total_moment += moment
        y_list.append(round(ys, 3))

    centroid = round(total_moment / total_area, 3) if total_area != 0 else 0
    return round(total_area, 3), round(total_moment, 3), y_list, centroid

def calculate_top_centroid(steel_list, diameters, H, covering, ds, top_row_spacing):
    """Calculate centroid for the top reinforcement bars."""
    total_area = 0
    total_moment = 0
    y_list = []

    for i, (num_bars, diameter) in enumerate(zip(steel_list, diameters)):
        As = math.pi * (0.5 * diameter) ** 2
        ys = H - (covering + ds + (diameter / 2) + i * (diameter + top_row_spacing))
        moment = As * num_bars * ys

        total_area += As * num_bars
        total_moment += moment
        y_list.append(round(ys, 3))

    centroid = round(total_moment / total_area, 3) if total_area != 0 else 0
    return round(total_area, 3), round(total_moment, 3), y_list, centroid

def plot_rc_beam_section(b, H, top_steel_list, bottom_steel_list, top_diameters, bottom_diameters, ds, covering, top_row_spacing, bottom_row_spacing):
    """Plot RC beam section with reinforcement."""
    bottom_steel_y = [
        covering + ds + bottom_diameters[i] / 2 + i * (bottom_diameters[i] + bottom_row_spacing)
        for i in range(len(bottom_steel_list))
    ]
    top_steel_y = [
        H - (covering + ds + top_diameters[i] / 2 + i * (top_diameters[i] + top_row_spacing))
        for i in range(len(top_steel_list))
    ]

    # Stirrup coordinates
    stirrup_outer_x = [covering, b - covering, b - covering, covering, covering]
    stirrup_outer_y = [covering, covering, H - covering, H - covering, covering]
    stirrup_inner_x = [covering + ds, b - covering - ds, b - covering - ds, covering + ds, covering + ds]
    stirrup_inner_y = [covering + ds, covering + ds, H - covering - ds, H - covering - ds, covering + ds]

    fig = go.Figure()

    # Beam section
    fig.add_shape(
        type="rect",
        x0=0, y0=0, x1=b, y1=H,
        line=dict(color="black", width=2),
        fillcolor="lightgray",
        opacity=0.5
    )

    # Stirrup lines
    fig.add_trace(go.Scatter(
        x=stirrup_outer_x, y=stirrup_outer_y, 
        mode="lines", 
        line=dict(color="blue", width=2), 
        showlegend=False,
        name="Stirrup"
    ))
    fig.add_trace(go.Scatter(
        x=stirrup_inner_x, y=stirrup_inner_y, 
        mode="lines", 
        line=dict(color="blue", width=2), 
        showlegend=False
    ))

    def distribute_steel_bars(steel_list, y_positions, diameters, color="red"):
        for i, y_position in enumerate(y_positions):
            num_bars = steel_list[i]
            diameter = diameters[i]
            
            if num_bars == 1:
                x_positions = [b / 2]
            elif num_bars == 2:
                x_positions = [covering + ds + diameter / 2, b - covering - ds - diameter / 2]
            else:
                spacing = (b - 2 * (covering + ds) - diameter) / (num_bars - 1)
                x_positions = [covering + ds + diameter / 2 + j * spacing for j in range(num_bars)]
            
            for x in x_positions:
                fig.add_shape(
                    type="circle",
                    x0=x - diameter / 2, y0=y_position - diameter / 2,
                    x1=x + diameter / 2, y1=y_position + diameter / 2,
                    line=dict(color=color, width=2), 
                    fillcolor=color
                )

    # Add steel bars
    if top_steel_list:
        distribute_steel_bars(top_steel_list, top_steel_y, top_diameters, "darkred")
    distribute_steel_bars(bottom_steel_list, bottom_steel_y, bottom_diameters, "red")

    # Add dimensions
    fig.add_annotation(
        x=b/2, y=-H*0.1,
        text=f"b = {b} cm",
        showarrow=False,
        font=dict(size=12, color="black")
    )
    fig.add_annotation(
        x=-b*0.15, y=H/2,
        text=f"H = {H} cm",
        showarrow=False,
        font=dict(size=12, color="black"),
        textangle=90
    )

    fig.update_layout(
        title="RC Beam Cross-Section",
        xaxis=dict(
            showgrid=True, gridwidth=0.5, gridcolor="lightgray",
            scaleanchor="y",
            tickmode="linear", dtick=5,
            range=[-b*0.2, b*1.2],
            title="Width (cm)"
        ),
        yaxis=dict(
            showgrid=True, gridwidth=0.5, gridcolor="lightgray",
            tickmode="linear", dtick=5,
            range=[-H*0.2, H*1.2],
            title="Height (cm)"
        ),
        width=600,
        height=400,
        margin=dict(l=50, r=50, t=50, b=50)
    )

    return fig

def hognestad_stress_strain(Ec, fc):
    """Generate Hognestad stress-strain relationship for concrete."""
    # Calculate key parameters
    e0 = 2 * fc / Ec  # Strain at peak stress
    ec = np.linspace(0, 0.005, 100)  # Strain range up to 0.5%
    
    fci_list = []
    for strain in ec:
        if strain <= e0:
            # Ascending branch
            fci = fc * (2 * strain / e0 - (strain / e0) ** 2)
        else:
            # Descending branch
            fci = fc * (1 - 0.15 * ((strain - e0) / (0.0038 - e0)))
            if fci < 0:
                fci = 0
        fci_list.append(fci)
    
    # Create plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ec * 1000,  # Convert to microstrain
        y=fci_list,
        mode='lines',
        name='Hognestad Model',
        line=dict(color='blue', width=3)
    ))
    
    fig.update_layout(
        title='Concrete Stress-Strain Relationship (Hognestad Model)',
        xaxis=dict(
            title='Strain (×10⁻³)',
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray'
        ),
        yaxis=dict(
            title='Stress (kg/cm²)',
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray'
        ),
        width=600,
        height=400
    )
    
    return ec, fci_list, e0, fig

def calculate_flexural_strength(As, As_prime, d, d_prime, b, fc, fy, beta_val):
    """Calculate nominal flexural strength of RC beam."""
    Es = 2.04e6  # Steel modulus (kg/cm²)
    
    # Balanced reinforcement ratio
    beta1_val = beta_val
    rho_b = 0.85 * beta1_val * fc / fy * (600 / (600 + fy))
    
    # Current reinforcement ratio
    rho = As / (b * d)
    
    # Check if section is over-reinforced
    if rho > rho_b:
        return None, "Over-reinforced section - not allowed", 0, 0, 0
    
    # For singly reinforced beam
    if As_prime == 0:
        # Calculate neutral axis depth
        c = As * fy / (0.85 * fc * beta1_val * b)
        
        # Check strain limits
        et = 0.003 * (d - c) / c  # Tension strain in steel
        
        if et < fy / Es:
            # Compression controlled
            phi = 0.65
        elif et >= 0.005:
            # Tension controlled
            phi = 0.9
        else:
            # Transition zone
            phi = 0.65 + (et - fy/Es) / (0.005 - fy/Es) * (0.9 - 0.65)
        
        # Nominal moment capacity
        a = beta1_val * c
        Mn = As * fy * (d - a/2) / 100000  # Convert to kg-m
        
        return Mn * phi, f"φMn = {Mn * phi:.2f} kg-m, φ = {phi:.3f}", Mn, phi, c
    
    # For doubly reinforced beam (simplified)
    else:
        # This is a simplified calculation - full analysis would be more complex
        c = (As - As_prime) * fy / (0.85 * fc * beta1_val * b)
        a = beta1_val * c
        
        # Check if compression steel yields
        fs_prime = 0.003 * Es * (c - d_prime) / c
        if fs_prime > fy:
            fs_prime = fy
        
        Mn1 = (As - As_prime) * fy * (d - a/2)
        Mn2 = As_prime * (fs_prime - 0.85 * fc) * (d - d_prime)
        Mn = (Mn1 + Mn2) / 100000  # Convert to kg-m
        
        # Simplified phi factor
        et = 0.003 * (d - c) / c
        if et >= 0.005:
            phi = 0.9
        else:
            phi = 0.75  # Conservative for doubly reinforced
        
        return Mn * phi, f"φMn = {Mn * phi:.2f} kg-m, φ = {phi:.3f}", Mn, phi, c

def moment_curvature_analysis(As, As_prime, d, d_prime, b, H, fc, fy, Es=2.04e6):
    """Perform moment-curvature analysis."""
    # Initialize arrays
    curvatures = []
    moments = []
    
    # Strain range for analysis
    max_strain = 0.005
    strain_increments = np.linspace(0.0005, max_strain, 50)
    
    beta1_val = beta1(fc)
    
    for ec in strain_increments:
        try:
            # Assume neutral axis position and iterate
            c_trial = d / 2  # Initial guess
            
            for iteration in range(50):  # Max iterations
                # Calculate strain in tension steel
                es = ec * (d - c_trial) / c_trial
                
                # Calculate strain in compression steel (if any)
                es_prime = 0
                if As_prime > 0:
                    es_prime = ec * (c_trial - d_prime) / c_trial
                
                # Steel stresses
                fs = min(es * Es, fy) if es > 0 else 0
                fs_prime = min(abs(es_prime) * Es, fy) if es_prime > 0 else 0
                
                # Forces
                Cc = 0.85 * fc * beta1_val * c_trial * b
                Ts = As * fs
                Cs = As_prime * fs_prime if As_prime > 0 else 0
                
                # Force equilibrium
                force_imbalance = Cc + Cs - Ts
                
                if abs(force_imbalance) < 0.1:  # Convergence criterion
                    break
                
                # Adjust c_trial
                if force_imbalance > 0:
                    c_trial *= 0.95
                else:
                    c_trial *= 1.05
                
                # Bounds check
                c_trial = max(0.1, min(c_trial, H))
            
            # Calculate moment
            a = beta1_val * c_trial
            M1 = Cc * (d - a/2)
            M2 = Cs * (d - d_prime) if As_prime > 0 else 0
            moment = (M1 + M2) / 100000  # Convert to kg-m
            
            # Calculate curvature
            curvature = ec / c_trial
            
            curvatures.append(curvature)
            moments.append(moment)
            
        except:
            continue
    
    return curvatures, moments

def plot_moment_curvature(curvatures, moments):
    """Plot moment-curvature diagram."""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=curvatures,
        y=moments,
        mode='lines+markers',
        name='Moment-Curvature',
        line=dict(color='red', width=3),
        marker=dict(size=4)
    ))
    
    fig.update_layout(
        title='Moment-Curvature Relationship',
        xaxis=dict(
            title='Curvature (1/cm)',
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray'
        ),
        yaxis=dict(
            title='Moment (kg-m)',
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray'
        ),
        width=700,
        height=400
    )
    
    return fig

def moment_capacity_analysis(b, d, d_prime, fc, fy, beta_val, As_prime=0):
    """Analyze moment capacity vs steel area with 0.1 cm² increments."""
    # Steel area range
    As_min = 0.1
    As_max = 15.0  # Reasonable maximum
    As_range = np.arange(As_min, As_max + 0.1, 0.1)
    
    moments = []
    phi_factors = []
    steel_ratios = []
    
    Es = 2.04e6
    rho_b = 0.85 * beta_val * fc / fy * (600 / (600 + fy))
    
    for As in As_range:
        try:
            result = calculate_flexural_strength(As, As_prime, d, d_prime, b, fc, fy, beta_val)
            
            if result[0] is not None:
                moments.append(result[0])
                phi_factors.append(result[3])
                steel_ratios.append(As / (b * d))
            else:
                # Over-reinforced - break the loop
                break
                
        except:
            break
    
    return As_range[:len(moments)], moments, phi_factors, steel_ratios, rho_b

def plot_moment_capacity_behavior(As_range, moments, phi_factors, steel_ratios, rho_b):
    """Plot moment capacity behavior analysis."""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Moment vs Steel Area', 'Phi Factor vs Steel Ratio', 
                       'Steel Ratio Analysis', 'Moment Efficiency'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Plot 1: Moment vs Steel Area
    fig.add_trace(
        go.Scatter(x=As_range, y=moments, mode='lines', name='Moment Capacity',
                  line=dict(color='blue', width=3)),
        row=1, col=1
    )
    
    # Plot 2: Phi factor vs Steel Ratio
    fig.add_trace(
        go.Scatter(x=steel_ratios, y=phi_factors, mode='lines', name='Phi Factor',
                  line=dict(color='green', width=3)),
        row=1, col=2
    )
    
    # Add balanced ratio line
    fig.add_vline(x=rho_b, line_dash="dash", line_color="red", 
                 annotation_text=f"ρb = {rho_b:.4f}", row=1, col=2)
    
    # Plot 3: Steel Ratio Analysis
    fig.add_trace(
        go.Scatter(x=As_range, y=steel_ratios, mode='lines', name='ρ = As/(bd)',
                  line=dict(color='orange', width=3)),
        row=2, col=1
    )
    
    fig.add_hline(y=rho_b, line_dash="dash", line_color="red", 
                 annotation_text="Balanced Ratio", row=2, col=1)
    
    # Plot 4: Moment Efficiency (Moment per unit steel)
    efficiency = [m/a for m, a in zip(moments, As_range)]
    fig.add_trace(
        go.Scatter(x=As_range, y=efficiency, mode='lines', name='Moment/Steel',
                  line=dict(color='purple', width=3)),
        row=2, col=2
    )
    
    # Update layout
    fig.update_xaxes(title_text="Steel Area (cm²)", row=1, col=1)
    fig.update_xaxes(title_text="Steel Ratio ρ", row=1, col=2)
    fig.update_xaxes(title_text="Steel Area (cm²)", row=2, col=1)
    fig.update_xaxes(title_text="Steel Area (cm²)", row=2, col=2)
    
    fig.update_yaxes(title_text="Moment (kg-m)", row=1, col=1)
    fig.update_yaxes(title_text="Phi Factor φ", row=1, col=2)
    fig.update_yaxes(title_text="Steel Ratio ρ", row=2, col=1)
    fig.update_yaxes(title_text="kg-m/cm²", row=2, col=2)
    
    fig.update_layout(
        title_text="Moment Capacity Behavior Analysis",
        showlegend=False,
        height=600
    )
    
    return fig

def interaction_diagram_PM(b, H, fc, fy, As_total, cover, Es=2.04e6):
    """Generate P-M interaction diagram for the beam section."""
    d = H - cover - 1.0  # Approximate effective depth
    beta1_val = beta1(fc)
    
    # Axial load range
    P_max = 0.85 * fc * b * H / 1000  # Maximum axial capacity in tons
    P_range = np.linspace(-P_max * 0.1, P_max * 0.8, 50)  # Include some tension
    
    moments = []
    
    for P in P_range:
        P_kg = P * 1000  # Convert to kg
        
        try:
            # For each axial load, find corresponding moment capacity
            # This is a simplified approach
            
            if P_kg >= 0:  # Compression
                # Approximate moment capacity with axial load
                # Using simplified interaction
                e = 0.1 * H  # Minimum eccentricity
                M = P_kg * e / 100000  # Convert to kg-m
                
                # Add flexural contribution
                As_eff = As_total
                c = (P_kg + As_eff * fy) / (0.85 * fc * beta1_val * b)
                
                if c > 0 and c < H:
                    a = beta1_val * c
                    M_flex = As_eff * fy * (d - a/2) / 100000
                    M += M_flex
                
            else:  # Tension
                M = 0
            
            moments.append(max(0, M))
            
        except:
            moments.append(0)
    
    return P_range, moments

def plot_interaction_diagram(P_range, moments):
    """Plot P-M interaction diagram."""
    fig = go.Figure()
    
    if len(P_range) > 0 and len(moments) > 0:
        fig.add_trace(go.Scatter(
            x=moments,
            y=P_range,
            mode='lines+markers',
            name='P-M Interaction',
            line=dict(color='darkblue', width=3),
            marker=dict(size=4)
        ))
        
        # Add balanced point indicator
        max_moment_idx = np.argmax(moments)
        fig.add_trace(go.Scatter(
            x=[moments[max_moment_idx]],
            y=[P_range[max_moment_idx]],
            mode='markers',
            name='Balanced Point',
            marker=dict(color='red', size=10, symbol='star')
        ))
    
    fig.update_layout(
        title='Axial Load - Moment Interaction Diagram',
        xaxis=dict(
            title='Moment (kg-m)',
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray'
        ),
        yaxis=dict(
            title='Axial Load (tons)',
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray'
        ),
        width=600,
        height=500
    )
    
    return fig

def calculate_required_steel(Mu, b, d, fc, fy):
    """Calculate required steel area for a given moment."""
    # Convert moment to kg-cm
    Mu_kg_cm = Mu * 100000
    
    # Assume phi = 0.9 for initial calculation
    phi = 0.9
    Mn_required = Mu_kg_cm / phi
    
    # For singly reinforced beam
    # Mn = As * fy * (d - a/2)
    # a = As * fy / (0.85 * fc * b)
    
    # Quadratic equation: As²*fy²/(1.7*fc*b) - As*fy*d + Mn = 0
    beta1_val = beta1(fc)
    
    a_coeff = fy**2 / (1.7 * fc * b)
    b_coeff = -fy * d
    c_coeff = Mn_required
    
    discriminant = b_coeff**2 - 4 * a_coeff * c_coeff
    
    if discriminant < 0:
        return None, "Moment too large for singly reinforced beam"
    
    As1 = (-b_coeff - math.sqrt(discriminant)) / (2 * a_coeff)
    As2 = (-b_coeff + math.sqrt(discriminant)) / (2 * a_coeff)
    
    # Take the smaller positive root
    As_required = min(As1, As2) if min(As1, As2) > 0 else max(As1, As2)
    
    # Check minimum reinforcement
    As_min = 1.4 * b * d / fy
    As_required = max(As_required, As_min)
    
    return As_required, f"Required As = {As_required:.2f} cm²"

# Initialize rebar database
@st.cache_data
def load_rebar_data():
    """Load rebar database."""
    rebar_data = {
        'Steel Rebar': ['DB6', 'DB9', 'DB12', 'DB16', 'DB20', 'DB25', 'DB28', 'DB32'],
        'Diameter (mm.)': [6, 9, 12, 16, 20, 25, 28, 32],
        'Area (cm²)': [0.28, 0.64, 1.13, 2.01, 3.14, 4.91, 6.16, 8.04],
        'Weight (kg/m)': [0.22, 0.50, 0.89, 1.58, 2.47, 3.85, 4.83, 6.31]
    }
    return pd.DataFrame(rebar_data).set_index('Steel Rebar')

# Main App
def main():
    st.title("🏗️ Advanced RC Beam Design - ACI Standard")
    st.markdown("*Enhanced with Moment-Curvature, Interaction Diagrams & Advanced Analysis*")
    st.markdown("---")

    # Load rebar database
    rebar_df = load_rebar_data()

    # Sidebar inputs
    st.sidebar.header("🔧 Beam Parameters")
    
    # Beam dimensions
    st.sidebar.subheader("Dimensions")
    b = st.sidebar.number_input("Beam width (b) [cm]", value=30.0, step=1.0, min_value=15.0)
    H = st.sidebar.number_input("Beam height (H) [cm]", value=60.0, step=1.0, min_value=25.0)
    
    # Material properties
    st.sidebar.subheader("Material Properties")
    fc = st.sidebar.selectbox("Concrete Strength f'c [kg/cm²]", [180, 200, 240, 280, 320, 350])
    grade_main = st.sidebar.selectbox("Main Rebar Grade", ["SD30", "SD40", "SD50"])
    grade_stirrup = st.sidebar.selectbox("Stirrup Grade", ["SR24", "SD40"])
    
    # Material properties calculation
    Ec = 15100 * (fc ** 0.5)
    Es = 2.04e6
    
    fy_values = {"SD30": 3000, "SD40": 4000, "SD50": 5000, "SR24": 2400}
    fy = fy_values[grade_main]
    fy_stirrup = fy_values[grade_stirrup]
    
    # Cover and stirrup
    st.sidebar.subheader("Cover & Stirrup")
    covering = st.sidebar.number_input("Concrete Cover [cm]", value=3.0, step=0.1, min_value=1.5)
    stirrup_type = st.sidebar.selectbox("Stirrup Type", rebar_df.index.tolist())
    ds = rebar_df.loc[stirrup_type, 'Diameter (mm.)'] / 10
    
    # Reinforcement layout
    st.sidebar.subheader("Reinforcement Layout")
    
    # Bottom steel
    bottom_rows = st.sidebar.number_input("Bottom Steel Rows", min_value=1, max_value=3, value=1)
    bottom_steel_list, bottom_diameters, bottom_steel_types = [], [], []
    
    for i in range(int(bottom_rows)):
        st.sidebar.write(f"**Bottom Row {i+1}**")
        n_bars = st.sidebar.number_input(f"Number of Bars", min_value=1, value=3, key=f"bot_n_{i}")
        rebar_choice = st.sidebar.selectbox(f"Rebar Type", rebar_df.index.tolist(), key=f"bot_rebar_{i}")
        dia_cm = rebar_df.loc[rebar_choice, 'Diameter (mm.)'] / 10
        
        bottom_steel_list.append(n_bars)
        bottom_diameters.append(dia_cm)
        bottom_steel_types.append(rebar_choice)
    
    bottom_row_spacing = st.sidebar.number_input("Bottom Row Spacing [cm]", value=2.0, step=0.1, min_value=1.0)
    
    # Top steel (compression)
    doubly_reinforced = st.sidebar.checkbox("Doubly Reinforced Beam")
    top_steel_list, top_diameters, top_steel_types = [], [], []
    
    if doubly_reinforced:
        top_rows = st.sidebar.number_input("Top Steel Rows", min_value=1, max_value=3, value=1)
        
        for i in range(int(top_rows)):
            st.sidebar.write(f"**Top Row {i+1}**")
            n_bars = st.sidebar.number_input(f"Number of Bars", min_value=1, value=2, key=f"top_n_{i}")
            rebar_choice = st.sidebar.selectbox(f"Rebar Type", rebar_df.index.tolist(), key=f"top_rebar_{i}")
            dia_cm = rebar_df.loc[rebar_choice, 'Diameter (mm.)'] / 10
            
            top_steel_list.append(n_bars)
            top_diameters.append(dia_cm)
            top_steel_types.append(rebar_choice)
        
        top_row_spacing = st.sidebar.number_input("Top Row Spacing [cm]", value=2.0, step=0.1, min_value=1.0)
    else:
        top_row_spacing = 2.0

    # Main content tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "📐 Beam Section", 
        "📊 Material Analysis", 
        "⚙️ Flexural Analysis",
        "📈 Moment-Curvature",
        "🎯 Capacity Analysis",
        "⚡ Interaction Diagram",
        "🔧 Steel Requirements",
        "📋 Design Summary"
    ])

    # Calculate basic properties for all tabs
    d = H - covering - ds - (bottom_diameters[0]/2 if bottom_diameters else 1.0)
    As = sum(n * rebar_df.loc[rebar_type, 'Area (cm²)'] 
            for n, rebar_type in zip(bottom_steel_list, bottom_steel_types))
    
    As_prime = 0
    d_prime = 0
    if doubly_reinforced and top_steel_list:
        d_prime = covering + ds + (top_diameters[0]/2 if top_diameters else 1.0)
        As_prime = sum(n * rebar_df.loc[rebar_type, 'Area (cm²)'] 
                      for n, rebar_type in zip(top_steel_list, top_steel_types))

    with tab1:
        st.header("Beam Cross-Section")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Plot beam section
            fig = plot_rc_beam_section(
                b, H, top_steel_list if doubly_reinforced else [], bottom_steel_list,
                top_diameters if doubly_reinforced else [], bottom_diameters,
                ds, covering, top_row_spacing, bottom_row_spacing
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Section Properties")
            st.write(f"**Width (b):** {b} cm")
            st.write(f"**Height (H):** {H} cm")
            st.write(f"**Cover:** {covering} cm")
            st.write(f"**Stirrup:** {stirrup_type} ({ds:.1f} cm)")
            st.write(f"**Effective depth (d):** {d:.1f} cm")
            
            if doubly_reinforced:
                st.write(f"**d' (compression steel):** {d_prime:.1f} cm")

        # Steel summary tables
        st.subheader("Reinforcement Summary")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Bottom Reinforcement (Tension)**")
            bottom_data = []
            total_As_bottom = 0
            
            for i, (n_bars, rebar_type, diameter) in enumerate(zip(bottom_steel_list, bottom_steel_types, bottom_diameters)):
                area_per_bar = rebar_df.loc[rebar_type, 'Area (cm²)']
                total_area = n_bars * area_per_bar
                total_As_bottom += total_area
                
                bottom_data.append({
                    'Layer': f'Layer {i+1}',
                    'Type': rebar_type,
                    'Quantity': n_bars,
                    'Area (cm²)': round(total_area, 2)
                })
            
            df_bottom = pd.DataFrame(bottom_data)
            st.dataframe(df_bottom, hide_index=True)
            st.write(f"**Total As = {total_As_bottom:.2f} cm²**")
        
        with col2:
            if doubly_reinforced and top_steel_list:
                st.write("**Top Reinforcement (Compression)**")
                top_data = []
                total_As_top = 0
                
                for i, (n_bars, rebar_type, diameter) in enumerate(zip(top_steel_list, top_steel_types, top_diameters)):
                    area_per_bar = rebar_df.loc[rebar_type, 'Area (cm²)']
                    total_area = n_bars * area_per_bar
                    total_As_top += total_area
                    
                    top_data.append({
                        'Layer': f'Layer {i+1}',
                        'Type': rebar_type,
                        'Quantity': n_bars,
                        'Area (cm²)': round(total_area, 2)
                    })
                
                df_top = pd.DataFrame(top_data)
                st.dataframe(df_top, hide_index=True)
                st.write(f"**Total As' = {total_As_top:.2f} cm²**")
            else:
                st.write("*No compression reinforcement*")

    with tab2:
        st.header("Material Properties Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Material Properties")
            
            material_props = pd.DataFrame({
                'Property': [
                    'Concrete Strength (f\'c)',
                    'Steel Yield Strength (fy)',
                    'Concrete Modulus (Ec)',
                    'Steel Modulus (Es)',
                    'β₁ Factor'
                ],
                'Value': [
                    f'{fc} kg/cm²',
                    f'{fy} kg/cm²',
                    f'{Ec:.0f} kg/cm²',
                    f'{Es:.0f} kg/cm²',
                    f'{beta1(fc):.3f}'
                ]
            })
            st.dataframe(material_props, hide_index=True)
        
        with col2:
            # Hognestad stress-strain curve
            ec, fci_list, e0, fig = hognestad_stress_strain(Ec, fc)
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.header("Flexural Strength Analysis")
        
        # Calculate flexural strength
        beta_val = beta1(fc)
        result = calculate_flexural_strength(As, As_prime, d, d_prime, b, fc, fy, beta_val)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Section Analysis")
            
            analysis_data = pd.DataFrame({
                'Parameter': [
                    'Effective depth (d)',
                    'Tension steel (As)',
                    'Compression steel (As\')',
                    'Reinforcement ratio (ρ)',
                    'Balanced ratio (ρb)',
                    'β₁ factor'
                ],
                'Value': [
                    f'{d:.1f} cm',
                    f'{As:.2f} cm²',
                    f'{As_prime:.2f} cm²' if As_prime > 0 else 'N/A',
                    f'{As/(b*d):.4f}',
                    f'{0.85 * beta_val * fc / fy * (600 / (600 + fy)):.4f}',
                    f'{beta_val:.3f}'
                ]
            })
            st.dataframe(analysis_data, hide_index=True)
        
        with col2:
            st.subheader("Strength Results")
            
            if result[0] is not None:
                st.success(f"✅ **Design Moment Capacity:**")
                st.write(f"**φMn = {result[0]:.2f} kg-m**")
                st.info(result[1])
                
                # Check reinforcement ratio
                rho = As / (b * d)
                rho_b = 0.85 * beta_val * fc / fy * (600 / (600 + fy))
                
                if rho < rho_b * 0.75:
                    st.success("✅ Under-reinforced section - Good design")
                elif rho < rho_b:
                    st.warning("⚠️ Moderately reinforced section")
                else:
                    st.error("❌ Over-reinforced section - Revise design")
            else:
                st.error("❌ " + result[1])

    with tab4:
        st.header("Moment-Curvature Analysis")
        
        if As > 0:
            with st.spinner("Calculating moment-curvature relationship..."):
                curvatures, moments = moment_curvature_analysis(As, As_prime, d, d_prime, b, H, fc, fy, Es)
            
            if curvatures and moments:
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    fig_mc = plot_moment_curvature(curvatures, moments)
                    st.plotly_chart(fig_mc, use_container_width=True)
                
                with col2:
                    st.subheader("Key Points")
                    if moments:
                        max_moment = max(moments)
                        max_idx = moments.index(max_moment)
                        max_curvature = curvatures[max_idx]
                        
                        st.write(f"**Maximum Moment:** {max_moment:.2f} kg-m")
                        st.write(f"**Curvature at Max:** {max_curvature:.6f} 1/cm")
                        
                        # Yield point approximation (assuming 70% of max moment)
                        yield_moment = max_moment * 0.7
                        yield_idx = next((i for i, m in enumerate(moments) if m >= yield_moment), 0)
                        yield_curvature = curvatures[yield_idx] if yield_idx < len(curvatures) else 0
                        
                        st.write(f"**Yield Moment (≈70%):** {yield_moment:.2f} kg-m")
                        st.write(f"**Yield Curvature:** {yield_curvature:.6f} 1/cm")
                        
                        # Ductility
                        if yield_curvature > 0:
                            ductility = max_curvature / yield_curvature
                            st.write(f"**Curvature Ductility:** {ductility:.2f}")
                
                # Summary table
                st.subheader("Moment-Curvature Data")
                mc_data = pd.DataFrame({
                    'Curvature (1/cm)': [f"{c:.6f}" for c in curvatures[::5]],  # Every 5th point
                    'Moment (kg-m)': [f"{m:.2f}" for m in moments[::5]]
                })
                st.dataframe(mc_data, hide_index=True)
            else:
                st.error("Could not generate moment-curvature data. Check section properties.")
        else:
            st.warning("Please define steel reinforcement in the sidebar.")

    with tab5:
        st.header("Moment Capacity Analysis")
        
        if As > 0:
            st.subheader("Capacity vs Steel Area (0.1 cm² increments)")
            
            with st.spinner("Analyzing moment capacity behavior..."):
                As_range, moments_cap, phi_factors, steel_ratios, rho_b = moment_capacity_analysis(
                    b, d, d_prime, fc, fy, beta1(fc), As_prime
                )
            
            if As_range.size > 0:
                # Plot behavior analysis
                fig_behavior = plot_moment_capacity_behavior(As_range, moments_cap, phi_factors, steel_ratios, rho_b)
                st.plotly_chart(fig_behavior, use_container_width=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Current Design Point")
                    current_ratio = As / (b * d)
                    st.write(f"**Current As:** {As:.2f} cm²")
                    st.write(f"**Current ρ:** {current_ratio:.4f}")
                    st.write(f"**Balanced ρb:** {rho_b:.4f}")
                    st.write(f"**Ratio ρ/ρb:** {current_ratio/rho_b:.3f}")
                    
                    if current_ratio < rho_b * 0.75:
                        st.success("✅ Under-reinforced (ρ < 0.75ρb)")
                    elif current_ratio < rho_b:
                        st.warning("⚠️ Moderately reinforced")
                    else:
                        st.error("❌ Over-reinforced (ρ > ρb)")
                
                with col2:
                    st.subheader("Optimization Insights")
                    
                    # Find optimal point (maximum efficiency)
                    efficiency = [m/a for m, a in zip(moments_cap, As_range)]
                    max_eff_idx = efficiency.index(max(efficiency))
                    optimal_As = As_range[max_eff_idx]
                    optimal_moment = moments_cap[max_eff_idx]
                    
                    st.write(f"**Optimal As:** {optimal_As:.1f} cm²")
                    st.write(f"**Max Efficiency:** {max(efficiency):.1f} kg-m/cm²")
                    st.write(f"**Moment at Optimal:** {optimal_moment:.1f} kg-m")
                    
                    if As < optimal_As * 0.9:
                        st.info("💡 Consider increasing steel for better efficiency")
                    elif As > optimal_As * 1.1:
                        st.info("💡 Current design may be over-conservative")
                
                # Data table
                st.subheader("Detailed Analysis Data")
                analysis_df = pd.DataFrame({
                    'Steel Area (cm²)': [f"{a:.1f}" for a in As_range[::2]],  # Every 2nd point
                    'Moment (kg-m)': [f"{m:.2f}" for m in moments_cap[::2]],
                    'Steel Ratio ρ': [f"{r:.4f}" for r in steel_ratios[::2]],
                    'Phi Factor φ': [f"{p:.3f}" for p in phi_factors[::2]],
                    'Efficiency': [f"{e:.1f}" for e in efficiency[::2]]
                })
                st.dataframe(analysis_df, hide_index=True)
        else:
            st.warning("Please define steel reinforcement in the sidebar.")

    with tab6:
        st.header("P-M Interaction Diagram")
        
        if As > 0:
            with st.spinner("Generating interaction diagram..."):
                P_range, M_range = interaction_diagram_PM(b, H, fc, fy, As, covering, Es)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                fig_interaction = plot_interaction_diagram(P_range, M_range)
                st.plotly_chart(fig_interaction, use_container_width=True)
            
            with col2:
                st.subheader("Interaction Analysis")
                
                max_moment = max(M_range) if len(M_range) > 0 else 0
                max_axial = max(P_range) if len(P_range) > 0 else 0
                balanced_idx = np.argmax(M_range) if len(M_range) > 0 else 0
                balanced_P = P_range[balanced_idx] if len(P_range) > 0 else 0
                
                st.write(f"**Pure Bending (P=0):** {M_range[len(M_range)//2]:.1f} kg-m" if len(M_range) > 0 else "**Pure Bending:** N/A")
                st.write(f"**Maximum Moment:** {max_moment:.1f} kg-m")
                st.write(f"**Balanced Point:** P = {balanced_P:.1f} tons")
                st.write(f"**Max Axial Load:** {max_axial:.1f} tons")
                
                st.subheader("Load Combinations")
                st.write("**Common Design Points:**")
                if len(M_range) > 0 and len(P_range) > 0:
                    st.write(f"• Pure flexure: M = {M_range[len(M_range)//2]:.1f} kg-m")
                    st.write(f"• Small axial: P = {max_axial*0.1:.1f} tons")
                    st.write(f"  M = {M_range[int(len(M_range)*0.2)]:.1f} kg-m")
                    st.write(f"• Moderate axial: P = {max_axial*0.3:.1f} tons")
                    st.write(f"  M = {M_range[int(len(M_range)*0.6)]:.1f} kg-m")
                else:
                    st.write("• Unable to calculate load combinations")
                
            # Interaction data table
            st.subheader("P-M Interaction Data")
            if len(P_range) > 0 and len(M_range) > 0:
                interaction_df = pd.DataFrame({
                    'Axial Load P (tons)': [f"{p:.2f}" for p in P_range[::3]],  # Every 3rd point
                    'Moment M (kg-m)': [f"{m:.1f}" for m in M_range[::3]]
                })
                st.dataframe(interaction_df, hide_index=True)
            else:
                st.write("No interaction data available")
        else:
            st.warning("Please define steel reinforcement in the sidebar.")

    with tab7:
        st.header("Steel Requirements Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Required vs Provided Steel")
            
            # Input moment for design
            design_moment = st.number_input(
                "Design Moment (kg-m)", 
                value=50.0, 
                step=5.0, 
                min_value=1.0,
                help="Enter the factored design moment"
            )
            
            # Calculate required steel
            As_required, req_message = calculate_required_steel(design_moment, b, d, fc, fy)
            
            if As_required:
                st.write(f"**Required Steel Area:** {As_required:.2f} cm²")
                st.write(f"**Provided Steel Area:** {As:.2f} cm²")
                
                ratio = As / As_required if As_required > 0 else 0
                st.write(f"**Provided/Required Ratio:** {ratio:.2f}")
                
                if ratio >= 1.0:
                    st.success(f"✅ Adequate steel provided ({((ratio-1)*100):.1f}% excess)")
                else:
                    st.error(f"❌ Insufficient steel ({((1-ratio)*100):.1f}% deficit)")
                    
                    # Suggest steel arrangement
                    st.subheader("Steel Arrangement Suggestions")
                    
                    # Find combinations that provide required area
                    suggestions = []
                    for rebar_type in rebar_df.index:
                        area_per_bar = rebar_df.loc[rebar_type, 'Area (cm²)']
                        n_bars_needed = math.ceil(As_required / area_per_bar)
                        
                        if n_bars_needed <= 8:  # Practical limit
                            total_provided = n_bars_needed * area_per_bar
                            excess = ((total_provided - As_required) / As_required) * 100
                            
                            suggestions.append({
                                'Rebar Type': rebar_type,
                                'Number of Bars': n_bars_needed,
                                'Total Area (cm²)': round(total_provided, 2),
                                'Excess (%)': round(excess, 1)
                            })
                    
                    suggestions_df = pd.DataFrame(suggestions)
                    suggestions_df = suggestions_df.sort_values('Excess (%)')
                    st.dataframe(suggestions_df, hide_index=True)
            else:
                st.error(req_message)
        
        with col2:
            st.subheader("Steel Ratio Analysis")
            
            # Current ratios
            rho_current = As / (b * d)
            rho_b = 0.85 * beta1(fc) * fc / fy * (600 / (600 + fy))
            rho_min = 1.4 / fy
            rho_max = 0.75 * rho_b
            
            ratio_data = pd.DataFrame({
                'Ratio Type': [
                    'Minimum (ρmin)',
                    'Current (ρ)',
                    'Maximum (0.75ρb)',
                    'Balanced (ρb)'
                ],
                'Value': [
                    f'{rho_min:.4f}',
                    f'{rho_current:.4f}',
                    f'{rho_max:.4f}',
                    f'{rho_b:.4f}'
                ],
                'Status': [
                    '✅' if rho_current >= rho_min else '❌',
                    '✅',
                    '✅' if rho_current <= rho_max else '❌',
                    '✅' if rho_current <= rho_b else '❌'
                ]
            })
            st.dataframe(ratio_data, hide_index=True)
            
            st.subheader("Design Recommendations")
            
            if rho_current < rho_min:
                st.error("❌ Below minimum reinforcement")
                min_As = rho_min * b * d
                st.write(f"Minimum required: {min_As:.2f} cm²")
            elif rho_current > rho_max:
                st.error("❌ Above maximum reinforcement")
                st.write("Consider:")
                st.write("• Increasing beam dimensions")
                st.write("• Using doubly reinforced design")
                st.write("• Higher strength concrete")
            elif rho_current > rho_b:
                st.error("❌ Over-reinforced section")
                st.write("Brittle failure mode - revise design")
            else:
                st.success("✅ Reinforcement within acceptable limits")
                
                if rho_current < rho_b * 0.5:
                    st.info("💡 Conservative design - could optimize")
                elif rho_current < rho_b * 0.75:
                    st.success("✅ Well-balanced design")
                else:
                    st.warning("⚠️ Approaching upper limit")

        # Steel comparison table
        st.subheader("Comprehensive Steel Analysis")
        
        comparison_data = []
        
        # Current design
        current_result = calculate_flexural_strength(As, As_prime, d, d_prime, b, fc, fy, beta1(fc))
        comparison_data.append({
            'Configuration': 'Current Design',
            'As (cm²)': f'{As:.2f}',
            'As\' (cm²)': f'{As_prime:.2f}' if As_prime > 0 else 'N/A',
            'ρ': f'{As/(b*d):.4f}',
            'φMn (kg-m)': f'{current_result[0]:.1f}' if current_result[0] else 'N/A',
            'Status': '✅' if current_result[0] else '❌'
        })
        
        # Minimum steel
        As_min_calc = 1.4 * b * d / fy
        min_result = calculate_flexural_strength(As_min_calc, 0, d, d_prime, b, fc, fy, beta1(fc))
        comparison_data.append({
            'Configuration': 'Minimum Steel',
            'As (cm²)': f'{As_min_calc:.2f}',
            'As\' (cm²)': 'N/A',
            'ρ': f'{As_min_calc/(b*d):.4f}',
            'φMn (kg-m)': f'{min_result[0]:.1f}' if min_result[0] else 'N/A',
            'Status': '✅' if min_result[0] else '❌'
        })
        
        # Balanced steel
        As_balanced = rho_b * b * d
        balanced_result = calculate_flexural_strength(As_balanced, 0, d, d_prime, b, fc, fy, beta1(fc))
        comparison_data.append({
            'Configuration': 'Balanced Steel',
            'As (cm²)': f'{As_balanced:.2f}',
            'As\' (cm²)': 'N/A',
            'ρ': f'{rho_b:.4f}',
            'φMn (kg-m)': f'{balanced_result[0]:.1f}' if balanced_result[0] else 'N/A',
            'Status': '⚠️'
        })
        
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, hide_index=True)

    with tab8:
        st.header("Comprehensive Design Summary")
        
        # Create comprehensive summary
        st.subheader("Project Information")
        st.write("**Design Code:** ACI 318 Building Code Requirements")
        st.write("**Analysis Type:** Advanced Flexural Design of RC Beam")
        st.write("**Analysis Features:** Moment-Curvature, P-M Interaction, Capacity Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Beam Specifications")
            
            summary_data = {
                'Section Dimensions': [f'{b} × {H} cm'],
                'Concrete Grade': [f'f\'c = {fc} kg/cm²'],
                'Steel Grade': [f'{grade_main} (fy = {fy} kg/cm²)'],
                'Cover': [f'{covering} cm'],
                'Effective Depth': [f'{d:.1f} cm']
            }
            
            if doubly_reinforced:
                summary_data.update({
                    'Reinforcement Type': ['Doubly Reinforced'],
                    'Tension Steel': [f'{As:.2f} cm² ({", ".join([f"{n}{t}" for n, t in zip(bottom_steel_list, bottom_steel_types)])})'],
                    'Compression Steel': [f'{As_prime:.2f} cm² ({", ".join([f"{n}{t}" for n, t in zip(top_steel_list, top_steel_types)])})']
                })
            else:
                summary_data.update({
                    'Reinforcement Type': ['Singly Reinforced'],
                    'Tension Steel': [f'{As:.2f} cm² ({", ".join([f"{n}{t}" for n, t in zip(bottom_steel_list, bottom_steel_types)])})']
                })
            
            for key, value in summary_data.items():
                st.write(f"**{key}:** {value[0]}")
        
        with col2:
            st.subheader("Design Results Summary")
            
            result = calculate_flexural_strength(As, As_prime, d, d_prime, b, fc, fy, beta1(fc))
            
            if result[0] is not None:
                st.write(f"**Nominal Moment Capacity:** φMn = {result[0]:.2f} kg-m")
                st.write(f"**Phi Factor:** φ = {result[3]:.3f}")
                st.write(f"**Steel Ratio:** ρ = {As/(b*d):.4f}")
                st.write(f"**Balanced Ratio:** ρb = {0.85 * beta1(fc) * fc / fy * (600 / (600 + fy)):.4f}")
                
                rho = As / (b * d)
                rho_b = 0.85 * beta1(fc) * fc / fy * (600 / (600 + fy))
                
                if rho < rho_b:
                    st.success("**Design Status:** ✅ Satisfactory")
                else:
                    st.error("**Design Status:** ❌ Over-reinforced")
            else:
                st.error("**Design Status:** ❌ Failed Analysis")
        
        # Performance metrics
        st.subheader("Performance Metrics")
        
        metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
        
        with metrics_col1:
            st.metric(
                "Moment Capacity", 
                f"{result[0]:.1f} kg-m" if result[0] else "N/A",
                help="Factored moment capacity (φMn)"
            )
            
        with metrics_col2:
            rho = As / (b * d) if As > 0 else 0
            st.metric(
                "Steel Ratio", 
                f"{rho:.4f}",
                help="Reinforcement ratio (ρ = As/bd)"
            )
            
        with metrics_col3:
            efficiency = result[0] / As if result[0] and As > 0 else 0
            st.metric(
                "Efficiency", 
                f"{efficiency:.1f} kg-m/cm²" if efficiency > 0 else "N/A",
                help="Moment capacity per unit steel area"
            )
        
        # Design verification checklist
        st.subheader("Design Verification Checklist")
        
        checklist_items = []
        
        # Check minimum steel
        rho_min = 1.4 / fy
        checklist_items.append({
            'Check': 'Minimum Reinforcement',
            'Requirement': f'ρ ≥ {rho_min:.4f}',
            'Actual': f'{rho:.4f}',
            'Status': '✅ Pass' if rho >= rho_min else '❌ Fail'
        })
        
        # Check maximum steel
        rho_max = 0.75 * 0.85 * beta1(fc) * fc / fy * (600 / (600 + fy))
        checklist_items.append({
            'Check': 'Maximum Reinforcement',
            'Requirement': f'ρ ≤ {rho_max:.4f}',
            'Actual': f'{rho:.4f}',
            'Status': '✅ Pass' if rho <= rho_max else '❌ Fail'
        })
        
        # Check spacing
        max_spacing = min(3 * H, 45)  # cm
        checklist_items.append({
            'Check': 'Bar Spacing',
            'Requirement': f'≤ {max_spacing:.0f} cm',
            'Actual': 'Check manually',
            'Status': '⚠️ Manual'
        })
        
        # Check cover
        min_cover = 2.5 if fc >= 280 else 3.0
        checklist_items.append({
            'Check': 'Concrete Cover',
            'Requirement': f'≥ {min_cover:.1f} cm',
            'Actual': f'{covering:.1f} cm',
            'Status': '✅ Pass' if covering >= min_cover else '❌ Fail'
        })
        
        # Check development length (simplified)
        checklist_items.append({
            'Check': 'Development Length',
            'Requirement': 'Check ACI 318',
            'Actual': 'Manual verification required',
            'Status': '⚠️ Manual'
        })
        
        checklist_df = pd.DataFrame(checklist_items)
        st.dataframe(checklist_df, hide_index=True)
        
        # Export summary
        st.subheader("Export Options")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📄 Generate PDF Report"):
                st.info("PDF generation feature - would integrate with reportlab library")
        
        with col2:
            if st.button("📊 Export Data to Excel"):
                st.info("Excel export feature - would integrate with openpyxl library")
        
        with col3:
            if st.button("📋 Copy Summary"):
                summary_text = f"""
RC Beam Design Summary - ACI 318
================================
Section: {b} × {H} cm
Concrete: f'c = {fc} kg/cm²
Steel: {grade_main} (fy = {fy} kg/cm²)
Reinforcement: {As:.2f} cm² tension
Moment Capacity: φMn = {result[0]:.2f} kg-m
Steel Ratio: ρ = {rho:.4f}
Status: {'Pass' if rho >= rho_min and rho <= rho_max else 'Fail'}
                """
                st.code(summary_text)

        # References and database
        st.subheader("Design References & Database")
        
        with st.expander("📚 Rebar Database"):
            st.dataframe(rebar_df, use_container_width=True)
        
        with st.expander("📖 ACI 318 Design Provisions"):
            st.write("""
            **Key Design Provisions:**
            
            **Strength Reduction Factors (Table 21.2.1):**
            - Tension-controlled sections: φ = 0.90
            - Compression-controlled sections: φ = 0.65
            - Transition zone: φ varies linearly between 0.65 and 0.90
            
            **Reinforcement Limits:**
            - Minimum: ρmin = 1.4/fy
            - Maximum: ρmax = 0.75ρb
            - Balanced: ρb = 0.85β₁(f'c/fy)(600/(600+fy))
            
            **Material Properties:**
            - Concrete: Ec = 15,100√f'c (kg/cm²)
            - Steel: Es = 2,040,000 kg/cm²
            - β₁ factor varies with concrete strength
            
            **Development Length:**
            - Basic: ld = (fy×ψt×ψe×ψs×λ)/(25×√f'c) × db
            - Minimum: 300 mm for #19 and smaller
            """)
        
        with st.expander("🔬 Analysis Methods"):
            st.write("""
            **Implemented Analysis Methods:**
            
            **1. Moment-Curvature Analysis:**
            - Strain compatibility method
            - Hognestad concrete stress-strain model
            - Elastic-perfectly plastic steel model
            - Iterative solution for neutral axis
            
            **2. P-M Interaction:**
            - Whitney stress block for concrete
            - Linear strain distribution assumption
            - Force and moment equilibrium
            
            **3. Capacity Analysis:**
            - Parametric study with steel area increments
            - Efficiency optimization analysis
            - Reinforcement ratio limits checking
            
            **4. Steel Requirements:**
            - Moment-area relationship
            - Minimum and maximum reinforcement checks
            - Practical arrangement suggestions
            """)
        
        with st.expander("⚠️ Limitations & Assumptions"):
            st.write("""
            **Design Assumptions:**
            1. Plane sections remain plane
            2. Perfect bond between concrete and steel
            3. Tensile strength of concrete neglected
            4. Linear elastic steel behavior up to yield
            5. Hognestad parabolic-linear concrete model
            6. Maximum concrete strain = 0.003
            
            **Analysis Limitations:**
            1. Does not include time-dependent effects
            2. Simplified P-M interaction (more detailed analysis needed for actual design)
            3. Does not check lateral stability
            4. Development length requires manual verification
            5. Shear design not included
            6. Serviceability checks not implemented
            
            **Recommended Verifications:**
            - Detailed development length calculations
            - Crack width and deflection checks
            - Shear and torsion design
            - Lateral-torsional buckling analysis
            - Construction and durability requirements
            """)

if __name__ == "__main__":
    main()