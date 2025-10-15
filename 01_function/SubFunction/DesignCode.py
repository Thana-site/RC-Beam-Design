import streamlit as st
import pandas as pd
import math
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from PIL import Image
import io

def beta1(fc):
    """Calculates β1 based on concrete compressive strength (fc in kg/cm²)."""
    if fc <= 280:
        return 0.85
    elif 280 < fc <= 550:
        return 0.85 - ((fc - 280) / 70) * 0.05
    else:
        return 0.65
    
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


#--------------------------------------------------

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