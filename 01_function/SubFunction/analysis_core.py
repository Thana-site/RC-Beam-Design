"""
Analysis Core Module
Contains all structural analysis calculations per ACI 318
"""

import numpy as np
from .material_config import (
    MaterialConfig, 
    calculate_beta1, 
    calculate_phi_factor,
    get_bar_area
)


def analyze_flexural_capacity(b_mm, h_mm, cover_mm, bars_df, fc_ksc, fy_ksc):
    """
    Analyze RC beam section for flexural capacity per ACI 318
    
    Parameters:
    -----------
    b_mm : float
        Beam width in mm
    h_mm : float
        Beam total height in mm
    cover_mm : float
        Concrete cover in mm
    bars_df : DataFrame
        Reinforcement data with columns: Diameter, Number, Layer
    fc_ksc : float
        Concrete strength in ksc
    fy_ksc : float
        Steel yield strength in ksc
    
    Returns:
    --------
    dict : Analysis results or None if error
    """
    
    if bars_df.empty or bars_df['Number of Bars'].sum() == 0:
        return None
    
    # Material properties
    fc_mpa = fc_ksc * 0.0980665
    fy_mpa = fy_ksc * 0.0980665
    Es = MaterialConfig.ES * 0.0980665  # Convert to MPa
    epsilon_c = MaterialConfig.EPSILON_C
    beta1 = calculate_beta1(fc_ksc)
    
    # Process reinforcement bars
    bar_data = []
    for idx, row in bars_df.iterrows():
        num_bars = int(row['Number of Bars'])
        if num_bars == 0:
            continue
        
        dia_mm = row['Diameter (mm)']
        As_single = get_bar_area(dia_mm)
        As_total = As_single * num_bars
        
        # Determine y-position from top
        layer = str(row['Layer Position'])
        if 'Top' in layer or 'top' in layer:
            y_mm = cover_mm + dia_mm / 2
        elif 'Bottom' in layer or 'bottom' in layer:
            y_mm = h_mm - cover_mm - dia_mm / 2
        elif 'Middle' in layer or 'Mid' in layer or 'middle' in layer:
            y_mm = h_mm / 2
        else:
            # Try to parse as number (custom y-distance from top)
            try:
                y_mm = float(layer)
            except:
                y_mm = h_mm - cover_mm - dia_mm / 2  # Default to bottom
        
        bar_data.append({
            'y_mm': y_mm,
            'd_mm': h_mm - y_mm,  # Distance from compression face to bar
            'As_mm2': As_total,
            'dia_mm': dia_mm,
            'num_bars': num_bars
        })
    
    if not bar_data:
        return None
    
    # Effective depth (to centroid of tension steel)
    tension_bars = [b for b in bar_data if b['y_mm'] > h_mm / 2]
    if tension_bars:
        d_mm = max([b['d_mm'] for b in tension_bars])
    else:
        d_mm = h_mm - cover_mm - 20  # Default assumption
    
    # Iterative solution for neutral axis depth
    c_mm = d_mm / 2  # Initial guess
    tolerance = 0.01  # Force equilibrium tolerance (N)
    max_iter = 100
    
    for iteration in range(max_iter):
        # Calculate compression force in concrete
        a_mm = beta1 * c_mm
        C_concrete = 0.85 * fc_mpa * b_mm * a_mm  # N
        
        # Calculate forces in steel bars
        T_steel = 0  # Tension force
        C_steel = 0  # Compression force in steel
        
        for bar in bar_data:
            y = bar['y_mm']
            As = bar['As_mm2']
            
            # Calculate strain at bar location
            epsilon_s = epsilon_c * (c_mm - y) / c_mm
            
            # Calculate stress (elastoplastic model)
            if abs(epsilon_s * Es) <= fy_mpa:
                fs = epsilon_s * Es
            else:
                fs = fy_mpa * np.sign(epsilon_s)
            
            # Force in bar
            F = As * fs  # N
            
            if y < c_mm:  # Compression zone
                C_steel += F
            else:  # Tension zone
                T_steel += abs(F)
        
        # Total forces
        C_total = C_concrete + C_steel
        force_error = abs(C_total - T_steel)
        
        # Check convergence
        if force_error < tolerance:
            break
        
        # Update neutral axis depth
        if C_total > T_steel:
            c_mm *= 0.98  # Reduce c
        else:
            c_mm *= 1.02  # Increase c
        
        # Bounds check
        c_mm = max(10, min(c_mm, h_mm - 10))
    
    # Calculate nominal moment capacity
    a_mm = beta1 * c_mm
    
    # Concrete contribution
    C_c = 0.85 * fc_mpa * b_mm * a_mm
    M_concrete = C_c * (d_mm - a_mm / 2)  # N·mm
    
    # Steel contribution
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
        M_steel += F * (d_mm - (h_mm - y))  # N·mm
    
    Mn_Nmm = abs(M_concrete + M_steel)
    Mn_tonfm = Mn_Nmm / 9.80665e6  # Convert to tonf·m
    
    # Calculate tension steel strain
    epsilon_t = epsilon_c * (d_mm - c_mm) / c_mm
    
    # Get phi factor and control type
    phi, control_type = calculate_phi_factor(epsilon_t)
    
    # Design moment capacity
    phi_Mn_tonfm = phi * Mn_tonfm
    
    # Prepare strain profile
    strain_profile = {
        'c_mm': c_mm,
        'a_mm': a_mm,
        'd_mm': d_mm,
        'epsilon_c': epsilon_c,
        'epsilon_t': epsilon_t,
        'bar_strains': [(b['y_mm'], epsilon_c * (c_mm - b['y_mm']) / c_mm) 
                        for b in bar_data]
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
    """
    Calculate shear capacity per ACI 318
    
    Parameters:
    -----------
    b_mm : float
        Beam width in mm
    h_mm : float
        Total beam height in mm
    d_mm : float
        Effective depth in mm
    fc_ksc : float
        Concrete strength in ksc
    stirrup_df : DataFrame
        Stirrup data with columns: Diameter, Legs, Spacing, Grade
    
    Returns:
    --------
    dict : Shear capacity results
    """
    fc_mpa = fc_ksc * 0.0980665
    
    # Concrete shear capacity per ACI 318
    # Vc = 0.17 * sqrt(f'c) * b * d
    Vc_N = 0.17 * np.sqrt(fc_mpa) * b_mm * d_mm
    
    # Steel shear capacity (if stirrups provided)
    Vs_N = 0
    if not stirrup_df.empty and stirrup_df['Number of Legs'].sum() > 0:
        for idx, row in stirrup_df.iterrows():
            dia_mm = row['Diameter (mm)']
            n_legs = int(row['Number of Legs'])
            spacing_mm = row['Spacing (mm)']
            fy_stirrup_ksc = row['Steel Grade (ksc)']
            
            if spacing_mm > 0 and n_legs > 0:
                Av = n_legs * get_bar_area(dia_mm)  # mm²
                fy_stirrup_mpa = fy_stirrup_ksc * 0.0980665
                Vs_N += Av * fy_stirrup_mpa * d_mm / spacing_mm
    
    # Total shear capacity
    Vn_N = Vc_N + Vs_N
    phi_shear = 0.75  # ACI 318 shear reduction factor
    phi_Vn_N = phi_shear * Vn_N
    
    # Convert to tonf
    Vn_tonf = Vn_N / 9806.65
    phi_Vn_tonf = phi_Vn_N / 9806.65
    Vc_tonf = Vc_N / 9806.65
    Vs_tonf = Vs_N / 9806.65
    
    return {
        'Vc_tonf': round(Vc_tonf, 2),
        'Vs_tonf': round(Vs_tonf, 2),
        'Vn_tonf': round(Vn_tonf, 2),
        'phi_Vn_tonf': round(phi_Vn_tonf, 2),
        'phi': phi_shear
    }


def calculate_minimum_reinforcement(b_mm, d_mm, fc_ksc, fy_ksc):
    """
    Calculate minimum reinforcement per ACI 318
    
    Parameters:
    -----------
    b_mm, d_mm : float
        Beam dimensions in mm
    fc_ksc, fy_ksc : float
        Material strengths in ksc
    
    Returns:
    --------
    float : Minimum steel area in mm²
    """
    fc_mpa = fc_ksc * 0.0980665
    fy_mpa = fy_ksc * 0.0980665
    
    # ACI 318: As,min = max(0.25*sqrt(f'c)*bw*d/fy, 1.4*bw*d/fy)
    As_min_1 = 0.25 * np.sqrt(fc_mpa) * b_mm * d_mm / fy_mpa
    As_min_2 = 1.4 * b_mm * d_mm / fy_mpa
    
    return max(As_min_1, As_min_2)


def calculate_maximum_reinforcement(b_mm, d_mm, fc_ksc, fy_ksc):
    """
    Calculate maximum reinforcement (0.75*ρ_balanced) per ACI 318
    
    Parameters:
    -----------
    b_mm, d_mm : float
        Beam dimensions in mm
    fc_ksc, fy_ksc : float
        Material strengths in ksc
    
    Returns:
    --------
    float : Maximum steel area in mm²
    """
    fc_mpa = fc_ksc * 0.0980665
    fy_mpa = fy_ksc * 0.0980665
    beta1 = calculate_beta1(fc_ksc)
    
    # Balanced reinforcement ratio
    rho_b = 0.85 * beta1 * fc_mpa / fy_mpa * (600 / (600 + fy_mpa))
    
    # Maximum allowed = 0.75 * rho_balanced
    rho_max = 0.75 * rho_b
    
    return rho_max * b_mm * d_mm
