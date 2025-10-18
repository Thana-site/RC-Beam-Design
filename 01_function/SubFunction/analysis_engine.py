"""
Analysis Engine Module
Core structural analysis calculations per ACI 318-19
"""

import numpy as np
from .material_properties import (
    MaterialProperties,
    calculate_beta1,
    calculate_phi_factor,
    get_bar_area
)


def analyze_flexural_capacity(b_mm, h_mm, cover_mm, bars_df, fc_ksc, fy_ksc):
    """
    Analyze flexural capacity of RC beam section
    
    Args:
        b_mm: Beam width (mm)
        h_mm: Beam height (mm)
        cover_mm: Concrete cover (mm)
        bars_df: DataFrame with reinforcement data
        fc_ksc: Concrete strength (ksc)
        fy_ksc: Steel yield strength (ksc)
        
    Returns:
        dict: Complete analysis results or None if invalid
    """
    
    if bars_df.empty or bars_df['Number of Bars'].sum() == 0:
        return None
    
    # Convert material properties to MPa
    fc_mpa = fc_ksc * 0.0980665
    fy_mpa = fy_ksc * 0.0980665
    Es = MaterialProperties.ES * 0.0980665
    epsilon_c = MaterialProperties.EPSILON_C_MAX
    beta1 = calculate_beta1(fc_ksc)
    
    # Process reinforcement bars
    bar_data = []
    for _, row in bars_df.iterrows():
        num_bars = int(row['Number of Bars'])
        if num_bars == 0:
            continue
        
        dia_mm = row['Diameter (mm)']
        As_single = get_bar_area(dia_mm)
        As_total = As_single * num_bars
        
        # Determine position from layer
        layer = str(row['Layer Position']).strip()
        if 'top' in layer.lower():
            y_mm = cover_mm + dia_mm / 2
        elif 'bottom' in layer.lower():
            y_mm = h_mm - cover_mm - dia_mm / 2
        elif 'mid' in layer.lower():
            y_mm = h_mm / 2
        else:
            try:
                y_mm = float(layer)
            except:
                y_mm = h_mm - cover_mm - dia_mm / 2
        
        bar_data.append({
            'y_mm': y_mm,
            'd_mm': h_mm - y_mm,
            'As_mm2': As_total,
            'dia_mm': dia_mm,
            'num_bars': num_bars
        })
    
    if not bar_data:
        return None
    
    # Effective depth to tension steel
    tension_bars = [b for b in bar_data if b['y_mm'] > h_mm / 2]
    d_mm = max([b['d_mm'] for b in tension_bars]) if tension_bars else h_mm - cover_mm - 20
    
    # Iterative solution for neutral axis
    c_mm = d_mm / 2
    tolerance = 0.01
    max_iterations = 100
    
    for _ in range(max_iterations):
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
        
        force_error = abs((C_concrete + C_steel) - T_steel)
        
        if force_error < tolerance:
            break
        
        if (C_concrete + C_steel) > T_steel:
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
    
    # Tension strain and phi factor
    epsilon_t = epsilon_c * (d_mm - c_mm) / c_mm
    phi, control_type = calculate_phi_factor(epsilon_t)
    phi_Mn_tonfm = phi * Mn_tonfm
    
    # Strain profile data
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
    """
    Calculate shear capacity per ACI 318-19
    
    Args:
        b_mm: Beam width (mm)
        h_mm: Beam height (mm)
        d_mm: Effective depth (mm)
        fc_ksc: Concrete strength (ksc)
        stirrup_df: DataFrame with stirrup data
        
    Returns:
        dict: Shear capacity results
    """
    fc_mpa = fc_ksc * 0.0980665
    
    # Concrete contribution
    Vc_N = 0.17 * np.sqrt(fc_mpa) * b_mm * d_mm
    
    # Steel contribution
    Vs_N = 0
    if not stirrup_df.empty and stirrup_df['Number of Legs'].sum() > 0:
        for _, row in stirrup_df.iterrows():
            dia_mm = row['Diameter (mm)']
            n_legs = int(row['Number of Legs'])
            spacing_mm = row['Spacing (mm)']
            fy_stirrup_ksc = row['Steel Grade (ksc)']
            
            if spacing_mm > 0 and n_legs > 0:
                Av = n_legs * get_bar_area(dia_mm)
                fy_stirrup_mpa = fy_stirrup_ksc * 0.0980665
                Vs_N += Av * fy_stirrup_mpa * d_mm / spacing_mm
    
    # Total capacity with phi factor
    Vn_N = Vc_N + Vs_N
    phi_shear = 0.75
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


def calculate_reinforcement_limits(b_mm, d_mm, fc_ksc, fy_ksc):
    """
    Calculate minimum and maximum reinforcement per ACI 318-19
    
    Returns:
        dict: Min and max steel areas
    """
    fc_mpa = fc_ksc * 0.0980665
    fy_mpa = fy_ksc * 0.0980665
    beta1 = calculate_beta1(fc_ksc)
    
    # Minimum reinforcement
    As_min_1 = 0.25 * np.sqrt(fc_mpa) * b_mm * d_mm / fy_mpa
    As_min_2 = 1.4 * b_mm * d_mm / fy_mpa
    As_min = max(As_min_1, As_min_2)
    
    # Maximum reinforcement (0.75 * balanced)
    rho_b = 0.85 * beta1 * fc_mpa / fy_mpa * (600 / (600 + fy_mpa))
    rho_max = 0.75 * rho_b
    As_max = rho_max * b_mm * d_mm
    
    return {
        'As_min_mm2': round(As_min, 0),
        'As_max_mm2': round(As_max, 0),
        'rho_min': round(As_min / (b_mm * d_mm), 6),
        'rho_max': round(rho_max, 6)
    }
