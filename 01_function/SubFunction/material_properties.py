"""
Material Properties Module
ACI 318 material parameters and conversions
"""

import numpy as np


class MaterialProperties:
    """ACI 318 material constants and properties"""
    
    # Default material strengths (ksc = kg/cm²)
    DEFAULT_FC = 280.0
    DEFAULT_FY_MAIN = 4200.0
    DEFAULT_FY_STIRRUP = 2800.0
    
    # Elastic modulus
    ES = 2.04e6  # ksc (Steel)
    
    # ACI 318 strain limits
    EPSILON_C_MAX = 0.003  # Ultimate concrete strain
    EPSILON_TY = 0.002     # Yield strain
    EPSILON_T_TENSION = 0.005  # Tension-controlled limit
    
    # Standard bar sizes
    BAR_DIAMETERS = [10, 12, 16, 20, 25, 28, 32]  # mm
    STIRRUP_DIAMETERS = [8, 10, 12]  # mm


def calculate_beta1(fc_ksc):
    """
    Calculate β₁ factor per ACI 318-19
    
    Args:
        fc_ksc: Concrete strength in ksc
        
    Returns:
        float: β₁ factor
    """
    fc_mpa = fc_ksc * 0.0980665
    
    if fc_mpa <= 28:
        return 0.85
    elif fc_mpa >= 55:
        return 0.65
    else:
        return 0.85 - 0.05 * (fc_mpa - 28) / 7


def calculate_phi_factor(epsilon_t):
    """
    Calculate strength reduction factor φ per ACI 318-19
    
    Args:
        epsilon_t: Tension steel strain
        
    Returns:
        tuple: (phi_factor, control_mode_string)
    """
    epsilon_ty = MaterialProperties.EPSILON_TY
    
    if epsilon_t >= 0.005:
        return 0.90, "Tension-Controlled ✓"
    elif epsilon_t <= epsilon_ty:
        return 0.65, "Compression-Controlled ⚠"
    else:
        phi = 0.65 + (epsilon_t - epsilon_ty) * (0.25 / (0.005 - epsilon_ty))
        return round(phi, 3), "Transition Zone ⚡"


def get_bar_area(diameter_mm):
    """Calculate bar area in mm²"""
    return np.pi * (diameter_mm ** 2) / 4


def convert_units(value, from_unit, to_unit):
    """Convert between common structural units"""
    conversions = {
        ('ksc', 'mpa'): 0.0980665,
        ('mpa', 'ksc'): 10.197162,
        ('tonf', 'n'): 9806.65,
        ('n', 'tonf'): 0.000101972,
        ('tonfm', 'nmm'): 9.80665e9,
        ('nmm', 'tonfm'): 1.01972e-10
    }
    
    key = (from_unit.lower(), to_unit.lower())
    if key in conversions:
        return value * conversions[key]
    return value
