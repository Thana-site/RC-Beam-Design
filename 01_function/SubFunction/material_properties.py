"""
Material Configuration Module
Handles material properties and ACI 318 code parameters
"""

import numpy as np


class MaterialConfig:
    """Material properties and ACI 318 constants"""
    
    # Default material properties (ksc units)
    DEFAULT_FC = 280.0  # Concrete strength (ksc)
    DEFAULT_FY_MAIN = 4200.0  # Main steel grade (ksc)
    DEFAULT_FY_STIRRUP = 2800.0  # Stirrup steel grade (ksc)
    
    # Steel modulus of elasticity
    ES = 2.04e6  # ksc (200 GPa)
    
    # ACI 318 strain limits
    EPSILON_C = 0.003  # Ultimate concrete strain
    EPSILON_TY = 0.002  # Yield strain transition
    EPSILON_T_LIMIT = 0.005  # Tension-controlled limit
    
    # Standard bar diameters (mm)
    BAR_SIZES = [10, 12, 16, 20, 25, 28, 32]
    
    # Stirrup sizes (mm)
    STIRRUP_SIZES = [8, 10, 12]


def calculate_beta1(fc_ksc):
    """
    Calculate β₁ factor based on concrete strength per ACI 318
    
    Parameters:
    -----------
    fc_ksc : float
        Concrete compressive strength in ksc (kg/cm²)
    
    Returns:
    --------
    float : β₁ factor
    """
    # Convert ksc to MPa (1 ksc = 0.0980665 MPa)
    fc_mpa = fc_ksc * 0.0980665
    
    if fc_mpa <= 28:
        return 0.85
    elif fc_mpa >= 55:
        return 0.65
    else:
        # Linear interpolation between 28 and 55 MPa
        return 0.85 - 0.05 * (fc_mpa - 28) / 7


def calculate_phi_factor(epsilon_t, epsilon_ty=MaterialConfig.EPSILON_TY):
    """
    Calculate strength reduction factor φ based on tension strain per ACI 318
    
    Parameters:
    -----------
    epsilon_t : float
        Tension steel strain
    epsilon_ty : float
        Yield strain (default 0.002)
    
    Returns:
    --------
    tuple : (phi_factor, control_type_string)
    """
    if epsilon_t >= 0.005:
        return 0.90, "Tension-Controlled ✓"
    elif epsilon_t <= epsilon_ty:
        return 0.65, "Compression-Controlled ⚠"
    else:
        # Transition zone - linear interpolation
        phi = 0.65 + (epsilon_t - epsilon_ty) * (0.25 / (0.005 - epsilon_ty))
        return round(phi, 3), "Transition Zone ⚡"


def get_bar_area(diameter_mm):
    """
    Calculate area of a single reinforcement bar
    
    Parameters:
    -----------
    diameter_mm : float
        Bar diameter in mm
    
    Returns:
    --------
    float : Bar area in mm²
    """
    return np.pi * (diameter_mm ** 2) / 4


def convert_ksc_to_mpa(ksc):
    """Convert ksc (kg/cm²) to MPa"""
    return ksc * 0.0980665


def convert_mpa_to_ksc(mpa):
    """Convert MPa to ksc (kg/cm²)"""
    return mpa / 0.0980665
