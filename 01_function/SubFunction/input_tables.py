"""
Input Tables Module
Handles reinforcement input tables using st.data_editor
"""

import streamlit as st
import pandas as pd
from .material_config import MaterialConfig


def create_main_reinforcement_table(key_suffix=""):
    """
    Create editable table for main reinforcement input
    
    Parameters:
    -----------
    key_suffix : str
        Suffix for session state key (for multiple sections)
    
    Returns:
    --------
    DataFrame : Updated reinforcement data
    """
    
    session_key = f'bars_df{key_suffix}'
    
    # Initialize default data if not exists
    if session_key not in st.session_state:
        st.session_state[session_key] = pd.DataFrame({
            'Diameter (mm)': [20, 20],
            'Number of Bars': [3, 2],
            'Layer Position': ['Bottom', 'Top']
        })
    
    st.markdown("#### 🔩 Main Reinforcement")
    
    edited_df = st.data_editor(
        st.session_state[session_key],
        num_rows="dynamic",
        use_container_width=True,
        key=f"bars_editor{key_suffix}",
        column_config={
            "Diameter (mm)": st.column_config.NumberColumn(
                "Diameter (mm)",
                help="Bar diameter in mm",
                min_value=10,
                max_value=40,
                step=2,
                format="%.0f",
                width="small"
            ),
            "Number of Bars": st.column_config.NumberColumn(
                "Number of Bars",
                help="Number of bars in this layer",
                min_value=0,
                max_value=20,
                step=1,
                format="%.0f",
                width="small"
            ),
            "Layer Position": st.column_config.TextColumn(
                "Layer Position",
                help="Enter: Top, Bottom, Middle, or custom y-distance from top (mm)",
                default="Bottom",
                width="medium"
            )
        },
        hide_index=True
    )
    
    st.session_state[session_key] = edited_df
    return edited_df


def create_stirrup_table(fy_stirrup_ksc, key_suffix=""):
    """
    Create editable table for stirrup reinforcement input
    
    Parameters:
    -----------
    fy_stirrup_ksc : float
        Stirrup steel grade
    key_suffix : str
        Suffix for session state key
    
    Returns:
    --------
    DataFrame : Updated stirrup data
    """
    
    session_key = f'stirrup_df{key_suffix}'
    
    # Initialize default data if not exists
    if session_key not in st.session_state:
        st.session_state[session_key] = pd.DataFrame({
            'Diameter (mm)': [10],
            'Number of Legs': [2],
            'Spacing (mm)': [150],
            'Steel Grade (ksc)': [fy_stirrup_ksc]
        })
    
    st.markdown("#### ⚡ Shear Reinforcement (Stirrups)")
    
    edited_df = st.data_editor(
        st.session_state[session_key],
        num_rows="dynamic",
        use_container_width=True,
        key=f"stirrup_editor{key_suffix}",
        column_config={
            "Diameter (mm)": st.column_config.NumberColumn(
                "Diameter (mm)",
                min_value=8,
                max_value=16,
                step=2,
                format="%.0f",
                width="small"
            ),
            "Number of Legs": st.column_config.NumberColumn(
                "Number of Legs",
                min_value=2,
                max_value=6,
                step=1,
                format="%.0f",
                width="small"
            ),
            "Spacing (mm)": st.column_config.NumberColumn(
                "Spacing (mm)",
                min_value=50,
                max_value=500,
                step=10,
                format="%.0f",
                width="small"
            ),
            "Steel Grade (ksc)": st.column_config.NumberColumn(
                "Steel Grade (ksc)",
                format="%.0f",
                width="small"
            )
        },
        hide_index=True
    )
    
    st.session_state[session_key] = edited_df
    return edited_df


def get_reinforcement_summary(bars_df):
    """
    Generate summary statistics for reinforcement
    
    Parameters:
    -----------
    bars_df : DataFrame
        Reinforcement data
    
    Returns:
    --------
    dict : Summary statistics
    """
    
    if bars_df.empty:
        return None
    
    total_bars = bars_df['Number of Bars'].sum()
    total_area = 0
    
    for idx, row in bars_df.iterrows():
        dia = row['Diameter (mm)']
        num = row['Number of Bars']
        area = (3.14159 * dia * dia / 4) * num
        total_area += area
    
    return {
        'total_bars': int(total_bars),
        'total_area_mm2': round(total_area, 0),
        'num_layers': len(bars_df)
    }
