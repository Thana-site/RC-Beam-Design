"""
Input Components Module
Editable tables for reinforcement input
"""

import streamlit as st
import pandas as pd


def create_reinforcement_table(key_suffix=""):
    """
    Create main reinforcement input table
    
    Args:
        key_suffix: Unique identifier for multiple tables
        
    Returns:
        DataFrame with reinforcement data
    """
    
    session_key = f'bars_df{key_suffix}'
    
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
                help="Bar diameter",
                min_value=10,
                max_value=40,
                step=2,
                format="%.0f"
            ),
            "Number of Bars": st.column_config.NumberColumn(
                "Number of Bars",
                help="Quantity",
                min_value=0,
                max_value=20,
                step=1,
                format="%.0f"
            ),
            "Layer Position": st.column_config.TextColumn(
                "Layer Position",
                help="Top / Bottom / Middle / custom y-value",
                default="Bottom"
            )
        },
        hide_index=True
    )
    
    st.session_state[session_key] = edited_df
    return edited_df


def create_stirrup_table(fy_stirrup, key_suffix=""):
    """
    Create stirrup reinforcement input table
    
    Args:
        fy_stirrup: Stirrup steel grade
        key_suffix: Unique identifier
        
    Returns:
        DataFrame with stirrup data
    """
    
    session_key = f'stirrup_df{key_suffix}'
    
    if session_key not in st.session_state:
        st.session_state[session_key] = pd.DataFrame({
            'Diameter (mm)': [10],
            'Number of Legs': [2],
            'Spacing (mm)': [150],
            'Steel Grade (ksc)': [fy_stirrup]
        })
    
    st.markdown("#### ⚡ Shear Reinforcement")
    
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
        },
        hide_index=True
    )
    
    st.session_state[session_key] = edited_df
    return edited_df
