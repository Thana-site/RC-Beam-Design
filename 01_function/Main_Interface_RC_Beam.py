"""
RC Beam Design & Analysis - Main Interface
ACI 318 Compliant Structural Analysis Tool
Refactored Modular Architecture
"""

import streamlit as st
import pandas as pd
import sys
from pathlib import Path

# Add the current directory to Python path for imports
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

# Import sub-function modules
from SubFunction.ui_layout import (
    apply_custom_css,
    render_header,
    render_sidebar_config,
    render_geometry_inputs,
    display_failure_mode_badge,
    display_capacity_results,
    display_detailed_parameters,
    render_footer
)
from SubFunction.input_tables import (
    create_main_reinforcement_table,
    create_stirrup_table
)
from SubFunction.analysis_core import (
    analyze_flexural_capacity,
    analyze_shear_capacity
)
from SubFunction.diagram_plot import (
    plot_section_and_strain,
    plot_three_section_comparison
)
from SubFunction.material_config import MaterialConfig


# ================================
# PAGE CONFIGURATION
# ================================
st.set_page_config(
    page_title="RC Beam Analysis - ACI 318",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ================================
# APPLY CUSTOM STYLING
# ================================
apply_custom_css()


# ================================
# RENDER HEADER
# ================================
render_header()


# ================================
# SIDEBAR CONFIGURATION
# ================================
config = render_sidebar_config()

analysis_mode = config['analysis_mode']
fc_ksc = config['fc_ksc']
fy_main_ksc = config['fy_main_ksc']
fy_stirrup_ksc = config['fy_stirrup_ksc']


# ================================
# MAIN CONTENT AREA
# ================================

# Geometry inputs
geometry = render_geometry_inputs()
b_mm = geometry['b_mm']
h_mm = geometry['h_mm']
cover_mm = geometry['cover_mm']

st.markdown("---")


# ================================
# SINGLE SECTION MODE
# ================================
if analysis_mode == "Single Section":
    
    # Reinforcement input tables
    st.markdown('<div class="input-section">', unsafe_allow_html=True)
    
    col_left, col_right = st.columns(2)
    
    with col_left:
        bars_df = create_main_reinforcement_table()
    
    with col_right:
        stirrup_df = create_stirrup_table(fy_stirrup_ksc)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Analysis button
    if st.button("🚀 Analyze Section", use_container_width=True, type="primary"):
        
        with st.spinner("🔄 Analyzing beam section..."):
            
            # Perform flexural analysis
            flex_results = analyze_flexural_capacity(
                b_mm, h_mm, cover_mm,
                bars_df,
                fc_ksc, fy_main_ksc
            )
            
            if flex_results is None:
                st.error("⚠️ Unable to analyze section. Please check reinforcement input.")
            else:
                # Store results in session state
                st.session_state.flex_results = flex_results
                
                # Perform shear analysis
                shear_results = analyze_shear_capacity(
                    b_mm, h_mm,
                    flex_results['d_mm'],
                    fc_ksc,
                    stirrup_df
                )
                st.session_state.shear_results = shear_results
                
                st.success("✅ Analysis completed successfully!")
    
    # Display results if available
    if 'flex_results' in st.session_state and st.session_state.flex_results:
        
        flex_results = st.session_state.flex_results
        
        st.markdown('<div class="results-container">', unsafe_allow_html=True)
        st.markdown("### 📊 Analysis Results")
        
        # Create and display visualization
        fig = plot_section_and_strain(
            b_mm, h_mm, cover_mm,
            flex_results['bar_data'],
            flex_results['strain_profile']
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Display failure mode badge
        display_failure_mode_badge(
            flex_results['control_type'],
            flex_results['phi'],
            flex_results['epsilon_t']
        )
        
        # Display capacity results
        if 'shear_results' in st.session_state:
            display_capacity_results(
                flex_results,
                st.session_state.shear_results
            )
        
        # Display detailed parameters
        display_detailed_parameters(flex_results)
        
        st.markdown('</div>', unsafe_allow_html=True)


# ================================
# THREE-SECTION MODE
# ================================
else:
    
    st.markdown("### 🔀 Three-Section Analysis (Start - Mid - End)")
    st.info("📌 Define reinforcement for each section separately.")
    
    # Create tabs for each section
    tab1, tab2, tab3 = st.tabs(["📍 Start Section", "📍 Mid Section", "📍 End Section"])
    
    for idx, (tab, section_name) in enumerate(
        zip([tab1, tab2, tab3], ["Start", "Mid", "End"])
    ):
        with tab:
            st.markdown(f"#### Configuration for {section_name} Section")
            
            st.markdown('<div class="input-section">', unsafe_allow_html=True)
            col_left, col_right = st.columns(2)
            
            with col_left:
                bars_df = create_main_reinforcement_table(f"_{section_name.lower()}")
            
            with col_right:
                stirrup_df = create_stirrup_table(
                    fy_stirrup_ksc,
                    f"_{section_name.lower()}"
                )
            st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Analyze all sections button
    if st.button("🚀 Analyze All Sections", use_container_width=True, type="primary"):
        
        with st.spinner("🔄 Analyzing all three sections..."):
            
            results_summary = []
            
            for section_name in ["Start", "Mid", "End"]:
                bars_key = f'bars_df_{section_name.lower()}'
                stirrup_key = f'stirrup_df_{section_name.lower()}'
                
                # Flexural analysis
                flex_results = analyze_flexural_capacity(
                    b_mm, h_mm, cover_mm,
                    st.session_state[bars_key],
                    fc_ksc, fy_main_ksc
                )
                
                if flex_results:
                    # Shear analysis
                    shear_results = analyze_shear_capacity(
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
            st.success("✅ Three-section analysis completed!")
    
    # Display three-section results
    if 'three_section_results' in st.session_state:
        
        st.markdown('<div class="results-container">', unsafe_allow_html=True)
        st.markdown("### 📊 Three-Section Analysis Summary")
        
        results_df = pd.DataFrame(st.session_state.three_section_results)
        
        st.dataframe(
            results_df,
            hide_index=True,
            use_container_width=True
        )
        
        # Visual comparison chart
        st.markdown("#### 📈 Capacity Comparison")
        
        fig_compare = plot_three_section_comparison(
            st.session_state.three_section_results
        )
        st.plotly_chart(fig_compare, use_container_width=True)
        
        # Export options
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
{'='*60}
Beam Geometry: {b_mm} × {h_mm} mm
Concrete: f'c = {fc_ksc} ksc
Steel: fy = {fy_main_ksc} ksc

RESULTS SUMMARY:
{'='*60}
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
        
        st.markdown('</div>', unsafe_allow_html=True)


# ================================
# FOOTER & DOCUMENTATION
# ================================
render_footer()

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #64748b; padding: 1rem;'>
    <p><strong>🏗️ RC Beam Design & Analysis v2.0</strong> | ACI 318 Compliant</p>
    <p style='font-size: 0.9em;'>
        <em>Refactored Modular Architecture | Professional Structural Analysis Tool</em>
    </p>
    <p style='font-size: 0.85rem; margin-top: 0.5rem;'>
        For educational and preliminary design purposes • Always verify with licensed engineer
    </p>
</div>
""", unsafe_allow_html=True)
