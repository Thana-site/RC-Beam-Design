"""
RC Beam Design & Analysis - Main Interface
ACI 318 Compliant Structural Analysis Tool
GitHub Repository Structure
"""

import streamlit as st
import pandas as pd

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



# ========================================
# PAGE CONFIGURATION
# ========================================

st.set_page_config(
    page_title="RC Beam Analysis - ACI 318",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ========================================
# APPLY STYLING
# ========================================

apply_custom_styling()


# ========================================
# HEADER
# ========================================

render_header()


# ========================================
# SIDEBAR CONFIGURATION
# ========================================

config = render_sidebar_controls()

# Extract configuration
analysis_mode = config['analysis_mode']
fc_ksc = config['fc_ksc']
fy_main = config['fy_main']
fy_stirrup = config['fy_stirrup']
b_mm = config['b_mm']
h_mm = config['h_mm']
cover_mm = config['cover_mm']


# ========================================
# MAIN CONTENT
# ========================================

# Display geometry summary
st.markdown("### 📐 Section Configuration")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Width (b)", f"{b_mm:.0f} mm")
with col2:
    st.metric("Height (h)", f"{h_mm:.0f} mm")
with col3:
    st.metric("Cover", f"{cover_mm:.0f} mm")
with col4:
    st.metric("Concrete f'c", f"{fc_ksc:.0f} ksc")

st.markdown("---")


# ========================================
# SINGLE SECTION MODE
# ========================================

if analysis_mode == "Single Section":
    
    st.markdown("### 🔧 Reinforcement Configuration")
    
    # Input tables in two columns
    st.markdown('<div class="input-card">', unsafe_allow_html=True)
    col_left, col_right = st.columns(2)
    
    with col_left:
        bars_df = create_reinforcement_table()
    
    with col_right:
        stirrup_df = create_stirrup_table(fy_stirrup)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Analysis button
    if st.button("🚀 **Analyze Section**", type="primary", use_container_width=True):
        
        with st.spinner("⚙️ Running structural analysis..."):
            
            # Flexural analysis
            flex_results = analyze_flexural_capacity(
                b_mm, h_mm, cover_mm,
                bars_df,
                fc_ksc, fy_main
            )
            
            if flex_results is None:
                st.error("⚠️ Analysis failed. Check reinforcement input.")
            else:
                # Store in session state
                st.session_state['flex_results'] = flex_results
                
                # Shear analysis
                shear_results = analyze_shear_capacity(
                    b_mm, h_mm,
                    flex_results['d_mm'],
                    fc_ksc,
                    stirrup_df
                )
                st.session_state['shear_results'] = shear_results
                
                st.success("✅ Analysis completed successfully!")
    
    # Display results if available
    if 'flex_results' in st.session_state and st.session_state['flex_results']:
        
        flex_results = st.session_state['flex_results']
        
        st.markdown('<div class="results-section">', unsafe_allow_html=True)
        st.markdown("## 📊 Analysis Results")
        
        # Combined diagram (section + strain side-by-side)
        fig = draw_section_and_strain(
            b_mm, h_mm, cover_mm,
            flex_results['bar_data'],
            flex_results['strain_profile']
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Failure mode badge
        display_failure_mode(
            flex_results['control_type'],
            flex_results['phi'],
            flex_results['epsilon_t']
        )
        
        # Capacity metrics
        if 'shear_results' in st.session_state:
            display_capacity_metrics(
                flex_results,
                st.session_state['shear_results']
            )
        
        # Detailed parameters
        display_detailed_results(flex_results)
        
        st.markdown('</div>', unsafe_allow_html=True)


# ========================================
# THREE-SECTION MODE
# ========================================

else:
    
    st.markdown("## 🔀 Three-Section Analysis")
    st.info("📌 Configure reinforcement for Start, Mid, and End sections")
    
    # Create tabs for each section
    tab1, tab2, tab3 = st.tabs(["📍 Start Section", "📍 Mid Section", "📍 End Section"])
    
    for tab, section_name in zip([tab1, tab2, tab3], ["Start", "Mid", "End"]):
        with tab:
            st.markdown(f"#### {section_name} Section Configuration")
            
            st.markdown('<div class="input-card">', unsafe_allow_html=True)
            col_left, col_right = st.columns(2)
            
            with col_left:
                bars_df = create_reinforcement_table(f"_{section_name.lower()}")
            
            with col_right:
                stirrup_df = create_stirrup_table(fy_stirrup, f"_{section_name.lower()}")
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Analyze all sections
    if st.button("🚀 **Analyze All Sections**", type="primary", use_container_width=True):
        
        with st.spinner("⚙️ Analyzing all three sections..."):
            
            results_summary = []
            
            for section_name in ["Start", "Mid", "End"]:
                bars_key = f'bars_df_{section_name.lower()}'
                stirrup_key = f'stirrup_df_{section_name.lower()}'
                
                # Flexural analysis
                flex_results = analyze_flexural_capacity(
                    b_mm, h_mm, cover_mm,
                    st.session_state[bars_key],
                    fc_ksc, fy_main
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
            
            st.session_state['three_section_results'] = results_summary
            st.success("✅ Three-section analysis completed!")
    
    # Display results
    if 'three_section_results' in st.session_state:
        
        st.markdown('<div class="results-section">', unsafe_allow_html=True)
        st.markdown("## 📊 Comparative Results")
        
        results_df = pd.DataFrame(st.session_state['three_section_results'])
        
        st.dataframe(
            results_df,
            hide_index=True,
            use_container_width=True
        )
        
        # Visual comparison
        st.markdown("#### 📈 Capacity Comparison Chart")
        fig_compare = plot_section_comparison(st.session_state['three_section_results'])
        st.plotly_chart(fig_compare, use_container_width=True)
        
        # Export options
        col1, col2 = st.columns(2)
        
        with col1:
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name="rc_beam_3section_analysis.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            summary_text = f"""RC BEAM THREE-SECTION ANALYSIS
{'='*70}
Geometry: {b_mm} × {h_mm} mm
Concrete: f'c = {fc_ksc} ksc
Steel: fy = {fy_main} ksc

RESULTS:
{'='*70}
"""
            for r in st.session_state['three_section_results']:
                summary_text += f"\n{r['Section']} Section:\n"
                summary_text += f"  φMn = {r['φMn (tonf·m)']} tonf·m\n"
                summary_text += f"  φVn = {r['φVn (tonf)']} tonf\n"
                summary_text += f"  Mode = {r['Control Type']}\n"
            
            st.download_button(
                label="📄 Download Report",
                data=summary_text,
                file_name="rc_beam_3section_report.txt",
                mime="text/plain",
                use_container_width=True
            )
        
        st.markdown('</div>', unsafe_allow_html=True)


# ========================================
# FOOTER
# ========================================

render_footer()


# ========================================
# INFO SECTION
# ========================================

with st.expander("ℹ️ About This Application"):
    st.markdown("""
    ### RC Beam Design & Analysis Tool
    
    **Design Standard:** ACI 318-19 (Building Code Requirements for Structural Concrete)
    
    **Features:**
    - ✅ Flexural capacity analysis (rectangular stress block method)
    - ✅ Shear capacity calculation
    - ✅ Strain distribution visualization
    - ✅ Failure mode identification (tension/transition/compression control)
    - ✅ Single section or three-section analysis
    - ✅ Interactive reinforcement configuration
    - ✅ Export results (CSV & text format)
    
    **Units:**
    - Dimensions: millimeters (mm)
    - Moments: tonf·m (metric ton-force × meter)
    - Stresses: ksc (kg/cm²)
    - Forces: tonf (metric ton-force)
    
    **Limitations:**
    - Rectangular sections only
    - Singly reinforced beams
    - Serviceability checks not included
    - Development length must be verified separately
    
    **Version:** 2.0.0 (Refactored Modular Architecture)
    
    ---
    
    *For educational and preliminary design purposes. Always verify with licensed professional engineer.*
    """)
