import streamlit as st
import pandas as pd
import numpy as np
from utils import load_data, train_and_predict, plot_approval_rates, plot_prediction_distribution
from fairness_metrics import calculate_fairness_metrics, detect_bias, get_mitigation_suggestions
from shap_analysis import run_shap_explanation, plot_shap_summary
import matplotlib.pyplot as plt

# Page Configuration
st.set_page_config(
    page_title="FairLens AI | Enterprise Audit Console",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# STEP 1 — Enterprise dark theme styling
st.markdown("""
<style>
    [data-testid="stAppViewContainer"] {
        background-color: #0F172A;
        color: #e4e1ed;
    }
    [data-testid="stSidebar"] {
        background-color: #020617;
    }
    .metric-card {
        background: #1e293b;
        padding: 20px;
        border-radius: 6px;
        border: 1px solid #334155;
        text-align: center;
    }
    .metric-title {
        font-size: 11px;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 8px;
    }
    .metric-value {
        font-size: 32px;
        font-weight: 700;
        color: white;
    }
    .section-card {
        background: #1e293b;
        padding: 20px;
        border-radius: 6px;
        border: 1px solid #334155;
    }
    h2, h3 {
        color: #f8fafc !important;
    }
</style>
""", unsafe_allow_html=True)

# STEP 2 — Enterprise top header bar
st.markdown("""
<div style="display:flex;
justify-content:space-between;
padding:16px;
background:#020617;
border-bottom:1px solid #334155;
margin-bottom: 20px;">
<h2 style="color:white; margin:0;">FairLens AI Enterprise Audit Console</h2>
<span style="color:#94a3b8; align-self:center;">
Lead Auditor Dashboard Active
</span>
</div>
""", unsafe_allow_html=True)

# STEP 9 — Search-style header input
st.text_input(
    "Search audits, datasets, or fairness metrics",
    placeholder="Search audits..."
)

# Initialize Session State
if 'df' not in st.session_state: st.session_state.df = None
if 'results' not in st.session_state: st.session_state.results = None

# STEP 3 — Sidebar Audit Control Panel
with st.sidebar:
    st.header("Audit Controls")
    st.success("System Status: Active")

    uploaded_file = st.file_uploader("Dataset Upload", type=["csv"])
    use_sample = st.checkbox("Use Sample Dataset", value=True if uploaded_file is None else False)
    
    if uploaded_file is not None:
        try:
            st.session_state.df = load_data(uploaded_file)
        except Exception as e:
            st.error("Dataset could not be loaded. Please upload a valid CSV.")
            st.stop()
    elif use_sample:
        try:
            st.session_state.df = load_data("sample_data.csv")
        except Exception as e:
            st.error("Dataset could not be loaded. Please upload a valid CSV.")
            st.stop()
        if 'target_col' not in st.session_state: st.session_state.target_col = 'hired'
        if 'sensitive_col' not in st.session_state: st.session_state.sensitive_col = 'gender'

    if st.session_state.df is not None:
        df = st.session_state.df
        def is_binary_numeric(col_name):
            unique_vals = set(df[col_name].dropna().unique())
            return unique_vals == {0, 1} or unique_vals == {0} or unique_vals == {1}

        binary_numeric_columns = [col for col in df.columns if is_binary_numeric(col)]
        categorical_columns = [col for col in df.columns if df[col].nunique() < 10]

        target_col = st.selectbox(
            "Target Column", 
            options=binary_numeric_columns,
            index=binary_numeric_columns.index(st.session_state.get('target_col', '')) if st.session_state.get('target_col' ) in binary_numeric_columns else 0
        )
        st.session_state.target_col = target_col

        sensitive_col = st.selectbox(
            "Sensitive Attribute", 
            options=categorical_columns,
            index=categorical_columns.index(st.session_state.get('sensitive_col', '')) if st.session_state.get('sensitive_col') in categorical_columns else 0
        )
        st.session_state.sensitive_col = sensitive_col

        st.markdown("---")
        
        if "prediction" in df.columns:
            st.success("Using existing prediction column for fairness analysis.")
        else:
            st.info("Training logistic regression model automatically.")

        has_prediction = "prediction" in df.columns
        if st.button("🚀 Run Analysis"):
            if not has_prediction:
                with st.spinner("Executing Audit Pipeline..."):
                    train_results = train_and_predict(df, target_col, sensitive_col)
                    if train_results:
                        st.session_state.df = train_results['df']
                        st.session_state.results = {'model': train_results['model'], 'X_test_encoded': train_results['X_test_encoded'], 'y_test': train_results['y_test']}
            else:
                st.session_state.results = {'model': None, 'use_existing': True}

# STEP 10 — Navigation tabs
if st.session_state.df is not None:
    df = st.session_state.df
    results_ready = 'results' in st.session_state and st.session_state.results is not None
    
    tabs = st.tabs(["📊 Fairness Metrics", "📈 Visualizations", "💡 Explainability", "📋 Recommendations", "📂 Data Explorer"])
    
    with tabs[0]:
        st.header("Fairness Metrics")
        if results_ready:
            target_col = st.session_state.target_col
            sensitive_col = st.session_state.sensitive_col
            analysis_df = df.dropna(subset=[target_col, sensitive_col, "prediction"])
            
            fairness_results = calculate_fairness_metrics(analysis_df[target_col], analysis_df["prediction"], analysis_df[sensitive_col])
            is_biased = detect_bias(fairness_results)
            
            # STEP 4 — Bias Alert Banner
            if is_biased:
                st.error("""
                ### Critical Bias Detected
                Demographic parity exceeds allowed threshold.  
                Immediate mitigation recommended.
                """)
            else:
                st.success("Fairness within acceptable thresholds")

            # STEP 5 — Enterprise Metric Cards
            dp_diff = float(fairness_results.get('demographic_parity_diff', 0.0))
            eo_diff = float(fairness_results.get('equal_opportunity_diff', 0.0))
            sr_ratio = float(fairness_results.get('selection_rate_ratio', 1.0))
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                <div class="metric-title">Demographic Parity Diff</div>
                <div class="metric-value">{dp_diff:.3f}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                <div class="metric-title">Equal Opportunity Diff</div>
                <div class="metric-value">{eo_diff:.3f}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                <div class="metric-title">Selection Rate Ratio</div>
                <div class="metric-value">{sr_ratio:.3f}</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("Awaiting audit execution. Please use the Audit Control panel to start.")

    with tabs[1]:
        st.header("Visualizations Dashboard")
        if results_ready:
            # STEP 6 — Charts + Recommendations split panel
            left_chart, right_panel = st.columns([2, 1])
            
            with left_chart:
                st.subheader("Distribution Analysis")
                fig1 = plot_approval_rates(fairness_results['selection_rate_by_group'], st.session_state.sensitive_col)
                if fig1: st.pyplot(fig1)
                
                fig2 = plot_prediction_distribution(df, "prediction")
                if fig2: st.pyplot(fig2)
            
            with right_panel:
                # Part of Step 6 & 7
                st.markdown("### Quick Recommendations")
                suggestions = get_mitigation_suggestions(fairness_results)
                for sug in suggestions:
                    st.info(f"**{sug['title']}**\n\n{sug['description']}")
        else:
            st.warning("Metrics unavailable until audit is complete.")

    with tabs[2]:
        st.header("Explainability Details")
        if results_ready and st.session_state.results.get('model') is not None:
            if st.button("Calculate SHAP Intelligence"):
                import shap
                model = st.session_state.results['model']
                X_test = st.session_state.results['X_test_encoded']
                fig, ax = plt.subplots()
                explainer = shap.Explainer(model, X_test)
                shap_v = explainer(X_test)
                shap.summary_plot(shap_v, X_test, show=False)
                st.pyplot(fig)
        else:
            st.info("SHAP metrics require a trained model instance.")

    with tabs[3]:
        # STEP 7 — Improved mitigation strategy panel
        st.markdown("### Mitigation Strategies")
        if results_ready:
            suggestions = get_mitigation_suggestions(fairness_results)
            for sug in suggestions:
                st.markdown(f"""
                <div class="section-card" style="margin-bottom:12px;">
                <h4 style="margin:0; color:#38bdf8;">{sug['title']}</h4>
                <p style="margin-top:8px; font-size:14px; color:#cbd5e1;">{sug['description']}</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("Run analysis to view mitigation strategies.")

    with tabs[4]:
        # STEP 8 — Enterprise Audit Table
        st.markdown("### Recent Audit Samples")
        st.dataframe(df.head(), width="stretch")
        st.markdown("---")
        st.subheader("Complete Data Register")
        st.dataframe(df, width="stretch")

else:
    st.info("Lead Auditor Dashboard initialized. Please upload a dataset to begin.")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#64748b; font-size:12px;">
Enterprise Compliant Audit Console | v2.4.1 Secure Build
</div>
""", unsafe_allow_html=True)
