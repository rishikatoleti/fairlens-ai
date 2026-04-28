import shap
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd

def run_shap_explanation(pipeline, X):
    """
    Generate SHAP values for a scikit-learn pipeline.
    We extract the preprocessor to transform X and the classifier for SHAP.
    """
    try:
        # Extract preprocessor and classifier from pipeline
        preprocessor = pipeline.named_steps['preprocessor']
        clf = pipeline.named_steps['classifier']
        
        # Transform the data
        X_transformed = preprocessor.transform(X)
        
        # Get feature names from preprocessor
        try:
            # For sklearn >= 1.0
            feature_names = preprocessor.get_feature_names_out()
        except:
            feature_names = [f"feature_{i}" for i in range(X_transformed.shape[1])]
            
        X_df = pd.DataFrame(X_transformed, columns=feature_names)
        
        # Generate SHAP values
        explainer = shap.Explainer(clf, X_df)
        shap_values = explainer(X_df)
        
        return shap_values, X_df
    except Exception as e:
        st.error(f"SHAP Error: {e}")
        return None, None

def plot_shap_summary(shap_values, X_df):
    """
    Plot SHAP summary plot.
    """
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.summary_plot(shap_values, X_df, show=False)
        plt.tight_layout()
        st.pyplot(fig)
    except Exception as e:
        st.error(f"SHAP Plotting Error: {e}")
