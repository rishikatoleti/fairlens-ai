import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def load_data(file):
    """Load dataset from a CSV file."""
    try:
        df = pd.read_csv(file)
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def train_and_predict(df, target_col, sensitive_col, prediction_col="prediction"):
    """
    Automated training pipeline:
    1. Drop missing values
    2. Encode categorical columns using pd.get_dummies
    3. Train LogisticRegression
    4. Store predictions in 'prediction' column
    """
    try:
        # Drop rows with missing values
        df = df.dropna(subset=[target_col, sensitive_col]).copy()
        
        # Features (X) and Target (y)
        # Drop target and prediction column if it exists in X
        cols_to_drop = [target_col]
        if prediction_col in df.columns:
            cols_to_drop.append(prediction_col)
            
        X = df.drop(columns=cols_to_drop)
        y = df[target_col]
        
        # One-hot encode using get_dummies
        X_encoded = pd.get_dummies(X, drop_first=True)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.3, random_state=42)
        
        # Train model
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_train, y_train)
        
        # Generate predictions for the entire encoded dataset (for dashboard simplicity)
        df[prediction_col] = clf.predict(X_encoded)
        
        return {
            'df': df,
            'model': clf,
            'X_test_encoded': X_test,
            'y_test': y_test
        }
        
    except Exception as e:
        st.error(f"Model Training Error: {e}")
        return None
import matplotlib.pyplot as plt
import seaborn as sns

def plot_approval_rates(selection_rate_by_group, sensitive_column):
    """
    Plot approval rate (positive outcome) by sensitive attribute.
    """
    try:
        fig, ax = plt.subplots(figsize=(6, 4))
        selection_rate_by_group.plot(kind="bar", ax=ax)
        ax.set_title("Selection Rate by Sensitive Group")
        ax.set_xlabel(sensitive_column)
        ax.set_ylabel("Selection Rate")
        plt.tight_layout()
        return fig
    except Exception as e:
        st.error(f"Plotting Error: {str(e)}")
        return None

def plot_prediction_distribution(df, prediction_column):
    """
    Plot histogram/distribution of model predictions.
    """
    try:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.countplot(x=df[prediction_column], ax=ax)
        ax.set_title("Prediction Distribution")
        ax.set_xlabel("Prediction")
        ax.set_ylabel("Count")
        plt.tight_layout()
        return fig
    except Exception as e:
        st.error(f"Plotting Error: {str(e)}")
        return None
