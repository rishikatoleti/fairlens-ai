# FairLens AI: Fairness Auditing Dashboard

FairLens AI is a Streamlit-based dashboard designed to help users detect and analyze bias in machine learning predictions.

## Key Features
- **Dataset Upload**: Support for CSV file uploads.
- **Fairness Metrics**: Calculate and visualize metrics such as Demographic Parity and Equalized Odds.
- **SHAP Explainability**: Use SHAP to understand how different features contribute to model predictions and potential bias.
- **Bias Detection**: Select target columns and sensitive attributes to identify disparities.

## Installation
1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the app:
   ```bash
   streamlit run app.py
   ```

## Project Structure
- `app.py`: Main entry point for the Streamlit application.
- `fairness_metrics.py`: Contains logic for calculating fairness metrics.
- `shap_analysis.py`: Contains logic for SHAP value generation and visualization.
- `utils.py`: Utility functions for data processing.
- `sample_data.csv`: A sample dataset for testing.

# FairLens AI Deployment

Run locally:

streamlit run app.py

Deploy using Streamlit Cloud:

https://share.streamlit.io/

Upload dataset and run fairness analysis.
