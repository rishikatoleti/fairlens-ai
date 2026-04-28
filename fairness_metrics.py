from fairlearn.metrics import (
    demographic_parity_difference,
    demographic_parity_ratio,
    equalized_odds_difference,
    selection_rate,
    true_positive_rate,
    MetricFrame
)
import pandas as pd

def calculate_fairness_metrics(y_true, y_pred, sensitive_features):
    """
    Calculate core fairness metrics using Fairlearn and return as a dictionary.
    """
    # Using MetricFrame to get per-group metrics
    metrics = {
        'selection_rate': selection_rate,
        'true_positive_rate': true_positive_rate
    }
    
    mf = MetricFrame(
        metrics=metrics,
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive_features
    )
    
    # Calculate Differences and Ratios
    dp_diff = demographic_parity_difference(y_true, y_pred, sensitive_features=sensitive_features)
    dp_ratio = demographic_parity_ratio(y_true, y_pred, sensitive_features=sensitive_features)
    
    # Equal Opportunity Difference involves True Positive Rate
    tpr_diff = mf.difference(method='between_groups')['true_positive_rate']
    
    return {
        'demographic_parity_diff': dp_diff,
        'selection_rate_ratio': dp_ratio,
        'equal_opportunity_diff': tpr_diff,
        'selection_rate_by_group': mf.by_group['selection_rate'],
        'tpr_by_group': mf.by_group['true_positive_rate'],
        'metric_frame': mf
    }

def detect_bias(fairness_results, threshold=0.1):
    """
    Check if bias is detected based on thresholds.
    """
    dp_diff = fairness_results['demographic_parity_diff']
    eo_diff = fairness_results['equal_opportunity_diff']
    
    is_biased = dp_diff > threshold or eo_diff > threshold
    
    return is_biased

def get_mitigation_suggestions(fairness_results, threshold=0.1):
    """
    Generate mitigation suggestions based on fairness results.
    """
    suggestions = []
    dp_diff = fairness_results['demographic_parity_diff']
    eo_diff = fairness_results['equal_opportunity_diff']
    
    if dp_diff > threshold:
        suggestions.append({
            'title': "Dataset Balancing",
            'description': "Demographic parity disparity detected. Consider oversampling underrepresented groups or using synthetic data generation (SMOTE) to balance the distribution of outcomes across groups."
        })
        
    if eo_diff > threshold:
        suggestions.append({
            'title': "Fairness-Aware Classifier",
            'description': "Equal opportunity disparity detected. Consider using post-processing techniques (like ThresholdOptimizer) or in-processing techniques (like ExponentiatedGradient) from the Fairlearn library."
        })
        
    if dp_diff > threshold * 2: # High influence
        suggestions.append({
            'title': "Feature Removal / Blindness",
            'description': "The sensitive attribute appears to have a very strong influence on predictions. Consider removing the sensitive attribute from the training features to reduce direct discrimination."
        })
        
    if not suggestions:
        suggestions.append({
            'title': "Continuous Monitoring",
            'description': "The model appears relatively fair. Recommendation is to continue monitoring fairness metrics as more data is collected."
        })
        
    return suggestions
