"""
Module B Drift Evaluator & Metrics Calculator
Evaluates predicted 168h drift against Safety Thresholds and computes evaluation metrics (MAE, RMSE).
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Dict, Any
from config import SAFETY_SLOPE_MARGIN_RATIO

def evaluate_drift_and_safety(
    df: pd.DataFrame,
    safety_ratio: float = SAFETY_SLOPE_MARGIN_RATIO,
) -> pd.DataFrame:
    """
    Evaluates predicted 168h value against safety slope criteria.
    
    Safety Threshold = Datasheet_Limit * SAFETY_SLOPE_MARGIN_RATIO (e.g. 80% of 50 uA = 40 uA)
    """
    res_df = df.copy()
    
    safety_threshold = res_df["Datasheet_Limit"] * safety_ratio
    res_df["Safety_Threshold_168h"] = safety_threshold.round(2)
    
    # Flag predicted breaches
    res_df["ModuleB_Safety_Breach_Flag"] = res_df["Predicted_Value_168h"] >= res_df["Safety_Threshold_168h"]
    res_df["ModuleB_Datasheet_Breach_Flag"] = res_df["Predicted_Value_168h"] >= res_df["Datasheet_Limit"]
    
    return res_df

def compute_model_performance_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Computes performance metrics (MAE, RMSE, R2) if ground truth Value_168h is available.
    """
    if "Value_168h" not in df.columns or "Predicted_Value_168h" not in df.columns:
        return {"mae": None, "rmse": None, "r2": None, "has_ground_truth": False}
        
    valid_mask = df["Value_168h"].notnull() & df["Predicted_Value_168h"].notnull()
    valid_df = df[valid_mask]
    
    if len(valid_df) == 0:
        return {"mae": None, "rmse": None, "r2": None, "has_ground_truth": False}
        
    y_true = valid_df["Value_168h"]
    y_pred = valid_df["Predicted_Value_168h"]
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    return {
        "mae": round(mae, 3),
        "rmse": round(rmse, 3),
        "r2": round(r2, 3),
        "total_samples_evaluated": len(valid_df),
        "has_ground_truth": True
    }
