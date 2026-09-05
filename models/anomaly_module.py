"""
Module A: Dynamic Anomaly Detection Interface
Combines lot-relative statistical analysis and unsupervised ML scoring.
"""

import pandas as pd
from models.robust_stats import compute_lot_robust_stats
from models.isolation_forest import train_run_isolation_forest
from config import ROBUST_Z_SCORE_THRESHOLD

def run_anomaly_module_a(
    df: pd.DataFrame,
    z_threshold: float = ROBUST_Z_SCORE_THRESHOLD,
) -> pd.DataFrame:
    """
    Executes Module A dynamic anomaly detection pipeline on the input DataFrame.
    """
    # Step 1: Compute Lot Robust Z-scores for 0h and 24h
    res_df = compute_lot_robust_stats(df, target_col="Value_0h")
    res_df = compute_lot_robust_stats(res_df, target_col="Value_24h")
    
    # Step 2: Fit Isolation Forest on early time points
    res_df, _ = train_run_isolation_forest(res_df, feature_cols=["Value_0h", "Value_24h"])
    
    # Step 3: Combined Peer Anomaly Flag
    # A component is flagged as a peer anomaly if:
    # 1. Robust Z-score > ROBUST_Z_SCORE_THRESHOLD at 24h OR 0h
    # OR 2. Isolation Forest marks it as ANOMALY
    z_flag = (res_df["Value_24h_Robust_Z"].abs() >= z_threshold) | \
             (res_df["Value_0h_Robust_Z"].abs() >= z_threshold)
             
    iso_flag = (res_df["IsoForest_Prediction"] == "ANOMALY")
    
    res_df["ModuleA_Peer_Anomaly_Flag"] = z_flag | iso_flag
    
    # Max Z-score for quick reference
    res_df["Max_Robust_Z"] = res_df[["Value_0h_Robust_Z", "Value_24h_Robust_Z"]].abs().max(axis=1)
    
    return res_df
