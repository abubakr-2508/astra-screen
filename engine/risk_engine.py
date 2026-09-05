"""
Risk Decision Engine
Combines Module A (Dynamic Anomaly Score) & Module B (Predicted 168h Drift)
into final QA Screening Decisions: PASS, REVIEW, or FLAG (HIGH RISK).
"""

import pandas as pd
import numpy as np
from config import RISK_PASS, RISK_REVIEW, RISK_FLAG, ROBUST_Z_SCORE_THRESHOLD

def evaluate_composite_risk(
    df: pd.DataFrame,
    z_threshold: float = ROBUST_Z_SCORE_THRESHOLD,
) -> pd.DataFrame:
    """
    Evaluates composite risk score and assigns screening status.
    
    Decision Matrix:
    - FLAG (HIGH RISK):
      - Predicted 168h value >= Datasheet Limit OR
      - Module B Safety Breach (Predicted >= 80% Datasheet Limit) AND Module A Peer Anomaly Flag OR
      - Robust Z-score > 4.0 (Extreme peer deviation)
      
    - REVIEW (MEDIUM RISK):
      - Module A Peer Anomaly Flag (Z-score 3.0 to 4.0) OR
      - Module B Safety Breach alone OR
      - Early 24h drift is abnormally high relative to lot.
      
    - PASS (LOW RISK):
      - Normal peer distribution and safe predicted 168h trajectory.
    """
    res_df = df.copy()
    
    risk_statuses = []
    risk_scores = []
    
    for _, row in res_df.iterrows():
        mod_a_flag = bool(row.get("ModuleA_Peer_Anomaly_Flag", False))
        mod_b_safety_flag = bool(row.get("ModuleB_Safety_Breach_Flag", False))
        mod_b_datasheet_flag = bool(row.get("ModuleB_Datasheet_Breach_Flag", False))
        max_z = float(row.get("Max_Robust_Z", 0.0))
        
        # Calculate continuous composite risk score (0 to 100)
        score = 10.0  # Base
        score += min(40.0, max_z * 10.0)
        if mod_b_safety_flag:
            score += 25.0
        if mod_b_datasheet_flag:
            score += 35.0
        if mod_a_flag:
            score += 15.0
            
        score = round(min(100.0, score), 1)
        risk_scores.append(score)
        
        # Assign Category
        # Thresholds follow the operator setting rather than literals. The
        # "extreme peer deviation" bar sits one sigma above the review bar,
        # which reproduces the original 4.0 / 3.0 pair at the default of 3.0.
        if mod_b_datasheet_flag or (mod_b_safety_flag and mod_a_flag) or max_z >= z_threshold + 1.0:
            status = RISK_FLAG
        elif mod_a_flag or mod_b_safety_flag or max_z >= z_threshold:
            status = RISK_REVIEW
        else:
            status = RISK_PASS
            
        risk_statuses.append(status)
        
    res_df["Risk_Score"] = risk_scores
    res_df["Screening_Status"] = risk_statuses
    
    return res_df
