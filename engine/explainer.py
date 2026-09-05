"""
Human-Interpretable Explainability Engine
Generates clear, auditable evidence cards and natural language bullet points
explaining screening decisions to QA inspectors.
"""

import pandas as pd
from typing import List, Dict, Any

from config import ROBUST_Z_SCORE_THRESHOLD, SAFETY_SLOPE_MARGIN_RATIO


def generate_component_explanation(
    row: pd.Series,
    z_threshold: float = ROBUST_Z_SCORE_THRESHOLD,
    safety_ratio: float = SAFETY_SLOPE_MARGIN_RATIO,
) -> List[str]:
    """
    Generates plain-language reasons explaining why a component was assigned its risk status.
    """
    reasons = []
    
    comp_id = row.get("Component_ID", "Unknown")
    lot_id = row.get("Lot_ID", "Unknown")
    val_0h = row.get("Value_0h", 0.0)
    val_24h = row.get("Value_24h", 0.0)
    val_24h_z = row.get("Value_24h_Robust_Z", 0.0)
    val_24h_med = row.get("Value_24h_Lot_Median", 0.0)
    pred_168h = row.get("Predicted_Value_168h", 0.0)
    safety_thresh = row.get("Safety_Threshold_168h", 0.0)
    limit = row.get("Datasheet_Limit", 0.0)
    status = row.get("Screening_Status", "PASS")
    
    if status == "PASS":
        reasons.append("Component measurements align with the manufacturing lot baseline.")
        reasons.append(f"Predicted 168h value ({pred_168h} µA) is safely below the safety threshold ({safety_thresh} µA).")
        return reasons

    # Peer anomaly reasons
    if abs(val_24h_z) >= z_threshold:
        reasons.append(
            f"Peer Anomaly: 24h measurement ({val_24h} µA) is significantly above Lot {lot_id} median ({val_24h_med} µA) "
            f"with a Robust Z-Score of {val_24h_z:+.2f} (Threshold = ±{z_threshold:.1f})."
        )
    elif row.get("IsoForest_Prediction") == "ANOMALY":
        reasons.append(f"Unsupervised Pattern Outlier: Multidimensional trajectory detected as anomalous relative to peer population.")

    # Drift & Future safety breach reasons
    if pred_168h >= limit:
        reasons.append(
            f"Predicted Limit Failure: Early burn-in trajectory predicts 168h value will reach {pred_168h} µA, "
            f"breaching absolute datasheet limit ({limit} µA)."
        )
    elif pred_168h >= safety_thresh:
        reasons.append(
            f"Safety Margin Breach: Predicted 168h value ({pred_168h} µA) exceeds the "
            f"{safety_ratio:.0%} safety margin threshold ({safety_thresh} µA)."
        )
        
    early_drift = round(val_24h - val_0h, 2)
    if early_drift > 5.0:
        reasons.append(f"Abnormal Early Drift: Parameter increased by +{early_drift} µA during first 24 hours of burn-in.")
        
    return reasons

def attach_explanations_to_df(
    df: pd.DataFrame,
    z_threshold: float = ROBUST_Z_SCORE_THRESHOLD,
    safety_ratio: float = SAFETY_SLOPE_MARGIN_RATIO,
) -> pd.DataFrame:
    """
    Appends formatted explanation lists and summary strings to each row in DataFrame.
    """
    res_df = df.copy()
    explanation_lists = []
    explanation_texts = []
    
    for _, row in res_df.iterrows():
        reasons = generate_component_explanation(row, z_threshold, safety_ratio)
        explanation_lists.append(reasons)
        explanation_texts.append(" | ".join(reasons))
        
    res_df["Explanation_List"] = explanation_lists
    res_df["Explanation_Summary"] = explanation_texts
    
    return res_df
