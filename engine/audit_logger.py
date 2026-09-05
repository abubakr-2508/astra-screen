"""
Audit Logger & Configuration Versioning Module
Logs screening runs, threshold settings, and component decisions for aerospace audit compliance.
"""

import json
import datetime
from pathlib import Path
from typing import Dict, Any, List
from config import DATA_DIR, ROBUST_Z_SCORE_THRESHOLD, SAFETY_SLOPE_MARGIN_RATIO

AUDIT_LOG_FILE = DATA_DIR / "screening_audit_trail.jsonl"

def log_screening_run(
    dataset_name: str,
    total_components: int,
    flagged_count: int,
    review_count: int,
    pass_count: int,
    metrics: Dict[str, Any],
    z_threshold: float = ROBUST_Z_SCORE_THRESHOLD,
    safety_ratio: float = SAFETY_SLOPE_MARGIN_RATIO,
) -> Dict[str, Any]:
    """
    Appends an immutable audit log entry for a screening execution.

    The thresholds MUST be passed in as the values the run actually used.
    Reading the config constants here would certify the defaults even when the
    operator screened at a different setting — the record would be a lie.
    """
    audit_entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "dataset": dataset_name,
        "total_components": total_components,
        "summary": {
            "PASS": pass_count,
            "REVIEW": review_count,
            "FLAG": flagged_count
        },
        "configured_thresholds": {
            "robust_z_score_threshold": z_threshold,
            "safety_margin_ratio": safety_ratio
        },
        "model_performance_metrics": metrics
    }
    
    with open(AUDIT_LOG_FILE, "a") as f:
        f.write(json.dumps(audit_entry) + "\n")
        
    return audit_entry

def get_audit_history(limit: int = 10) -> List[Dict[str, Any]]:
    """
    Retrieves recent audit log entries.
    """
    if not AUDIT_LOG_FILE.exists():
        return []
        
    entries = []
    with open(AUDIT_LOG_FILE, "r") as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))
                
    return entries[-limit:]
