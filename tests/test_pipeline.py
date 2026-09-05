"""
Unit & Integration Test Suite for Component Burn-In Anomaly Detection Pipeline
"""

import unittest
import pandas as pd
import numpy as np

from data.synthetic_generator import generate_synthetic_burnin_data
from data.ingestion import load_burnin_dataset
from data.validation import validate_burnin_dataset
from models.robust_stats import compute_lot_robust_stats
from models.isolation_forest import train_run_isolation_forest
from models.drift_predictor import DriftPredictor168h
from engine.risk_engine import evaluate_composite_risk
from engine.explainer import generate_component_explanation
from pipeline import run_full_screening_pipeline

class TestBurnInScreeningPipeline(unittest.TestCase):

    def setUp(self):
        self.df_synthetic = generate_synthetic_burnin_data(num_lots=3, components_per_lot=20, seed=42)

    def test_01_synthetic_generator(self):
        self.assertEqual(len(self.df_synthetic), 60)
        self.assertIn("Component_ID", self.df_synthetic.columns)
        self.assertIn("Value_0h", self.df_synthetic.columns)
        self.assertIn("Value_168h", self.df_synthetic.columns)

    def test_02_data_validation(self):
        clean_df, summary = validate_burnin_dataset(self.df_synthetic)
        self.assertTrue(summary["is_valid"])
        self.assertEqual(len(clean_df), 60)

    def test_03_robust_statistics(self):
        df_stats = compute_lot_robust_stats(self.df_synthetic, target_col="Value_24h")
        self.assertIn("Value_24h_Robust_Z", df_stats.columns)
        self.assertIn("Value_24h_Lot_Median", df_stats.columns)

    def test_04_isolation_forest(self):
        df_iso, _ = train_run_isolation_forest(self.df_synthetic)
        self.assertIn("IsoForest_Prediction", df_iso.columns)
        self.assertIn("IsoForest_Score", df_iso.columns)

    def test_05_drift_predictor(self):
        predictor = DriftPredictor168h()
        train_res = predictor.fit(self.df_synthetic)
        self.assertLess(train_res["train_mae"], 10.0)
        
        df_pred = predictor.predict(self.df_synthetic)
        self.assertIn("Predicted_Value_168h", df_pred.columns)

    def test_06_end_to_end_pipeline(self):
        processed_df, summary = run_full_screening_pipeline(
            self.df_synthetic, dataset_name="Test Run", log_run=False
        )  # log_run=False: a unit test must not append to the tracked audit trail
        self.assertEqual(len(processed_df), 60)
        self.assertIn("Screening_Status", processed_df.columns)
        self.assertIn("Explanation_Summary", processed_df.columns)
        
        # Verify status categories exist
        statuses = processed_df["Screening_Status"].unique()
        self.assertTrue(any(s in ["PASS", "REVIEW", "FLAG (HIGH RISK)"] for s in statuses))

if __name__ == "__main__":
    unittest.main()
