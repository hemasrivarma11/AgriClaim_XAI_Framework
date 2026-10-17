import os
import json
import numpy as np

class EvidenceFusionEngine:
    def __init__(self, config_path="configs/fusion_config.json"):
        self.config_path = config_path
        self.crop_weight_profiles = {}
        self.load_weights()

    def load_weights(self):
        """Loads optimized weights from fusion_config.json."""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    self.crop_weight_profiles = json.load(f)
                return
            except Exception as e:
                print(f"Error reading weights config: {e}. Reverting to default profiles.")

        self.crop_weight_profiles = {
            "rice": {"alpha": 0.30, "beta": 0.40, "gamma": 0.30},
            "cotton": {"alpha": 0.50, "beta": 0.20, "gamma": 0.30},
            "wheat": {"alpha": 0.30, "beta": 0.20, "gamma": 0.50},
            "maize": {"alpha": 0.40, "beta": 0.25, "gamma": 0.35},
            "soybean": {"alpha": 0.35, "beta": 0.30, "gamma": 0.35},
            "sugarcane": {"alpha": 0.35, "beta": 0.35, "gamma": 0.30},
            "default": {"alpha": 0.40, "beta": 0.30, "gamma": 0.30}
        }
        self.save_weights()

    def save_weights(self):
        """Saves current weight profiles to fusion_config.json."""
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        try:
            with open(self.config_path, 'w') as f:
                json.dump(self.crop_weight_profiles, f, indent=4)
        except Exception as e:
            print(f"Error saving weights config: {e}")

    def get_weights(self, crop_type):
        """Get weights for a specific crop, fallback to default."""
        crop_clean = str(crop_type).lower().strip()
        return self.crop_weight_profiles.get(crop_clean, self.crop_weight_profiles.get("default", {"alpha": 0.4, "beta": 0.3, "gamma": 0.3}))

    def fuse_evidence(self, visual_evidence, context_evidence, severity_evidence, crop_type):
        """
        Fuses multimodal evidence vectors into a Fused Evidence Vector.
        No decision, claim category, or thresholds are evaluated inside this engine.
        """
        weights = self.get_weights(crop_type)
        alpha = weights["alpha"]
        beta = weights["beta"]
        gamma = weights["gamma"]

        disease = visual_evidence.get("disease", "Healthy").lower()
        confidence_score = visual_evidence.get("confidence", 0.0)

        # 1. Visual Score (zero if healthy)
        v_score = 0.0 if "healthy" in disease else confidence_score

        # 2. Context Score (CCI)
        c_score = context_evidence.get("cci", 0.0)

        # 3. Severity Score
        severity = severity_evidence.get("Severity", 0.0)
        yield_loss = severity_evidence.get("YieldLoss", 0.0)
        s_score = 0.0 if "healthy" in disease else ((severity * 0.5) + (yield_loss * 0.5))

        # Compute Fused Score
        fused_score = (alpha * v_score) + (beta * c_score) + (gamma * s_score)
        fused_score = float(np.clip(fused_score, 0.0, 1.0))

        return {
            "visual_evidence": visual_evidence,
            "context_evidence": context_evidence,
            "severity_evidence": severity_evidence,
            "fused_vector": {
                "fused_score": fused_score,
                "visual_score": v_score,
                "context_score": c_score,
                "severity_score": s_score,
                "weights_used": weights
            }
        }
