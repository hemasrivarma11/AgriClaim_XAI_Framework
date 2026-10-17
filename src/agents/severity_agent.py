import numpy as np

class SeverityAgent:
    def __init__(self):
        # Crop yield vulnerability factor
        self.crop_vulnerability = {
            "rice": 0.75,   # High vulnerability
            "cotton": 0.65, # Moderate vulnerability
            "wheat": 0.80,  # Critical vulnerability to leaf rust
            "default": 0.70
        }

    def estimate(self, visual_evidence, context_evidence, metadata):
        """
        Estimates the damage severity category, affected area, and potential yield loss deterministically.
        """
        disease = visual_evidence.get("disease", "Healthy").lower()
        vis_severity = visual_evidence.get("severity", 0.0)
        cci = context_evidence.get("cci", 0.0)
        crop_type = metadata.get("crop_type", "Rice").lower()

        if "healthy" in disease:
            severity_val = 0.0
            affected_area = 0.0
            yield_loss_val = 0.0
            sev_category = "Healthy"
            risk_level = "Low"
        else:
            severity_val = vis_severity
            
            # Estimate Affected Area: deterministic scaling of visual severity
            affected_area = float(np.clip(vis_severity * 0.95, 0.0, 1.0))
            
            # Map severity value to Category
            if severity_val <= 0.20:
                sev_category = "Healthy"
            elif severity_val <= 0.40:
                sev_category = "Mild"
            elif severity_val <= 0.60:
                sev_category = "Moderate"
            elif severity_val <= 0.80:
                sev_category = "Severe"
            else:
                sev_category = "Critical"

            # Compute Yield Loss: Y_L = Severity * AffectedArea * Vulnerability + (1 - CCI) * 0.10
            vuln_factor = self.crop_vulnerability.get(crop_type, self.crop_vulnerability["default"])
            yield_loss_val = (severity_val * affected_area * vuln_factor) + ((1.0 - cci) * 0.10)
            yield_loss_val = float(np.clip(yield_loss_val, 0.0, 0.95))

            # Assign Risk Level
            if yield_loss_val >= 0.60:
                risk_level = "High"
            elif yield_loss_val >= 0.30:
                risk_level = "Moderate"
            else:
                risk_level = "Low"

        return {
            # Standardized Capitalized Keys (Task 2)
            "Severity": severity_val,
            "AffectedArea": affected_area,
            "YieldLoss": yield_loss_val,
            "RiskLevel": risk_level,
            
            # Backward Compatibility Keys
            "severity": severity_val,
            "severity_category": sev_category,
            "yield_loss": yield_loss_val,
            "risk_category": risk_level,
            "vulnerability_multiplier": self.crop_vulnerability.get(crop_type, 0.70)
        }
