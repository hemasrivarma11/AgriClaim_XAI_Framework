import os
import sys

# Adjust path to import packages from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from insurance.evidence_fusion_engine import EvidenceFusionEngine

class ClaimReasoningAgent:
    def __init__(self):
        self.evidence_fusion_engine = EvidenceFusionEngine()

    def process_claim(self, visual_evidence, context_evidence, severity_evidence, metadata):
        """
        Processes claim by first calling the EvidenceFusionEngine to get the Fused Evidence Vector,
        then evaluates thresholds, assigns category, and generates Reason Codes.
        """
        crop_type = metadata.get("crop_type", "Rice").lower()
        
        # Fuse evidence
        fusion_pkg = self.evidence_fusion_engine.fuse_evidence(
            visual_evidence=visual_evidence,
            context_evidence=context_evidence,
            severity_evidence=severity_evidence,
            crop_type=crop_type
        )
        
        fused_vector = fusion_pkg["fused_vector"]
        crs = fused_vector["fused_score"]
        
        disease = visual_evidence.get("disease", "Healthy").lower()
        yield_loss = severity_evidence.get("YieldLoss", 0.0)
        cci = context_evidence.get("cci", 0.0)
        
        reason_codes = []
        
        # Decision Logic & Reason Codes
        if "invalid" in disease or "non-crop" in disease:
            decision = "Reject"
            claim_category = "No Compensation (Invalid Image)"
            reason_codes.append("OOD Verification Failed: Non-Crop Image Uploaded")
        elif "healthy" in disease:
            decision = "Reject"
            claim_category = "No Compensation"
            reason_codes.append("Zero Visual Damage: Crop classified as Healthy")
        elif crs >= 0.80:
            decision = "Approve"
            if yield_loss < 0.30:
                claim_category = "Low Compensation"
            elif yield_loss < 0.60:
                claim_category = "Moderate Compensation"
            else:
                claim_category = "High Compensation"
                
            # Populate positive reason codes
            if visual_evidence.get("confidence", 0.0) >= 0.85:
                reason_codes.append("High Visual Confidence")
            if cci >= 0.80:
                reason_codes.append("Context Consistency Above Threshold")
            if yield_loss >= 0.50:
                reason_codes.append("High Estimated Yield Loss")
            if "impossibility_flag" not in context_evidence or not context_evidence["impossibility_flag"]:
                reason_codes.append("Weather Matches Damage Pattern")
        elif crs >= 0.50:
            decision = "Manual Review"
            claim_category = "Pending Assessment"
            
            # Populate warning reason codes
            if cci < 0.60:
                reason_codes.append("Low Context Consistency")
            if visual_evidence.get("confidence", 0.0) < 0.80:
                reason_codes.append("Medium Visual Confidence")
            if context_evidence.get("impossibility_flag", False):
                reason_codes.append("Weather Does Not Match")
            else:
                reason_codes.append("Discrepancy In Environmental Metadata")
        else:
            decision = "Reject"
            claim_category = "No Compensation"
            reason_codes.append("Low Multimodal Evidence Score")
            if cci < 0.50:
                reason_codes.append("Environmental Context Inconsistent")

        return {
            "CRS": crs,
            "Decision": decision,
            "ClaimCategory": claim_category,
            "ReasonCodes": reason_codes,
            "Evidence": {
                "weights_used": fused_vector["weights_used"],
                "components": {
                    "visual_score": fused_vector["visual_score"],
                    "context_score": fused_vector["context_score"],
                    "severity_score": fused_vector["severity_score"]
                }
            }
        }
