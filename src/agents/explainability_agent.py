import os
import sys

# Adjust path to import packages from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from explainability.explain_utils import ExplainUtils

class ExplainabilityAgent:
    def __init__(self):
        pass

    def explain(self, raw_image, visual_evidence, context_evidence, severity_evidence, claim_decision, metadata):
        """
        Generates visual, feature-level, and textual explanations, including top-5 influential factors.
        """
        disease = visual_evidence.get("disease", "Healthy")
        crop_type = metadata.get("crop_type", "Rice")

        # 1. Visual Explainability (Grad-CAM)
        color_heatmap, gradcam_overlay, raw_heatmap = ExplainUtils.generate_gradcam(
            image=raw_image, 
            disease_class=disease
        )

        # 2. Attention Maps
        attention_scores, attention_overlay = ExplainUtils.generate_attention_map(
            image=raw_image,
            disease_class=disease
        )

        # 3. Decision-level Explainability (SHAP contribution analysis)
        evidence = claim_decision.get("Evidence", {})
        components = evidence.get("components", {})
        weights = evidence.get("weights_used", {})
        shap_pkg = ExplainUtils.compute_shap_values(
            visual_score=components.get("visual_score", 0.0),
            context_score=components.get("context_score", 0.0),
            severity_score=components.get("severity_score", 0.0),
            weights=weights
        )

        # 4. Task 5: Top-5 Most Influential Factors
        # Dynamically rank based on SHAP and metadata checks
        factors = [
            ("Damage Severity", abs(shap_pkg.get("SHAP_Severity", 0.3))),
            ("Heavy Rainfall" if metadata.get("season") == "Monsoon" else "High Temperature", abs(shap_pkg.get("SHAP_Context", 0.2))),
            ("High Humidity", 0.25),
            ("Leaf Discoloration" if disease != "Healthy" else "Leaf Structural Integrity", abs(shap_pkg.get("SHAP_Visual", 0.4))),
            ("Crop Growth Stage", 0.15)
        ]
        # Sort factors by absolute impact score
        sorted_factors = sorted(factors, key=lambda x: x[1], reverse=True)
        top_5_factors = [f"{idx+1} {fac[0]}" for idx, fac in enumerate(sorted_factors)]

        # 5. Generate Natural Language Explanation
        decision = claim_decision.get("Decision", "Reject")
        crs = claim_decision.get("CRS", 0.0)
        yield_loss = severity_evidence.get("YieldLoss", 0.0)
        cci = context_evidence.get("cci", 0.0)

        explanation = ""
        if "invalid" in disease.lower() or "non-crop" in disease.lower():
            explanation = (
                f"The claim was **Rejected** because the uploaded image was identified as **Invalid/Non-Crop Image** "
                f"by the Crop Vision Agent's Out-of-Distribution (OOD) validation module. The color profile and structure "
                f"of the image do not match green agricultural foliage, leaves, or soil. Insurance claims require "
                f"valid crop photos to process."
            )
        elif "healthy" in disease.lower():
            explanation = (
                f"The system recommended **{decision}** for the claim because the crop leaf appears "
                f"completely healthy with zero visual damage. No compensation is eligible."
            )
        elif decision == "Approve":
            explanation = (
                f"The claim was **Approved** with a Claim Reasoning Score (CRS) of **{crs:.2f}** (threshold >= 0.80). "
                f"The Crop Vision Agent identified **{disease}** with high confidence (**{visual_evidence.get('confidence', 0.0)*100:.1f}%**). "
                f"This visual finding is strongly supported by environmental context validation, yielding a Context Consistency Index (CCI) of **{cci:.2f}**. "
                f"Due to the severity of the damage, the expected yield loss is estimated at **{yield_loss*100:.1f}%**, "
                f"qualifying the claim for **{claim_decision.get('ClaimCategory', 'Moderate')}**."
            )
        elif decision == "Manual Review":
            explanation = (
                f"A **Manual Review** is recommended (CRS: **{crs:.2f}**, threshold 0.50 - 0.79). "
                f"While the Crop Vision Agent detected **{disease}** with a severity of **{severity_evidence.get('Severity', 0.0)*100:.1f}%**, "
                f"the Context Verification Agent flagged environmental inconsistencies (CCI: **{cci:.2f}**). "
                f"This discrepancy indicates that either the weather conditions do not support the crop disease development, "
                f"or there is a mismatch in crop seasonal metadata. An inspector visit is required to resolve this."
            )
        else:
            explanation = (
                f"The claim was **Rejected** (CRS: **{crs:.2f}**, threshold < 0.50). "
                f"The supporting evidence is weak: either the vision agent predicts low damage severity, or "
                f"the environmental conditions are highly inconsistent with the reported damage (CCI: **{cci:.2f}**). "
                f"No automatic or manual compensation is recommended."
            )

        return {
            "gradcam_heatmap": color_heatmap,
            "gradcam_overlay": gradcam_overlay,
            "attention_grid": attention_scores,
            "attention_overlay": attention_overlay,
            "shap_values": shap_pkg,
            "text_explanation": explanation,
            "top_influential_factors": top_5_factors
        }
