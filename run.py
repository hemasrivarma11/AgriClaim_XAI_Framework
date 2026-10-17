import os
import sys
import time
import argparse
from datetime import datetime

# Adjust path to import packages from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from agents.data_agent import DataIntegrationAgent
from agents.vision_agent import CropVisionAgent
from agents.context_agent import ContextVerificationAgent
from agents.severity_agent import SeverityAgent
from agents.insurance_agent import ClaimReasoningAgent
from agents.explainability_agent import ExplainabilityAgent
from agents.report_agent import ReportAgent
from utils.metrics_logger import MetricsLogger

def run_pipeline(image_path, crop_type, season, lat, lon, suspected_damage=None, output_dir="08_Results"):
    print("\n" + "="*50)
    print("      AgriClaim-XAI Execution Pipeline")
    print("="*50)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Input Image: {image_path}")
    print(f"Crop Type: {crop_type} | Season: {season}")
    print(f"GPS Coordinates: ({lat}, {lon})")
    print(f"Suspected Damage (Context Mode): {suspected_damage}")
    print("-"*50)

    # Make output directory
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    # Instantiate agents
    print("[1/7] Initializing Data Integration Agent...")
    data_agent = DataIntegrationAgent()
    
    print("[2/7] Initializing Crop Vision Agent...")
    vision_agent = CropVisionAgent()
    
    print("[3/7] Initializing Context Verification Agent...")
    context_agent = ContextVerificationAgent()
    
    print("[4/7] Initializing Damage Severity Agent...")
    severity_agent = SeverityAgent()
    
    print("[5/7] Initializing Claim Reasoning Agent...")
    claim_agent = ClaimReasoningAgent()
    
    print("[6/7] Initializing Explainability Agent...")
    xai_agent = ExplainabilityAgent()
    
    print("[7/7] Initializing Report Generation Agent...")
    report_agent = ReportAgent()

    t_start = time.time()

    # Step 1: Preprocessing and Data Integration
    print("\n[Step 1] Running Data Integration...")
    gps_coords = {"latitude": lat, "longitude": lon}
    metadata = {
        "crop_type": crop_type,
        "season": season,
        "date": datetime.now().strftime("%Y-%m-%d")
    }
    processed_dataset = data_agent.process(image_path, gps_coords, metadata, suspected_damage)
    print(f" - Resized image shape: {processed_dataset['resized_image'].shape}")
    print(f" - Weather conditions retrieved: {processed_dataset['weather']}")
    print(f" - Metadata validated: {processed_dataset['metadata']}")

    # Step 2: Crop Vision analysis
    print("\n[Step 2] Running Crop Vision Agent (Swin Transformer)...")
    visual_evidence = vision_agent.analyze(processed_dataset, image_path)
    print(f" - Backbone: {visual_evidence['swin_backbone']}")
    print(f" - Detected Disease: {visual_evidence['disease']}")
    print(f" - Severity Estimate: {visual_evidence['severity']*100:.2f}%")
    print(f" - Model Confidence: {visual_evidence['confidence']*100:.2f}%")

    # Step 3: Context Verification
    print("\n[Step 3] Running Context Verification Agent...")
    context_evidence = context_agent.verify(visual_evidence, processed_dataset["weather"], processed_dataset["metadata"], suspected_damage=suspected_damage)
    print(f" - Context Consistency Index (CCI): {context_evidence['cci']:.4f}")
    print(f" - Verification Logs:")
    for key, val in context_evidence["logs"].items():
        # Clean unicode characters/emojis for Windows terminal printing safety
        clean_val = str(val).encode('ascii', 'ignore').decode('ascii')
        print(f"    * {key}: {clean_val}")

    # Step 4: Damage Severity & Yield Loss Estimation
    print("\n[Step 4] Running Damage Severity & Yield Loss Agent...")
    severity_evidence = severity_agent.estimate(visual_evidence, context_evidence, processed_dataset["metadata"])
    print(f" - Categorized Severity: {severity_evidence['severity_category']}")
    print(f" - Expected Yield Loss: {severity_evidence['yield_loss']*100:.2f}%")
    print(f" - Risk Category: {severity_evidence['risk_category']}")

    # Step 5: Claim Reasoning
    print("\n[Step 5] Running Claim Reasoning Agent (Evidence Fusion)...")
    claim_decision = claim_agent.process_claim(visual_evidence, context_evidence, severity_evidence, processed_dataset["metadata"])
    print(f" - Claim Reasoning Score (CRS): {claim_decision['CRS']:.4f}")
    print(f" - Decision Output: {claim_decision['Decision']}")
    print(f" - Compensation Category: {claim_decision['ClaimCategory']}")
    print(f" - Fusion Weights used: {claim_decision.get('Evidence', {}).get('weights_used')}")

    # Step 6: Explainability
    print("\n[Step 6] Running Explainability Agent...")
    explainability_pkg = xai_agent.explain(
        raw_image=processed_dataset["raw_image"],
        visual_evidence=visual_evidence,
        context_evidence=context_evidence,
        severity_evidence=severity_evidence,
        claim_decision=claim_decision,
        metadata=processed_dataset["metadata"]
    )
    print(f" - Human Explanation: {explainability_pkg['text_explanation']}")
    print(f" - SHAP values: {explainability_pkg['shap_values']}")

    # Step 7: Report Generation
    print("\n[Step 7] Running Report Generation Agent...")
    report_pkg = report_agent.generate(
        claim_decision=claim_decision,
        visual_evidence=visual_evidence,
        context_evidence=context_evidence,
        severity_evidence=severity_evidence,
        explainability_pkg=explainability_pkg,
        metadata=processed_dataset["metadata"]
    )

    # Export Reports
    json_path = os.path.join(output_dir, f"{base_name}_claim_report.json")
    html_path = os.path.join(output_dir, f"{base_name}_claim_report.html")
    
    report_agent.export_json(report_pkg, json_path)
    report_agent.export_html(report_pkg, html_path)
    print(f" - Exported JSON report: [report]({json_path})")
    print(f" - Exported HTML report: [report]({html_path})")
    
    # Calculate execution time and log
    exec_time = time.time() - t_start
    logger = MetricsLogger()
    logger.log_prediction(
        image_id=base_name,
        crop_type=crop_type,
        visual_evidence=visual_evidence.get("disease", "Healthy"),
        confidence_score=visual_evidence.get("confidence", 0.0),
        severity=severity_evidence.get("Severity", 0.0),
        yield_loss=severity_evidence.get("YieldLoss", 0.0),
        cci=context_evidence.get("cci", 0.0),
        claim_reasoning_score=claim_decision.get("CRS", 0.0),
        decision=claim_decision.get("Decision", "Reject"),
        claim_category=claim_decision.get("ClaimCategory", "No Compensation"),
        execution_time=exec_time
    )
    print(f" - Prediction metrics logged to outputs/results/predictions.csv & predictions.json")
    
    print("="*50)
    print("      AgriClaim-XAI Execution Completed")
    print("="*50 + "\n")

    return report_pkg

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AgriClaim-XAI Framework CLI Pipeline")
    parser.add_argument("--image", type=str, default="samples/rice_leaf_blight.png", help="Path to crop image")
    parser.add_argument("--crop", type=str, default="Rice", help="Crop type (Rice, Cotton, Wheat)")
    parser.add_argument("--season", type=str, default="Monsoon", help="Season (Monsoon, Winter, Summer)")
    parser.add_argument("--lat", type=float, default=17.3850, help="Latitude")
    parser.add_argument("--lon", type=float, default=78.4867, help="Longitude")
    parser.add_argument("--damage", type=str, default="blight", help="Suspected damage (forces weather simulation context)")
    parser.add_argument("--test", action="store_true", help="Run quick automated validation test")

    args = parser.parse_args()

    if args.test:
        print("Running quick automated pipeline validation tests...")
        # Validate using healthy crop sample
        run_pipeline(
            image_path="samples/healthy_crop.png",
            crop_type="Rice",
            season="Monsoon",
            lat=17.3850,
            lon=78.4867,
            suspected_damage="healthy"
        )
        # Validate using diseased crop sample (mismatched weather to test manual review trigger)
        run_pipeline(
            image_path="samples/rice_leaf_blight.png",
            crop_type="Rice",
            season="Monsoon",
            lat=17.3850,
            lon=78.4867,
            suspected_damage="drought" # Causes extreme weather mismatch
        )
        print("Verification completed successfully!")
    else:
        run_pipeline(
            image_path=args.image,
            crop_type=args.crop,
            season=args.season,
            lat=args.lat,
            lon=args.lon,
            suspected_damage=args.damage
        )
