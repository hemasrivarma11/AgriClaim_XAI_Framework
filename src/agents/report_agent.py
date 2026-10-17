import json
from datetime import datetime

class ReportAgent:
    def __init__(self):
        pass

    def generate(self, claim_decision, visual_evidence, context_evidence, severity_evidence, explainability_pkg, metadata):
        """
        Generates distinct reports for farmers and insurance officers.
        Farmer reports are simple. Officer reports are evidence-oriented and highly detailed.
        """
        crop_type = metadata.get("crop_type", "Rice").capitalize()
        disease = visual_evidence.get("disease", "Healthy")
        decision = claim_decision.get("Decision", "Reject")
        claim_cat = claim_decision.get("ClaimCategory", "No Compensation")
        yield_loss = severity_evidence.get("YieldLoss", 0.0)
        affected_area = severity_evidence.get("AffectedArea", 0.0)
        cci = context_evidence.get("cci", 0.0)
        crs = claim_decision.get("CRS", 0.0)
        reason_codes = claim_decision.get("ReasonCodes", [])
        
        if disease.lower() == "healthy":
            status = "Healthy Crop"
        elif "invalid" in disease.lower():
            status = "Invalid/Non-Crop Image Uploaded"
        else:
            status = f"Affected by {disease}"

        # 1. Farmer Report (simple language, unchanged style)
        farmer_report = {
            "title": "Crop Insurance Claim Report",
            "date": metadata.get("date", datetime.now().strftime("%Y-%m-%d")),
            "crop": crop_type,
            "status": status,
            "severity_level": "N/A" if "invalid" in disease.lower() else severity_evidence.get("severity_category", "Healthy"),
            "estimated_yield_loss": "0.0%" if "invalid" in disease.lower() else f"{yield_loss * 100:.1f}%",
            "insurance_recommendation": "Ineligible (Invalid Image)" if "invalid" in disease.lower() else ("Eligible for Payout" if decision == "Approve" else ("Pending Verification" if decision == "Manual Review" else "Ineligible for Payout")),
            "claim_grade": claim_cat,
            "action_required": (
                "Please upload a valid crop or foliage photograph to process this claim." if "invalid" in disease.lower() else (
                    "No action required. Your crop is healthy." if disease.lower() == "healthy" else (
                        "None. Your claim has been pre-approved. Compensation will be processed automatically." if decision == "Approve" else (
                            "Assessor Visit Required. An insurance officer will visit your field within 7 days to verify conditions." if decision == "Manual Review" else
                            "Appeal Claim. If you believe this is incorrect, you may request a manual survey at your local agency."
                        )
                    )
                )
            )
        }

        # 2. Officer Report (Task 6: Evidence-oriented detailed report)
        officer_report = {
            "report_metadata": {
                "report_id": f"ACX-{int(datetime.now().timestamp())}",
                "generated_at": datetime.now().isoformat(),
                "gps_coordinates": f"{metadata.get('latitude', 20.0):.4f}, {metadata.get('longitude', 78.0):.4f}",
                "season": metadata.get("season", "Monsoon"),
                "crop_growth_stage": metadata.get("growth_stage", "Vegetative")
            },
            "evidence_summaries": {
                "visual_evidence_summary": {
                    "disease_classification": disease,
                    "confidence_score": f"{visual_evidence.get('confidence', 0.0) * 100:.2f}%",
                    "visual_features_status": "Abnormal Lesions Detected" if disease.lower() != "healthy" else "Pristine Foliage Structures"
                },
                "context_evidence_summary": {
                    "meteorological_conditions": f"Temp: {metadata.get('temperature', 25.0)}C, Rain: {metadata.get('rainfall', 50.0)}mm, Humidity: {metadata.get('humidity', 70.0)}%",
                    "consistency_check_status": "Valid Context" if cci >= 0.70 else "Mismatched Context Profile",
                    "context_consistency_index": f"{cci:.4f}"
                },
                "severity_evidence_summary": {
                    "affected_area_percentage": f"{affected_area * 100:.1f}%",
                    "crop_vulnerability_index": f"{severity_evidence.get('vulnerability_multiplier', 0.70):.2f}",
                    "estimated_yield_loss_percentage": f"{yield_loss * 100:.1f}%"
                }
            },
            "analytical_fusion_metrics": {
                "claim_reasoning_score": f"{crs:.4f}",
                "reason_codes": reason_codes,
                "weights_applied": claim_decision.get("Evidence", {}).get("weights_used", {}),
                "top_5_decision_influencers": explainability_pkg.get("top_influential_factors", [])
            },
            "final_recommendation": {
                "decision": decision,
                "claim_category": claim_cat,
                "assessor_instruction": "Trigger automatic payment processing" if decision == "Approve" else ("Schedule prompt physical field inspection" if decision == "Manual Review" else "Reject claim file immediately")
            }
        }

        # Compile package
        report_pkg = {
            "farmer": farmer_report,
            "officer": officer_report
        }

        return report_pkg

    def export_json(self, report_pkg, filepath):
        """Export the reports as JSON."""
        try:
            with open(filepath, 'w') as f:
                json.dump(report_pkg, f, indent=4)
            return True
        except Exception as e:
            print(f"Error exporting report: {e}")
            return False
            
    def export_html(self, report_pkg, filepath):
        """Export as styled HTML for presentation."""
        farmer = report_pkg["farmer"]
        officer = report_pkg["officer"]
        
        # Color mapping for badges
        badge_color = "#3498db"
        if officer["final_recommendation"]["decision"] == "Approve":
            badge_color = "#2ecc71"
        elif officer["final_recommendation"]["decision"] == "Reject":
            badge_color = "#e74c3c"
            
        html_content = f"""
        <html>
        <head>
            <meta charset="utf-8">
            <style>
                body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f8f9fa; color: #333; margin: 20px; }}
                .container {{ max-width: 900px; margin: 0 auto; background: white; padding: 30px; border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.05); }}
                h1 {{ color: #2c3e50; text-align: center; border-bottom: 2px solid #ecf0f1; padding-bottom: 10px; }}
                h2 {{ color: #16a085; border-bottom: 1px solid #eee; padding-bottom: 5px; margin-top: 30px; }}
                .card {{ background: #fdfefe; border: 1px solid #e5e8e8; border-radius: 8px; padding: 20px; margin-bottom: 15px; }}
                .badge {{ display: inline-block; padding: 6px 12px; font-weight: bold; border-radius: 20px; color: white; background-color: {badge_color}; }}
                table {{ width: 100%; border-collapse: collapse; margin-top: 15px; }}
                th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #f2f2f2; color: #2c3e50; }}
                .section-title {{ font-weight: bold; color: #2c3e50; margin-top: 15px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>AgriClaim-XAI Assessment Report</h1>
                <p style="text-align: center; color: #7f8c8d;">Generated on {farmer["date"]}</p>
                
                <h2>Farmer Report Summary</h2>
                <div class="card">
                    <p><strong>Crop Type:</strong> {farmer["crop"]}</p>
                    <p><strong>Crop Status:</strong> {farmer["status"]}</p>
                    <p><strong>Damage Severity Level:</strong> {farmer["severity_level"]}</p>
                    <p><strong>Estimated Yield Loss:</strong> {farmer["estimated_yield_loss"]}</p>
                    <p><strong>Insurance Decision:</strong> <span class="badge">{farmer["insurance_recommendation"]}</span></p>
                    <p><strong>Compensation Grade:</strong> {farmer["claim_grade"]}</p>
                    <p><strong>Action Required:</strong> <span style="color: #e67e22; font-weight: bold;">{farmer["action_required"]}</span></p>
                </div>
                
                <h2>Insurance Officer Analytical Report</h2>
                <div class="card">
                    <p><strong>Report Reference:</strong> {officer["report_metadata"]["report_id"]}</p>
                    <p><strong>Location Coordinates:</strong> {officer["report_metadata"]["gps_coordinates"]}</p>
                    <p><strong>Growth Stage:</strong> {officer["report_metadata"]["crop_growth_stage"]}</p>
                    <p><strong>Season:</strong> {officer["report_metadata"]["season"]}</p>
                    
                    <h3 class="section-title">Evidence Summaries</h3>
                    <table>
                        <tr><th>Dimension</th><th>Details</th><th>Metric / Score</th></tr>
                        <tr>
                            <td><strong>Visual Evidence</strong></td>
                            <td>Detected: {officer["evidence_summaries"]["visual_evidence_summary"]["disease_classification"]}</td>
                            <td>Confidence: {officer["evidence_summaries"]["visual_evidence_summary"]["confidence_score"]}</td>
                        </tr>
                        <tr>
                            <td><strong>Context Evidence</strong></td>
                            <td>Status: {officer["evidence_summaries"]["context_evidence_summary"]["consistency_check_status"]}</td>
                            <td>CCI: {officer["evidence_summaries"]["context_evidence_summary"]["context_consistency_index"]}</td>
                        </tr>
                        <tr>
                            <td><strong>Severity Evidence</strong></td>
                            <td>Affected Area: {officer["evidence_summaries"]["severity_evidence_summary"]["affected_area_percentage"]}</td>
                            <td>Yield Loss: {officer["evidence_summaries"]["severity_evidence_summary"]["estimated_yield_loss_percentage"]}</td>
                        </tr>
                    </table>

                    <h3 class="section-title">Fusion Reasoning</h3>
                    <p><strong>Claim Reasoning Score (CRS):</strong> {officer["analytical_fusion_metrics"]["claim_reasoning_score"]}</p>
                    <p><strong>Reason Codes:</strong></p>
                    <ul>
                        {"".join(f"<li>{code}</li>" for code in officer["analytical_fusion_metrics"]["reason_codes"])}
                    </ul>
                    <p><strong>Top Influential Factors:</strong></p>
                    <ol>
                        {"".join(f"<li>{factor}</li>" for factor in officer["analytical_fusion_metrics"]["top_5_decision_influencers"])}
                    </ol>

                    <h3 class="section-title">Final Decision & Recommendation</h3>
                    <p><strong>Final Decision:</strong> <span class="badge">{officer["final_recommendation"]["decision"]}</span></p>
                    <p><strong>Claim Category:</strong> {officer["final_recommendation"]["claim_category"]}</p>
                    <p><strong>Assessor Instruction:</strong> <em>{officer["final_recommendation"]["assessor_instruction"]}</em></p>
                </div>
            </div>
        </body>
        </html>
        """
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(html_content)
            return True
        except Exception as e:
            print(f"Error exporting HTML: {e}")
            return False
