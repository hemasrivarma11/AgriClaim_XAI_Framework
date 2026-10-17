import os
import csv
import json
from datetime import datetime

class MetricsLogger:
    def __init__(self, csv_path="outputs/results/predictions.csv", json_path="outputs/results/predictions.json"):
        self.csv_path = csv_path
        self.json_path = json_path
        
        # Ensure directories exist
        os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.json_path), exist_ok=True)

    def log_prediction(self, image_id, crop_type, visual_evidence, confidence_score, severity, yield_loss, cci, claim_reasoning_score, decision, claim_category, execution_time):
        """Appends a new record to CSV and JSON logs using consistent IEEE paper terminology (Task 7)."""
        timestamp = datetime.now().isoformat()
        
        record = {
            "Image ID": image_id,
            "Crop Type": crop_type,
            "Visual Evidence": visual_evidence,
            "Confidence Score": float(confidence_score),
            "Damage Severity": float(severity),
            "Yield Loss": float(yield_loss),
            "Context Consistency Index (CCI)": float(cci),
            "Claim Reasoning Score (CRS)": float(claim_reasoning_score),
            "Insurance Decision": decision,
            "Claim Category": claim_category,
            "Execution Time": float(execution_time),
            "Timestamp": timestamp
        }

        # 1. Append to CSV
        file_exists = os.path.exists(self.csv_path)
        headers = list(record.keys())
        
        try:
            with open(self.csv_path, mode='a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=headers)
                if not file_exists:
                    writer.writeheader()
                writer.writerow(record)
        except Exception as e:
            print(f"Error writing to CSV log: {e}")

        # 2. Append to JSON
        try:
            records = []
            if os.path.exists(self.json_path):
                with open(self.json_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if content:
                        records = json.loads(content)
            
            records.append(record)
            
            with open(self.json_path, 'w', encoding='utf-8') as f:
                json.dump(records, f, indent=4)
        except Exception as e:
            print(f"Error writing to JSON log: {e}")
