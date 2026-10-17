import sys
sys.path.append('src')
from agents.context_agent import ContextVerificationAgent

a = ContextVerificationAgent()

print("=== TEST 1: Flood Damage in Summer (should REJECT, CCI=0.05) ===")
r = a.verify(
    {"disease": "Leaf Blight", "confidence": 0.95, "severity": 0.7},
    {"temperature": 43.0, "rainfall": 300.0, "humidity": 90.0, "wind_speed": 12.0},
    {"crop_type": "Rice", "season": "Summer", "date": "2026-07-02", "latitude": 17.3, "longitude": 78.4},
    suspected_damage="Flood Damage"
)
print("CCI:", r["cci"])
print("Impossible:", r["impossibility_flag"])
print("Reason:", r["impossibility_reason"])

print()
print("=== TEST 2: Drought Damage in Monsoon (should REJECT, CCI=0.05) ===")
r2 = a.verify(
    {"disease": "Leaf Blight", "confidence": 0.93, "severity": 0.6},
    {"temperature": 25.0, "rainfall": 200.0, "humidity": 85.0, "wind_speed": 10.0},
    {"crop_type": "Cotton", "season": "Monsoon", "date": "2026-07-02", "latitude": 17.3, "longitude": 78.4},
    suspected_damage="Drought Damage"
)
print("CCI:", r2["cci"])
print("Impossible:", r2["impossibility_flag"])
print("Reason:", r2["impossibility_reason"])

print()
print("=== TEST 3: Flood Damage in Monsoon (should PASS normally) ===")
r3 = a.verify(
    {"disease": "Leaf Blight", "confidence": 0.96, "severity": 0.74},
    {"temperature": 28.0, "rainfall": 320.0, "humidity": 92.0, "wind_speed": 15.0},
    {"crop_type": "Rice", "season": "Monsoon", "date": "2026-07-02", "latitude": 17.3, "longitude": 78.4},
    suspected_damage="Flood Damage"
)
print("CCI:", round(r3["cci"], 4))
print("Impossible:", r3["impossibility_flag"])
print("Verified:", r3["verified_context"])
