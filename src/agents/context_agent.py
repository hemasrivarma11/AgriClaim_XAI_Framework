import numpy as np

class ContextVerificationAgent:
    def __init__(self):
        # Weights for the CCI calculation (Task 3: 5 components, 0.20 each)
        self.weights = {
            "w_weather": 0.20,
            "w_location": 0.20,
            "w_season": 0.20,
            "w_metadata": 0.20,
            "w_growth_stage": 0.20
        }

        # Season-damage impossibility matrix
        self.impossibility_rules = [
            {
                "seasons": ["summer"],
                "damages": ["flood", "waterlog"],
                "reason":  "Flood damage is climatologically impossible during Summer — no monsoon rainfall occurs. Claim is physically inconsistent."
            },
            {
                "seasons": ["winter"],
                "damages": ["flood", "waterlog"],
                "reason":  "Flood damage during Winter is extremely unlikely in most agricultural belts. Seasonal context strongly contradicts the claimed damage."
            },
            {
                "seasons": ["monsoon"],
                "damages": ["drought", "heat stress", "wilt"],
                "reason":  "Drought damage during the Monsoon season is climatologically impossible — high rainfall prevents soil moisture deficit."
            },
            {
                "seasons": ["summer"],
                "damages": ["frost", "freeze"],
                "reason":  "Frost or freeze damage cannot occur in Summer conditions. Claim is climatologically impossible."
            },
            {
                "seasons": ["summer", "monsoon"],
                "damages": ["snow"],
                "reason":  "Snow damage is impossible during Summer/Monsoon seasons in agricultural zones."
            },
        ]

    def _check_season_damage_impossibility(self, season, disease, suspected_damage_str=""):
        """
        Returns (is_impossible: bool, reason: str).
        Checks both the vision-detected disease label and the user-declared
        suspected damage string against the impossibility matrix.
        """
        combined_text = (disease + " " + suspected_damage_str).lower()
        season_lower = season.lower()

        for rule in self.impossibility_rules:
            if season_lower in rule["seasons"]:
                for dmg_kw in rule["damages"]:
                    if dmg_kw in combined_text:
                        return True, rule["reason"]
        return False, ""

    def verify(self, visual_evidence, weather_data, metadata, suspected_damage=""):
        """
        Validates the visual findings against environmental weather and metadata.
        Computes the Context Consistency Index (CCI) incorporating Crop Growth Stage.
        """
        disease   = visual_evidence.get("disease", "Healthy").lower()
        crop_type = metadata.get("crop_type", "Rice").lower()
        season    = metadata.get("season", "Monsoon").lower()

        temp     = weather_data.get("temperature", 25.0)
        rainfall = weather_data.get("rainfall", 50.0)
        humidity = weather_data.get("humidity", 70.0)
        wind     = weather_data.get("wind_speed", 10.0)

        # ── STEP 0: Season-Damage Impossibility Check ──────────────────────────
        impossibility_flag, impossibility_reason = self._check_season_damage_impossibility(
            season, disease, suspected_damage
        )

        if impossibility_flag:
            return {
                "cci": 0.05,
                "scores": {
                    "weather_score":  0.05,
                    "location_score": 0.05,
                    "season_score":   0.05,
                    "metadata_score": 0.05,
                    "growth_stage_score": 0.05
                },
                "logs": {
                    "weather_check":  ["❌ IMPOSSIBLE COMBINATION DETECTED — see Season Check."],
                    "location_check": "GPS not evaluated — claim blocked at season-damage validation.",
                    "season_check":   f"❌ REJECTED: {impossibility_reason}",
                    "metadata_check": "Not evaluated.",
                    "growth_stage_check": "Not evaluated."
                },
                "impossibility_flag":   True,
                "impossibility_reason": impossibility_reason,
                "verified_context":     False
            }

        # ── STEP 1: Weather Consistency Score (S_w) ────────────────────────────
        s_w = 0.95
        weather_logs = []

        if "blight" in disease:
            if humidity > 80:
                s_w = float(np.clip(humidity / 100.0, 0.0, 1.0))
                weather_logs.append("Humidity matches Leaf Blight development profile (>80%).")
            elif humidity < 50:
                s_w = 0.25
                weather_logs.append("Humidity is too low for Leaf Blight development (<50%).")
            else:
                s_w = 0.65
                weather_logs.append("Humidity is moderate, moderately supportive of blight.")
            if temp < 20:
                s_w = min(s_w, 0.45)
                weather_logs.append("Temperature is cooler than typical fungal growth ranges (<20°C).")

        elif "pest" in disease:
            if temp > 35:
                s_w = 0.58
                weather_logs.append("Extreme heat (>35°C) slightly suppresses pest activity.")
            elif temp < 12:
                s_w = 0.32
                weather_logs.append("Low temperature (<12°C) slows down pest lifecycle significantly.")
            else:
                s_w = 0.92
                weather_logs.append("Temperature is ideal for pest reproduction and crop feeding.")

        elif "rust" in disease:
            if humidity > 75 and 15 <= temp <= 25:
                s_w = 0.95
                weather_logs.append("Rust spores thrive in damp leaf surfaces and mild temperatures.")
            elif temp > 32:
                s_w = 0.38
                weather_logs.append("Temperatures above 32°C restrict rust growth.")
            else:
                s_w = 0.78
                weather_logs.append("Ambient weather is moderately consistent with rust infection.")

        elif "healthy" in disease:
            if temp > 45 or rainfall > 300:
                s_w = 0.68
                weather_logs.append("Weather is severe (very hot/high rain) but plant is reported healthy.")
            else:
                s_w = 0.96
                weather_logs.append("Normal weather conditions, consistent with crop health.")

        # ── STEP 2: Location Consistency Score (S_l) ───────────────────────────
        lat = metadata.get("latitude", 20.0)
        lon = metadata.get("longitude", 78.0)
        if 8.0 <= lat <= 38.0 and 68.0 <= lon <= 97.0:
            s_l = 0.97
            location_logs = "GPS coordinates valid within agricultural belts."
        else:
            s_l = 0.65
            location_logs = "GPS location lies outside primary regional reference range."

        # ── STEP 3: Seasonal Consistency Score (S_s) ───────────────────────────
        s_s = 0.90
        season_logs = ""
        if crop_type == "rice":
            if season == "monsoon":
                s_s = 0.96
                season_logs = "Monsoon is the primary sowing season for Rice crop."
            elif season == "summer":
                s_s = 0.75
                season_logs = "Rice requires high irrigation when grown in Summer."
            else:
                s_s = 0.60
                season_logs = "Rice cultivation is atypical in Winter season."
        elif crop_type == "cotton":
            if season in ["monsoon", "summer"]:
                s_s = 0.95
                season_logs = "Sowing/growth season aligns with cotton crop timeline."
            else:
                s_s = 0.48
                season_logs = "Winter harvested Cotton is rare in this region."
        elif crop_type == "wheat":
            if season == "winter":
                s_s = 0.97
                season_logs = "Wheat is a Rabi crop and matches Winter sowing season."
            else:
                s_s = 0.28
                season_logs = "Wheat cannot survive high heat and rain of Monsoon/Summer."
        else:
            s_s = 0.80
            season_logs = f"Season cross-check passed for {crop_type.capitalize()} (using general profile)."

        # ── STEP 4: Metadata Consistency Score (S_m) ───────────────────────────
        date = metadata.get("date", "")
        if date and len(date) >= 10:
            s_m = 0.96
            metadata_logs = "Metadata parameters fully completed and structured."
        else:
            s_m = 0.45
            metadata_logs = "Incomplete or corrupted timestamp/metadata record."

        # ── STEP 5: Growth Stage Consistency Score (S_gs) ──────────────────────
        growth_stage = metadata.get("growth_stage", "Vegetative").lower()
        valid_stages = ["seedling", "vegetative", "flowering", "fruiting", "harvest"]
        
        if growth_stage in valid_stages:
            s_gs = 0.96
            growth_stage_logs = f"Growth stage '{growth_stage.capitalize()}' is valid and compatible with seasonal cultivation cycle."
            
            # Cross-validation rules
            if "rust" in disease and growth_stage == "seedling":
                s_gs = 0.48
                growth_stage_logs = "Discrepancy: Leaf Rust is rare during early Seedling stage."
            elif "blight" in disease and growth_stage == "harvest":
                s_gs = 0.58
                growth_stage_logs = "Discrepancy: Fungal blight claims at Harvest stage are atypical."
        else:
            s_gs = 0.38
            growth_stage_logs = f"Unknown or invalid growth stage '{growth_stage}' reported."

        # ── STEP 6: Compute CCI (Task 3: 5 components weighted equally at 0.20) ──
        w_w = self.weights["w_weather"]
        w_l = self.weights["w_location"]
        w_s = self.weights["w_season"]
        w_m = self.weights["w_metadata"]
        w_gs = self.weights["w_growth_stage"]

        cci = (w_w * s_w) + (w_l * s_l) + (w_s * s_s) + (w_m * s_m) + (w_gs * s_gs)
        cci = float(np.clip(cci, 0.0, 1.0))

        return {
            "cci": cci,
            "scores": {
                "weather_score":  s_w,
                "location_score": s_l,
                "season_score":   s_s,
                "metadata_score": s_m,
                "growth_stage_score": s_gs
            },
            "logs": {
                "weather_check":  weather_logs,
                "location_check": location_logs,
                "season_check":   season_logs,
                "metadata_check": metadata_logs,
                "growth_stage_check": growth_stage_logs
            },
            "impossibility_flag":   False,
            "impossibility_reason": "",
            "verified_context":     cci >= 0.70
        }
