import random
from datetime import datetime

class WeatherSimulator:
    def __init__(self):
        # Base climatology by season
        self.climatology = {
            "Monsoon": {
                "temp_range": (24, 32),
                "rainfall_range": (150, 400), # mm monthly/recent
                "humidity_range": (80, 95),
                "wind_range": (10, 25)
            },
            "Winter": {
                "temp_range": (10, 20),
                "rainfall_range": (5, 40),
                "humidity_range": (40, 60),
                "wind_range": (5, 15)
            },
            "Summer": {
                "temp_range": (30, 43),
                "rainfall_range": (0, 20),
                "humidity_range": (20, 45),
                "wind_range": (8, 20)
            }
        }

    def get_weather(self, gps_coords, date_str, season, crop_type, suspected_damage=None):
        """
        Generate realistic weather parameters.
        Anomalies are injected based on suspected_damage to showcase Context Verification.
        """
        lat, lon = gps_coords.get("latitude", 20.0), gps_coords.get("longitude", 78.0)
        
        # Get base values from season
        base = self.climatology.get(season, self.climatology["Summer"])
        
        temp = round(random.uniform(*base["temp_range"]), 1)
        rainfall = round(random.uniform(*base["rainfall_range"]), 1)
        humidity = round(random.uniform(*base["humidity_range"]), 1)
        wind_speed = round(random.uniform(*base["wind_range"]), 1)
        
        # Override to inject consistent or inconsistent anomalies for demo purposes
        if suspected_damage:
            dmg_lower = suspected_damage.lower()
            if "flood" in dmg_lower or "waterlogging" in dmg_lower:
                rainfall = round(random.uniform(250, 450), 1)
                humidity = round(random.uniform(90, 98), 1)
                temp = round(random.uniform(22, 28), 1)
            elif "drought" in dmg_lower or "heat" in dmg_lower or "wilt" in dmg_lower:
                rainfall = 0.0
                temp = round(random.uniform(38, 46), 1)
                humidity = round(random.uniform(10, 25), 1)
            elif "blight" in dmg_lower or "fungal" in dmg_lower:
                # Fungal diseases thrive in high humidity and warm temp
                temp = round(random.uniform(26, 33), 1)
                humidity = round(random.uniform(85, 95), 1)
                rainfall = round(random.uniform(40, 120), 1)
            elif "healthy" in dmg_lower:
                # Ideal weather for growing
                if season == "Monsoon":
                    rainfall = round(random.uniform(100, 200), 1)
                    temp = round(random.uniform(26, 30), 1)
                    humidity = round(random.uniform(70, 85), 1)
                elif season == "Winter":
                    rainfall = round(random.uniform(10, 30), 1)
                    temp = round(random.uniform(15, 22), 1)
                    humidity = round(random.uniform(50, 65), 1)
                else:
                    rainfall = round(random.uniform(5, 15), 1)
                    temp = round(random.uniform(28, 33), 1)
                    humidity = round(random.uniform(40, 55), 1)

        return {
            "temperature": temp,
            "rainfall": rainfall,
            "humidity": humidity,
            "wind_speed": wind_speed,
            "location_ref": f"District/State simulated for ({lat:.4f}, {lon:.4f})",
            "station_id": f"AGRI_STN_{int(lat*100)}_{int(lon*100)}"
        }
