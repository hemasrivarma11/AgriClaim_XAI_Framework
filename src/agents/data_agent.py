import os
import sys
from datetime import datetime

# Adjust path to import packages from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from preprocessing.image_processor import ImageProcessor
from preprocessing.weather_simulator import WeatherSimulator

class DataIntegrationAgent:
    def __init__(self):
        self.image_processor = ImageProcessor()
        self.weather_simulator = WeatherSimulator()

    def process(self, image_path, gps_coords, metadata, suspected_damage=None):
        """
        Processes and synchronizes crop image, weather data, and metadata.
        """
        # 1. Image preprocessing
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Crop image not found at: {image_path}")
            
        img_data = self.image_processor.preprocess_pipeline(image_path)

        # 2. Metadata validation
        crop_type = metadata.get("crop_type", "Rice").capitalize()
        season = metadata.get("season", "Monsoon").capitalize()
        date_str = metadata.get("date", datetime.now().strftime("%Y-%m-%d"))

        # Default values if missing
        validated_metadata = {
            "crop_type": crop_type,
            "season": season,
            "date": date_str,
            "latitude": float(gps_coords.get("latitude", 20.0)),
            "longitude": float(gps_coords.get("longitude", 78.0)),
            "timestamp": metadata.get("timestamp", datetime.now().isoformat())
        }

        # 3. Weather retrieval
        weather_data = self.weather_simulator.get_weather(
            gps_coords=gps_coords,
            date_str=date_str,
            season=season,
            crop_type=crop_type,
            suspected_damage=suspected_damage
        )

        return {
            "processed_image": img_data["normalized"],
            "raw_image": img_data["raw"],
            "resized_image": img_data["resized"],
            "image_patches": img_data["patches"],
            "weather": weather_data,
            "metadata": validated_metadata
        }
