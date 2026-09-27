import os
import pandas as pd
from typing import Dict, Any, List, Optional


class WeatherDataLoader:
    """Handles loading, cleaning, and querying the local weather dataset."""

    def __init__(self, csv_path: str = "data/06_agent_weather_tool_data.csv"):
        self.csv_path = csv_path
        self._df: Optional[pd.DataFrame] = None
        self._load_and_validate_data()

    def _load_and_validate_data(self) -> None:
        """Loads and normalizes CSV data with schema validation."""
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(
                f"Dataset not found at path: '{self.csv_path}'. "
                "Ensure the file exists in the data/ directory."
            )

        try:
            df = pd.read_csv(self.csv_path)

            # Ensure expected columns exist
            required_columns = {
                "city",
                "date",
                "temperature_c",
                "rainfall_mm",
                "humidity_pct",
            }
            if not required_columns.issubset(set(df.columns)):
                missing = required_columns - set(df.columns)
                raise ValueError(f"Missing required columns in CSV: {missing}")

            # Normalize data types and clean strings
            df["city"] = df["city"].astype(str).str.strip().str.title()
            df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
            df["temperature_c"] = pd.to_numeric(
                df["temperature_c"], errors="coerce"
            )
            df["rainfall_mm"] = pd.to_numeric(
                df["rainfall_mm"], errors="coerce"
            )
            df["humidity_pct"] = pd.to_numeric(
                df["humidity_pct"], errors="coerce"
            )

            # Drop missing essential records
            self._df = df.dropna(
                subset=["city", "date", "temperature_c"]
            ).copy()

        except Exception as e:
            raise RuntimeError(f"Error initializing WeatherDataLoader: {str(e)}")

    def get_city_weather(
        self, city: str, date: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Queries metrics for a specific city, optionally filtered by date (YYYY-MM-DD)."""
        if self._df is None or self._df.empty:
            return []

        formatted_city = city.strip().title()
        filtered_df = self._df[self._df["city"] == formatted_city]

        if date:
            formatted_date = date.strip()
            filtered_df = filtered_df[filtered_df["date"] == formatted_date]

        return filtered_df.to_dict(orient="records")

    def get_city_stats(self, city: str) -> Dict[str, Any]:
        """Calculates aggregations for a target city."""
        if self._df is None or self._df.empty:
            return {"error": "Dataset unavailable"}

        formatted_city = city.strip().title()
        city_df = self._df[self._df["city"] == formatted_city]

        if city_df.empty:
            return {
                "found": False,
                "message": f"No records found for city '{formatted_city}' in local dataset.",
            }

        return {
            "found": True,
            "city": formatted_city,
            "record_count": int(len(city_df)),
            "average_temperature": round(
                float(city_df["temperature_c"].mean()), 2
            ),
            "maximum_temperature": round(
                float(city_df["temperature_c"].max()), 2
            ),
            "minimum_temperature": round(
                float(city_df["temperature_c"].min()), 2
            ),
            "total_rainfall": round(float(city_df["rainfall_mm"].sum()), 2),
            "average_humidity": round(
                float(city_df["humidity_pct"].mean()), 2
            ),
        }

    def list_available_cities(self) -> List[str]:
        """Returns unique list of cities in dataset for strict grounding."""
        if self._df is None:
            return []
        return sorted(self._df["city"].unique().tolist())