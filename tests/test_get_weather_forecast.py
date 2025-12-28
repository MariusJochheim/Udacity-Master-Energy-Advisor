from datetime import datetime

import tools


class FixedDatetime(datetime):
    """Frozen datetime for deterministic weather generation."""

    @classmethod
    def now(cls, tz=None):
        return cls(2024, 5, 1, 8, 0, 0, tzinfo=tz)


def test_get_weather_forecast_structure_and_determinism(monkeypatch):
    # Patch datetime in tools to keep forecasts stable
    monkeypatch.setattr(tools, "datetime", FixedDatetime)
    result = tools.get_weather_forecast.func("Berlin, DE", days=2)
    repeat_result = tools.get_weather_forecast.func("Berlin, DE", days=2)

    assert result == repeat_result, "Forecast should be deterministic for same inputs"
    assert result["location"] == "Berlin, DE"
    assert result["forecast_days"] == 2
    assert len(result["hourly"]) == 48

    first_hour = result["hourly"][0]
    assert first_hour["timestamp"].startswith("2024-05-01T08:00:00")
    assert first_hour["condition"] is not None
    assert first_hour["humidity"] is not None
    assert first_hour["wind_speed"] is not None

    current = result["current"]
    assert current["temperature_c"] == first_hour["temperature_c"]
    assert current["condition"] == first_hour["condition"]
    assert current["humidity"] == first_hour["humidity"]
    assert current["wind_speed"] == first_hour["wind_speed"]
