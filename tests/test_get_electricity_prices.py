import tools


def test_get_electricity_prices_structure_and_rules():
    result = tools.get_electricity_prices.func("2024-05-01")
    repeat_result = tools.get_electricity_prices.func("2024-05-01")

    assert result == repeat_result, "Pricing should be deterministic for same inputs"
    assert result["date"] == "2024-05-01"
    assert result["pricing_type"] == "time_of_use"
    assert result["currency"] == "USD"
    assert result["unit"] == "per_kWh"
    assert len(result["hourly_rates"]) == 24

    for entry in result["hourly_rates"]:
        assert 0 <= entry["hour"] <= 23
        assert entry["rate"] > 0
        if 6 <= entry["hour"] < 22:
            assert entry["period"] == "peak"
            assert entry["demand_charge"] > 0
        else:
            assert entry["period"] == "off_peak"
            assert entry["demand_charge"] == 0
