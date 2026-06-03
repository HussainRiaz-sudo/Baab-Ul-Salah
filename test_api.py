from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_nearby():
    # Example coordinates in Lahore (should be close to Ibn-e-Adam)
    response = client.get("/masjids/nearby?lat=31.46&lon=74.26")
    assert response.status_code == 200
    data = response.json()
    print(" Nearby Masjids Response:")
    for m in data:
        print(f"   - {m['masjid_name']} ({m['distance_km']} km)")
    assert len(data) == 5 # Default limit is 5
    assert data[0]['masjid_name'] == 'Ibn-e-Adam Masjid' # It is the closest to these coords

def test_predict_range():
    response = client.post(
        "/predict/range",
        json={
            "masjid_id": 1,
            "latitude": 31.4567628,
            "longitude": 74.2525003,
            "start_date": "2026-01-01",
            "end_date": "2026-01-31"
        }
    )
    assert response.status_code == 200
    data = response.json()
    print(f"\n Calendar Range Response: {len(data)} days received.")
    print(f"   First day: {data[0]['date']} -> Fajr: {data[0]['fajr_jamaat']}")
    print(f"   Last day: {data[-1]['date']} -> Fajr: {data[-1]['fajr_jamaat']}")
    assert len(data) == 31

if __name__ == "__main__":
    print("Running Tests...\n")
    test_nearby()
    test_predict_range()
    print("\n✅ All Local API Tests Passed!")
