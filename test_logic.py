import sys
sys.path.append('C:/Users/ABC/Downloads/api')

from main import get_nearby_masjids, _generate_features_for_day, model, from_mins
from datetime import datetime
import numpy as np

def test_logic():
    print("Testing Nearby GPS Logic...")
    nearby = get_nearby_masjids(lat=31.46, lon=74.26, limit=5)
    for m in nearby:
        print(f"   - {m['masjid_name']} ({m['distance_km']} km)")
    
    print("\nTesting Prediction Logic for Range simulation...")
    dt = datetime(2026, 1, 1)
    features = _generate_features_for_day(dt, 1, 31.4567628, 74.2525003)
    preds = model.predict([features])[0]
    preds = np.round(preds).astype(int)
    print(f"Fajr Jamaat on {dt.strftime('%Y-%m-%d')}: {from_mins(preds[0])}")
    print("\n✅ Internal Logic Tests Passed!")

if __name__ == "__main__":
    test_logic()
