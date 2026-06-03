from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import math
from datetime import datetime, timedelta
from praytimes import PrayTimes
from hijridate import convert
from fastapi.middleware.cors import CORSMiddleware
import os
from typing import Optional

app = FastAPI(
    title="Masjid Jamaat Prediction API",
    description="API for Jamaat predictions, bulk calendar fetching, and GPS nearby masjids.",
    version="5.0.0"
)

# Allow CORS for frontend and backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- MASJID DATABASE ---
MASJIDS = [
    {'masjid_id': 1, 'masjid_name': 'Ibn-e-Adam Masjid', 'latitude': 31.4567628, 'longitude': 74.2525003},
    {'masjid_id': 2, 'masjid_name': 'Shan-e-Islam Masjid', 'latitude': 31.5088462, 'longitude': 74.3535366},
    {'masjid_id': 3, 'masjid_name': 'Muzammil Masjid', 'latitude': 31.49729, 'longitude': 74.35664},
    {'masjid_id': 4, 'masjid_name': 'Al-Habib Masjid', 'latitude': 31.535879, 'longitude': 74.3790217},
    {'masjid_id': 5, 'masjid_name': 'Khalid Masjid', 'latitude': 31.5000111, 'longitude': 74.3665154},
    {'masjid_id': 6, 'masjid_name': 'Ravi Hotel Masjid', 'latitude': 31.6145895, 'longitude': 74.2916412},
    {'masjid_id': 7, 'masjid_name': 'M7C Masjid', 'latitude': 31.3504382, 'longitude': 74.2461831},
    {'masjid_id': 8, 'masjid_name': 'Jamia Masjid Rehman', 'latitude': 31.5808333, 'longitude': 74.4665918},
    {'masjid_id': 9, 'masjid_name': 'Grand Jamia Mosque', 'latitude': 31.4829403, 'longitude': 74.3343893},
    {'masjid_id': 10, 'masjid_name': ' Jamia Masjid Dha Phase 4', 'latitude': 31.464632, 'longitude': 74.3839531},
    {'masjid_id': 11, 'masjid_name': ' Jamia Masjid Mustafa', 'latitude': 31.495198, 'longitude': 74.3469482},
    {'masjid_id': 12, 'masjid_name': ' Madina Masjid', 'latitude': 31.4855552, 'longitude': 74.3470166},
    {'masjid_id': 13, 'masjid_name': ' Jamia Masjid Ghausia Rizwia', 'latitude': 31.5440233, 'longitude': 74.3505727},
    {'masjid_id': 14, 'masjid_name': ' Jamia Masjid Muhammadiya Rizwia', 'latitude': 31.5028744, 'longitude': 74.3471544},
    {'masjid_id': 15, 'masjid_name': ' Jamia Masjid Aziz Rashid', 'latitude': 31.5480595, 'longitude': 74.2844224},
    {'masjid_id': 16, 'masjid_name': ' Jamia Masjid Ahsan Raheem', 'latitude': 31.4999791, 'longitude': 74.3523614}
]

# --- LOAD MODEL ---
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'masjid_model.pkl')
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# --- INITIALIZE PRAYTIMES ---
PT = PrayTimes('Karachi')
PT.adjust({'asr': 'Hanafi'})

# --- SCHEMAS ---
class PredictionRequest(BaseModel):
    masjid_name: str
    date: str  # Format: YYYY-MM-DD

class RangePredictionRequest(BaseModel):
    masjid_name: str
    start_date: str  # Format: YYYY-MM-DD
    end_date: str    # Format: YYYY-MM-DD

class PredictionResponse(BaseModel):
    masjid_name: str
    date: str
    fajr_jamaat: str
    zuhr_jamaat: str
    asr_jamaat: str
    maghrib_jamaat: str
    isha_jamaat: str

# --- HELPER FUNCTIONS ---
def to_mins(hhmm):
    h, m = map(int, hhmm.split(':'))
    return h * 60 + m

def from_mins(mins):
    mins = int(mins) % 1440
    return f"{mins // 60:02d}:{mins % 60:02d}"

def is_ramadan(y, m, d):
    return convert.Gregorian(y, m, d).to_hijri().month == 9

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculates distance between two GPS points in kilometers"""
    R = 6371.0  # Earth radius in kilometers
    dLat = math.radians(lat2 - lat1)
    dLon = math.radians(lon2 - lon1)
    a = math.sin(dLat / 2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dLon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def _lookup_masjid(name: str):
    search_name = name.strip().lower()
    for m in MASJIDS:
        if m['masjid_name'].strip().lower() == search_name:
            return m
    return None

def _generate_features_for_day(dt, masjid_id, lat, lon):
    y, m, d = dt.year, dt.month, dt.day
    waqt = PT.getTimes((y, m, d), (lat, lon), 5)
    return [
        masjid_id,
        int(is_ramadan(y, m, d)),
        int(dt.weekday() == 4),
        dt.timetuple().tm_yday,
        to_mins(waqt['fajr']),
        to_mins(waqt['dhuhr']),
        to_mins(waqt['asr']),
        to_mins(waqt['maghrib']),
        to_mins(waqt['isha'])
    ]

# --- ENDPOINTS ---
@app.get("/")
def read_root():
    return {"message": "Masjid Jamaat API. See /docs for endpoints."}

@app.get("/masjids")
def get_all_masjids():
    """Returns all available masjids"""
    return MASJIDS

@app.get("/masjids/nearby")
def get_nearby_masjids(lat: float = Query(..., description="User's latitude"), 
                       lon: float = Query(..., description="User's longitude"), 
                       limit: int = Query(5, description="Number of closest masjids to return")):
    """Returns masjids sorted by distance to the user's GPS location"""
    distances = []
    for m in MASJIDS:
        dist = haversine_distance(lat, lon, m['latitude'], m['longitude'])
        m_copy = m.copy()
        m_copy['distance_km'] = round(dist, 2)
        distances.append(m_copy)
    
    distances.sort(key=lambda x: x['distance_km'])
    return distances[:limit]


@app.get("/masjids/{name}")
def get_masjid_by_name(name: str):
    """Returns details of a specific masjid by its name"""
    masjid = _lookup_masjid(name)
    if not masjid:
        raise HTTPException(status_code=404, detail=f"Masjid '{name}' not found.")
    return masjid


@app.post("/predict/range", response_model=list[PredictionResponse])
def predict_range(req: RangePredictionRequest):
    """Predicts Jamaat timings for an entire date range (e.g. for a full year calendar)"""
    if model is None:
        raise HTTPException(status_code=500, detail="Model is not loaded.")
    
    # Lookup Masjid
    masjid_details = _lookup_masjid(req.masjid_name)
    if not masjid_details:
        raise HTTPException(status_code=404, detail=f"Masjid '{req.masjid_name}' not found.")
    
    try:
        start_dt = datetime.strptime(req.start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(req.end_date, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD.")
    
    if end_dt < start_dt:
        raise HTTPException(status_code=400, detail="end_date must be after start_date")
    
    delta_days = (end_dt - start_dt).days + 1
    if delta_days > 400: # Protect against massive payloads (like 10 years at once)
        raise HTTPException(status_code=400, detail="Maximum range is 400 days")
    
    dates = []
    features_list = []
    
    # Bulk generate features
    current = start_dt
    for _ in range(delta_days):
        dates.append(current.strftime("%Y-%m-%d"))
        features_list.append(_generate_features_for_day(current, masjid_details['masjid_id'], masjid_details['latitude'], masjid_details['longitude']))
        current += timedelta(days=1)
    
    # Bulk predict
    predictions = model.predict(features_list)
    predictions = np.round(predictions).astype(int)
    
    response = []
    for i in range(len(dates)):
        response.append(PredictionResponse(
            masjid_name=masjid_details['masjid_name'],
            date=dates[i],
            fajr_jamaat=from_mins(predictions[i][0]),
            zuhr_jamaat=from_mins(predictions[i][1]),
            asr_jamaat=from_mins(predictions[i][2]),
            maghrib_jamaat=from_mins(predictions[i][3]),
            isha_jamaat=from_mins(predictions[i][4])
        ))
    
    return response

@app.post("/predict", response_model=PredictionResponse)
def predict_single_day(req: PredictionRequest):
    """Predicts Jamaat timings for a single day"""
    if model is None:
        raise HTTPException(status_code=500, detail="Model is not loaded.")
    
    # Lookup Masjid
    masjid_details = _lookup_masjid(req.masjid_name)
    if not masjid_details:
        raise HTTPException(status_code=404, detail=f"Masjid '{req.masjid_name}' not found.")
    
    try:
        dt = datetime.strptime(req.date, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD.")
    
    features = _generate_features_for_day(dt, masjid_details['masjid_id'], masjid_details['latitude'], masjid_details['longitude'])
    
    predictions = model.predict([features])[0]
    predictions = np.round(predictions).astype(int)
    
    return PredictionResponse(
        masjid_name=masjid_details['masjid_name'],
        date=req.date,
        fajr_jamaat=from_mins(predictions[0]),
        zuhr_jamaat=from_mins(predictions[1]),
        asr_jamaat=from_mins(predictions[2]),
        maghrib_jamaat=from_mins(predictions[3]),
        isha_jamaat=from_mins(predictions[4])
    )
