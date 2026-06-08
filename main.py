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

# Load .env variables manually if .env file exists in the directory
env_path = os.path.join(os.path.dirname(__file__), '.env')
if os.path.exists(env_path):
    with open(env_path, 'r', encoding='utf-8') as f:
        for line in f:
            if '=' in line and not line.strip().startswith('#'):
                parts = line.strip().split('=', 1)
                os.environ[parts[0].strip()] = parts[1].strip()

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
    {'masjid_id': 16, 'masjid_name': ' Jamia Masjid Ahsan Raheem', 'latitude': 31.4999791, 'longitude': 74.3523614},
    {'masjid_id': 17, 'masjid_name': 'Jamia Masjid Bilal (DHA Phase 3)', 'latitude': 31.4725, 'longitude': 74.3735},
    {'masjid_id': 18, 'masjid_name': 'Jamia Masjid Rehmat (Model Town)', 'latitude': 31.4795, 'longitude': 74.3225},
    {'masjid_id': 19, 'masjid_name': 'Jamia Masjid Al-Shafi (Johar Town)', 'latitude': 31.4642, 'longitude': 74.2795},
    {'masjid_id': 20, 'masjid_name': 'Masjid Al-Fatah (Gulberg)', 'latitude': 31.5115, 'longitude': 74.3498},
    {'masjid_id': 21, 'masjid_name': 'Jamia Masjid Aqsa (Faisal Town)', 'latitude': 31.4792, 'longitude': 74.3072},
    {'masjid_id': 22, 'masjid_name': 'Masjid Al-Khair (Wapda Town)', 'latitude': 31.4285, 'longitude': 74.2612},
    {'masjid_id': 23, 'masjid_name': 'Jamia Masjid Siddique (Samanabad)', 'latitude': 31.5342, 'longitude': 74.3148},
    {'masjid_id': 24, 'masjid_name': 'Masjid Al-Madina (Garden Town)', 'latitude': 31.4995, 'longitude': 74.3218},
    {'masjid_id': 25, 'masjid_name': 'Jamia Masjid Anwar-e-Madina (Green Town)', 'latitude': 31.4395, 'longitude': 74.3082},
    {'masjid_id': 26, 'masjid_name': 'Masjid Al-Quds (Gulshan-e-Ravi)', 'latitude': 31.5462, 'longitude': 74.2952},
    {'masjid_id': 27, 'masjid_name': 'Jamia Masjid Farooq-e-Azam (Township)', 'latitude': 31.4525, 'longitude': 74.3052},
    {'masjid_id': 28, 'masjid_name': 'Jamia Masjid Hanfia (Sabzazar)', 'latitude': 31.5252, 'longitude': 74.2852},
    {'masjid_id': 29, 'masjid_name': 'Masjid Ibrahim (Bahria Town)', 'latitude': 31.3752, 'longitude': 74.1912},
    {'masjid_id': 30, 'masjid_name': 'Jamia Masjid Khalid Bin Walid (Cavalry Ground)', 'latitude': 31.5052, 'longitude': 74.3725},
    {'masjid_id': 31, 'masjid_name': 'Jamia Masjid Usman (Garhi Shahu)', 'latitude': 31.5542, 'longitude': 74.3512},
    {'masjid_id': 32, 'masjid_name': 'Masjid Bilal (Allama Iqbal Town)', 'latitude': 31.5152, 'longitude': 74.2925},
    {'masjid_id': 33, 'masjid_name': 'Jamia Masjid Imam Abu Hanifa (Tajpura)', 'latitude': 31.5695, 'longitude': 74.4095},
    {'masjid_id': 34, 'masjid_name': 'Jamia Masjid Riaz-ul-Jannah (Mughalpura)', 'latitude': 31.5625, 'longitude': 74.3825},
    {'masjid_id': 35, 'masjid_name': 'Jamia Masjid Taqwa (Nishtar Colony)', 'latitude': 31.4125, 'longitude': 74.3652},
    {'masjid_id': 36, 'masjid_name': 'Jamia Masjid Noor (Shalamar)', 'latitude': 31.5782, 'longitude': 74.3792},
    {'masjid_id': 37, 'masjid_name': 'Jamia Masjid Hazrat Bilal (Chung)', 'latitude': 31.4152, 'longitude': 74.2025},
    {'masjid_id': 38, 'masjid_name': 'Jamia Masjid Quba (Cantt)', 'latitude': 31.5195, 'longitude': 74.4025},
    {'masjid_id': 39, 'masjid_name': 'Jamia Masjid Ghausia (Baghbanpura)', 'latitude': 31.5752, 'longitude': 74.3692},
    {'masjid_id': 40, 'masjid_name': 'Jamia Masjid Rehmania (Ichhra)', 'latitude': 31.5262, 'longitude': 74.3282},
    {'masjid_id': 41, 'masjid_name': 'Jamia Masjid Fatima (Johar Town Phase 2)', 'latitude': 31.4552, 'longitude': 74.2662},
    {'masjid_id': 42, 'masjid_name': 'Jamia Masjid Al-Haram (DHA Phase 6)', 'latitude': 31.4695, 'longitude': 74.4425},
    {'masjid_id': 43, 'masjid_name': 'Jamia Masjid Madni (Model Town Extension)', 'latitude': 31.4855, 'longitude': 74.3332},
    {'masjid_id': 44, 'masjid_name': 'Jamia Masjid Khizra (Samanabad)', 'latitude': 31.5292, 'longitude': 74.3052},
    {'masjid_id': 45, 'masjid_name': 'Jamia Masjid Toheed (Valencia Town)', 'latitude': 31.3995, 'longitude': 74.2502},
    {'masjid_id': 46, 'masjid_name': 'Jamia Masjid Aqsa (Sanda)', 'latitude': 31.5642, 'longitude': 74.2892},
    {'masjid_id': 47, 'masjid_name': 'Jamia Masjid Mustafa (Shadman)', 'latitude': 31.5352, 'longitude': 74.3322},
    {'masjid_id': 48, 'masjid_name': 'Jamia Masjid Abu Bakar (DHA Phase 5)', 'latitude': 31.4625, 'longitude': 74.4122},
    {'masjid_id': 49, 'masjid_name': 'Badshahi Mosque (Walled City)', 'latitude': 31.5880, 'longitude': 74.3108},
    {'masjid_id': 50, 'masjid_name': 'Wazir Khan Mosque (Walled City)', 'latitude': 31.5831, 'longitude': 74.3236}
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
    hours = mins // 60
    minutes = mins % 60
    period = "AM" if hours < 12 else "PM"
    
    display_hours = hours % 12
    if display_hours == 0:
        display_hours = 12
        
    return f"{display_hours:02d}:{minutes:02d} {period}"

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

# --- CHATBOT DATA AND ENDPOINT ---
class ChatRequest(BaseModel):
    message: str

CHAT_QA = [
    {
        "keywords": ["project", "goal", "about", "purpose", "what is this", "work"],
        "answer": "🕌 **Masjid Jamaat Timing Predictor** is a machine learning backend designed to predict the daily congregational (Jamaat) prayer times for masjids in Lahore.\n\nIt helps users find the exact start times of congregation prayers which normally change dynamically based on season, Ramadan, and local mosque rules."
    },
    {
        "keywords": ["model", "algorithm", "random forest", "regressor", "ml", "method"],
        "answer": "🤖 We use a **Random Forest Regressor** (100 trees, random state 42) from Scikit-Learn. It is trained as a **Multi-Output Regressor** to predict all 5 daily prayers (Fajr, Zuhr, Asr, Maghrib, Isha) in a single inference pass."
    },
    {
        "keywords": ["dataset", "rows", "size", "data", "how many masjids", "total", "samples"],
        "answer": "📊 The dataset covers **50 masjids** across Lahore (16 original starting masjids + 34 new verified locations).\n\nFor training, we generated a full year of daily data (2025), yielding **18,250 rows** (365 days × 50 masjids) with precise waqt boundaries and offsets."
    },
    {
        "keywords": ["accuracy", "mae", "error", "performance", "metrics", "minute", "minutes"],
        "answer": "📈 **Model Metrics:**\n* **Training MAE:** **0.198 minutes** (~12 seconds average deviation).\n* **Testing MAE:** **0.58 minutes** (~35 seconds average deviation).\n* **Interval Match Rate:** **99.4%** accuracy in predicting the correct 15-minute rounded boundaries."
    },
    {
        "keywords": ["feature", "input", "dimension", "predictor", "columns", "features"],
        "answer": "🔑 The model uses **9 features** to predict Jamaat timings:\n1. `masjid_id` (encodes local mosque offset rules)\n2. `is_ramadan` (binary flag for Ramadan calendar shift)\n3. `is_friday` (binary flag for Friday Jumu'ah shift)\n4. `day_of_year` (1-365, captures seasonal sunlight cycles)\n5. `fajr_waqt_mins` (Fajr start time from midnight)\n6. `dhuhr_waqt_mins` (Dhuhr start time)\n7. `asr_waqt_mins` (Asr Hanafi start time)\n8. `maghrib_waqt_mins` (Maghrib start time)\n9. `isha_waqt_mins` (Isha start time)"
    },
    {
        "keywords": ["target", "output", "prediction", "predicts", "prayers", "targets"],
        "answer": "🎯 The model outputs **5 continuous targets** (multi-output regression) representing the Jamaat timings in minutes from midnight:\n1. `fajr_jamaat_mins`\n2. `zuhr_jamaat_mins` (or Jumu'ah on Fridays)\n3. `asr_jamaat_mins`\n4. `maghrib_jamaat_mins`\n5. `isha_jamaat_mins`"
    },
    {
        "keywords": ["offset", "rule", "calculation", "formula", "jamaat time", "how is jamaat", "offsets", "rules"],
        "answer": "⚙️ **Jamaat Calculation Offset Rules:**\n* **Fajr**: Waqt +60 mins (normal) or +15 mins (Ramadan).\n* **Zuhr / Jumu'ah**: Fixed at `13:30` (default) or customized per masjid (e.g. `13:45`).\n* **Asr**: Waqt +35 mins, rounded to the nearest 15 minutes.\n* **Maghrib**: Waqt +10 mins (normal) or +15 mins (Ramadan).\n* **Isha**: Waqt +30 mins, rounded to the nearest 15 minutes."
    },
    {
        "keywords": ["nearby", "gps", "distance", "haversine", "closest", "km", "location", "locations"],
        "answer": "🌐 **Nearby Masjid Search:**\nWe use the **Haversine Formula** to compute the great-circle distance between the user's GPS coordinates and the masjids:\n\n`d = 2 * R * arcsin(sqrt(sin²(Δlat/2) + cos(lat1)*cos(lat2)*sin²(Δlon/2)))`\n\nwhere `R = 6371` km. The `/masjids/nearby` endpoint returns the closest masjids sorted by distance."
    },
    {
        "keywords": ["endpoints", "api", "routes", "url", "requests", "paths"],
        "answer": "🔌 **Available API Endpoints:**\n* `GET /` - Health check\n* `GET /masjids` - Retrieve all 50 masjids\n* `GET /masjids/nearby` - Get nearby masjids using GPS coordinates\n* `GET /masjids/{name}` - Search masjid by name\n* `POST /predict` - Predict timings for a single day\n* `POST /predict/range` - Bulk predict a date range (up to 400 days)\n* `POST /chat` - Interactive chatbot endpoint"
    },
    {
        "keywords": ["why random forest", "decision tree", "neural network", "why rf"],
        "answer": "🌳 **Why Random Forest?**\nDecision tree ensembles are uniquely suited for this task because the target offsets contain **discontinuous step-like roundings** (e.g., rounding to the nearest 15-minute interval). Neural networks tend to smooth these thresholds out, leading to larger rounding errors, while Random Forest learns the exact step boundaries perfectly."
    },
    {
        "keywords": ["grand jamia", "id 9", "coordinates of 9", "jamia mosque"],
        "answer": "🕌 **Grand Jamia Mosque (ID 9)** is configured at its original starting coordinates: **Latitude 31.4829403, Longitude 74.3343893** to preserve dataset consistency."
    },
    {
        "keywords": ["lahore", "masjids in lahore", "geographic", "coordinates"],
        "answer": "📍 All 50 masjids in our database are strictly verified Lahore locations. The dataset includes 16 starting masjids and 34 newly added locations spanning major neighborhoods like Gulberg, Johar Town, DHA, Model Town, Township, Samanabad, and the historic Walled City (e.g. Badshahi and Wazir Khan Mosques)."
    }
]

@app.post("/chat")
def chat_bot(req: ChatRequest):
    """Chatbot endpoint to answer questions about the Masjid Predictor system"""
    import re
    cleaned = re.sub(r'[^\w\s]', '', req.message.lower())
    
    best_match = None
    best_score = 0
    
    for qa in CHAT_QA:
        # Calculate score: count how many keywords from this QA are in the cleaned message
        score = sum(1 for kw in qa["keywords"] if kw in cleaned)
        if score > best_score:
            best_score = score
            best_match = qa
            
    if best_score > 0:
        return {"response": best_match["answer"]}
        
    return {
        "response": "🤖 I'm specialized in the **Masjid Jamaat Timing Predictor**! You can ask me about:\n\n"
                    "* **The ML Model**: Random Forest architecture, parameters, and features.\n"
                    "* **Dataset & Accuracy**: MAE scores, training data size, and Lahore masjids.\n"
                    "* **Offset Rules**: How prayer offsets (normal vs. Ramadan) are calculated.\n"
                    "* **API Endpoints**: Details on endpoints like `/predict` and `/masjids/nearby`.\n\n"
                    "Try asking: *'What algorithm is used?'* or *'What is the model accuracy?'*"
    }

# --- ADVANCED GEMINI PRO CHATBOT INTEGRATION ---
import google.generativeai as genai

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

class ConversationTurn(BaseModel):
    role: str  # "user" or "model"
    text: str

class GeminiChatRequest(BaseModel):
    message: str
    history: Optional[list[ConversationTurn]] = None

def _get_masjid_timings_context_for_today(query_text: str) -> str:
    """Uses the loaded Random Forest model to predict today's timings for context injection"""
    query_lower = query_text.lower()
    timing_words = ["time", "timing", "namaz", "prayer", "jamaat", "when", "fajr", "zuhr", "dhuhr", "asr", "maghrib", "isha", "jumuah"]
    mentions_timing = any(w in query_lower for w in timing_words)
    
    # Check if a specific masjid is mentioned
    mentioned_masjid = None
    for m in MASJIDS:
        if m['masjid_name'].lower() in query_lower:
            mentioned_masjid = m
            break
            
    if not mentions_timing and not mentioned_masjid:
        return ""
        
    if model is None:
        return "Note: The machine learning model is currently not loaded on the backend."
        
    dt = datetime.now()
    context_lines = [
        f"Today's Date: {dt.strftime('%Y-%m-%d')} ({dt.strftime('%A')})",
        "Islamic Month Context: Ramadan is active if is_ramadan is 1."
    ]
    
    # Predict for the mentioned masjid, or default to first 5
    target_masjids = [mentioned_masjid] if mentioned_masjid else MASJIDS[:5]
    
    context_lines.append("\nCurrent predicted congregation (Jamaat) timings for relevant Lahore masjids:")
    for m in target_masjids:
        features = _generate_features_for_day(dt, m['masjid_id'], m['latitude'], m['longitude'])
        preds = model.predict([features])[0]
        preds = np.round(preds).astype(int)
        
        context_lines.append(
            f"- **{m['masjid_name']}** (ID: {m['masjid_id']}): "
            f"Fajr: {from_mins(preds[0])}, "
            f"Zuhr: {from_mins(preds[1])}, "
            f"Asr: {from_mins(preds[2])}, "
            f"Maghrib: {from_mins(preds[3])}, "
            f"Isha: {from_mins(preds[4])}"
        )
        
    return "\n".join(context_lines)

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
XAI_API_KEY = os.environ.get("XAI_API_KEY")

@app.post("/chat/gemini")
def chat_gemini(req: GeminiChatRequest):
    """Answers user query using xAI (Grok), Groq (Llama 3), or Gemini Pro, enriched with model predicted timings (RAG)"""
    # 1. Fetch predicted timing context if needed
    timing_context = _get_masjid_timings_context_for_today(req.message)
    
    # 2. Build Islamic and app-aware System Prompt
    system_prompt = (
        "You are e.Baab-ul-Salah Assistant, a friendly and polite Islamic guide for users of the e.Baab-ul-Salah application.\n"
        "Your goal is to answer queries about Islamic practices, daily prayers, fiqh basics, sunnah, and masjid congregation (Jamaat) timings in Lahore.\n\n"
        "Guidelines:\n"
        "- Do NOT discuss technical developer jargon (such as Random Forest, training metrics, MAE, dataset rows, or REST endpoints) to the user. Keep it simple and focused on user-facing features.\n"
        "- Explain App Features naturally:\n"
        "  * **Nearby Masjids**: The app uses your GPS location to calculate real-world distances and show you the closest masjids sorted by distance.\n"
        "  * **Smart Timings**: The app uses an advanced AI engine in the background to automatically predict and display the correct congregation (Jamaat) timings for 50 Lahore masjids, adapting to seasons and Ramadan changes.\n"
        "  * **Prayer Reminders**: Users can enable customized notifications and alerts for Jamaat timings.\n"
        "- Utilize the context below for any timings or masjid queries. If the user asks about a masjid not in the context, guide them to search for it in the app's directory.\n"
        "- Answer politely, clearly, and respectfully. If unsure about a fiqh or timing question, state it humbly.\n\n"
    )
    if timing_context:
        system_prompt += f"Active Database Context:\n{timing_context}\n\n"

    # --- PROVIDER 0: xAI GROK API ---
    if XAI_API_KEY:
        try:
            import httpx
            # Format history for OpenAI chat format
            messages = [{"role": "system", "content": system_prompt}]
            if req.history:
                for turn in req.history:
                    role = "user" if turn.role == "user" else "assistant"
                    messages.append({"role": role, "content": turn.text})
            messages.append({"role": "user", "content": req.message})
            
            headers = {
                "Authorization": f"Bearer {XAI_API_KEY}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": "grok-beta",  # Grok-2 Developer Beta Model
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 1024
            }
            
            with httpx.Client() as client:
                response = client.post(
                    "https://api.x.ai/v1/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=20.0
                )
                
            if response.status_code == 200:
                res_data = response.json()
                reply = res_data["choices"][0]["message"]["content"]
                return {"response": reply}
            else:
                print(f"xAI API returned error status: {response.status_code}, detail: {response.text}")
        except Exception as e:
            print(f"Failed to communicate with xAI: {str(e)}")

    # --- PROVIDER 1: GROQ CLOUD API (100% Free, Blazing Fast, No Billing Needed) ---
    if GROQ_API_KEY:
        try:
            import httpx
            # Format history for OpenAI chat format
            messages = [{"role": "system", "content": system_prompt}]
            if req.history:
                for turn in req.history:
                    role = "user" if turn.role == "user" else "assistant"
                    messages.append({"role": role, "content": turn.text})
            messages.append({"role": "user", "content": req.message})
            
            headers = {
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": "llama-3.1-8b-instant",  # Free tier default high-performance model
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 1024
            }
            
            with httpx.Client() as client:
                response = client.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=15.0
                )
                
            if response.status_code == 200:
                res_data = response.json()
                reply = res_data["choices"][0]["message"]["content"]
                return {"response": reply}
            else:
                print(f"Groq API returned error status: {response.status_code}, detail: {response.text}")
        except Exception as e:
            print(f"Failed to communicate with Groq: {str(e)}")

    # --- PROVIDER 2: GEMINI PRO API ---
    if GEMINI_API_KEY:
        try:
            # Load Gemini Model (using gemini-2.0-flash for high free quota limits)
            gemini_model = genai.GenerativeModel(model_name="gemini-2.0-flash")
            
            # Initialize chat history
            history = []
            if req.history:
                for turn in req.history:
                    role = "user" if turn.role == "user" else "model"
                    history.append({
                        "role": role,
                        "parts": [turn.text]
                    })
                    
            chat = gemini_model.start_chat(history=history)
            
            # Generate response
            combined_prompt = f"{system_prompt}\n\nUser Question: {req.message}"
            response = chat.send_message(combined_prompt)
            return {"response": response.text}
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Gemini API Exception: {str(e)}")

    # If neither key is configured
    raise HTTPException(
        status_code=500, 
        detail="No active AI API keys (XAI_API_KEY, GROQ_API_KEY, or GEMINI_API_KEY) found in the environment configurations."
    )
