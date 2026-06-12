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

# --- CHATBOT INTEGRATION ---
import google.generativeai as genai

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
XAI_API_KEY = os.environ.get("XAI_API_KEY")

class ConversationTurn(BaseModel):
    role: str  # "user" or "model"
    text: str

class ChatRequest(BaseModel):
    message: str
    mode: Optional[str] = "standard"

class GeminiChatRequest(BaseModel):
    message: str
    history: Optional[list[ConversationTurn]] = None
    mode: Optional[str] = "standard"

def get_islamic_chat_response(message: str, history: Optional[list[ConversationTurn]] = None, mode: str = "standard") -> str:
    """Answers user query using Groq (Llama 3) or Gemini Pro, acting as a general Islamic chatbot or Deen Buddy for kids"""
    # 1. Build System Prompt based on Mode
    if mode == "kids":
        system_prompt = (
            "You are Deen Buddy, a patient, warm, and friendly Islamic tutor for kids.\n"
            "Your goal is to answer questions about Islam, explain Wudhu (ablution) and daily prayers, "
            "and tell stories of the Prophets in a simple, engaging, and age-appropriate way.\n\n"
            "Guidelines:\n"
            "- Use simple language, short sentences, and friendly analogies.\n"
            "- Use kid-friendly emojis (e.g. 🌟, 🎈, 💦, 🚢, 🐦, 🕌, 🐜) to make the text lively and interactive.\n"
            "- When explaining concepts like Wudhu, explain the steps simply and encourage them.\n"
            "- Always start with a warm child-friendly greeting, like 'Assalamu Alaikum little friend! 🌟' or 'Hey there! Assalamu Alaikum! 👋'.\n"
            "- Always close with a sweet sign-off, like 'Remember, Allah loves you! 🌟' or 'Keep learning and smiling! 😊'.\n"
            "- ADULT / 18+ / PUBERTY BOUNDARY: Never answer questions related to puberty, periods, menstruation, sexuality, intimacy, or any other adult (18+) topics. If a child asks about these, do not explain them. Instead, respond with a fun, age-appropriate diversion that redirects their attention, strictly using this text format:\n"
            "  'Oh, that is a special question to ask your parents or older family members when you grow a bit older! 🌟 But guess what? Did you know that Prophet Sulaiman (Solomon) could talk to animals, and he once had a conversation with a tiny ant! 🐜 If you could talk to any animal, which one would it be? 🎈'\n"
            "- ABSOLUTE BOUNDARY: Never answer questions about non-Islamic topics (such as coding, computer programming, mathematics, general science, video games, general news, recipes, etc.). You must strictly refuse to answer and only output: 'That sounds interesting, but as your Deen Buddy, I love to talk about our beautiful Deen (Islam), the Prophets, and Wudhu! Let\'s talk about that instead! 🎈'. Do not write any code, solve any math, or provide general answers under any circumstances.\n"
            "- Never mention GPS, map coordinates, or app statistics. If they ask about local mosques, tell them to ask their parents to help them find one using the app's map! 🕌\n"
        )
    else:
        system_prompt = (
            "You are e.Baab-ul-Salah Assistant, a dedicated, polite, and highly specialized Islamic chatbot.\n"
            "Your goal is to answer questions strictly about Islamic practices, daily prayers (Namaz), Wudhu (ablution), "
            "authentic Hadiths, Sunnah, Quranic virtues, and basic Fiqh rules.\n\n"
            "Crucial Behavior & Formatting Guidelines:\n"
            "- Always begin your response with a respectful Islamic greeting, such as 'Assalamu Alaikum wa Rahmatullahi wa Barakatuh'.\n"
            "- Always close your response with a brief, polite Islamic prayer or sign-off, such as 'May Allah guide us all' or 'May Allah accept our deeds'.\n"
            "- Format Quranic verses or Hadiths in clear blockquotes (using '>').\n"
            "- Use clean markdown headers (###), lists, and bold text to make answers readable.\n"
            "- STRICT BOUNDARY: If the user asks about general, non-Islamic topics (e.g. computer programming, coding, math, general science, news, cooking recipes, etc.), politely decline, stating that you are dedicated solely to Islamic guidance and cannot assist with unrelated subjects.\n"
            "- Do not mention or explain any GPS, geolocation, nearby search, or map features under any circumstances. If a user asks about local mosque timings or GPS, explain that you cannot assist with location services or specific local timings, and guide them to use the app's native navigation features.\n"
            "- If you are unsure of a Fiqh question, state so humbly rather than speculating.\n\n"
        )

    # --- PROVIDER 0: xAI GROK API ---
    if XAI_API_KEY:
        try:
            import httpx
            messages = [{"role": "system", "content": system_prompt}]
            if history:
                for turn in history:
                    role = "user" if turn.role == "user" else "assistant"
                    messages.append({"role": role, "content": turn.text})
            messages.append({"role": "user", "content": message})
            
            headers = {
                "Authorization": f"Bearer {XAI_API_KEY}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": "grok-beta",
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
                return res_data["choices"][0]["message"]["content"]
            else:
                print(f"xAI API returned error status: {response.status_code}, detail: {response.text}")
        except Exception as e:
            print(f"Failed to communicate with xAI: {str(e)}")

    # --- PROVIDER 1: GROQ CLOUD API (100% Free, Blazing Fast) ---
    if GROQ_API_KEY:
        try:
            import httpx
            messages = [{"role": "system", "content": system_prompt}]
            if history:
                for turn in history:
                    role = "user" if turn.role == "user" else "assistant"
                    messages.append({"role": role, "content": turn.text})
            messages.append({"role": "user", "content": message})
            
            headers = {
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": "llama-3.1-8b-instant",
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
                return res_data["choices"][0]["message"]["content"]
            else:
                print(f"Groq API returned error status: {response.status_code}, detail: {response.text}")
        except Exception as e:
            print(f"Failed to communicate with Groq: {str(e)}")

    # --- PROVIDER 2: GEMINI PRO API ---
    if GEMINI_API_KEY:
        try:
            gemini_model = genai.GenerativeModel(model_name="gemini-2.0-flash")
            history_gemini = []
            if history:
                for turn in history:
                    role = "user" if turn.role == "user" else "model"
                    history_gemini.append({
                        "role": role,
                        "parts": [turn.text]
                    })
                    
            chat = gemini_model.start_chat(history=history_gemini)
            combined_prompt = f"{system_prompt}\n\nUser Question: {message}"
            response = chat.send_message(combined_prompt)
            return response.text
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Gemini API Exception: {str(e)}")

    # If neither key is configured
    raise HTTPException(
        status_code=500, 
        detail="No active AI API keys (XAI_API_KEY, GROQ_API_KEY, or GEMINI_API_KEY) found in the environment configurations."
    )

@app.post("/chat")
def chat_bot(req: ChatRequest):
    """General Islamic chatbot endpoint using the shared AI model pipeline"""
    response_text = get_islamic_chat_response(req.message, None, req.mode)
    return {"response": response_text}

@app.post("/chat/gemini")
def chat_gemini(req: GeminiChatRequest):
    """Answers user query using Groq or Gemini Pro, acting as a general Islamic chatbot"""
    response_text = get_islamic_chat_response(req.message, req.history, req.mode)
    return {"response": response_text}

# --- AI PREDICTIVE PRAYER ANALYTICS ---
class PrayerLog(BaseModel):
    prayer_name: str  # "Fajr", "Zuhr", "Asr", "Maghrib", "Isha"
    date: str        # "YYYY-MM-DD"
    status: str      # "attended_jamaat", "prayed_alone", "missed"

class AnalyticsRequest(BaseModel):
    history: list[PrayerLog]

@app.post("/analytics/predictive-advice")
def predictive_advice(req: AnalyticsRequest):
    """Analyzes personal prayer logs over time to generate tailored habit-building insights"""
    if not req.history:
        return {"insights": []}

    # Group records by prayer name
    # We will compute attendance stats
    stats = {}
    for entry in req.history:
        name = entry.prayer_name.strip().capitalize()
        # Parse date to check if weekday (0-4 represent Mon-Fri)
        is_weekday = False
        try:
            dt = datetime.strptime(entry.date, "%Y-%m-%d")
            is_weekday = dt.weekday() < 5
        except Exception:
            pass

        if name not in stats:
            stats[name] = {
                "total_weekday": 0, "attended_weekday": 0, "missed_weekday": 0, "alone_weekday": 0,
                "total_weekend": 0, "attended_weekend": 0, "missed_weekend": 0, "alone_weekend": 0,
                "total": 0, "attended": 0, "missed": 0, "alone": 0
            }

        s = stats[name]
        s["total"] += 1
        if entry.status == "attended_jamaat":
            s["attended"] += 1
        elif entry.status == "missed":
            s["missed"] += 1
        else:
            s["alone"] += 1

        if is_weekday:
            s["total_weekday"] += 1
            if entry.status == "attended_jamaat":
                s["attended_weekday"] += 1
            elif entry.status == "missed":
                s["missed_weekday"] += 1
            else:
                s["alone_weekday"] += 1
        else:
            s["total_weekend"] += 1
            if entry.status == "attended_jamaat":
                s["attended_weekend"] += 1
            elif entry.status == "missed":
                s["missed_weekend"] += 1
            else:
                s["alone_weekend"] += 1

    insights = []

    # Heuristic Rule checks
    for prayer, s in stats.items():
        # 1. Weekday Asr/Zuhr drop due to work/traffic
        if prayer in ["Asr", "Zuhr"] and s["total_weekday"] >= 3:
            missed_or_alone_rate = (s["missed_weekday"] + s["alone_weekday"]) / s["total_weekday"]
            if missed_or_alone_rate > 0.4:
                insights.append({
                    "prayer_name": prayer,
                    "pain_point": "Weekday Work/Traffic Friction",
                    "advice": f"We noticed you often miss Jamaat for {prayer} on weekdays. This is common due to rush-hour office traffic and meeting times. Would you like to enable a smart travel alert 25 minutes before prayer time to plan your journey?",
                    "suggested_reminder_offset_mins": 25
                })
                continue # Avoid duplicate alerts for same prayer

        # 2. Fajr wake-up drop
        if prayer == "Fajr" and s["total"] >= 3:
            missed_rate = s["missed"] / s["total"]
            if missed_rate > 0.4:
                insights.append({
                    "prayer_name": prayer,
                    "pain_point": "Fajr Morning Wake-up",
                    "advice": "We noticed Fajr is your most missed congregational prayer. Establishing Fajr brings immense barakah to your day. Would you like to schedule an automated alarm 15 minutes before Fajr Iqamah starts?",
                    "suggested_reminder_offset_mins": 15
                })
                continue

        # 3. Weekend Alone drop for Maghrib/Isha
        if prayer in ["Maghrib", "Isha"] and s["total_weekend"] >= 2:
            alone_rate = s["alone_weekend"] / s["total_weekend"]
            if alone_rate > 0.4:
                insights.append({
                    "prayer_name": prayer,
                    "pain_point": "Weekend Alone Prayers",
                    "advice": f"You frequently pray {prayer} alone on weekends. Weekend evenings are a great opportunity to visit the mosque with family. Would you like a mosque arrival alert 15 minutes before congregational prayer?",
                    "suggested_reminder_offset_mins": 15
                })
                continue

    return {"insights": insights}


# --- SEASONAL TIME-SHIFT PREDICTIVE MODELING ---
class ConfiguredPrayerTimes(BaseModel):
    fajr_jamaat: str
    zuhr_jamaat: str
    asr_jamaat: str
    maghrib_jamaat: str
    isha_jamaat: str

class MasjidDriftRequest(BaseModel):
    masjid_name: str
    date: str  # Format: YYYY-MM-DD
    configured_times: ConfiguredPrayerTimes

class DriftCheckRequest(BaseModel):
    masjids: list[MasjidDriftRequest]

def parse_time_to_mins(time_str: str) -> int:
    """Parses a time string like '05:00 AM', '17:30', or '5:00 PM' into minutes from midnight"""
    time_str = time_str.strip().upper()
    if "AM" in time_str or "PM" in time_str:
        meridian = "PM" if "PM" in time_str else "AM"
        clean_time = time_str.replace("AM", "").replace("PM", "").strip()
        parts = clean_time.split(":")
        h = int(parts[0])
        m = int(parts[1])
        if meridian == "PM" and h < 12:
            h += 12
        elif meridian == "AM" and h == 12:
            h = 0
        return h * 60 + m
    else:
        parts = time_str.split(":")
        return int(parts[0]) * 60 + int(parts[1])

@app.post("/analytics/seasonal-drift-check")
def seasonal_drift_check(req: DriftCheckRequest):
    """Studies historical timetable variations and flags masjids lagging behind seasonal adjustments"""
    if model is None:
        raise HTTPException(status_code=500, detail="Model is not loaded.")

    anomalies = []
    thresholds = {
        "Fajr": 15,
        "Zuhr": 15,
        "Asr": 15,
        "Maghrib": 10,
        "Isha": 15
    }

    for item in req.masjids:
        masjid_details = _lookup_masjid(item.masjid_name)
        if not masjid_details:
            continue

        try:
            dt = datetime.strptime(item.date, "%Y-%m-%d")
        except ValueError:
            continue

        # Get astronomical start times
        y, m, d = dt.year, dt.month, dt.day
        waqt = PT.getTimes((y, m, d), (masjid_details['latitude'], masjid_details['longitude']), 5)
        astro = {
            "Fajr": to_mins(waqt['fajr']),
            "Zuhr": to_mins(waqt['dhuhr']),
            "Asr": to_mins(waqt['asr']),
            "Maghrib": to_mins(waqt['maghrib']),
            "Isha": to_mins(waqt['isha'])
        }

        # Predict standard/expected Jamaat times via ML model
        features = _generate_features_for_day(dt, masjid_details['masjid_id'], masjid_details['latitude'], masjid_details['longitude'])
        predictions = model.predict([features])[0]
        predictions = np.round(predictions).astype(int)
        predicted = {
            "Fajr": int(predictions[0]),
            "Zuhr": int(predictions[1]),
            "Asr": int(predictions[2]),
            "Maghrib": int(predictions[3]),
            "Isha": int(predictions[4])
        }

        # Parse configured times
        cfg = item.configured_times
        configured = {
            "Fajr": parse_time_to_mins(cfg.fajr_jamaat),
            "Zuhr": parse_time_to_mins(cfg.zuhr_jamaat),
            "Asr": parse_time_to_mins(cfg.asr_jamaat),
            "Maghrib": parse_time_to_mins(cfg.maghrib_jamaat),
            "Isha": parse_time_to_mins(cfg.isha_jamaat)
        }

        # Check each prayer for drift or invalid status
        for prayer in ["Fajr", "Zuhr", "Asr", "Maghrib", "Isha"]:
            cfg_mins = configured[prayer]
            pred_mins = predicted[prayer]
            astro_mins = astro[prayer]

            # 1. Astronomical check (configured time is before prayer start time)
            if cfg_mins < astro_mins:
                anomalies.append({
                    "masjid_name": masjid_details['masjid_name'],
                    "date": item.date,
                    "prayer_name": prayer,
                    "configured_time": from_mins(cfg_mins),
                    "predicted_time": from_mins(pred_mins),
                    "astronomical_start": from_mins(astro_mins),
                    "drift_mins": astro_mins - cfg_mins,
                    "status": "Invalid",
                    "message": f"{prayer} Jamaat ({from_mins(cfg_mins)}) is configured before its astronomical start time of {from_mins(astro_mins)}."
                })
            else:
                # 2. Model prediction drift check
                drift = cfg_mins - pred_mins
                if abs(drift) > thresholds[prayer]:
                    status = "Lagging" if drift > 0 else "Leading"
                    anomalies.append({
                        "masjid_name": masjid_details['masjid_name'],
                        "date": item.date,
                        "prayer_name": prayer,
                        "configured_time": from_mins(cfg_mins),
                        "predicted_time": from_mins(pred_mins),
                        "astronomical_start": from_mins(astro_mins),
                        "drift_mins": abs(drift),
                        "status": status,
                        "message": f"{prayer} Jamaat ({from_mins(cfg_mins)}) is {status.lower()} behind seasonal adjustments by {abs(drift)} minutes (Predicted: {from_mins(pred_mins)})."
                    })

    return {"anomalies": anomalies}


# --- INTERACTIVE WEEKLY REFLECTION & PLAN GENERATOR ---
class WeeklySummaryRequest(BaseModel):
    history: list[PrayerLog]

class NonJamaatPrayerDetail(BaseModel):
    prayer_name: str
    missed_count: int
    alone_count: int
    total_logged: int

class WeeklySummaryResponse(BaseModel):
    non_jamaat_prayers: list[NonJamaatPrayerDetail]

class GeneratePlanRequest(BaseModel):
    prayer_name: str
    status: str  # "missed" or "prayed_alone"
    reason: str  # "work_meetings", "traffic_travel", "sleep_fatigue", "forgetfulness", "no_masjid_nearby"

class HabitPlanResponse(BaseModel):
    prayer_name: str
    status: str
    reason: str
    plan: str
    suggested_reminder_offset_mins: int

@app.post("/analytics/weekly-summary", response_model=WeeklySummaryResponse)
def weekly_summary(req: WeeklySummaryRequest):
    """Analyzes a week of prayer history to identify prayers that were missed or prayed alone"""
    counts = {}
    # Initialize counts for all 5 prayers
    for name in ["Fajr", "Zuhr", "Asr", "Maghrib", "Isha"]:
        counts[name] = {"missed": 0, "alone": 0, "total": 0}

    for entry in req.history:
        name = entry.prayer_name.strip().capitalize()
        if name in counts:
            counts[name]["total"] += 1
            if entry.status == "missed":
                counts[name]["missed"] += 1
            elif entry.status == "prayed_alone":
                counts[name]["alone"] += 1

    non_jamaat = []
    for name, c in counts.items():
        if c["missed"] > 0 or c["alone"] > 0:
            non_jamaat.append(NonJamaatPrayerDetail(
                prayer_name=name,
                missed_count=c["missed"],
                alone_count=c["alone"],
                total_logged=c["total"]
            ))

    return WeeklySummaryResponse(non_jamaat_prayers=non_jamaat)

@app.post("/analytics/generate-plan", response_model=HabitPlanResponse)
def generate_plan(req: GeneratePlanRequest):
    """Generates a tailored habit-building plan based on the reason selected by the user"""
    prayer = req.prayer_name.strip().capitalize()
    status = req.status.strip().lower()
    reason = req.reason.strip().lower()

    # Base offsets and plans based on user reason and status
    plans = {
        "work_meetings": {
            "missed": f"To prevent work or chores from causing you to completely miss {prayer}, try setting a calendar block for 10 minutes. A gentle reminder 10 minutes before Iqamah will help you transition from your tasks.",
            "prayed_alone": f"Since you prayed {prayer} alone due to work, consider scheduling a reminder 15 minutes before congregation. Try inviting colleagues or family members nearby to pray together in congregation.",
            "offset": 15
        },
        "traffic_travel": {
            "missed": f"Commuting during {prayer} time is a challenge. We suggest setting a smart travel alert 25 minutes before prayer to plan your departure or locate a mosque on your route.",
            "prayed_alone": f"To avoid praying {prayer} alone when traveling, set a mosque-arrival alert 20 minutes early. Use the app's 5km finder to locate a nearby mosque with active congregation.",
            "offset": 20
        },
        "sleep_fatigue": {
            "missed": f"Struggling to wake up for {prayer} is common. We recommend placing your alarm far from your bed and setting an automated alert 15 minutes before Iqamah starts. Ask a family member or friend to call you.",
            "prayed_alone": f"If morning or late-night fatigue makes you pray {prayer} alone, set a wakeup alert 15 minutes before congregation. Linking up with a mosque-going buddy will help you stay accountable.",
            "offset": 15
        },
        "forgetfulness": {
            "missed": f"It is easy to get distracted and miss {prayer}. We recommend enabling a recurring notification 5 minutes before Azan to bring your awareness back to the prayer time.",
            "prayed_alone": f"To ensure you don't miss the Jamaat for {prayer} due to distraction, set an alert 10 minutes before congregation so you can get ready and walk to the mosque.",
            "offset": 10
        },
        "no_masjid_nearby": {
            "missed": f"If you missed {prayer} because you couldn't find a mosque, enable a masjid-locator notification 15 minutes early to guide you using the 5km radius navigation.",
            "prayed_alone": f"To pray {prayer} in congregation when in an unfamiliar area, use the app's 5km radius finder and set a masjid-arrival alert 15 minutes before the Iqamah starts.",
            "offset": 15
        }
    }

    # Fallback plan if reason is unknown
    default_plan = {
        "missed": f"Building consistency for {prayer} starts with small steps. Set a reminder 10 minutes before the prayer to prepare yourself.",
        "prayed_alone": f"Try to visit the local masjid for {prayer}. Set an alert 15 minutes before the congregation to give you enough time to travel.",
        "offset": 15
    }

    selected = plans.get(reason, default_plan)
    plan_text = selected.get(status, selected.get("missed"))
    offset = selected.get("offset", 15)

    return HabitPlanResponse(
        prayer_name=prayer,
        status=status,
        reason=reason,
        plan=plan_text,
        suggested_reminder_offset_mins=offset
    )


