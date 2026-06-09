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
            "- Use kid-friendly emojis (e.g. 🌟, 🎈, 💦, 🚢, 🐦, 🕌) to make the text lively and interactive.\n"
            "- When explaining concepts like Wudhu, explain the steps simply and encourage them.\n"
            "- Always start with a warm child-friendly greeting, like 'Assalamu Alaikum little friend! 🌟' or 'Hey there! Assalamu Alaikum! 👋'.\n"
            "- Always close with a sweet sign-off, like 'Remember, Allah loves you! 🌟' or 'Keep learning and smiling! 😊'.\n"
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
