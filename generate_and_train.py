import pandas as pd
import numpy as np
from datetime import date, timedelta
from praytimes import PrayTimes
from hijridate import convert
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

print("⏳ Initializing PrayTimes...")
PT = PrayTimes('Karachi')
PT.adjust({'asr': 'Hanafi'})

DEFAULT_OFFSETS = {
    'fajr_normal':      60,
    'fajr_ramadan':     15,
    'zuhr_fixed':       '13:30',
    'jumuah_fixed':     '13:30',
    'asr_normal':       35,
    'maghrib_normal':   10,
    'maghrib_ramadan':  15,
    'isha_normal':      30,
}

MASJIDS = [
    # Original 16 Masjids (Verified Coordinates)
    {'id': 1, 'name': 'Ibn-e-Adam Masjid', 'lat': 31.4567628, 'lon': 74.2525003, 'offsets': {**DEFAULT_OFFSETS, 'zuhr_fixed': '13:45', 'jumuah_fixed': '14:15'}},
    {'id': 2, 'name': 'Shan-e-Islam Masjid', 'lat': 31.5088462, 'lon': 74.3535366, 'offsets': {**DEFAULT_OFFSETS, 'jumuah_fixed': '13:40'}},
    {'id': 3, 'name': 'Muzammil Masjid', 'lat': 31.49729, 'lon': 74.35664, 'offsets': {**DEFAULT_OFFSETS}},
    {'id': 4, 'name': 'Al-Habib Masjid', 'lat': 31.535879, 'lon': 74.3790217, 'offsets': {**DEFAULT_OFFSETS}},
    {'id': 5, 'name': 'Khalid Masjid', 'lat': 31.5000111, 'lon': 74.3665154, 'offsets': {**DEFAULT_OFFSETS}},
    {'id': 6, 'name': 'Ravi Hotel Masjid', 'lat': 31.6145895, 'lon': 74.2916412, 'offsets': {**DEFAULT_OFFSETS}},
    {'id': 7, 'name': 'M7C Masjid', 'lat': 31.3504382, 'lon': 74.2461831, 'offsets': {**DEFAULT_OFFSETS}},
    {'id': 8, 'name': 'Jamia Masjid Rehman', 'lat': 31.5808333, 'lon': 74.4665918, 'offsets': {**DEFAULT_OFFSETS}},
    {'id': 9, 'name': 'Grand Jamia Mosque', 'lat': 31.4829403, 'lon': 74.3343893, 'offsets': {**DEFAULT_OFFSETS, 'zuhr_fixed': '13:45'}},
    {'id': 10, 'name': ' Jamia Masjid Dha Phase 4', 'lat': 31.464632, 'lon': 74.3839531, 'offsets': {**DEFAULT_OFFSETS}},
    {'id': 11, 'name': ' Jamia Masjid Mustafa', 'lat': 31.495198, 'lon': 74.3469482, 'offsets': {**DEFAULT_OFFSETS}},
    {'id': 12, 'name': ' Madina Masjid', 'lat': 31.4855552, 'lon': 74.3470166, 'offsets': {**DEFAULT_OFFSETS, 'zuhr_fixed': '13:45'}},
    {'id': 13, 'name': ' Jamia Masjid Ghausia Rizwia', 'lat': 31.5440233, 'lon': 74.3505727, 'offsets': {**DEFAULT_OFFSETS, 'jumuah_fixed': '14:15'}},
    {'id': 14, 'name': ' Jamia Masjid Muhammadiya Rizwia', 'lat': 31.5028744, 'lon': 74.3471544, 'offsets': {**DEFAULT_OFFSETS, 'jumuah_fixed': '14:30'}},
    {'id': 15, 'name': ' Jamia Masjid Aziz Rashid', 'lat': 31.5480595, 'lon': 74.2844224, 'offsets': {**DEFAULT_OFFSETS}},
    {'id': 16, 'name': ' Jamia Masjid Ahsan Raheem', 'lat': 31.4999791, 'lon': 74.3523614, 'offsets': {**DEFAULT_OFFSETS, 'jumuah_fixed': '14:15'}},

    # 34 New Masjids (Pinpoint Lahore Locations)
    {'id': 17, 'name': 'Jamia Masjid Bilal (DHA Phase 3)', 'lat': 31.4725, 'lon': 74.3735, 'offsets': {**DEFAULT_OFFSETS}},
    {'id': 18, 'name': 'Jamia Masjid Rehmat (Model Town)', 'lat': 31.4795, 'lon': 74.3225, 'offsets': {**DEFAULT_OFFSETS}},
    {'id': 19, 'name': 'Jamia Masjid Al-Shafi (Johar Town)', 'lat': 31.4642, 'lon': 74.2795, 'offsets': {**DEFAULT_OFFSETS}},
    {'id': 20, 'name': 'Masjid Al-Fatah (Gulberg)', 'lat': 31.5115, 'lon': 74.3498, 'offsets': {**DEFAULT_OFFSETS, 'zuhr_fixed': '13:45'}},
    {'id': 21, 'name': 'Jamia Masjid Aqsa (Faisal Town)', 'lat': 31.4792, 'lon': 74.3072, 'offsets': {**DEFAULT_OFFSETS}},
    {'id': 22, 'name': 'Masjid Al-Khair (Wapda Town)', 'lat': 31.4285, 'lon': 74.2612, 'offsets': {**DEFAULT_OFFSETS}},
    {'id': 23, 'name': 'Jamia Masjid Siddique (Samanabad)', 'lat': 31.5342, 'lon': 74.3148, 'offsets': {**DEFAULT_OFFSETS}},
    {'id': 24, 'name': 'Masjid Al-Madina (Garden Town)', 'lat': 31.4995, 'lon': 74.3218, 'offsets': {**DEFAULT_OFFSETS}},
    {'id': 25, 'name': 'Jamia Masjid Anwar-e-Madina (Green Town)', 'lat': 31.4395, 'lon': 74.3082, 'offsets': {**DEFAULT_OFFSETS}},
    {'id': 26, 'name': 'Masjid Al-Quds (Gulshan-e-Ravi)', 'lat': 31.5462, 'lon': 74.2952, 'offsets': {**DEFAULT_OFFSETS}},
    {'id': 27, 'name': 'Jamia Masjid Farooq-e-Azam (Township)', 'lat': 31.4525, 'lon': 74.3052, 'offsets': {**DEFAULT_OFFSETS}},
    {'id': 28, 'name': 'Jamia Masjid Hanfia (Sabzazar)', 'lat': 31.5252, 'lon': 74.2852, 'offsets': {**DEFAULT_OFFSETS}},
    {'id': 29, 'name': 'Masjid Ibrahim (Bahria Town)', 'lat': 31.3752, 'lon': 74.1912, 'offsets': {**DEFAULT_OFFSETS, 'zuhr_fixed': '13:45', 'jumuah_fixed': '14:00'}},
    {'id': 30, 'name': 'Jamia Masjid Khalid Bin Walid (Cavalry Ground)', 'lat': 31.5052, 'lon': 74.3725, 'offsets': {**DEFAULT_OFFSETS}},
    {'id': 31, 'name': 'Jamia Masjid Usman (Garhi Shahu)', 'lat': 31.5542, 'lon': 74.3512, 'offsets': {**DEFAULT_OFFSETS}},
    {'id': 32, 'name': 'Masjid Bilal (Allama Iqbal Town)', 'lat': 31.5152, 'lon': 74.2925, 'offsets': {**DEFAULT_OFFSETS}},
    {'id': 33, 'name': 'Jamia Masjid Imam Abu Hanifa (Tajpura)', 'lat': 31.5695, 'lon': 74.4095, 'offsets': {**DEFAULT_OFFSETS}},
    {'id': 34, 'name': 'Jamia Masjid Riaz-ul-Jannah (Mughalpura)', 'lat': 31.5625, 'lon': 74.3825, 'offsets': {**DEFAULT_OFFSETS}},
    {'id': 35, 'name': 'Jamia Masjid Taqwa (Nishtar Colony)', 'lat': 31.4125, 'lon': 74.3652, 'offsets': {**DEFAULT_OFFSETS}},
    {'id': 36, 'name': 'Jamia Masjid Noor (Shalamar)', 'lat': 31.5782, 'lon': 74.3792, 'offsets': {**DEFAULT_OFFSETS}},
    {'id': 37, 'name': 'Jamia Masjid Hazrat Bilal (Chung)', 'lat': 31.4152, 'lon': 74.2025, 'offsets': {**DEFAULT_OFFSETS}},
    {'id': 38, 'name': 'Jamia Masjid Quba (Cantt)', 'lat': 31.5195, 'lon': 74.4025, 'offsets': {**DEFAULT_OFFSETS, 'jumuah_fixed': '13:45'}},
    {'id': 39, 'name': 'Jamia Masjid Ghausia (Baghbanpura)', 'lat': 31.5752, 'lon': 74.3692, 'offsets': {**DEFAULT_OFFSETS}},
    {'id': 40, 'name': 'Jamia Masjid Rehmania (Ichhra)', 'lat': 31.5262, 'lon': 74.3282, 'offsets': {**DEFAULT_OFFSETS}},
    {'id': 41, 'name': 'Jamia Masjid Fatima (Johar Town Phase 2)', 'lat': 31.4552, 'lon': 74.2662, 'offsets': {**DEFAULT_OFFSETS}},
    {'id': 42, 'name': 'Jamia Masjid Al-Haram (DHA Phase 6)', 'lat': 31.4695, 'lon': 74.4425, 'offsets': {**DEFAULT_OFFSETS}},
    {'id': 43, 'name': 'Jamia Masjid Madni (Model Town Extension)', 'lat': 31.4855, 'lon': 74.3332, 'offsets': {**DEFAULT_OFFSETS, 'zuhr_fixed': '13:45'}},
    {'id': 44, 'name': 'Jamia Masjid Khizra (Samanabad)', 'lat': 31.5292, 'lon': 74.3052, 'offsets': {**DEFAULT_OFFSETS}},
    {'id': 45, 'name': 'Jamia Masjid Toheed (Valencia Town)', 'lat': 31.3995, 'lon': 74.2502, 'offsets': {**DEFAULT_OFFSETS}},
    {'id': 46, 'name': 'Jamia Masjid Aqsa (Sanda)', 'lat': 31.5642, 'lon': 74.2892, 'offsets': {**DEFAULT_OFFSETS}},
    {'id': 47, 'name': 'Jamia Masjid Mustafa (Shadman)', 'lat': 31.5352, 'lon': 74.3322, 'offsets': {**DEFAULT_OFFSETS}},
    {'id': 48, 'name': 'Jamia Masjid Abu Bakar (DHA Phase 5)', 'lat': 31.4625, 'lon': 74.4122, 'offsets': {**DEFAULT_OFFSETS, 'zuhr_fixed': '13:45', 'jumuah_fixed': '14:15'}},
    {'id': 49, 'name': 'Badshahi Mosque (Walled City)', 'lat': 31.5880, 'lon': 74.3108, 'offsets': {**DEFAULT_OFFSETS, 'zuhr_fixed': '13:45', 'jumuah_fixed': '14:00'}},
    {'id': 50, 'name': 'Wazir Khan Mosque (Walled City)', 'lat': 31.5831, 'lon': 74.3236, 'offsets': {**DEFAULT_OFFSETS, 'zuhr_fixed': '13:45', 'jumuah_fixed': '14:00'}},
]

def to_mins(hhmm):
    h, m = map(int, hhmm.split(':'))
    return h * 60 + m

def from_mins(mins):
    mins = int(mins) % 1440
    return f"{mins // 60:02d}:{mins % 60:02d}"

def floor_n(mins, n):
    return (mins // n) * n

def is_ramadan(y, m, d):
    return convert.Gregorian(y, m, d).to_hijri().month == 9

def get_hijri_info(y, m, d):
    return convert.Gregorian(y, m, d).to_hijri()

def compute_jamaat(waqt_times, is_ram, is_fri, offsets):
    # FAJR
    fajr_w = to_mins(waqt_times['fajr'])
    if is_ram:
        fajr_j = fajr_w + offsets['fajr_ramadan']
    else:
        fajr_j = floor_n(fajr_w + offsets['fajr_normal'], 5)

    # ZUHR / JUMU'AH
    zuhr_j = offsets['jumuah_fixed'] if is_fri else offsets['zuhr_fixed']

    # ASR
    asr_w = to_mins(waqt_times['asr'])
    asr_j = floor_n(asr_w + offsets['asr_normal'], 15)
    asr_j = max(to_mins('15:45'), min(to_mins('17:30'), asr_j))

    # MAGHRIB
    magh_w = to_mins(waqt_times['maghrib'])
    magh_j = magh_w + (offsets['maghrib_ramadan'] if is_ram else offsets['maghrib_normal'])

    # ISHA
    isha_w = to_mins(waqt_times['isha'])
    isha_j = floor_n(isha_w + offsets['isha_normal'], 15)
    isha_j = max(to_mins('18:45'), min(to_mins('21:00'), isha_j))

    return {
        'fajr_jamaat':    from_mins(fajr_j),
        'zuhr_jamaat':    zuhr_j,
        'asr_jamaat':     from_mins(asr_j),
        'maghrib_jamaat': from_mins(magh_j),
        'isha_jamaat':    from_mins(isha_j),
    }

def generate_year_data(year):
    print(f"⏳ Generating data for year {year}...")
    start_date = date(year, 1, 1)
    end_date = date(year, 12, 31)
    
    rows = []
    for masjid in MASJIDS:
        current = start_date
        while current <= end_date:
            y, m, d = current.year, current.month, current.day
            waqt = PT.getTimes((y, m, d), (masjid['lat'], masjid['lon']), 5)
            ram = is_ramadan(y, m, d)
            fri = current.weekday() == 4
            jamaat = compute_jamaat(waqt, ram, fri, masjid['offsets'])
            hijri = get_hijri_info(y, m, d)
            
            row = {
                'masjid_id':            masjid['id'],
                'masjid_name':          masjid['name'],
                'latitude':             masjid['lat'],
                'longitude':            masjid['lon'],
                'date':                 current.strftime('%Y-%m-%d'),
                'day_of_week':          current.strftime('%A'),
                'day_of_year':          current.timetuple().tm_yday,
                'month':                m,
                'month_name':           current.strftime('%B'),
                'hijri_date':           str(hijri),
                'hijri_month':          hijri.month,
                'hijri_month_name':     hijri.month_name(),
                'is_ramadan':           int(ram),
                'is_friday':            int(fri),
                'fajr_waqt':            waqt['fajr'],
                'sunrise':              waqt['sunrise'],
                'dhuhr_waqt':           waqt['dhuhr'],
                'asr_waqt':             waqt['asr'],
                'maghrib_waqt':         waqt['maghrib'],
                'isha_waqt':            waqt['isha'],
                'fajr_jamaat':          jamaat['fajr_jamaat'],
                'zuhr_jamaat':          jamaat['zuhr_jamaat'],
                'asr_jamaat':           jamaat['asr_jamaat'],
                'maghrib_jamaat':       jamaat['maghrib_jamaat'],
                'isha_jamaat':          jamaat['isha_jamaat'],
                'fajr_waqt_mins':       to_mins(waqt['fajr']),
                'dhuhr_waqt_mins':      to_mins(waqt['dhuhr']),
                'asr_waqt_mins':        to_mins(waqt['asr']),
                'maghrib_waqt_mins':    to_mins(waqt['maghrib']),
                'isha_waqt_mins':       to_mins(waqt['isha']),
                'fajr_jamaat_mins':     to_mins(jamaat['fajr_jamaat']),
                'zuhr_jamaat_mins':     to_mins(jamaat['zuhr_jamaat']),
                'asr_jamaat_mins':      to_mins(jamaat['asr_jamaat']),
                'maghrib_jamaat_mins':  to_mins(jamaat['maghrib_jamaat']),
                'isha_jamaat_mins':     to_mins(jamaat['isha_jamaat']),
            }
            rows.append(row)
            current += timedelta(days=1)
            
    return pd.DataFrame(rows)

# 1. Generate 2025 dataset
df_2025 = generate_year_data(2025)
csv_2025_path = 'C:/Users/ABC/Downloads/masjid_prayer_times_2025_ALL.csv'
df_2025.to_csv(csv_2025_path, index=False)
print(f"✅ Saved 2025 training data ({len(df_2025)} rows) to: {csv_2025_path}")

# 2. Train model
print("⏳ Training model on 50 masjids...")
features = [
    'masjid_id', 'is_ramadan', 'is_friday', 'day_of_year',
    'fajr_waqt_mins', 'dhuhr_waqt_mins', 'asr_waqt_mins', 'maghrib_waqt_mins', 'isha_waqt_mins'
]
targets = [
    'fajr_jamaat_mins', 'zuhr_jamaat_mins', 'asr_jamaat_mins', 'maghrib_jamaat_mins', 'isha_jamaat_mins'
]

X_train = df_2025[features]
y_train = df_2025[targets]

model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# Evaluate model
preds = model.predict(X_train)
mae = mean_absolute_error(y_train, preds)
print(f"✅ Model Training Complete! MAE: {mae:.4f} minutes")

# Save model
model_save_path = 'C:/Users/ABC/Downloads/api/masjid_model.pkl'
joblib.dump(model, model_save_path)
print(f"✅ Model saved to: {model_save_path}")

# 3. Generate 2026 predictions
df_2026 = generate_year_data(2026)
X_test = df_2026[features]
predicted_mins = model.predict(X_test)
predicted_mins = np.round(predicted_mins).astype(int)

df_2026['fajr_jamaat_mins'] = predicted_mins[:, 0]
df_2026['zuhr_jamaat_mins'] = predicted_mins[:, 1]
df_2026['asr_jamaat_mins'] = predicted_mins[:, 2]
df_2026['maghrib_jamaat_mins'] = predicted_mins[:, 3]
df_2026['isha_jamaat_mins'] = predicted_mins[:, 4]

df_2026['fajr_jamaat'] = df_2026['fajr_jamaat_mins'].apply(from_mins)
df_2026['zuhr_jamaat'] = df_2026['zuhr_jamaat_mins'].apply(from_mins)
df_2026['asr_jamaat'] = df_2026['asr_jamaat_mins'].apply(from_mins)
df_2026['maghrib_jamaat'] = df_2026['maghrib_jamaat_mins'].apply(from_mins)
df_2026['isha_jamaat'] = df_2026['isha_jamaat_mins'].apply(from_mins)

csv_2026_path = 'C:/Users/ABC/Downloads/masjid_prayer_times_2026_PREDICTED.csv'
df_2026.to_csv(csv_2026_path, index=False)
print(f"✅ Saved 2026 predicted data ({len(df_2026)} rows) to: {csv_2026_path}")

# Save excel
excel_2026_path = 'C:/Users/ABC/Downloads/masjid_prayer_times_2026_PREDICTED_BY_MASJID.xlsx'
with pd.ExcelWriter(excel_2026_path, engine='openpyxl') as writer:
    df_2026.to_excel(writer, sheet_name='ALL_MASJIDS', index=False)
    for m in df_2026['masjid_name'].unique():
        masjid_df = df_2026[df_2026['masjid_name'] == m]
        sheet_name = m[:31]
        masjid_df.to_excel(writer, sheet_name=sheet_name, index=False)

print(f"✅ Saved 2026 Excel sheets to: {excel_2026_path}")
print("🎉 Retraining and data generation pipeline complete!")
