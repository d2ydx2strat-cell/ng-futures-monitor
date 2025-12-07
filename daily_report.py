import os
import smtplib
import ssl
from email.message import EmailMessage
import yfinance as yf
import requests
import pandas as pd
from datetime import datetime

# --- CONFIG ---
EIA_KEY = os.environ.get("EIA_API_KEY")
EMAIL_SENDER = os.environ.get("EMAIL_USER")
EMAIL_PASSWORD = os.environ.get("EMAIL_PASS")
EMAIL_RECEIVER = os.environ.get("EMAIL_USER") # Sending to yourself

def get_market_data():
    try:
        # Fetch Henry Hub and TTF Proxy
        tickers = ['NG=F', 'TTF=F', 'EURUSD=X']
        df = yf.download(tickers, period="5d", interval="1d")['Close']
        
        last_hh = df['NG=F'].iloc[-1]
        last_ttf_eur = df['TTF=F'].iloc[-1]
        fx = df['EURUSD=X'].iloc[-1]
        
        # Convert TTF to USD/MMBtu
        last_ttf_usd = (last_ttf_eur * fx) / 3.412
        spread = last_ttf_usd - last_hh
        
        return last_hh, last_ttf_usd, spread
    except:
        return 0, 0, 0

def get_storage_data():
    if not EIA_KEY: return "No API Key"
    try:
        url = "https://api.eia.gov/v2/natural-gas/stor/wkly/data/"
        params = {
            "api_key": EIA_KEY,
            "frequency": "weekly",
            "data[0]": "value",
            "facets[series][]": "NG.NW2_EPG0_SWO_R48_BCF.W",
            "sort[0][column]": "period",
            "sort[0][direction]": "desc",
            "length": 1
        }
        r = requests.get(url, params=params).json()
        val = r['response']['data'][0]['value']
        return f"{val} Bcf"
    except:
        return "Fetch Error"

def get_weather_outlook():
    # Simple check for Chicago (Midwest Proxy)
    try:
        url = "https://api.open-meteo.com/v1/forecast?latitude=41.85&longitude=-87.62&daily=temperature_2m_mean&forecast_days=7"
        r = requests.get(url).json()
        temps = r['daily']['temperature_2m_mean']
        
        # Calculate HDD (65F base). API returns Celsius.
        hdds = []
        for t_c in temps:
            t_f = (t_c * 9/5) + 32
            hdds.append(max(0, 65 - t_f))
        
        total_hdd = sum(hdds)
        return f"{total_hdd:.0f} HDDs (7-Day Chicago)"
    except:
        return "N/A"

def send_email():
    hh, ttf, spread = get_market_data()
    storage = get_storage_data()
    weather = get_weather_outlook()
    
    date_str = datetime.now().strftime("%Y-%m-%d")
    
    subject = f"NG Morning Report: Spread ${spread:.2f} | HH ${hh:.2f}"
    
    body = f"""
    NG TRADING SNAPSHOT - {date_str}
    --------------------------------
    
    1. MARKET (Arb Window)
    Henry Hub:   ${hh:.2f}
    TTF (EU):    ${ttf:.2f}
    Spread:      ${spread:.2f} 
    (Spread > $8 usually supports max LNG exports)

    2. STORAGE (Lower 48)
    Latest Level: {storage}
    
    3. WEATHER (Demand Proxy)
    Chicago 7-Day Forecast: {weather}
    
    --------------------------------
    Sent automatically via GitHub Actions
    """

    msg = EmailMessage()
    msg.set_content(body)
    msg['Subject'] = subject
    msg['From'] = EMAIL_SENDER
    msg['To'] = EMAIL_RECEIVER

    context = ssl.create_default_context()
    with smtplib.SMTP_SSL('smtp.gmail.com', 465, context=context) as smtp:
        smtp.login(EMAIL_SENDER, EMAIL_PASSWORD)
        smtp.send_message(msg)
    
    print("Email Sent Successfully")

if __name__ == "__main__":
    send_email()