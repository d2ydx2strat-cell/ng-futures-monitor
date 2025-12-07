import os
import smtplib
import ssl
from email.message import EmailMessage
import yfinance as yf
import requests
import pandas as pd
from datetime import datetime

# --- CONFIG ---
# Hardcoded Key as requested.
EIA_KEY = "KzzwPVmMSTVCI3pQbpL9calvF4CqGgEbwWy0qqXV" 

# These remain as GitHub Secrets for security (EMAIL_USER, EMAIL_PASS)
EMAIL_SENDER = os.environ.get("EMAIL_USER")
EMAIL_PASSWORD = os.environ.get("EMAIL_PASS")
EMAIL_RECEIVER = os.environ.get("EMAIL_USER") # Sending to yourself

def get_market_data():
    """Fetches NG futures, TTF futures, and FX data."""
    try:
        tickers = ['NG=F', 'TTF=F', 'EURUSD=X']
        df = yf.download(tickers, period="5d", interval="1d")['Close']
        
        last_hh = df['NG=F'].iloc[-1]
        last_ttf_eur = df['TTF=F'].iloc[-1]
        fx = df['EURUSD=X'].iloc[-1]
        
        # Convert TTF (EUR/MWh) to USD/MMBtu (using 3.412 conversion factor)
        last_ttf_usd = (last_ttf_eur * fx) / 3.412
        spread = last_ttf_usd - last_hh
        
        return last_hh, last_ttf_usd, spread
    except:
        return 0, 0, 0

def get_storage_data():
    """Fetches Weekly Midwest Region Storage Data from EIA."""
    if not EIA_KEY: 
        return "No API Key"
        
    url = "https://api.eia.gov/v2/natural-gas/stor/wkly/data/"
    
    # CONFIRMED WORKING & SIMPLIFIED SERIES ID: Midwest Region
    WORKING_SERIES_ID = "NW2_EPG0_SWO_R32_BCF"

    params = {
        "api_key": EIA_KEY, 
        "frequency": "weekly",
        "data[0]": "value", 
        "facets[series][]": WORKING_SERIES_ID, 
        "sort[0][column]": "period", 
        "sort[0][direction]": "desc", 
        "offset": 0,
        "length": 1 # Fetch only the single latest record
    }
    
    try:
        r = requests.get(url, params=params).json()
        
        if 'error' in r:
            return f"EIA Error: {r['error']}"
        
        if 'response' in r and 'data' in r['response'] and r['response']['data']:
            val = r['response']['data'][0]['value']
            return f"{val} Bcf (Midwest)"
        else:
            return "Fetch Error (Data Empty)"
    except Exception as e:
        return f"Fetch Error: {e}"

def get_weather_outlook():
    """Calculates 7-day Heating Degree Days (HDD) for Chicago (Midwest Proxy)."""
    try:
        # Simple check for Chicago (Midwest Proxy)
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
    """Gathers all data and sends the formatted email."""
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

    2. STORAGE 
    Latest Level: {storage}
    
    3. WEATHER (Demand Proxy)
    Chicago 7-Day Forecast: {weather}
    
    --------------------------------
    Sent automatically via GitHub Actions
    """

    if not EMAIL_SENDER or not EMAIL_PASSWORD:
        print("ERROR: Email user or password is not set in environment variables.")
        return

    msg = EmailMessage()
    msg.set_content(body)
    msg['Subject'] = subject
    msg['From'] = EMAIL_SENDER
    msg['To'] = EMAIL_RECEIVER

    context = ssl.create_default_context()
    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465, context=context) as smtp:
            smtp.login(EMAIL_SENDER, EMAIL_PASSWORD)
            smtp.send_message(msg)
        print("Email Sent Successfully")
    except Exception as e:
        print(f"Email Sending Failed: {e}")

if __name__ == "__main__":
    send_email()
