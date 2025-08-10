from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import requests
import torch
from torch.nn.functional import softmax
import google.generativeai as genai
import os
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# =========================
# Load environment variables
# =========================
load_dotenv()

# =========================
# FastAPI App
# =========================
app = FastAPI(
    title="Green Minds MCP Server",
    description="AI-powered mood & mental wellness API for the Puch AI x OpenAI Hackathon.",
    version="2.2.0" # Final Fix Version
)

# =========================
# API Keys Configuration
# =========================
GEMINI_API_KEYS = [key for key in [os.getenv("GEMINI_API_KEY_1"), os.getenv("GEMINI_API_KEY_2")] if key]
if not GEMINI_API_KEYS: raise RuntimeError("No Gemini API keys found.")
genai.configure(api_key=GEMINI_API_KEYS[0])
print(f"✅ Using Gemini API key #1")

RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY")

# =========================
# Global Model Variables
# =========================
goemotions_tokenizer, goemotions_model, sentiment_tokenizer, sentiment_model, emotions = None, None, None, None, None

# =========================
# Startup Event - Load Models
# =========================
@app.on_event("startup")
def load_models():
    """Load ML models at app startup."""
    global goemotions_tokenizer, goemotions_model, sentiment_tokenizer, sentiment_model, emotions
    print("📥 Loading models on startup...")
    goemotions_model_name = "mrm8488/distilroberta-finetuned-go_emotions"
    sentiment_model_name = "distilbert-base-uncased-finetuned-sst-2-english"

    goemotions_tokenizer = AutoTokenizer.from_pretrained(goemotions_model_name)
    goemotions_model = AutoModelForSequenceClassification.from_pretrained(goemotions_model_name)
    sentiment_tokenizer = AutoTokenizer.from_pretrained(sentiment_model_name)
    sentiment_model = AutoModelForSequenceClassification.from_pretrained(sentiment_model_name)

    emotions_url = "https://raw.githubusercontent.com/google-research/google-research/master/goemotions/data/emotions.txt"
    emotions = requests.get(emotions_url).text.strip().split('\n')
    print("✅ Models loaded successfully.")

# =========================
# Request & Response Schemas
# =========================
class MoodRequest(BaseModel):
    text: str

class StoryRequest(BaseModel):
    mood: str

class BearerToken(BaseModel):
    bearer_token: str

# =========================
# API Endpoints
# =========================
# THIS IS THE FIX: The root endpoint now accepts both GET and POST requests.
@app.get("/", response_class=HTMLResponse, include_in_schema=False)
@app.post("/", response_class=HTMLResponse, include_in_schema=False)
def home():
    """A simple HTML landing page that handles GET and POST for verification."""
    return """
    <html>
        <head><title>Green Minds MCP</title>
            <style>
                body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; text-align: center; background: #f0fdf4; padding: 40px; color: #333; }
                .container { max-width: 600px; margin: auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
                h1 { color: #16a34a; } p { font-size: 1.1em; color: #555; }
                a { display: inline-block; margin-top: 20px; padding: 12px 25px; background: #22c55e; color: white; text-decoration: none; border-radius: 5px; font-weight: bold; transition: background-color 0.3s; }
                a:hover { background: #16a34a; }
            </style>
        </head>
        <body><div class="container">
            <h1>🌿 Green Minds MCP Server</h1>
            <p>Your AI-powered mood & mental wellness companion is running.</p>
            <a href="/docs">🚀 View API Documentation</a>
        </div></body>
    </html>
    """

@app.post("/validate", tags=["MCP Compliance"])
async def validate_server(token: BearerToken):
    """Mandatory endpoint for the Puch MCP platform to verify server ownership."""
    YOUR_PHONE_NUMBER = "919876543210" # <-- IMPORTANT: REPLACE WITH YOURS
    print(f"Received validation request with token: {token.bearer_token}")
    return {"phone_number": YOUR_PHONE_NUMBER}

@app.get("/healthz", tags=["Utilities"])
def health_check():
    """A simple health check endpoint to confirm the server is running."""
    return {"status": "ok"}

@app.post("/analyze_mood", tags=["Core AI Tools"])
def analyze_mood(req: MoodRequest):
    """Analyzes text to determine emotions and mental state."""
    if not req.text.strip(): raise HTTPException(status_code=400, detail="Text cannot be empty.")
    inputs = goemotions_tokenizer(req.text, return_tensors="pt", truncation=True, padding=True)
    outputs = goemotions_model(**inputs)
    probs = softmax(outputs.logits, dim=1)
    top_probs, top_ids = torch.topk(probs, 3)
    emotions_list = [(emotions[i], float(top_probs[0][idx].detach())) for idx, i in enumerate(top_ids[0])]
    
    inputs = sentiment_tokenizer(req.text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    outputs = sentiment_model(**inputs)
    probs = softmax(outputs.logits, dim=1)
    if torch.argmax(probs) == 0:
        mental_state, confidence = "depressed", probs[0][0].item()
    else:
        mental_state, confidence = "non-depressed", probs[0][1].item()
        
    return {"emotions": emotions_list, "mental_state": mental_state, "confidence": confidence}

@app.post("/get_ai_story", tags=["Core AI Tools"])
def get_ai_story(req: StoryRequest):
    """Generates a short motivational story based on mood."""
    if not req.mood.strip(): raise HTTPException(status_code=400, detail="Mood cannot be empty.")
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        prompt = f"Write a comforting, uplifting short story (~300 words) for someone feeling {req.mood}. Keep it positive and encouraging."
        response = model.generate_content(prompt)
        return {"story": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate story: {e}")

@app.get("/get_daily_wisdom", tags=["Core AI Tools"])
def daily_wisdom():
    """Fetches a daily verse from the Bhagavad Gita."""
    if not RAPIDAPI_KEY: return {"error": "RapidAPI key is not configured."}
    url = "https://bhagavad-gita3.p.rapidapi.com/v2/chapters/2/verses/47/"
    headers = {"X-RapidAPI-Key": RAPIDAPI_KEY, "X-RapidAPI-Host": "bhagavad-gita3.p.rapidapi.com"}
    try:
        r = requests.get(url, headers=headers)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Could not fetch shlok: {e}")