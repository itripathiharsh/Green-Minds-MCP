from fastapi import FastAPI, HTTPException, Query, Request
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
app = FastAPI(title="Green Minds MCP Server", description="AI-powered mood & mental wellness API", version="1.0")

# =========================
# Gemini API Configuration
# =========================
GEMINI_API_KEYS = [
    os.getenv("GEMINI_API_KEY_1"),
    os.getenv("GEMINI_API_KEY_2"),
    os.getenv("GEMINI_API_KEY_3"),
]
GEMINI_API_KEYS = [key for key in GEMINI_API_KEYS if key]
if not GEMINI_API_KEYS:
    raise RuntimeError("No Gemini API keys found.")

current_key_index = 0

def configure_gemini():
    """Set the active Gemini API key."""
    global current_key_index
    try:
        genai.configure(api_key=GEMINI_API_KEYS[current_key_index])
        print(f"✅ Using Gemini API key #{current_key_index + 1}")
    except IndexError:
        raise RuntimeError("Invalid Gemini API key index.")

def rotate_gemini_key():
    """Rotate to the next Gemini API key if one fails."""
    global current_key_index
    current_key_index += 1
    if current_key_index >= len(GEMINI_API_KEYS):
        raise RuntimeError("All Gemini API keys have failed.")
    configure_gemini()

# Initial configuration
configure_gemini()

# =========================
# Other API Keys
# =========================
RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY")

# =========================
# Global Model Variables
# =========================
goemotions_tokenizer = None
goemotions_model = None
sentiment_tokenizer = None
sentiment_model = None
emotions = None

# =========================
# Startup Event - Load Models
# =========================
@app.on_event("startup")
def load_models():
    """Load ML models at app startup."""
    global goemotions_tokenizer, goemotions_model, sentiment_tokenizer, sentiment_model, emotions
    print("📥 Loading models on startup...")

    # GoEmotions Model
    goemotions_tokenizer = AutoTokenizer.from_pretrained("monologg/bert-base-cased-goemotions-original")
    goemotions_model = AutoModelForSequenceClassification.from_pretrained("monologg/bert-base-cased-goemotions-original")

    # Sentiment Analysis Model
    sentiment_tokenizer = AutoTokenizer.from_pretrained("finiteautomata/bertweet-base-sentiment-analysis")
    sentiment_model = AutoModelForSequenceClassification.from_pretrained("finiteautomata/bertweet-base-sentiment-analysis")

    # Emotion labels
    emotions_url = "https://raw.githubusercontent.com/google-research/google-research/master/goemotions/data/emotions.txt"
    emotions_response = requests.get(emotions_url)
    emotions_response.raise_for_status()
    emotions = emotions_response.text.strip().split('\n')

    print("✅ Models loaded successfully.")

# =========================
# Helper Functions
# =========================
def detect_emotion(text: str):
    """Detect top 3 emotions from text."""
    inputs = goemotions_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = goemotions_model(**inputs)
    probs = softmax(outputs.logits, dim=1)
    top_probs, top_ids = torch.topk(probs, 3)
    return [(emotions[i], float(top_probs[0][idx].detach())) for idx, i in enumerate(top_ids[0])]

def detect_mental_state(text: str):
    """Classify mental state as depressed or non-depressed."""
    inputs = sentiment_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    outputs = sentiment_model(**inputs)
    probs = softmax(outputs.logits, dim=1)
    neg_prob = probs[0][0].item()

    if torch.argmax(probs) == 0:
        return "depressed", neg_prob
    else:
        non_dep_prob = probs[0][1].item() + probs[0][2].item()
        return "non-depressed", non_dep_prob

def get_daily_wisdom():
    """Fetch a Bhagavad Gita verse via RapidAPI."""
    if not RAPIDAPI_KEY:
        return {"error": "RapidAPI key is not configured."}
    url = "https://bhagavad-gita3.p.rapidapi.com/v2/chapters/2/verses/47/"
    headers = {
        "X-RapidAPI-Key": RAPIDAPI_KEY,
        "X-RapidAPI-Host": "bhagavad-gita3.p.rapidapi.com"
    }
    try:
        r = requests.get(url, headers=headers)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.RequestException as e:
        return {"error": f"Could not fetch shlok: {e}"}

def generate_ai_story(mood: str):
    """Generate an uplifting short story using Gemini."""
    for _ in range(len(GEMINI_API_KEYS)):
        try:
            model = genai.GenerativeModel("gemini-1.5-flash")
            prompt = f"Write a comforting, uplifting short story (~300 words) for someone feeling {mood}. Keep it positive and encouraging."
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f"⚠️ Gemini API key #{current_key_index + 1} failed: {e}")
            try:
                rotate_gemini_key()
            except RuntimeError as re:
                return f"Error: {str(re)}"
    return f"Error: All Gemini API keys failed for mood '{mood}'."

# =========================
# Request Schemas
# =========================
class MoodRequest(BaseModel):
    text: str

class StoryRequest(BaseModel):
    mood: str

# =========================
# Routes
# =========================
@app.get("/", response_class=HTMLResponse)
def home():
    """Landing page for the API."""
    return """
    <html>
        <head>
            <title>Green Minds MCP</title>
            <style>
                body { font-family: Arial; text-align: center; background: #f4fff7; padding: 50px; }
                h1 { color: #2e8b57; }
                p { font-size: 18px; }
                a { padding: 10px 20px; background: #2e8b57; color: white; text-decoration: none; border-radius: 5px; }
                a:hover { background: #246b46; }
            </style>
        </head>
        <body>
            <h1>🌿 Green Minds MCP</h1>
            <p>Your AI-powered mood & mental wellness companion.</p>
            <a href="/docs">🚀 Try the API</a>
        </body>
    </html>
    """

@app.get("/healthz")
def health_check():
    """Health check endpoint."""
    return {"status": "ok"}

@app.post("/analyze_mood")
def analyze_mood(req: MoodRequest):
    """Analyze emotions & mental state from input text."""
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
    emotions_list = detect_emotion(text)
    mental_state, confidence = detect_mental_state(text)
    return {"emotions": emotions_list, "mental_state": mental_state, "confidence": confidence}

@app.post("/get_ai_story")
def get_ai_story(req: StoryRequest):
    """Generate a short motivational story based on mood."""
    mood = req.mood.strip()
    if not mood:
        raise HTTPException(status_code=400, detail="Mood cannot be empty.")
    story = generate_ai_story(mood)
    return {"story": story}

@app.get("/get_daily_wisdom")
def daily_wisdom():
    """Fetch a daily verse from the Bhagavad Gita."""
    return get_daily_wisdom()
