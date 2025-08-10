from fastapi import FastAPI, HTTPException, Security, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
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
    version="2.3.0"
)

# =========================
# Security Scheme
# =========================
security = HTTPBearer()

# =========================
# API Keys Configuration
# =========================
GEMINI_API_KEYS = [key for key in [os.getenv("GEMINI_API_KEY_1"), os.getenv("GEMINI_API_KEY_2")] if key]
if not GEMINI_API_KEYS:
    raise RuntimeError("No Gemini API keys found.")
genai.configure(api_key=GEMINI_API_KEYS[0])
print(f"✅ Using Gemini API key #1")

RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY")

# =========================
# Global Model Variables (Lazy Load)
# =========================
emotion_model = None
emotion_tokenizer = None
sentiment_model = None
sentiment_tokenizer = None

# =========================
# Helper Functions - Lazy Load Models
# =========================
def get_emotion_model():
    global emotion_model, emotion_tokenizer
    if emotion_model is None or emotion_tokenizer is None:
        print("📥 Loading emotion model on demand...")
        model_name = "bhadresh-savani/distilbert-base-uncased-emotion"
        emotion_tokenizer = AutoTokenizer.from_pretrained(model_name)
        emotion_model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return emotion_model, emotion_tokenizer


def get_sentiment_model():
    global sentiment_model, sentiment_tokenizer
    if sentiment_model is None or sentiment_tokenizer is None:
        print("📥 Loading sentiment model on demand...")
        model_name = "distilbert-base-uncased-finetuned-sst-2-english"
        sentiment_tokenizer = AutoTokenizer.from_pretrained(model_name)
        sentiment_model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return sentiment_model, sentiment_tokenizer


def detect_emotion(text: str):
    model, tokenizer = get_emotion_model()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    probs = softmax(outputs.logits, dim=1)
    top_probs, top_ids = torch.topk(probs, 3)
    labels = model.config.id2label
    return [(labels[i.item()], top_probs[0][idx].item()) for idx, i in enumerate(top_ids[0])]


def detect_mental_state(text: str):
    model, tokenizer = get_sentiment_model()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    outputs = model(**inputs)
    probs = softmax(outputs.logits, dim=1)
    if torch.argmax(probs) == 0:
        return "depressed", probs[0][0].item()
    else:
        return "non-depressed", probs[0][1].item()


# =========================
# Request & Response Schemas
# =========================
class MoodRequest(BaseModel):
    text: str


class StoryRequest(BaseModel):
    mood: str


# =========================
# API Endpoints
# =========================
@app.get("/", response_class=HTMLResponse, include_in_schema=False)
@app.post("/", response_class=HTMLResponse, include_in_schema=False)
def home():
    """Simple landing page."""
    return """
    <html>
        <head>
            <title>Green Minds MCP</title>
            <style>
                body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif; 
                       text-align: center; background: #f0fdf4; padding: 40px; color: #333; }
                .container { max-width: 600px; margin: auto; background: white; padding: 30px; border-radius: 10px; 
                             box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
                h1 { color: #16a34a; } 
                p { font-size: 1.1em; color: #555; }
                a { display: inline-block; margin-top: 20px; padding: 12px 25px; background: #22c55e; color: white; 
                    text-decoration: none; border-radius: 5px; font-weight: bold; transition: background-color 0.3s; }
                a:hover { background: #16a34a; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🌿 Green Minds MCP Server</h1>
                <p>Your AI-powered mood & mental wellness companion is running.</p>
                <a href="/docs">🚀 View API Documentation</a>
            </div>
        </body>
    </html>
    """


@app.get("/health", include_in_schema=False)
def health():
    """Fast health check endpoint for bots."""
    return {"status": "ok", "server": "Green Minds MCP", "version": "2.3.0"}


@app.post("/validate", tags=["MCP Compliance"])
async def validate_server(credentials: HTTPAuthorizationCredentials = Security(security)):
    """
    MCP validation endpoint. Will now respond instantly.
    """
    token = credentials.credentials.strip()

    # Optional: Load from env for better security
    ALLOWED_TOKENS = ["mytoken123"]  # Replace with os.getenv("VALID_TOKENS").split(",") if needed

    if token not in ALLOWED_TOKENS:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return {"phone_number": "919565249247"}


@app.post("/analyze_mood", tags=["Core AI Tools"])
def analyze_mood(req: MoodRequest):
    """Analyzes text to determine emotions and mental state."""
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
    
    emotions_list = detect_emotion(req.text)
    mental_state, confidence = detect_mental_state(req.text)
    
    return {
        "emotions": emotions_list,
        "mental_state": mental_state,
        "confidence": confidence
    }


@app.post("/get_ai_story", tags=["Core AI Tools"])
def get_ai_story(req: StoryRequest):
    """Generates a short motivational story based on mood."""
    if not req.mood.strip():
        raise HTTPException(status_code=400, detail="Mood cannot be empty.")
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
    if not RAPIDAPI_KEY:
        raise HTTPException(status_code=500, detail="RapidAPI key is not configured.")
    
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
        raise HTTPException(status_code=500, detail=f"Could not fetch shloka: {e}")
