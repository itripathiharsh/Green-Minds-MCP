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
    version="1.1.0"
)

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

    # Using lighter, distilled models to prevent memory errors on free hosting tiers
    goemotions_model_name = "mrm8488/distilroberta-finetuned-go_emotions"
    sentiment_model_name = "distilbert-base-uncased-finetuned-sst-2-english"

    # GoEmotions Model
    goemotions_tokenizer = AutoTokenizer.from_pretrained(goemotions_model_name)
    goemotions_model = AutoModelForSequenceClassification.from_pretrained(goemotions_model_name)

    # Sentiment Analysis Model
    sentiment_tokenizer = AutoTokenizer.from_pretrained(sentiment_model_name)
    sentiment_model = AutoModelForSequenceClassification.from_pretrained(sentiment_model_name)

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
    
    # For 'distilbert-base-uncased-finetuned-sst-2-english', the labels are: 0 -> NEGATIVE, 1 -> POSITIVE
    neg_prob = probs[0][0].item()
    pos_prob = probs[0][1].item()
    
    if torch.argmax(probs) == 0: # If Negative is the highest probability
        return "depressed", neg_prob
    else:
        return "non-depressed", pos_prob

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
# API Endpoints
# =========================
@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def home():
    """A simple HTML landing page for the API root."""
    return """
    <html>
        <head>
            <title>Green Minds MCP</title>
            <style>
                body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; text-align: center; background: #f0fdf4; padding: 40px; color: #333; }
                .container { max-width: 600px; margin: auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
                h1 { color: #16a34a; }
                p { font-size: 1.1em; color: #555; }
                a { display: inline-block; margin-top: 20px; padding: 12px 25px; background: #22c55e; color: white; text-decoration: none; border-radius: 5px; font-weight: bold; transition: background-color 0.3s; }
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

@app.get("/healthz", tags=["Utilities"])
def health_check():
    """A simple health check endpoint to confirm the server is running."""
    return {"status": "ok"}

@app.post("/analyze_mood", tags=["Core AI Tools"])
def analyze_mood(req: MoodRequest):
    """Analyzes text to determine emotions and mental state."""
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
    emotions_list = detect_emotion(text)
    mental_state, confidence = detect_mental_state(text)
    return {"emotions": emotions_list, "mental_state": mental_state, "confidence": confidence}

@app.post("/get_ai_story", tags=["Core AI Tools"])
def get_ai_story(req: StoryRequest):
    """Generates a short motivational story based on mood."""
    mood = req.mood.strip()
    if not mood:
        raise HTTPException(status_code=400, detail="Mood cannot be empty.")
    story = generate_ai_story(mood)
    return {"story": story}

@app.get("/get_daily_wisdom", tags=["Core AI Tools"])
def daily_wisdom():
    """Fetches a daily verse from the Bhagavad Gita."""
    return get_daily_wisdom()