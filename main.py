from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import torch
from torch.nn.functional import softmax
import google.generativeai as genai
import os
from dotenv import load_dotenv
from fastapi.responses import HTMLResponse

load_dotenv()

app = FastAPI(title="Green Minds MCP Server")

# ===== CONFIG =====
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
    global current_key_index
    genai.configure(api_key=GEMINI_API_KEYS[current_key_index])
    print(f"✅ Using Gemini API key #{current_key_index+1}")

def rotate_gemini_key():
    global current_key_index
    current_key_index += 1
    if current_key_index >= len(GEMINI_API_KEYS):
        raise RuntimeError("All Gemini API keys have failed.")
    configure_gemini()

configure_gemini()

# ===== Other API Keys =====
RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY", "fb219e70e5msh16e46246e5d06ebp1b198bjsnd1485b3213b4")

# ===== Lazy Model Loading =====
goemotions_tokenizer = None
goemotions_model = None
sentiment_tokenizer = None
sentiment_model = None
emotions = None

def load_models():
    global goemotions_tokenizer, goemotions_model, sentiment_tokenizer, sentiment_model, emotions
    if goemotions_tokenizer is None:
        print("📥 Loading models for the first time...")
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        goemotions_tokenizer = AutoTokenizer.from_pretrained("monologg/bert-base-cased-goemotions-original")
        goemotions_model = AutoModelForSequenceClassification.from_pretrained("monologg/bert-base-cased-goemotions-original")
        sentiment_tokenizer = AutoTokenizer.from_pretrained("finiteautomata/bertweet-base-sentiment-analysis")
        sentiment_model = AutoModelForSequenceClassification.from_pretrained("finiteautomata/bertweet-base-sentiment-analysis")
        emotions_url = "https://raw.githubusercontent.com/google-research/google-research/master/goemotions/data/emotions.txt"
        emotions = requests.get(emotions_url).text.strip().split('\n')
        print("✅ Models loaded successfully.")

def detect_emotion(text):
    load_models()
    inputs = goemotions_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = goemotions_model(**inputs)
    probs = softmax(outputs.logits, dim=1)
    top_probs, top_ids = torch.topk(probs, 3)
    return [(emotions[i], float(top_probs[0][idx].detach())) for idx, i in enumerate(top_ids[0])]

def detect_mental_state(text):
    load_models()
    inputs = sentiment_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    outputs = sentiment_model(**inputs)
    probs = softmax(outputs.logits, dim=1)
    neg_prob = probs[0][0].item()
    if neg_prob > 0.5 or torch.argmax(probs) == 0:
        return "depressed", neg_prob
    else:
        non_dep_prob = probs[0][1].item() + probs[0][2].item()
        return "non-depressed", non_dep_prob

def get_daily_wisdom():
    url = "https://bhagavad-gita3.p.rapidapi.com/v2/chapters/2/verses/47/"
    headers = {
        "X-RapidAPI-Key": RAPIDAPI_KEY,
        "X-RapidAPI-Host": "bhagavad-gita3.p.rapidapi.com"
    }
    r = requests.get(url, headers=headers)
    if r.status_code == 200:
        return r.json()
    else:
        return {"error": "Could not fetch shlok"}

def generate_ai_story(mood):
    for attempt in range(len(GEMINI_API_KEYS)):
        try:
            model = genai.GenerativeModel("gemini-1.5-flash")
            prompt = f"Write a comforting, uplifting short story (~300 words) for someone feeling {mood}. Keep it positive and encouraging."
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f"⚠️ Gemini API key #{current_key_index+1} failed: {e}")
            try:
                rotate_gemini_key()
            except RuntimeError as re:
                return f"Error: {str(re)}"
    return f"Error: All Gemini API keys failed for mood '{mood}'."

# ===== Schemas =====
class MoodRequest(BaseModel):
    text: str

class StoryRequest(BaseModel):
    mood: str

# ===== Root Endpoint for Web Browsers =====
@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
        <head>
            <title>Green Minds MCP API</title>
        </head>
        <body style="font-family: Arial; margin: 20px;">
            <h1>🌱 Green Minds MCP API</h1>
            <p>Welcome! This API provides mood analysis, AI-generated stories, and daily wisdom.</p>
            <h2>Available Endpoints</h2>
            <ul>
                <li><b>GET</b> <code>/healthz</code> → Health check</li>
                <li><b>POST</b> <code>/analyze_mood</code> → Analyze mood (JSON: {"text": "your text"})</li>
                <li><b>POST</b> <code>/get_ai_story</code> → Get AI-generated story (JSON: {"mood": "happy"})</li>
                <li><b>GET</b> <code>/get_daily_wisdom</code> → Get Bhagavad Gita wisdom</li>
            </ul>
            <h3>Example Usage</h3>
            <pre>
curl -X POST https://green-minds-mcp.onrender.com/analyze_mood \\
-H "Content-Type: application/json" \\
-d '{"text": "I feel great today"}'
            </pre>
        </body>
    </html>
    """

@app.get("/healthz")
def health_check():
    return {"status": "ok"}

@app.post("/analyze_mood")
def analyze_mood(req: MoodRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text input cannot be empty.")
    emotions_list = detect_emotion(req.text)
    mental_state, confidence = detect_mental_state(req.text)
    return {
        "emotions": emotions_list,
        "mental_state": mental_state,
        "confidence": confidence
    }

@app.post("/get_ai_story")
def get_ai_story(req: StoryRequest):
    if not req.mood.strip():
        raise HTTPException(status_code=400, detail="Mood cannot be empty.")
    story = generate_ai_story(req.mood)
    return {"story": story}

@app.get("/get_daily_wisdom")
def daily_wisdom():
    return get_daily_wisdom()
