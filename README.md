# 🌱 Green Minds MCP Server

**Hackathon Project for Puch AI x OpenAI — Phase 1: Be Wozniak**

Green Minds MCP Server is an AI-powered **Mental Wellness & Mood Analysis MCP Server** built to help people understand and improve their emotional well-being.
It detects emotions, identifies mental states, shares uplifting AI-generated stories, and provides daily wisdom — all through simple API endpoints.

---

## Try API AT https://the-arthur-morgan-green-minds-mcp.hf.space/

## 🚀 Features

* **Mood & Emotion Analysis**
  Detects top emotions from text using the **GoEmotions** model.

* **Mental State Detection**
  Classifies if the user is in a *depressed* or *non-depressed* state.

* **AI Story Generation**
  Uses **Gemini API** to create uplifting, mood-specific short stories.

* **Daily Wisdom API**
  Fetches motivational verses from the Bhagavad Gita.

* **Multiple API Key Support**
  Automatically rotates through multiple Gemini API keys if one fails.

---

## 📦 Tech Stack

* **FastAPI** — REST API framework
* **Hugging Face Transformers** — for NLP models
* **PyTorch** — for model inference
* **Google Gemini API** — AI story generation
* **Requests** — API calls for wisdom data

---

## 📂 API Endpoints

### `POST /analyze_mood`

Analyze emotions & mental state.

**Example request:**

```json
{
  "text": "I am feeling amazing today!"
}
```

**Example response:**

```json
{
  "emotions": [["excitement", 0.71], ["admiration", 0.26], ["joy", 0.006]],
  "mental_state": "non-depressed",
  "confidence": 0.998
}
```

---

### `POST /get_ai_story`

Generate a mood-specific uplifting story.

**Example request:**

```json
{
  "mood": "feeling sad"
}
```

**Example response:**

```json
{
  "story": "Once upon a time..."
}
```

---

### `GET /get_daily_wisdom`

Returns daily wisdom from the Bhagavad Gita.

**Example response:**

```json
{
  "verse": "You have the right to work, but never to the fruit of work..."
}
```

---

## ⚙️ Installation & Run Locally

### 1️⃣ Clone the Repo

```bash
git clone https://github.com/itripathiharsh/Green-Minds-MCP.git
cd Green-Minds-MCP
```

### 2️⃣ Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Set Environment Variables

Create a `.env` file in the root folder:

```env
GEMINI_API_KEY_1=your_key_here
GEMINI_API_KEY_2=backup_key_here
RAPIDAPI_KEY=your_rapidapi_key_here
```

### 5️⃣ Run Server

```bash
uvicorn main:app --reload
```

Swagger Docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## 🎯 Hackathon Impact

Green Minds MCP Server can be integrated into:

* 🧠 **Mental Health Chatbots**
* 💬 **Therapy Companion Apps**
* 📊 **Wellness Tracking Platforms**
* 📜 **Motivational Content Generators**

By providing AI-powered emotional insights & instant uplifting content, it can support mental wellness at scale.

---


## 👨‍💻 Author

**Harsh Vardhan Tripathi**
GitHub: [@itripathiharsh](https://github.com/itripathiharsh)

