# =====================================================
# 🧠 Advanced AI Mental Health Chatbot (OpenRouter Integration)
# =====================================================

import os
import sys


_python313 = r"C:\Users\HP\AppData\Local\Programs\Python\Python313\python.exe"
if os.path.exists(_python313) and os.path.abspath(sys.executable).lower() != _python313.lower():
    os.execv(_python313, [_python313, os.path.abspath(__file__), *sys.argv[1:]])


import json, re, requests
from collections import deque
from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template
try:
    from flask_cors import CORS
except ModuleNotFoundError:
    def CORS(app, *args, **kwargs):
        return app
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

# =====================================================
# 1️⃣ Load Environment Variables
# =====================================================
load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
MODEL_PATH = os.getenv("MODEL_PATH", "emotion_model/models/transformer_model")

if not OPENROUTER_API_KEY:
    raise ValueError("❌ No OpenRouter API key found in .env file")

print("🔑 OpenRouter API Key Loaded Successfully!")

# =====================================================
# 2️⃣ Flask Setup
# =====================================================
app = Flask(__name__)
CORS(app)

# =====================================================
# 3️⃣ Load Emotion Detection Model
# =====================================================
print("🔄 Loading Emotion Detection Model...")
tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_PATH)
emo_model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
emo_model.eval()
print("✅ Emotion Detection Model Loaded Successfully!")

# Load label map if available
label_map_path = os.path.join(MODEL_PATH, "label_map.json")
if os.path.exists(label_map_path):
    with open(label_map_path, "r") as f:
        label_map = json.load(f)
    id_to_label = {int(v): k for k, v in label_map.items()}
else:
    id_to_label = {0: "joy", 1: "love", 2: "anger", 3: "fear", 4: "sadness", 5: "neutral"}

# Short-term conversation memory
conversation_memory = deque(maxlen=8)

# =====================================================
# 4️⃣ Emotion Mapping
# =====================================================
BROAD_MAP = {
    "admiration": "joy", "amusement": "joy", "approval": "joy", "gratitude": "joy", "joy": "joy", "relief": "joy",
    "disappointment": "sadness", "grief": "sadness", "remorse": "sadness", "sadness": "sadness",
    "anger": "anger", "annoyance": "anger", "disgust": "anger", "disapproval": "anger",
    "nervousness": "fear", "fear": "fear", "anxiety": "fear",
    "love": "love", "caring": "love", "desire": "love",
    "curiosity": "neutral", "realization": "neutral", "neutral": "neutral", "surprise": "neutral", "confusion": "neutral"
}

def map_fine_to_broad(fine):
    return BROAD_MAP.get(fine, "neutral")

def detect_emotion(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        logits = emo_model(**inputs).logits
        pred = int(torch.argmax(logits, dim=-1).cpu().numpy()[0])
    fine = id_to_label.get(pred, "neutral")
    broad = map_fine_to_broad(fine)
    return broad, fine

# =====================================================
# 5️⃣ Safety Filters
# =====================================================
def is_crisis(text):
    t = text.lower()
    return bool(re.search(r"\b(suicide|kill myself|end my life|want to die|self harm)\b", t))

def is_explicit(text):
    t = text.lower()
    return bool(re.search(r"\b(horny|sex|porn|nude|fuck|dick|boobs)\b", t))

# =====================================================
# 6️⃣ OpenRouter LLM Response Generator
# =====================================================
def generate_llm_reply(message, emotion, fine_emotion, context):
    system_prompt = (
        "You are an empathetic AI mental health companion. "
        "Provide emotional support, motivation, and practical self-care guidance. "
        "Avoid repetition and medical or explicit advice."
    )

    convo_context = "\n".join([f"User: {m['user']}\nBot: {m['bot']}" for m in context])

    user_prompt = (
        f"Recent conversation:\n{convo_context}\n\n"
        f"User message: {message}\n"
        f"Detected emotion: {emotion} ({fine_emotion})\n\n"
        f"Reply with empathy, 2–4 sentences, offering comfort and practical suggestions."
    )

    payload = {
        "model": "meta-llama/llama-3.1-70b-instruct",  # ✅ you can swap with gpt-4o-mini or claude-3-haiku
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.8,
        "max_tokens": 300
    }

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "HTTP-Referer": "http://localhost:5000",
        "X-Title": "AI Mental Health Companion",
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(f"{BASE_URL}/chat/completions", json=payload, headers=headers, timeout=30)
        data = response.json()
        reply_text = data["choices"][0]["message"]["content"].strip()
        return reply_text
    except Exception as e:
        print("❌ OpenRouter Error:", e)
        return f"⚠️ LLM Error: {str(e)}"

# =====================================================
# 7️⃣ Generate Chatbot Response
# =====================================================
def generate_response(user_message):
    if is_crisis(user_message):
        return {
            "emotion": "crisis",
            "reply": (
                "I'm really sorry you're feeling this way. You're not alone. "
                "Please reach out right now — in India, call AASRA at 91-9820466726 "
                "or visit findahelpline.com ❤️"
            ),
        }

    if is_explicit(user_message):
        return {
            "emotion": "sensitive",
            "reply": (
                "I can’t discuss explicit topics, but I can help you talk about your emotions safely. "
                "Would you like to tell me what’s bothering you?"
            ),
        }

    broad_emotion, fine_emotion = detect_emotion(user_message)
    context = list(conversation_memory)[-5:]
    reply = generate_llm_reply(user_message, broad_emotion, fine_emotion, context)

    conversation_memory.append({"user": user_message, "bot": reply})

    return {"emotion": broad_emotion, "reply": reply}

# =====================================================
# 8️⃣ Flask Endpoints
# =====================================================
@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True)
    msg = data.get("message", "").strip()
    if not msg:
        return jsonify({"error": "Message cannot be empty"}), 400
    result = generate_response(msg)
    return jsonify(result)

@app.route("/")
def index():
    return render_template("index.html")

# =====================================================
# 9️⃣ Run Server
# =====================================================
if __name__ == "__main__":
    app.run(debug=True)
