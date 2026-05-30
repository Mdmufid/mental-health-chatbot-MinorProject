# 🧠 Mental Health Chatbot — Minor Project

> An AI-powered mental health companion that detects your emotion in real time and responds with genuine empathy — built as a college minor project using **DistilBERT**, **OpenRouter (LLaMA 3.1 70B)**, **Flask**, and **Gradio**, deployable on **Hugging Face Spaces**.

---

## 📸 Preview

> _Add a screenshot or screen recording of the Gradio UI here to give visitors an instant feel for the app._

---

## 🌟 What Makes This Special

Unlike a standard rule-based chatbot, this system **detects the user's emotional state** before generating a reply. Every response is personalised to how the user actually feels — not just what they say. The local DistilBERT model runs offline (no API cost), while the LLaMA 3.1 70B model via OpenRouter handles the empathetic response generation.

---

## 🏗️ How It Works

```
 User types a message
        │
        ▼
 ┌─────────────────────────────────────┐
 │  Local DistilBERT Emotion Detector  │  ← runs on-device, no API call
 │  emotion_model/models/transformer_  │
 │  Classifies into 6 emotions:        │
 │  joy / love / anger / fear /        │
 │  sadness / neutral                  │
 └──────────────────┬──────────────────┘
                    │  detected emotion
                    ▼
 ┌─────────────────────────────────────┐
 │  OpenRouter API — LLaMA 3.1 70B     │  ← cloud LLM
 │  System: "You are a kind and        │
 │  empathetic mental health chatbot." │
 │  User: "User feels {emotion}.       │
 │  Message: {original message}"       │
 └──────────────────┬──────────────────┘
                    │
                    ▼
 Reply returned as: "(sadness) I hear you..."
```

1. **Emotion Detection** — The user's message is tokenised (max 128 tokens) and passed through a fine-tuned `DistilBertForSequenceClassification` model that classifies it into one of six emotion categories.
2. **Emotion-Conditioned LLM Prompt** — The detected emotion is injected directly into the LLM system prompt, so LLaMA 3.1 70B can tailor its tone and response accordingly.
3. **Conversation Memory** — A rolling `deque(maxlen=5)` keeps the last 5 exchanges in memory for conversational context.
4. **Response** — The chatbot replies with the emotion label prepended, e.g. `(fear) That sounds really overwhelming — let's take this one step at a time.`

---

## ✨ Features

- 🎭 **Real-time emotion detection** — fine-tuned DistilBERT classifying 6 emotions locally
- 🤖 **LLM-powered empathetic responses** — Meta LLaMA 3.1 70B via OpenRouter
- 💡 **Emotion-conditioned prompting** — every reply is shaped by the detected emotion
- 🧠 **Conversation memory** — rolling 5-message context window
- 🌐 **Flask REST backend** — with CORS enabled for API access
- 🖥️ **Gradio web UI** — titled *"🧠 Mental Health Companion"*, no frontend code needed
- ☁️ **Hugging Face Spaces ready** — designed for one-click cloud deployment
- 🔐 **dotenv config** — API key and model path managed via `.env`
- 🎨 **Custom UI** — HTML + CSS frontend in the `app/` folder

---

## 📁 Project Structure

```
mental-health-chatbot-MinorProject/
├── app.py                              # 🚀 Main entry point
│                                       #    Loads DistilBERT model, defines
│                                       #    detect_emotion() & generate_reply(),
│                                       #    launches Gradio interface
│
├── requirements.txt                    # Python dependencies (7 packages)
├── .gitignore
├── README.md
│
├── app/                                # Flask web application
│   ├── __init__.py                     # Flask app factory
│   ├── routes.py                       # HTTP route handlers
│   ├── templates/                      # Jinja2 HTML templates
│   │   └── index.html                  # Main chat UI page
│   └── static/                         # Frontend assets
│       └── css/
│           └── style.css               # Custom styles
│
├── emotion_model/                      # Local DistilBERT model
│   └── models/
│       └── transformer_model/
│           ├── config.json             # Model architecture config
│           ├── pytorch_model.bin       # Fine-tuned weights
│           ├── tokenizer_config.json   # Tokenizer settings
│           ├── vocab.txt               # Vocabulary file
│           └── label_map.json          # {emotion: id} mapping
│
├── data/                               # Training & evaluation datasets
│   └── emotions.csv                    # Labelled emotion dataset
│
└── results/                            # Model training outputs
    ├── training_metrics.json           # Loss, accuracy per epoch
    └── confusion_matrix.png            # Evaluation visualisation
```

---

## 🛠️ Tech Stack

| Layer                  | Technology                                            |
|------------------------|-------------------------------------------------------|
| Emotion Classification | DistilBERT (`DistilBertForSequenceClassification`)    |
| LLM Backend            | Meta LLaMA 3.1 70B Instruct via OpenRouter API        |
| ML Framework           | PyTorch + Hugging Face Transformers                   |
| Web UI                 | Gradio (`gr.Interface`)                               |
| REST Backend           | Flask + Flask-CORS                                    |
| Additional AI          | Google Generative AI (`google-generativeai`)          |
| Config Management      | python-dotenv                                         |
| Deployment             | Hugging Face Spaces                                   |
| Frontend               | HTML5 + CSS3 (custom templates in `app/`)             |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- An [OpenRouter](https://openrouter.ai/) account and API key (free tier available)
- The `emotion_model/` folder with trained DistilBERT weights

### 1. Clone the Repository

```bash
git clone https://github.com/Mdmufid/mental-health-chatbot-MinorProject.git
cd mental-health-chatbot-MinorProject
```

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the project root:

```env
OPENROUTER_API_KEY=your_openrouter_api_key_here
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
MODEL_PATH=emotion_model/models/transformer_model
```

> 🔑 Get your free API key at [openrouter.ai](https://openrouter.ai/). The free tier supports LLaMA 3.1 70B.

### 5. Run the App

```bash
python app.py
```

The Gradio interface will launch at **http://localhost:7860** and open automatically in your browser.

---

## ☁️ Deploy on Hugging Face Spaces

1. Create a new Space at [huggingface.co/spaces](https://huggingface.co/spaces)
2. Set the SDK to **Gradio**
3. Push the repository to your Space:
   ```bash
   git remote add space https://huggingface.co/spaces/<your-username>/<space-name>
   git push space main
   ```
4. Go to **Settings → Repository Secrets** and add:
   - `OPENROUTER_API_KEY`
   - `OPENROUTER_BASE_URL`
   - `MODEL_PATH`

> **Note:** The `emotion_model/` weights must be committed to the repo or downloaded at runtime (e.g. via `gdown` from Google Drive) for the Space to function.

---

## 📦 Dependencies

```
Flask
flask-cors
torch
transformers
python-dotenv
google-generativeai
gradio
```

```bash
pip install -r requirements.txt
```

---

## 🎭 Emotion Classes

The DistilBERT model classifies each message into one of six emotion categories:

| ID | Label     | Example trigger                       |
|----|-----------|---------------------------------------|
| 0  | `joy`     | "I just got great news!"              |
| 1  | `love`    | "I miss my family so much."           |
| 2  | `anger`   | "I'm so frustrated right now."        |
| 3  | `fear`    | "I'm scared about what might happen." |
| 4  | `sadness` | "I've been feeling really low lately."|
| 5  | `neutral` | "Can you tell me more about anxiety?" |

Labels are loaded at runtime from `emotion_model/models/transformer_model/label_map.json`.

---

## 🔒 Environment Variables

| Variable               | Required | Description                                 | Default                          |
|------------------------|----------|---------------------------------------------|----------------------------------|
| `OPENROUTER_API_KEY`   | ✅ Yes   | Your OpenRouter API key                     | —                                |
| `OPENROUTER_BASE_URL`  | ❌ No    | OpenRouter API base URL                     | `https://openrouter.ai/api/v1`   |
| `MODEL_PATH`           | ❌ No    | Path to the local DistilBERT model directory| `emotion_model/models/transformer_model` |

---

## ⚠️ Disclaimer

This chatbot is an **academic minor project** built for educational and research purposes. It is **not** a substitute for professional mental health advice, diagnosis, or treatment.

If you or someone you know is experiencing a mental health crisis, please reach out to a qualified mental health professional or a crisis helpline in your region. In India you can contact **iCall** at 9152987821 or **Vandrevala Foundation** at 1860-2662-345 (24/7).

---

## 🤝 Contributing

Contributions are welcome!

1. Fork the repository
2. Create a new branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m "Add your feature"`
4. Push to your branch: `git push origin feature/your-feature`
5. Open a Pull Request

---

## 📄 License

This project is open source and free to use for educational purposes.

---

## 👤 Author

**Md Mufid Alam**
GitHub: [@Mdmufid](https://github.com/Mdmufid)
