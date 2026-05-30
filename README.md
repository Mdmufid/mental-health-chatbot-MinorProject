# Mental Health Chatbot

🧠 Mental Health Chatbot is a Flask-based AI companion that combines a local emotion model with OpenRouter for supportive replies.

## Features

- Flask web app with a chat UI
- Local emotion detection with DistilBERT
- Safety filters for crisis and explicit content
- Conversation memory for short context

## Run Locally

1. Install dependencies:

	```bash
	python -m pip install -r requirements.txt
	```

2. Add your API key in `.env`:

	```env
	OPENROUTER_API_KEY=your_key_here
	OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
	MODEL_PATH=emotion_model/models/transformer_model
	```

3. Start the app:

	```bash
	python app/chatbot_api.py
	```

4. Open `http://127.0.0.1:5000` in your browser.

## Project Structure

- `app/chatbot_api.py` - main Flask chatbot app
- `app/templates/index.html` - chat UI template
- `app/static/style.css` - frontend styling
- `emotion_model/models/transformer_model` - saved emotion model
- `emotion_model/train_transformer.py` - transformer training script
- `emotion_model/train_baseline.py` - baseline training script
- `emotion_model/preprocess.py` - dataset preprocessing

## Notes

- The project currently runs from `app/chatbot_api.py`.
- `app.py` was an older launcher and has been removed.
