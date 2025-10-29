import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
import os

# --- Load processed training data ---
train_path = r"D:\MentalHealthChatbot\data\processed\train.csv"
df = pd.read_csv(train_path)

# Clean dataset
df = df.dropna(subset=["text", "label"])
df["text"] = df["text"].astype(str)

print("Columns:", df.columns)
print("Sample rows:")
print(df.head())

# --- Train model ---
print("🚀 Training Logistic Regression model...")
X = df["text"]
y = df["label"]

vectorizer = TfidfVectorizer(max_features=5000)
X_tfidf = vectorizer.fit_transform(X)

model = LogisticRegression(max_iter=1000)
model.fit(X_tfidf, y)

# --- Save model ---
os.makedirs("emotion_model/models", exist_ok=True)
joblib.dump((vectorizer, model), "emotion_model/models/baseline_tfidf_lr.pkl")

print("✅ Model trained successfully!")
print("💾 Model saved to emotion_model/models/baseline_tfidf_lr.pkl")



# import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import classification_report
# import joblib
# import json
# from pathlib import Path

# # Paths
# train_path = Path("data/processed/train.csv")
# val_path   = Path("data/processed/val.csv")
# model_dir  = Path("emotion_model/models")
# results_dir = Path("results")
# model_dir.mkdir(parents=True, exist_ok=True)
# results_dir.mkdir(parents=True, exist_ok=True)

# print("📂 Loading data...")
# train_df = pd.read_csv(train_path)
# val_df   = pd.read_csv(val_path)

# # 🧹 Clean NaN or empty rows
# train_df = train_df.dropna(subset=["text", "label"])
# val_df = val_df.dropna(subset=["text", "label"])
# train_df = train_df[train_df["text"].str.strip() != ""]
# val_df = val_df[val_df["text"].str.strip() != ""]

# # ----------------------------
# # TEXT VECTORISATION
# # ----------------------------
# print("⚙️  Converting text to TF-IDF vectors...")
# vectorizer = TfidfVectorizer(max_features=15000, ngram_range=(1,2))
# X_train = vectorizer.fit_transform(train_df["text"])
# X_val   = vectorizer.transform(val_df["text"])
# y_train = train_df["label"]
# y_val   = val_df["label"]

# # ----------------------------
# # TRAINING
# # ----------------------------
# print("🚀 Training Logistic Regression...")
# clf = LogisticRegression(max_iter=1000, n_jobs=-1)
# clf.fit(X_train, y_train)

# # ----------------------------
# # EVALUATION
# # ----------------------------
# print("📊 Evaluating...")
# y_pred = clf.predict(X_val)
# report = classification_report(y_val, y_pred, output_dict=True)
# print(classification_report(y_val, y_pred))

# # Save metrics and model
# with open(results_dir / "baseline_metrics.json", "w") as f:
#     json.dump(report, f, indent=2)
# joblib.dump((vectorizer, clf), model_dir / "baseline_tfidf_lr.pkl")

# print("\n✅ Model trained and saved to emotion_model/models/baseline_tfidf_lr.pkl")
# print("📄 Evaluation metrics saved to results/baseline_metrics.json")
