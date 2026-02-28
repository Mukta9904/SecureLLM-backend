import pandas as pd
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from datasets import load_dataset

# --- CONFIGURATION ---
MODEL_DIR = "backend/models"
os.makedirs(MODEL_DIR, exist_ok=True)

print("⏳ 1. Loading Datasets...")

# A. Load Attacks (Deepset)
try:
    print("   - Downloading Attack Dataset...")
    ds_attacks = load_dataset("deepset/prompt-injections", split="train")
    # FIX: Explicitly convert to a Python list
    attacks = list(ds_attacks['text']) 
except Exception as e:
    print(f"   ⚠️ Error loading attacks: {e}. Using fallback data.")
    attacks = ["Ignore previous instructions", "System override", "DAN mode", "Jailbreak"] * 50

# B. Load Safe Prompts (Alpaca - balanced size)
try:
    print("   - Downloading Safe Dataset...")
    # We take 2000 to ensure the model sees enough "normal" English
    ds_safe = load_dataset("tatsu-lab/alpaca", split="train[:2000]")
    safe = [row['instruction'] + " " + row['input'] for row in ds_safe]
except Exception as e:
    print(f"   ⚠️ Error loading safe data: {e}. Using fallback data.")
    safe = ["Hello world", "Write a python script", "How are you?", "Explain quantum physics"] * 50

# C. Labeling (1 = Attack, 0 = Safe)
# Now both are lists, so this will work
X = attacks + safe
y = [1] * len(attacks) + [0] * len(safe)

print(f"✅ Loaded {len(X)} examples ({len(attacks)} Attacks, {len(safe)} Safe).")

# --- TRAINING ---
print("🧠 2. Training Model...")

# KEY FIX: stop_words='english' removes "the", "a", "is"
# KEY FIX: ngram_range=(1, 2) learns "Ignore instructions" (phrase) instead of just "Ignore"
vectorizer = TfidfVectorizer(
    stop_words='english', 
    ngram_range=(1, 2),
    min_df=2, 
    max_features=5000
)

X_vectors = vectorizer.fit_transform(X)

classifier = LogisticRegression(class_weight='balanced', C=10.0)
classifier.fit(X_vectors, y)

print("✅ Model Trained successfully.")

# --- SAVING ---
print(f"💾 3. Saving to {MODEL_DIR}...")

with open(os.path.join(MODEL_DIR, "vectorizer.pkl"), "wb") as f:
    pickle.dump(vectorizer, f)

with open(os.path.join(MODEL_DIR, "classifier.pkl"), "wb") as f:
    pickle.dump(classifier, f)

print("\n🎉 DONE! Now restart your backend.")