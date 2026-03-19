import pickle
import os
import numpy as np

class SecureScanner:
    def __init__(self):
        print("🛡️ Booting Aegis Secure Scanner...")

        # --- 1. ROBUST PATH RESOLUTION ---
        current_file_path = os.path.abspath(__file__)
        
        # Step up the directory tree to find the root backend folder
        security_dir = os.path.dirname(current_file_path)
        app_dir = os.path.dirname(security_dir)
        base_dir = os.path.dirname(app_dir) 
        
        model_dir = os.path.join(base_dir, "new_models")
        
        # --- 2. LOAD MODELS WITH ERROR HANDLING ---
        try:
            with open(os.path.join(model_dir, "vectorizer.pkl"), "rb") as f:
                self.vectorizer = pickle.load(f)
            with open(os.path.join(model_dir, "classifier.pkl"), "rb") as f:
                self.classifier = pickle.load(f)
            print(f"✅ Models loaded successfully from: {model_dir}")
        except FileNotFoundError:
            print(f"❌ ERROR: Could not find models.")
            print(f"🔍 The scanner was looking here: {model_dir}")
            print("Make sure your 'models' folder is placed in that exact location.")
        
        # Extract features for Explainability (XAI)
        self.feature_names = np.array(self.vectorizer.get_feature_names_out())
        self.coefficients = self.classifier.coef_[0]

    def scan(self, text: str, threshold: float = 0.30):
        text_lower = text.lower()
        
        # --- LAYER 1: SIGNATURE CHECK ---
        known_signatures = ["do anything now", "dan", "jailbreak", "dev mode", "chaosgpt"]
        for sig in known_signatures:
            if sig in text_lower:
                return False, 1.0, [sig, "signature_match"]

        # --- LAYER 2: ML MODEL ---
        vector = self.vectorizer.transform([text])
        risk_score = self.classifier.predict_proba(vector)[0][1]
        
        # 3. Decision (NOW DYNAMIC!)
        is_safe = risk_score < threshold
        
        triggers = []
        if not is_safe:
            nonzero_indices = vector.nonzero()[1]
            if len(nonzero_indices) > 0:
                word_scores = [(self.feature_names[i], self.coefficients[i]) for i in nonzero_indices]
                word_scores.sort(key=lambda x: x[1], reverse=True)
                triggers = [word for word, score in word_scores[:3] if score > 0]

        return is_safe, float(risk_score), triggers