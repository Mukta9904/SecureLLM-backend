import pickle
import os
import numpy as np

class SecureScanner:
    def __init__(self):
        print("🛡️ Booting Aegis Secure Scanner...")
        
        # Point to backend/models folder
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        model_dir = os.path.join(base_dir, "models")
        
        with open(os.path.join(model_dir, "vectorizer.pkl"), "rb") as f:
            self.vectorizer = pickle.load(f)
        with open(os.path.join(model_dir, "classifier.pkl"), "rb") as f:
            self.classifier = pickle.load(f)
        
        # Extract features for Explainability (XAI)
        self.feature_names = np.array(self.vectorizer.get_feature_names_out())
        self.coefficients = self.classifier.coef_[0]

    def scan(self, text: str):
        # 1. Transform text
        vector = self.vectorizer.transform([text])
        
        # 2. Get Risk Score
        risk_score = self.classifier.predict_proba(vector)[0][1]
        
        # 3. Decision (Block if score > 0.65)
        is_safe = risk_score < 0.65
        
        # 4. Explainability: Find the specific "Trigger Words"
        triggers = []
        if not is_safe:
            nonzero_indices = vector.nonzero()[1]
            if len(nonzero_indices) > 0:
                word_scores = [(self.feature_names[i], self.coefficients[i]) for i in nonzero_indices]
                word_scores.sort(key=lambda x: x[1], reverse=True)
                # Keep top 3 dangerous words found in this prompt
                triggers = [word for word, score in word_scores[:3] if score > 0]

        return is_safe, float(risk_score), triggers