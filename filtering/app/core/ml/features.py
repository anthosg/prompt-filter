import numpy as np
import scipy.stats
from sklearn.base import BaseEstimator, TransformerMixin

class TextStatsTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        features = []
        for text in X:
            text_str = str(text)
            length = len(text_str)
            
            special_chars = sum(1 for c in text_str if not c.isalnum() and not c.isspace())
            special_ratio = special_chars / max(1, length)
            
            encoded = text_str.encode('utf-8', 'ignore')
            if len(encoded) > 0:
                char_counts = np.bincount(np.frombuffer(encoded, dtype=np.uint8))
                probs = char_counts[char_counts > 0] / np.sum(char_counts)
                entropy = float(scipy.stats.entropy(probs))
            else:
                entropy = 0.0
                
            features.append([length, special_ratio, entropy])
        return np.array(features)