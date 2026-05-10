# utils/nlp_engine.py
from textblob import TextBlob
from deep_translator import GoogleTranslator
from langdetect import detect
import pandas as pd

class SentimentAnalyzer:
    def __init__(self):
        """Initialize the modern translation API."""
        self.translator = GoogleTranslator(source='auto', target='en')

    def detect_and_translate(self, text):
        """Detects language and translates to English if necessary."""
        try:
            text_str = str(text).strip()
            if not text_str:
                return "unknown", ""

            lang = detect(text_str)
            
            if lang != 'en':
                translated = self.translator.translate(text_str)
                return lang, translated
            return lang, text_str
            
        except Exception as e:
            return "unknown", str(text)

    def analyze_text(self, text):
        """Returns polarity, subjectivity, and sentiment label."""
        if not text:
             return {"polarity": 0, "subjectivity": 0, "label": "Neutral", "emoji": "⚪"}

        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        
        if polarity > 0.1:
            label, emoji = "Positive", "🟢"
        elif polarity < -0.1:
            label, emoji = "Negative", "🔴"
        else:
            label, emoji = "Neutral", "⚪"
            
        return {"polarity": polarity, "subjectivity": subjectivity, "label": label, "emoji": emoji}

    def batch_analyze(self, dataframe, text_column):
        """Analyzes an entire dataset at once."""
        results = []
        for index, row in dataframe.iterrows():
            original_text = row[text_column]
            
            if pd.isna(original_text):
                continue
                
            lang, translated = self.detect_and_translate(original_text)
            analysis = self.analyze_text(translated)
            
            results.append({
                "Original Text": original_text,
                "Detected Language": lang.upper(),
                "Translated (English)": translated,
                "Sentiment": analysis['label'],
                "Polarity Score": round(analysis['polarity'], 3),
                "Subjectivity Score": round(analysis['subjectivity'], 3),
                "Text Length": len(str(original_text)) # Added for 3D visualization
            })
            
        return pd.DataFrame(results)