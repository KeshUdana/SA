# finbert_sentiment.py
import torch
from torch.nn.functional import softmax

from transformers import AutoTokenizer, AutoModelForSequenceClassification


model_name = "ProsusAI/finbert"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
label_map = {0: "positive", 1: "negative", 2: "neutral"}

def classify_sentiment(text: str):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = softmax(logits, dim=1)
    pred = torch.argmax(probs).item()
    return label_map[pred], float(probs[0][pred])
