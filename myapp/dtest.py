diarycontent= '''

'''

import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from torch.nn.functional import softmax

# Load pre-trained DistilBERT model and tokenizer
model_name = 'distilbert-base-uncased'
tokenizer = DistilBertTokenizer.from_pretrained(model_name)
model = DistilBertForSequenceClassification.from_pretrained(model_name)

# Define emotions
emotions = ["anger", "joy", "sadness", "fear"]

def detect_emotion(text):
    # Tokenize input text
    inputs = tokenizer(text, return_tensors="pt", truncation=True)

    # Forward pass through the model
    outputs = model(**inputs)

    # Extract logits from the output
    logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]

    # Apply softmax to get probabilities
    probabilities = softmax(logits, dim=1).detach().numpy()[0]

    # Get the predicted emotion
    predicted_emotion_index = int(torch.argmax(logits, dim=1))
    predicted_emotion = emotions[predicted_emotion_index]

    return predicted_emotion, probabilities

# Example usage
text_to_analyze =diarycontent
predicted_emotion, probabilities = detect_emotion(text_to_analyze)

# Display results
print(f"Predicted Emotion: {predicted_emotion}")
print("Emotion Probabilities:")
for emotion, probability in zip(emotions, probabilities):
    print(f"{emotion}: {probability:.4f}")
