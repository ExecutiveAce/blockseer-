import pandas as pd
from modules.emotion_intent_runner import classify_emotion_and_intent
from modules.preprocessing import preprocess_text, clean_text
from modules.classifiers.emotion_classifier import classify_emotion
from modules.classifiers.intent_classifier import classify_intent

# Simulated data for testing (replace with actual data loading process)
data = [
    {"text": "I’m definitely going to buy more BTC if it drops.", "timestamp": "2025-04-18T10:00:00Z"},
    {"text": "This is terrifying. I'm going to sell everything.", "timestamp": "2025-04-18T11:00:00Z"},
    {"text": "Just watching the market for now.", "timestamp": "2025-04-18T12:00:00Z"},
    {"text": "Bitcoin is going to the moon!!", "timestamp": "2025-04-18T13:00:00Z"},
    {"text": None, "timestamp": "2025-04-18T14:00:00Z"},
    {"text": "123456", "timestamp": "2025-04-18T15:00:00Z"},
]

# Convert data into DataFrame
df = pd.DataFrame(data)

# Apply preprocessing
df['text'] = df['text'].apply(preprocess_text)
df['text'] = df['text'].apply(clean_text)

# Filter out rows where the text is invalid (None, empty, or junk)
filtered_df = df.dropna(subset=['text'])  # Drop rows with 'None' or NaN in 'text'
filtered_df = filtered_df[filtered_df['text'].str.strip().apply(lambda x: len(x) > 5)]  # Remove short texts

# Apply emotion and intent classification
filtered_df['emotion'] = filtered_df['text'].apply(classify_emotion)
filtered_df['intent'] = filtered_df['text'].apply(classify_intent)

# Print or save the cleaned and classified DataFrame
print(filtered_df[['text', 'emotion', 'intent']])

# Optional: You can also save the output to a file (like CSV)
# filtered_df.to_csv('classified_data.csv', index=False)
