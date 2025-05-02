import pandas as pd
from modules.preprocessing import preprocess_text, clean_text
from modules.emotion_intent_runner import classify_emotion_and_intent
from modules.aggregator import run_aggregator

def run_pipeline(input_file: str, output_file: str = "data/classified_data.csv", aggregated_output: str = "data/aggregated_summary.csv"):
    # Step 1: Load the raw data
    try:
        df = pd.read_csv(input_file)
        print("[Pipeline] Raw data loaded.")
    except Exception as e:
        print(f"[Pipeline Error] Loading raw data failed: {e}")
        return

    # Step 2: Preprocess text data
    df['cleaned_text'] = df['text'].apply(lambda x: clean_text(preprocess_text(x)))
    df.dropna(subset=['cleaned_text'], inplace=True)  # Drop rows where text was invalid or None

    print("[Pipeline] Text preprocessing complete.")

    # Step 3: Classify emotion and intent
    df[['emotion', 'intent']] = df['cleaned_text'].apply(classify_emotion_and_intent)

    # Step 4: Save classified data to CSV
    df.to_csv(output_file, index=False)
    print(f"[Pipeline] Classified data saved to {output_file}")

    # Step 5: Run emotion-intent aggregation
    run_aggregator(output_file, aggregated_output)
    print("[Pipeline] Aggregation complete and saved to", aggregated_output)

# Example usage:
if __name__ == "__main__":
    input_file = "data/raw_data.csv"  # Assuming you have some raw input data CSV
    run_pipeline(input_file)
