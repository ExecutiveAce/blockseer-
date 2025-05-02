from modules.fetching import fetch_binance_data, insert_into_db
from modules.trainer import train_model
from modules.predictor import run_predictions

def run_blockseer_pipeline():
    print("🚀 Running full BLOCKSEER pipeline")
    data = fetch_binance_data()
    insert_into_db(data)
    train_model()
    run_predictions()

if __name__ == "__main__":
    run_blockseer_pipeline()
