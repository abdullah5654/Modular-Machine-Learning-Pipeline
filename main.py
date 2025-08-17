import os
from preprocessing import load_and_split_data, preprocess_data
from train_model import train_and_tune
from evaluate import evaluate_model, load_pipeline_and_preprocess

# Ensure artifacts directory exists
os.makedirs("artifacts", exist_ok=True)

DATA_PATH = "data.csv"
TARGET = "MEDV"   # change target according to dataset

def main():
    # Step 1: Load & preprocess
    train_set, test_set = load_and_split_data(DATA_PATH, TARGET)
    X_train_prepared, y_train = preprocess_data(train_set, TARGET)

    # Step 2: Train model
    model, best_params = train_and_tune(X_train_prepared, y_train, model_type="RandomForest")
    print("Best Params:", best_params)

    # Step 3: Evaluate
    X_test_prepared, y_test = load_pipeline_and_preprocess(test_set, TARGET)
    evaluate_model(model, X_test_prepared, y_test)

if __name__ == "__main__":
    main()
