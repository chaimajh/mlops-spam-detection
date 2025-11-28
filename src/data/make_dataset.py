import os
import pandas as pd
from sklearn.model_selection import train_test_split
import yaml

def load_params():
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)
    return params

def main():
    # Load parameters
    params = load_params()
    test_size = params["base"]["test_size"]
    random_state = params["base"]["random_state"]

    # Load raw dataset
    raw_path = "data/raw/spam.csv"
    print(f"Loading raw dataset from {raw_path}...")
    df = pd.read_csv(raw_path, encoding="latin-1")

    # Clean useless columns
    df = df[["v1", "v2"]]
    df.columns = ["label", "text"]

    # Split train/test
    train_df, test_df = train_test_split(
        df, 
        test_size=test_size, 
        random_state=random_state,
        stratify=df["label"]
    )

    # Save
    os.makedirs("data/processed", exist_ok=True)

    train_df.to_csv("data/processed/train.csv", index=False)
    test_df.to_csv("data/processed/test.csv", index=False)

    print("Dataset processed and saved to data/processed/")

if __name__ == "__main__":
    main()
