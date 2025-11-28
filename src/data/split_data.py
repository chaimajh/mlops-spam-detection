import pandas as pd
from sklearn.model_selection import train_test_split
import os

def main():
    raw_path = "data/raw/spam.csv"

    df = pd.read_csv(raw_path, encoding="latin1")

    # Renommer les colonnes du dataset spam
    df = df.rename(columns={"v1": "label", "v2": "text"})
    df = df[["label", "text"]]

    train, test = train_test_split(df, test_size=0.2, random_state=42)

    os.makedirs("data/processed", exist_ok=True)

    train.to_csv("data/processed/train.csv", index=False)
    test.to_csv("data/processed/test.csv", index=False)

if __name__ == "__main__":
    main()
