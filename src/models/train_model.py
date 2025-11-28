import os
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score

def train():

    # --- Create required folders ---
    os.makedirs("models", exist_ok=True)
    os.makedirs("metrics", exist_ok=True)

    print("Loading data...")
    train = pd.read_csv("data/processed/train.csv")
    test = pd.read_csv("data/processed/test.csv")

    X_train, y_train = train["text"], train["label"]
    X_test, y_test = test["text"], test["label"]

    print("Vectorizing text...")
    vectorizer = TfidfVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    models = {
        "logreg": LogisticRegression(max_iter=300),
        "svm": LinearSVC(),
        "randomforest": RandomForestClassifier()
    }

    results = []

    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train_vec, y_train)
        preds = model.predict(X_test_vec)

        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="weighted")

        # Save each model
        joblib.dump(model, f"models/{name}.joblib")

        results.append([name, acc, f1])

    # Save metrics
    df = pd.DataFrame(results, columns=["model", "accuracy", "f1_score"])
    df.to_csv("metrics/scores.csv", index=False)

    # Choose best model
    best = df.sort_values("accuracy", ascending=False).iloc[0]
    best_name = best["model"]

    print(f"Best model is: {best_name}")

    # Save best model + vectorizer
    joblib.dump(models[best_name], "models/best_model.joblib")
    joblib.dump(vectorizer, "models/best_vectorizer.joblib")

    print("Training done! Best model saved.")

if __name__ == "__main__":
    train()
