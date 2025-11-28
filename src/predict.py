import joblib
import sys

def predict(text):
    model = joblib.load("models/best_model.joblib")
    vectorizer = joblib.load("models/best_vectorizer.joblib")

    X = vectorizer.transform([text])
    pred = model.predict(X)[0]

    return "SPAM" if pred == 1 else "HAM"

if __name__ == "__main__":
    msg = " ".join(sys.argv[1:])
    print(predict(msg))
