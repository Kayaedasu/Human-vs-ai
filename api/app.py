from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

tfidf = joblib.load("tfidf_vectorizer.pkl")
log_model = joblib.load("logistic_model.pkl")
svm_model = joblib.load("svm_model.pkl")
rf_model = joblib.load("rf_model.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    text = data["text"]

    X = tfidf.transform([text])

    log_pred = log_model.predict_proba(X)[0]
    svm_pred = svm_model.predict_proba(X)[0]
    rf_pred = rf_model.predict_proba(X)[0]

    result = {
        "logistic_regression": {
            "ai": float(log_pred[1]),
            "human": float(log_pred[0])
        },
        "svm": {
            "ai": float(svm_pred[1]),
            "human": float(svm_pred[0])
        },
        "random_forest": {
            "ai": float(rf_pred[1]),
            "human": float(rf_pred[0])
        }
    }

    return jsonify(result)

if __name__ == "__main__":
    app.run(port=5000)
