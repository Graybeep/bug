import joblib
import json
import argparse

def predict_bug(description):
    # Load models and encoders
    try:
        vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
        severity_model = joblib.load('models/best_severity_model.pkl')
        priority_model = joblib.load('models/best_priority_model.pkl')
        encoders = joblib.load('models/label_encoders.pkl')
    except FileNotFoundError:
        print("Models not found. Please run 05_modeling.py first.")
        return

    # Process input
    X_input = vectorizer.transform([description]).toarray()
    
    # Predict
    sev_pred = severity_model.predict(X_input)[0]
    pri_pred = priority_model.predict(X_input)[0]
    
    # Decode
    severity = encoders['Severity'].inverse_transform([sev_pred])[0]
    priority = encoders['Priority'].inverse_transform([pri_pred])[0]
    
    result = {
        'Description': description,
        'Predicted_Severity': severity,
        'Predicted_Priority': priority
    }
    
    print("\n--- Bug Prediction ---")
    print(json.dumps(result, indent=4))
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict Bug Severity and Priority")
    parser.add_argument('--desc', type=str, default="The application throws a NullPointerException during checkout, causing a total failure of the payment process.", help="Bug description text")
    args = parser.parse_args()
    
    predict_bug(args.desc)
