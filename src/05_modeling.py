import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import os
import json

def train_and_evaluate():
    print("Loading preprocessed data...")
    df = pd.read_csv('data/bug_reports_processed.csv')
    
    # We will use the 'Description' as the feature to predict 'Severity' and 'Priority'
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    X = vectorizer.fit_transform(df['Description']).toarray()
    
    # Save the vectorizer
    os.makedirs('models', exist_ok=True)
    joblib.dump(vectorizer, 'models/tfidf_vectorizer.pkl')

    targets = ['Severity_encoded', 'Priority_encoded']
    models = {
        'Naive Bayes': MultinomialNB(),
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(),
        'Random Forest': RandomForestClassifier(n_estimators=100),
        'SVM': SVC(probability=True)
    }

    results = {}

    for target in targets:
        print(f"\nTraining models for {target}...")
        y = df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        target_results = {}
        best_model_name = None
        best_f1 = -1
        best_model = None
        
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            
            target_results[name] = {
                'Accuracy': acc,
                'Precision': prec,
                'Recall': rec,
                'F1-Score': f1
            }
            
            if f1 > best_f1:
                best_f1 = f1
                best_model_name = name
                best_model = model
                
        results[target] = target_results
        print(f"Best model for {target}: {best_model_name} (F1: {best_f1:.4f})")
        
        # Save the best model
        target_name = target.split('_')[0]
        joblib.dump(best_model, f'models/best_{target_name.lower()}_model.pkl')

    with open('data/model_evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    print("Model training and evaluation complete.")

if __name__ == "__main__":
    train_and_evaluate()
