import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib
import os

def preprocess_data(file_path='data/bug_reports.csv'):
    print("Loading dataset...")
    df = pd.read_csv(file_path)
    
    # 3. Data Preprocessing
    print("Initial shape:", df.shape)
    
    # Handle missing values (if any)
    df.fillna({'Description': ''}, inplace=True)
    df.dropna(subset=['Summary', 'Status', 'Severity', 'Priority'], inplace=True)
    
    # Remove duplicates (exact duplicates based on Bug ID if they exist)
    df.drop_duplicates(subset=['Bug ID'], inplace=True)
    print("Shape after basic cleaning:", df.shape)

    # Encode categorical data
    encoders = {}
    categorical_cols = ['Status', 'Severity', 'Priority', 'Resolution']
    for col in categorical_cols:
        le = LabelEncoder()
        df[col + '_encoded'] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
    
    # Save encoders for later prediction
    os.makedirs('models', exist_ok=True)
    joblib.dump(encoders, 'models/label_encoders.pkl')
    
    # Save preprocessed data
    df.to_csv('data/bug_reports_processed.csv', index=False)
    print("Preprocessed data saved to data/bug_reports_processed.csv")
    return df

if __name__ == "__main__":
    preprocess_data()
