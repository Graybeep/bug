import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json

def detect_duplicates(file_path='data/bug_reports.csv', threshold=0.85):
    df = pd.read_csv(file_path)
    # Fill NA descriptions
    df['Description'] = df['Description'].fillna('')
    
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(df['Description'])
    
    # Calculate cosine similarity
    similarity_matrix = cosine_similarity(tfidf_matrix)
    
    duplicates = []
    num_bugs = len(df)
    
    # Find pairs with similarity > threshold
    for i in range(num_bugs):
        for j in range(i + 1, num_bugs):
            if similarity_matrix[i, j] > threshold:
                duplicates.append({
                    'Bug_1': df.iloc[i]['Bug ID'],
                    'Bug_2': df.iloc[j]['Bug ID'],
                    'Similarity': float(similarity_matrix[i, j])
                })
    
    print(f"Found {len(duplicates)} potential duplicate pairs based on description similarity (threshold > {threshold}).")
    with open('data/potential_duplicates.json', 'w') as f:
        json.dump(duplicates, f, indent=4)
        
    return duplicates

if __name__ == "__main__":
    detect_duplicates()
