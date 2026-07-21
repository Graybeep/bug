import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def visualize_data(file_path='data/bug_reports.csv'):
    df = pd.read_csv(file_path)
    os.makedirs('visualizations', exist_ok=True)

    sns.set_theme(style="whitegrid")

    # 1. Bug Status Distribution
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x='Status', palette='viridis', order=df['Status'].value_counts().index)
    plt.title('Bug Status Distribution')
    plt.savefig('visualizations/bug_status.png')
    plt.close()

    # 2. Severity Distribution
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x='Severity', palette='magma', order=['Trivial', 'Minor', 'Major', 'Critical'])
    plt.title('Severity Distribution')
    plt.savefig('visualizations/severity_distribution.png')
    plt.close()

    # 3. Priority Distribution
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x='Priority', palette='plasma', order=['P5', 'P4', 'P3', 'P2', 'P1'])
    plt.title('Priority Distribution')
    plt.savefig('visualizations/priority_distribution.png')
    plt.close()

    # 4. Duplicate Bugs (based on Resolution)
    plt.figure(figsize=(6, 6))
    duplicate_counts = df['Resolution'].apply(lambda x: 'Duplicate' if x == 'Duplicate' else 'Not Duplicate').value_counts()
    plt.pie(duplicate_counts, labels=duplicate_counts.index, autopct='%1.1f%%', colors=['#66b3ff','#ff9999'])
    plt.title('Proportion of Duplicate Bugs')
    plt.savefig('visualizations/duplicate_bugs.png')
    plt.close()
    
    print("Visualizations saved in 'visualizations/' directory.")

if __name__ == "__main__":
    visualize_data()
