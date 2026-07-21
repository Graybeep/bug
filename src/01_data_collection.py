import pandas as pd
import numpy as np
import random
import os

def generate_bug_dataset(num_records=1000):
    statuses = ['New', 'Assigned', 'In Progress', 'Resolved', 'Closed']
    severities = ['Trivial', 'Minor', 'Major', 'Critical']
    priorities = ['P5', 'P4', 'P3', 'P2', 'P1']
    resolutions = ['Fixed', 'Duplicate', 'Won\'t Fix', 'Cannot Reproduce', 'Not a Bug']
    
    summaries = [
        "App crashes on login", "UI misalignment in header", "Database connection timeout",
        "Null pointer exception in module X", "Feature Y is not responsive",
        "Typo in the settings menu", "Memory leak during file upload",
        "API returning 500 error", "Button click has no effect", "Performance degradation on load"
    ]
    
    descriptions = [
        "When attempting to log in with valid credentials, the application throws an exception and crashes.",
        "The header elements are overlapping on mobile screens.",
        "The connection to the database times out after 30 seconds of inactivity.",
        "A NullPointerException is thrown when clicking the submit button in module X.",
        "Feature Y does not resize properly when the window is scaled down.",
        "There is a spelling mistake in the user profile settings page.",
        "Uploading a file larger than 10MB causes a steady increase in RAM usage.",
        "The /api/v1/data endpoint returns a 500 Internal Server Error intermittently.",
        "Clicking the 'Save' button in the dialog does not trigger any network request.",
        "Loading the main dashboard takes over 10 seconds, which is a severe degradation."
    ]

    data = []
    for i in range(1, num_records + 1):
        # Create some intentional duplicates
        is_duplicate = random.random() < 0.1
        if is_duplicate and i > 10:
            dup_idx = random.randint(0, len(data) - 1)
            summary = data[dup_idx]['Summary']
            desc = data[dup_idx]['Description']
            res = 'Duplicate'
            status = 'Closed'
        else:
            idx = random.randint(0, len(summaries) - 1)
            summary = summaries[idx] + f" - Case {random.randint(100, 999)}"
            desc = descriptions[idx]
            res = random.choice(resolutions)
            status = random.choice(statuses)
            
            # If resolution is Duplicate, ensure status is closed or resolved
            if res == 'Duplicate':
                status = random.choice(['Resolved', 'Closed'])
            
            # If status is New or Assigned, resolution is usually None
            if status in ['New', 'Assigned', 'In Progress']:
                res = 'None'

        record = {
            'Bug ID': f"BUG-{i:04d}",
            'Summary': summary,
            'Description': desc,
            'Status': status,
            'Severity': random.choice(severities),
            'Priority': random.choice(priorities),
            'Resolution': res
        }
        data.append(record)
        
    df = pd.DataFrame(data)
    
    os.makedirs('data', exist_ok=True)
    file_path = 'data/bug_reports.csv'
    df.to_csv(file_path, index=False)
    print(f"Dataset with {num_records} records generated and saved to {file_path}")

if __name__ == "__main__":
    generate_bug_dataset(1500)
