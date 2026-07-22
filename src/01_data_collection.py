# ==============================================================================
# 01_data_collection.py
# Task 1 & 2: Data Collection and Dataset Connection
# Generates a synthetic bug report dataset simulating data from Bugzilla/JIRA/GitHub.
# Output: data/bug_reports.csv
# ==============================================================================

import pandas as pd
import numpy as np
import random
import os
from datetime import datetime, timedelta

def generate_bug_dataset(num_records=1500):
    """Generate a synthetic bug report dataset with realistic fields."""

    random.seed(42)
    np.random.seed(42)

    # --- Domain Data ---
    statuses     = ['New', 'Assigned', 'In Progress', 'Resolved', 'Closed']
    severities   = ['Trivial', 'Minor', 'Major', 'Critical']
    priorities   = ['P5', 'P4', 'P3', 'P2', 'P1']
    resolutions  = ['Fixed', 'Duplicate', "Won't Fix", 'Cannot Reproduce', 'Not a Bug']
    modules      = ['Authentication', 'Payment', 'Dashboard', 'Reporting', 'API', 'Database', 'UI/UX', 'Notifications']
    developers   = ['Alice Johnson', 'Bob Smith', 'Carol White', 'David Lee',
                    'Eva Martinez', 'Frank Brown', 'Grace Kim', 'Henry Wilson']

    summaries = [
        "App crashes on login",
        "UI misalignment in header",
        "Database connection timeout",
        "Null pointer exception in module X",
        "Feature Y is not responsive",
        "Typo in the settings menu",
        "Memory leak during file upload",
        "API returning 500 error",
        "Button click has no effect",
        "Performance degradation on load",
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
        "Loading the main dashboard takes over 10 seconds, which is a severe degradation.",
    ]

    # --- Date range: last 12 months ---
    end_date   = datetime(2024, 12, 31)
    start_date = datetime(2024, 1, 1)
    date_range = (end_date - start_date).days

    data = []
    for i in range(1, num_records + 1):
        # ~10% intentional duplicates
        is_duplicate = random.random() < 0.10
        if is_duplicate and i > 10:
            dup_idx  = random.randint(0, len(data) - 1)
            summary  = data[dup_idx]['Summary']
            desc     = data[dup_idx]['Description']
            res      = 'Duplicate'
            status   = 'Closed'
        else:
            idx     = random.randint(0, len(summaries) - 1)
            summary = summaries[idx] + f" - Case {random.randint(100, 999)}"
            desc    = descriptions[idx]
            res     = random.choice(resolutions)
            status  = random.choice(statuses)

            # Resolution logic consistency
            if res == 'Duplicate':
                status = random.choice(['Resolved', 'Closed'])
            if status in ['New', 'Assigned', 'In Progress']:
                res = 'None'

        reported_date = (start_date + timedelta(days=random.randint(0, date_range))).strftime('%Y-%m-%d')

        record = {
            'Bug ID':        f"BUG-{i:04d}",
            'Summary':       summary,
            'Description':   desc,
            'Status':        status,
            'Severity':      random.choice(severities),
            'Priority':      random.choice(priorities),
            'Resolution':    res,
            'Assigned_To':   random.choice(developers),
            'Module':        random.choice(modules),
            'Reported_Date': reported_date,
        }
        data.append(record)

    df = pd.DataFrame(data)

    os.makedirs('data', exist_ok=True)
    file_path = 'data/bug_reports.csv'
    df.to_csv(file_path, index=False)

    print("=" * 60)
    print("  TASK 1 & 2: Data Collection & Dataset Connection")
    print("=" * 60)
    print(f"  Source       : Synthetic (simulating Bugzilla/JIRA/GitHub)")
    print(f"  Total Records: {num_records}")
    print(f"  Columns      : {list(df.columns)}")
    print(f"  Output File  : {file_path}")
    print("=" * 60)
    print(df.head(5).to_string(index=False))
    print("=" * 60)

if __name__ == "__main__":
    generate_bug_dataset(1500)
