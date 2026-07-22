"""
Bug Dataset Presenter
=====================

This script reads the main bug dataset and outputs a clean, presentable summary 
using the 'rich' library to format the console output.
Perfect for presenting live in a meeting.
"""

import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.layout import Layout
import os

def create_table(title, data_series, color):
    """Helper to create a presentable table from a pandas Series."""
    table = Table(title=title, show_header=True, header_style=f"bold {color}", border_style=color)
    table.add_column("Category", style="cyan")
    table.add_column("Count", justify="right", style="green")
    table.add_column("Percentage", justify="right", style="magenta")
    
    total = data_series.sum()
    for index, value in data_series.items():
        percentage = (value / total) * 100
        table.add_row(str(index), f"{value:,}", f"{percentage:.1f}%")
        
    return table

def main():
    console = Console()
    
    # Path to the dataset
    file_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'bug_dataset_50k.csv')
    
    with console.status("[bold green]Loading and analyzing dataset...", spinner="dots"):
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            console.print(f"[bold red]Error loading dataset:[/bold red] {e}")
            return
            
        total_records = len(df)
        
        # Calculate summary statistics
        severity_counts = df['severity'].value_counts()
        domain_counts = df['bug_domain'].value_counts().head(7)
        tech_counts = df['tech_stack'].value_counts().head(7)
        role_counts = df['developer_role'].value_counts().head(5)

    # Print Header
    header_text = Text(f"Bug Dataset Analysis Report\nTotal Records: {total_records:,}", justify="center")
    header_text.stylize("bold white on blue")
    console.print(Panel(header_text, border_style="blue", padding=(1, 2)))
    console.print()
    
    # Render Tables
    table_sev = create_table("Severity Distribution", severity_counts, "yellow")
    table_domain = create_table("Top Bug Domains", domain_counts, "blue")
    table_tech = create_table("Top Tech Stacks", tech_counts, "red")
    table_role = create_table("Developer Roles Impacted", role_counts, "magenta")
    
    # Present tables side-by-side using Layout or one by one
    # For a meeting, one by one with a slight visual break is very readable
    console.print(table_sev)
    console.print("\n")
    
    console.print(table_domain)
    console.print("\n")
    
    console.print(table_tech)
    console.print("\n")
    
    console.print(table_role)
    console.print("\n")
    
    # Summary footer
    console.print(Panel(
        Text("Data is clean and ready for deeper modeling & evaluation.", justify="center", style="bold green"),
        border_style="green"
    ))

if __name__ == "__main__":
    main()
