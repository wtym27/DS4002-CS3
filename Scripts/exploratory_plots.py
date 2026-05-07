#!/usr/bin/env python3
"""
Script: exploratory_plots.py
Purpose: Generate exploratory visualizations of the airlines review dataset.
This script creates interactive bar charts to visualize the distribution of reviews across different airlines
and the distribution of customer classes (Business, Economy, Other). 

Output plots are saved to output/ directory:
- reviews_per_airline.png — number of reviews per airline
- customers_by_class.png — counts of customers mapped to Business, Economy, or Other
"""
from pathlib import Path
import argparse

# Parse --show flag early so we can set matplotlib GUI backend before pyplot is imported for interactive graph to display when requested
early = argparse.ArgumentParser(add_help=False)
early.add_argument("--show", action="store_true")
early_args, _ = early.parse_known_args()

if early_args.show:
    try:
        import matplotlib

        matplotlib.use("TkAgg")
    except Exception:
        # if setting the GUI backend fails, continue with default
        pass

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def main(show: bool = False, csv_path: str = "data/airlines_reviews.csv"):
    # Load the dataset from CSV file
    csv = Path(csv_path)
    df = pd.read_csv(csv)

    #retrieves sentiment data
    sentiment_csv = csv.parent / "airlines_reviews_with_sentiment.csv"
    df_sentiment = None
    if sentiment_csv.exists():
        df_sentiment = pd.read_csv(sentiment_csv)

    out = Path("output")
    out.mkdir(exist_ok=True)

    # Plot 1: Reviews per Airline:This visualization shows how many reviews each airline has received, helping identify potential data imbalances
    if "Airline" not in df.columns:
        raise KeyError("missing 'Airline' column")

    # Count the number of reviews for each airline
    counts = df["Airline"].value_counts()
    
    # Create a horizontal bar chart with dynamic height to accommodate all airlines
    plt.figure(figsize=(10, max(4, len(counts) * 0.25)))
    sns.barplot(x=counts.values, y=counts.index, palette="viridis")
    plt.xlabel("Number of reviews")
    plt.ylabel("Airline")
    plt.title("Reviews per Airline")
    plt.tight_layout()
    
    # Save the plot
    a = out / "reviews_per_airline.png"
    plt.savefig(a, dpi=150)
    if show:
        plt.show()
    plt.close()

    # Plot 2: Customers by Class: This visualization shows the distribution of customer types (Business/Economy/Other),which is important for understanding class representation in reviews
    if "Class" not in df.columns:
        raise KeyError("missing 'Class' column")

    # Normalize class labels to three categories: Business, Economy, or Other. This handles variations in naming and missing values from the dataset
    def cls_label(x):
        if isinstance(x, str):
            lx = x.lower()
            if "business" in lx:
                return "Business"
            if "econom" in lx:
                return "Economy"
        return "Other"

    # Apply classification labeing and properly order with all categories present
    cls_counts = df["Class"].apply(cls_label).value_counts()
    cls_counts = cls_counts.reindex(["Business", "Economy", "Other"]).fillna(0)

    # Create a bar chart showing customer distribution by class
    plt.figure(figsize=(6, 4))
    sns.barplot(x=cls_counts.index, y=cls_counts.values, palette="magma")
    plt.xlabel("Class")
    plt.ylabel("Number of customers")
    plt.title("Customers by Class (Business / Economy / Other)")
    
    # Add count labels on top of bars for clarity
    for i, v in enumerate(cls_counts.values):
        plt.text(i, v + max(cls_counts.values) * 0.01, str(int(v)), ha="center")
    plt.tight_layout()
    
    # Save the plot
    b = out / "customers_by_class.png"
    plt.savefig(b, dpi=150)
    if show:
        plt.show()
    plt.close()

    #sentiment plot
    if df_sentiment is not None and "sentiment" in df_sentiment.columns:
        sentiment_counts = df_sentiment["sentiment"].str.capitalize().value_counts()
        plt.figure(figsize=(8, 5))
        colors = {"Positive": "green", "Neutral": "gray", "Negative": "red"}
        sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette=colors)
        plt.xlabel("Sentiment")
        plt.ylabel("Number of Reviews")
        plt.title("Distribution of Sentiment in Airline Reviews")
        for i, v in enumerate(sentiment_counts.values):
            plt.text(i, v + max(cls_counts.values) * 0.01, str(int(v)), ha="center")
        plt.tight_layout()
        c = out / "sentiment_distribution.png"
        plt.savefig(c, dpi=150)
        if show:
            plt.show()
        plt.close()
        print(a, b, c)
    else:
        print(a, b)


if __name__ == "__main__":
    # Parse command-line arguments for user customization
    parser = argparse.ArgumentParser()
    parser.add_argument("--show", action="store_true", help="Show plots interactively")
    parser.add_argument("--csv", default="data/airlines_reviews.csv", help="Path to the CSV file")
    args = parser.parse_args()
    
    # Run the main function with interactive display and specified CSV path
    main(show=args.show, csv_path=args.csv)
