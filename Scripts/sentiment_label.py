"""
Script: sentiment_label.py
Purpose: Add sentiment labels to airline review data using a pre-trained BERT model.

This script processes the raw airlines review dataset and performs sentiment analysis on each 
review text. The sentiment labels (POSITIVE, NEGATIVE, NEUTRAL) are added as a new column 
to the dataset, enabling analysis of customer sentiment patterns.

Input: data/airlines_reviews.csv (raw review data with 'Reviews' column)
Output: data/airlines_reviews_with_sentiment.csv (data with sentiment labels as an appended column)
"""

import math
import pandas as pd
from transformers import pipeline
import torch
from tqdm import tqdm

def adjust_df(input_file, output_file, columns=None):
    if columns is None:
        columns = ["Airline", "Reviews", "Verified", "Class"]
    df = pd.read_csv(input_file)
    df = df[columns]
    df.to_csv(output_file, index=False)

def add_sentiment_column(input_file, output_file, batch_size=32):
    # Load the input CSV file
    print(f"Loading {input_file}...")
    df = pd.read_csv(input_file)

    # Validate that the Reviews column exists in the data
    if 'Reviews' not in df.columns:
        raise ValueError("Error: 'Reviews' column not found in the CSV.")

    # Initialize the sentiment analysis model using a pre-trained BERT model fine-tuned for sentiment analysis
    model_name = "MarieAngeA13/Sentiment-Analysis-BERT"
    print(f"Loading model: {model_name}...")

    # Check for GPU availability and use it if possible for faster processing
    device = 0 if torch.cuda.is_available() else -1
    if device == 0:
        print("Using GPU")
    else:
        print("Using CPU")

    # Create sentiment analysis pipeline with model configuration, truncation ensures reviews longer than 512 tokens are handled properly
    sentiment_pipeline = pipeline(
        "sentiment-analysis",
        model=model_name,
        tokenizer=model_name,
        device=device,
        truncation=True,
        max_length=512
    )

    print("Analyzing sentiments...")

    # Convert reviews to a list for batch processing
    reviews = df['Reviews'].astype(str).tolist()
    n = len(reviews)
    num_batches = math.ceil(n / batch_size)

    sentiments = []

    # Process reviews in batches to optimize memory usage and speed
    for i in tqdm(range(num_batches), desc="Batches", unit="batch"):
        start = i * batch_size
        end = min(start + batch_size, n)
        batch_texts = reviews[start:end]
        batch_results = sentiment_pipeline(batch_texts)
        sentiments.extend([r["label"] for r in batch_results])

    # Validate that sentiment analysis completed successfully and ensure the number of sentiment labels matches the number of reviews
    if len(sentiments) != n:
        raise RuntimeError("Sentiment list length does not match number of rows.")

    # Add sentiment column to the dataframe
    df["sentiment"] = sentiments

    # Save the new dataset with sentiment labels
    print(f"Saving results to {output_file}...")
    df.to_csv(output_file, index=False)
    print("Done!")

if __name__ == "__main__":
    # Set paths for input and output files
    input_csv = "data/airlines_reviews.csv"
    cleaned_csv = "data/airlines_reviews_cleaned.csv"
    output_csv = "data/airlines_reviews_with_sentiment.csv"
    
    # Run the sentiment analysis with batch processing on our data 
    add_sentiment_column(input_csv, output_csv, batch_size=32)
    adjust_df(output_csv, cleaned_csv)
