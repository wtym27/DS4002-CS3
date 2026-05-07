# DS4002-CS3
DS4002-CS3
Airline Sentiment Analysis Using BERT and Chi-Squared Testing

A DS 4002 Case Study by William McLaughlin

Contents of Repository

This repository contains all materials necessary to complete the Airline Sentiment Analysis case study. The data folder contains the airline review dataset used throughout the project. The scripts folder contains Python scripts for sentiment labeling, exploratory analysis, visualization, and chi-squared testing. The supplemental_materials folder contains supporting articles, tutorials, and reference materials related to BERT sentiment analysis and Hugging Face pipelines, airline seating classes, and chi-squared testing. The repository also includes the hook document (CS3 Hook Airlines Reviews.pdf) and rubric (Case Study Rubric Airline Reviews.pdf) that guide students through the case study.

Software and Platform

This project was developed using Python 3.10+ in VS Code on Windows and MacOS. Main libraries used include pandas, matplotlib, scipy, transformers, torch, and tqdm.

Project Folder Map
DS4002-CS3

data/
   airlines_reviews.csv
   airlines_reviews_with_sentiment.csv

scripts/
   sentiment_label.py
   sentiment_by_airline.py
   exploratory_plots.py
   chi_squared_test.py

supplemental_materials/

CS3 Hook Airlines Reviews.pdf
Case Study Rubric Airline Reviews.pdf
README.md
LICENSE

Instructions for Reproducing Results

First, clone the repository. Then, install the required libraries with:

pip install pandas matplotlib scipy transformers torch tqdm

The airline review dataset is already included in the data folder, so no additional preprocessing is required. Run sentiment_label.py to generate sentiment labels using a pre-trained BERT model from Hugging Face. Then run sentiment_by_airline.py and exploratory_plots.py to create visualizations and explore sentiment trends across airlines and passenger classes. Finally, run chi_squared_test.py to perform a chi-squared test examining whether passenger class and customer sentiment are statistically related.

After reproducing the analysis, students should reflect on their findings by discussing whether passenger class appeared to influence customer sentiment, whether the statistical results were significant, limitations of using a pre-trained sentiment model, and what they learned from combining machine learning with statistical testing.

Reference Materials

The supplemental materials folder contains introductory and technical references related to BERT sentiment analysis and Hugging Face pipelines, airline reviews, passenger seating classes, and chi-squared testing to help students understand both the technical and real-world motivation behind the project.