"""
This script performs the chi-squared test and prints the results. It also prints out other useful statistics, such as the contingency table and proportions. 
You can run it by entering the command 'python3 chi_squared_test' in the command line after activating the environment and importing the packages. 
You can also run it by pressing the triangle to run program in VScode. 
"""
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

#Imports data from the csv with the sentiment column
df = pd.read_csv('../data/airlines_reviews_with_sentiment.csv')

#Filters for Business and Economy class only
df_filtered = df[df['Class'].isin(['Business Class', 'Economy Class'])].copy()

#Creates and prints contingency table for sentiment by class
contingency_table = pd.crosstab(df_filtered['Class'], df_filtered['sentiment'])
print("Contingency Table:")
print(contingency_table)
print("\n")

#Calculates and prints proportions of positive reviews for each class
print("Proportions of Positive Reviews by Class:")
for class_type in ['Business Class', 'Economy Class']:
    class_data = df_filtered[df_filtered['Class'] == class_type]
    total = len(class_data)
    positive = len(class_data[class_data['sentiment'] == 'positive'])
    proportion = positive / total if total > 0 else 0
    print(f"{class_type}: {positive}/{total} = {proportion:.4f} ({proportion*100:.2f}%)")

#Performs chi-squared test and prints all the results
chi2, p_value, dof, expected_freq = chi2_contingency(contingency_table)
print("\nChi-Squared Test Results:")
print(f"Chi-squared statistic: {chi2:.4f}")
print(f"P-value: {p_value:.4f}")
print(f"Degrees of freedom: {dof}")
print(f"\nExpected frequencies:")
print(pd.DataFrame(expected_freq, index=contingency_table.index, columns=contingency_table.columns))
