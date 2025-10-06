import pandas as pd

# Load dataset
df = pd.read_csv('Phishing_Email.csv')

# Peek at first 5 rows
print("First 5 emails:")
print(df.head())

# Check size and labels
print("\nDataset shape (rows, columns):", df.shape)
print("Label breakdown:")
print(df['Email Type'].value_counts())

# Basic stats
print("\nAverage email length (chars):", df['Email Text'].str.len().mean())