#preprocessing heart.csv
import pandas as pd

# Load original CSV
df = pd.read_csv('heart.csv')

# One-hot encode the 'thal' column
df_encoded = pd.get_dummies(df, columns=['thal']) #This will create new columns for each unique value in 'thal'

# Save new CSV with one-hot encoded 'thal'
df_encoded.to_csv('heart_processed_data.csv', index=False)
