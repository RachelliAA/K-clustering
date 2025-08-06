# #preprocessing heart.csv one hot encoding
# import pandas as pd

# # Load original CSV
# df = pd.read_csv('heart.csv')

# # One-hot encode the 'thal' column
# df_encoded = pd.get_dummies(df, columns=['thal']) #This will create new columns for each unique value in 'thal'

# # Save new CSV with one-hot encoded 'thal'
# df_encoded.to_csv('heart_processed_data2.csv', index=False)



# this also takes out the 2 rows with 1 0r 2 in thal
import pandas as pd

# Load original CSV
df = pd.read_csv('heart.csv')

# Ensure 'thal' is treated as string
df['thal'] = df['thal'].astype(str)

# Manually loop and mark rows to delete
rows_to_drop = []

for index, row in df.iterrows():
    if row['thal'] in ['1', '2']:
        rows_to_drop.append(index)

# Drop the marked rows
df.drop(index=rows_to_drop, inplace=True)

# One-hot encode the cleaned 'thal' column
df_encoded = pd.get_dummies(df, columns=['thal'])

# Save cleaned and encoded data
df_encoded.to_csv('heart_processed_data.csv', index=False)

print(f"Dropped {len(rows_to_drop)} rows where thal was '1' or '2'.")
