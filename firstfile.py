import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler





import pandas as pd
from sklearn.preprocessing import LabelEncoder

def convert_categorical_to_numeric(df):
    """
    Converts non-numeric (categorical) columns in a DataFrame to numeric using Label Encoding.
    
    Parameters:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: A DataFrame with all columns converted to numeric.
    """
    label_encoders = {}
    
    for col in df.columns:
        if df[col].dtype == 'object':
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le  # Store encoder for possible inverse_transform
    
    return df, label_encoders



# # Load the dataset
df = pd.read_csv('heart.csv')

# # Convert categorical columns
df_numeric, encoders = convert_categorical_to_numeric(df)

print(df_numeric.head())

df= df_numeric.copy()
# Load your dataset
#df = pd.read_csv('heart.csv')  # Replace with your actual dataset path

# Assume last column is label (if unsupervised, you can ignore it)
X = df.iloc[:, :-1].values

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Reduce to 2D using PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Plot the data
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c='gray', alpha=0.7)
plt.title('PCA Projection (2D)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.grid(True)
plt.show()
