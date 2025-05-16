# import pandas as pd
# from sklearn.preprocessing import StandardScaler, LabelEncoder
# from sklearn.cluster import KMeans
# from sklearn.decomposition import PCA
# from sklearn.metrics import accuracy_score, confusion_matrix
# from scipy.stats import mode
# import numpy as np
# import matplotlib.pyplot as plt

# # Load your dataset
# df = pd.read_csv("heart_numeric.csv")

# # # Convert categorical columns to numeric
# # def convert_categorical_to_numeric(df):
# #     for col in df.columns:
# #         if df[col].dtype == 'object':
# #             df[col] = LabelEncoder().fit_transform(df[col])
# #     return df

# # df = convert_categorical_to_numeric(df)

# # Separate features and target
# X = df.drop(columns=["target"])
# y_true = df["target"].values

# # Standardize features
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# # Run KMeans
# k = 3#len(np.unique(y_true))  # Choose number of clusters based on number of classes
# kmeans = KMeans(n_clusters=k, random_state=42)
# y_kmeans = kmeans.fit_predict(X_scaled)
# print(f"KMeans Labels: {np.unique(y_kmeans)}")

# # Map cluster labels to true labels
# def map_clusters_to_labels(y_true, y_pred):
#     labels = np.zeros_like(y_pred) # Initialize with zeros
#     for i in range(len(np.unique(y_pred))):
#         mask = (y_pred == i)
#         labels[mask] = mode(y_true[mask], keepdims=True)[0]
#     return labels

# y_pred_mapped = map_clusters_to_labels(y_true, y_kmeans)

# # Evaluate
# accuracy = accuracy_score(y_true, y_pred_mapped)
# cm = confusion_matrix(y_true, y_pred_mapped)

# print(f"KMeans Accuracy: {accuracy:.2f}")
# print("Confusion Matrix:")
# print(cm)

# # Visualize
# pca = PCA(n_components=8)
# X_pca = pca.fit_transform(X_scaled)

# plt.figure(figsize=(8, 6))
# plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_pred_mapped, cmap='viridis', alpha=0.6)
# plt.title(f'KMeans Clustering (mapped to true labels), Accuracy = {accuracy:.2f}')
# plt.xlabel("PC1")
# plt.ylabel("PC2")
# plt.grid(True)
# plt.colorbar(label='Predicted Cluster (Mapped)')
# plt.show()



# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import MinMaxScaler, Normalizer, RobustScaler, StandardScaler
# from sklearn.cluster import KMeans
# from sklearn.metrics import confusion_matrix, accuracy_score
# from scipy.stats import mode
# import matplotlib.pyplot as plt

# # Load dataset
# df = pd.read_csv("heart_numeric.csv")  # Adjust path if needed

# # Separate features and labels
# X = df.drop(columns=["target"])
# y_true = df["target"].values

# # Standardize features
# scaler = RobustScaler() #Normalizer() #MinMaxScaler() #StandardScaler()
# #scaler = minMaxscaller() #StandardScaler()
# X_scaled = scaler.fit_transform(X)

# # Run KMeans
# k = len(np.unique(y_true)) #=2 # Set number of clusters equal to number of unique target classes
# kmeans = KMeans(n_clusters=k, random_state=42)
# y_kmeans = kmeans.fit_predict(X_scaled)

# # Function to map predicted clusters to true labels
# def map_clusters_to_labels(y_true, y_pred):
#     labels = np.zeros_like(y_pred)
#     for i in range(len(np.unique(y_pred))):
#         mask = (y_pred == i)
#         labels[mask] = mode(y_true[mask], keepdims=True)[0]
#     return labels

# y_pred_mapped = map_clusters_to_labels(y_true, y_kmeans)

# # Standard Accuracy
# accuracy = accuracy_score(y_true, y_pred_mapped)
# print(f"Standard Accuracy: {accuracy:.2f}")

# # Confusion Matrix
# cm = confusion_matrix(y_true, y_pred_mapped)
# print("Confusion Matrix:")
# print(cm)

# # Weighted Classification Accuracy
# def weighted_classification_accuracy(y_true, y_pred):
#     cm = confusion_matrix(y_true, y_pred)
#     total_samples = np.sum(cm)
#     weighted_acc = 0
#     for i in range(len(cm)):
#         class_samples = np.sum(cm[i])
#         correct_predictions = cm[i, i]
#         class_weight = class_samples / total_samples
#         class_accuracy = correct_predictions / class_samples if class_samples > 0 else 0
#         weighted_acc += class_weight * class_accuracy
#     return weighted_acc * 100

# wca = weighted_classification_accuracy(y_true, y_pred_mapped)
# print(f"Weighted Classification Accuracy: {wca:.2f}%")

# # Optional: PCA for visualization
# from sklearn.decomposition import PCA

# pca = PCA(n_components=2)
# X_pca = pca.fit_transform(X_scaled)

# plt.figure(figsize=(8, 6))
# plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_pred_mapped, cmap='viridis', alpha=0.7)
# plt.title(f'K-Means Clustering (mapped), Accuracy = {accuracy:.2f}, WCA = {wca:.2f}%')
# plt.xlabel("PCA Component 1")
# plt.ylabel("PCA Component 2")
# plt.colorbar(label='Predicted Cluster (Mapped)')
# plt.grid(True)
# plt.show()

import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Load dataset
df = pd.read_csv("heart_processed_data.csv")

# Only use relevant features (13 features)
X = df.drop(columns=["target"])
y_true = df["target"].values

# Normalize features using MinMaxScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Run KMeans with 2 clusters (as per paper)
kmeans = KMeans(n_clusters=2, random_state=0)
y_kmeans = kmeans.fit_predict(X_scaled)

# Raw accuracy (just to observe, doesn't mean much here)
accuracy = accuracy_score(y_true, y_kmeans)
cm = confusion_matrix(y_true, y_kmeans)

print(f"Raw (Unmapped) Accuracy: {accuracy:.2f}")
print("Confusion Matrix (Unmapped):")
print(cm)

# Weighted Classification Accuracy
def weighted_classification_accuracy(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    total = np.sum(cm)
    correct = 0
    for cluster in range(cm.shape[1]):
        correct += np.max(cm[:, cluster])
    return 100 * correct / total

wca = weighted_classification_accuracy(y_true, y_kmeans)
print(f"Weighted Classification Accuracy: {wca:.2f}%")

# Visualize
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_kmeans, cmap='viridis', alpha=0.7)
plt.title(f'K-Means Clustering (Raw), Accuracy = {accuracy:.2f}')
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.colorbar(label='Cluster Label')
plt.grid(True)
plt.show()
