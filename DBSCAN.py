import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics import accuracy_score
from scipy.stats import mode

# 1. Load the liver dataset
# df = pd.read_csv("liver.csv")  # Ensure 'selector' is the ground-truth label

# # 2. Separate features and labels
# X = df.drop(columns=['selector'])  # Features
# y_true = df['selector'].to_numpy()  # Ground truth labels


# 1. Load the heart dataset
df = pd.read_csv("heart_numeric.csv")  # Ensure 'selector' is the ground-truth label

# 2. Separate features and labels
X = df.drop(columns=['target'])  # Features
y_true = df['target'].to_numpy()  # Ground truth labels

# 3. Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Apply DBSCAN
# dbscan = DBSCAN(eps=1.1, min_samples=1) # for liver 67.54%
dbscan = DBSCAN(eps=5, min_samples=3) # for heart 72.52%
y_pred = dbscan.fit_predict(X_scaled)

# 5. Filter out noise (-1 label)
mask = y_pred != -1
y_pred_filtered = y_pred[mask]
y_true_filtered = y_true[mask]

# 6. Map DBSCAN cluster labels to actual labels using majority voting
labels_map = {}
for cluster_id in np.unique(y_pred_filtered):
    true_labels = y_true_filtered[y_pred_filtered == cluster_id]
    if len(true_labels) > 0:
        most_common = mode(true_labels, keepdims=True)[0][0]
        labels_map[cluster_id] = most_common

# 7. Generate mapped predictions
mapped_preds = np.array([labels_map[cluster] for cluster in y_pred_filtered])

# 8. Calculate accuracy
accuracy = accuracy_score(y_true_filtered, mapped_preds)
print(f"Classification Accuracy (DBSCAN): {accuracy:.4f}")


# ================================================
# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import StandardScaler
# from sklearn.neighbors import NearestNeighbors
# import matplotlib.pyplot as plt

# # 1. Load dataset
# df = pd.read_csv("heart_numeric.csv")
# X = df.drop(columns=['target'])  # Adjust column name if needed

# # 2. Standardize
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# # 3. k-distance plot
# min_samples = 4  # You can tune this
# neighbors = NearestNeighbors(n_neighbors=min_samples)
# neighbors_fit = neighbors.fit(X_scaled)
# distances, indices = neighbors_fit.kneighbors(X_scaled)

# k_distances = np.sort(distances[:, -1])
# plt.figure(figsize=(8, 5))
# plt.plot(k_distances)
# plt.axhline(y=k_distances[int(len(k_distances) * 0.9)], color='r', linestyle='--', label='90th percentile')
# plt.title(f'k-Distance Graph (k={min_samples - 1})')
# plt.xlabel('Points sorted by distance')
# plt.ylabel(f'{min_samples}-NN Distance')
# plt.legend()
# plt.grid(True)
# plt.show()
