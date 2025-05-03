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
df = pd.read_csv("heart.csv")  # Ensure 'selector' is the ground-truth label

# 2. Separate features and labels
X = df.drop(columns=['target'])  # Features
y_true = df['target'].to_numpy()  # Ground truth labels

# 3. Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Apply DBSCAN
dbscan = DBSCAN(eps=1.1, min_samples=1)
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
# from sklearn.preprocessing import StandardScaler
# from sklearn.cluster import DBSCAN
# from sklearn.metrics import accuracy_score
# from scipy.stats import mode
# from sklearn.decomposition import PCA
# import matplotlib.pyplot as plt
# import numpy as np

# # Load dataset
# df = pd.read_csv("liver.csv")

# # Split features and labels
# X = df.drop(columns=['selector'])
# y_true = df['selector']

# # Standardize
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# # Optional: reduce dimensionality to denoise data (can help DBSCAN)
# pca = PCA(n_components=6)
# X_pca = pca.fit_transform(X_scaled)

# # Tuning DBSCAN: try a few values
# best_accuracy = 0
# best_eps = None
# best_min_samples = None
# best_pred = None

# for eps in [1.5, 2, 2.5, 3, 3.5, 4]:
#     for min_samples in [3, 5, 10]:
#         dbscan = DBSCAN(eps=eps, min_samples=min_samples)
#         y_pred = dbscan.fit_predict(X_pca)

#         # Filter noise
#         mask = y_pred != -1
#         if np.sum(mask) == 0:
#             continue

#         y_pred_filtered = y_pred[mask]
#         y_true_filtered = y_true[mask].to_numpy()

#         # Map cluster labels
#         labels_map = {}
#         for cluster in np.unique(y_pred_filtered):
#             mask_cluster = y_pred_filtered == cluster
#             true_labels = y_true_filtered[mask_cluster]
#             if len(true_labels) == 0:
#                 continue
#             most_common = mode(true_labels, keepdims=True)[0][0]
#             labels_map[cluster] = most_common

#         mapped_preds = np.array([labels_map[cluster] for cluster in y_pred_filtered])
#         accuracy = accuracy_score(y_true_filtered, mapped_preds)

#         if accuracy > best_accuracy:
#             best_accuracy = accuracy
#             best_eps = eps
#             best_min_samples = min_samples
#             best_pred = y_pred

# print(f"Best DBSCAN Accuracy: {best_accuracy:.4f} with eps={best_eps}, min_samples={best_min_samples}")

# ====================================================
# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import StandardScaler
# from sklearn.neighbors import NearestNeighbors
# import matplotlib.pyplot as plt

# # 1. Load dataset
# df = pd.read_csv("liver.csv")
# X = df.drop(columns=['selector'])  # Adjust column name if needed

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
