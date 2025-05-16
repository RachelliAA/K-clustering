# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from minisom import MiniSom
# from sklearn.preprocessing import MinMaxScaler

# # === Load CSV ===
# df = pd.read_csv("liver.csv")

# # === Drop label column if present (assume last column is label) ===
# if df.columns[-1].lower() in ['selector', 'class', 'label', 'target']:
#     X = df.iloc[:, :-1].values
#     y = df.iloc[:, -1].values
# else:
#     X = df.values
#     y = None

# # === Scale the Data ===
# scaler = MinMaxScaler()
# X_scaled = scaler.fit_transform(X)

# # === Define and Train SOM ===
# som_shape = (6, 6)  # You can adjust this based on dataset size
# som = MiniSom(som_shape[0], som_shape[1], X_scaled.shape[1], sigma=1.0, learning_rate=0.5)
# som.random_weights_init(X_scaled)
# som.train_batch(X_scaled, num_iteration=1000)

# # === Get BMUs and Cluster Labels ===
# bmu_indices = np.array([som.winner(x) for x in X_scaled])
# som_labels = bmu_indices[:, 0] * som_shape[1] + bmu_indices[:, 1]

# # === Add SOM Labels to DataFrame ===
# df['som_cluster'] = som_labels

# # === Visualize Heatmaps ===
# plt.figure(figsize=(12, 8))
# for i in range(X.shape[1]):
#     plt.subplot(2, (X.shape[1] + 1) // 2, i + 1)
#     feature_map = som.distance_map().T  # transpose for correct orientation
#     sns.heatmap(feature_map, cmap='coolwarm', cbar=True)
#     plt.title(df.columns[i])
# plt.tight_layout()
# plt.suptitle("SOM Distance Map for Each Feature", fontsize=16, y=1.05)
# plt.show()

# # === Print Cluster Assignment ===
# print("SOM cluster label counts:")
# print(df['som_cluster'].value_counts())

# # === Optional: Save output ===
# df.to_csv("liver_with_som_clusters.csv", index=False)


# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from minisom import MiniSom
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.cluster import KMeans
# from sklearn.metrics import accuracy_score
# from scipy.stats import mode
# from matplotlib.patches import Circle
# from matplotlib import cm

# # === Load Dataset ===
# df = pd.read_csv("liver.csv")

# # === Separate Features and Label ===
# if df.columns[-1].lower() in ['selector', 'class', 'label', 'target']:
#     X = df.iloc[:, :-1].values
#     y = df.iloc[:, -1].values.astype(int)
#     feature_names = df.columns[:-1]
# else:
#     X = df.values
#     y = None
#     feature_names = df.columns

# # === Normalize Features ===
# scaler = MinMaxScaler()
# X_scaled = scaler.fit_transform(X)

# # === Train SOM ===
# som_shape = (6, 6)
# som = MiniSom(som_shape[0], som_shape[1], X.shape[1], sigma=1.0, learning_rate=0.5)
# som.random_weights_init(X_scaled)
# som.train_batch(X_scaled, 1000)

# # === Visualization Function (component planes) ===
# import matplotlib.pyplot as plt
# from matplotlib.patches import Circle
# from matplotlib import cm
# import numpy as np

# def plot_som_component_planes(som, feature_names):
#     weights = som.get_weights()  # Shape: [x, y, features]
#     grid_x, grid_y, num_features = weights.shape
    
#     for feature_index in range(num_features):
#         data = weights[:, :, feature_index]
        
#         plt.figure(figsize=(6, 6))
#         ax = plt.gca()  # Get the current axes
#         plt.title(feature_names[feature_index], fontsize=14)

#         # Normalize to real feature scale (not 0–1)
#         vmin = data.min()
#         vmax = data.max()

#         for x in range(grid_x):
#             for y in range(grid_y):
#                 value = data[x, y]
#                 color = cm.jet((value - vmin) / (vmax - vmin))  # scale color
#                 circ = Circle((x, y), radius=0.45, color=color)
#                 ax.add_patch(circ)

#         # Axis config
#         ax.set_xlim(-0.5, grid_x - 0.5)
#         ax.set_ylim(-0.5, grid_y - 0.5)
#         ax.set_xticks([])
#         ax.set_yticks([])
#         ax.set_aspect('equal')
#         ax.invert_yaxis()

#         # Colorbar using real feature values
#         sm = plt.cm.ScalarMappable(cmap='jet', norm=plt.Normalize(vmin=vmin, vmax=vmax))
#         sm.set_array([])  # This prevents a warning when using colorbar with no data
#         cbar = plt.colorbar(sm, ax=ax, orientation='vertical')
#         cbar.set_label(feature_names[feature_index])

#         plt.tight_layout()
#         plt.show()

# plot_som_component_planes(som, feature_names)
# # === SOM BMU Clustering (convert 2D BMU to flat cluster ID) ===
# bmu_coords = np.array([som.winner(x) for x in X_scaled])
# som_labels = bmu_coords[:, 0] * som_shape[1] + bmu_coords[:, 1]
# df['som_cluster'] = som_labels

# # === Use number of unique BMUs as K for KMeans ===
# k = len(np.unique(som_labels))
# kmeans = KMeans(n_clusters=k, random_state=42)
# cluster_labels = kmeans.fit_predict(X_scaled)

# df['kmeans_cluster'] = cluster_labels

# # === Accuracy Metrics ===
# def compute_accuracy(true_labels, cluster_labels):
#     labels = np.zeros_like(cluster_labels)
#     for i in np.unique(cluster_labels):
#         mask = cluster_labels == i
#         labels[mask] = mode(true_labels[mask], keepdims=False)[0]
#     return accuracy_score(true_labels, labels)

# def compute_weighted_accuracy(true_labels, cluster_labels):
#     total = len(true_labels)
#     weighted_acc = 0
#     for i in np.unique(cluster_labels):
#         mask = cluster_labels == i
#         acc = np.sum(mode(true_labels[mask], keepdims=False)[0] == true_labels[mask]) / len(mask)
#         weighted_acc += len(mask) * acc
#     return weighted_acc / total

# # === Compute and Print Results ===
# if y is not None:
#     acc = compute_accuracy(y, cluster_labels)
#     w_acc = compute_weighted_accuracy(y, cluster_labels)
#     print(f"\n🔍 Clustering Accuracy: {acc * 100:.2f}%")
#     print(f"📊 Weighted Accuracy:  {w_acc * 100:.2f}%")
# else:
#     print("No true class labels found. Cannot compute accuracy.")


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from minisom import MiniSom

# Load dataset
df = pd.read_csv('liver.csv')

# Normalize data
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(df)

# Initialize and train SOM
som_x, som_y = 6, 6  # Grid size
som = MiniSom(x=som_x, y=som_y, input_len=data_scaled.shape[1], sigma=1.0, learning_rate=0.5)
som.random_weights_init(data_scaled)
som.train_random(data_scaled, num_iteration=1000)

# --------------------------
# 1. U-Matrix Visualization
# --------------------------
plt.figure(figsize=(7, 7))
plt.title("SOM Distance Map (U-Matrix)", fontsize=14)
u_matrix = som.distance_map().T  # transpose for correct orientation
plt.imshow(u_matrix, cmap='bone_r', origin='lower')
plt.colorbar(label='Neuron Distance')

# Overlay neuron locations
for x in range(som_x):
    for y in range(som_y):
        plt.plot(x + 0.5, y + 0.5, 'o', markerfacecolor='none', markeredgecolor='black')

plt.xticks([])
plt.yticks([])
plt.tight_layout()
plt.show()

# -------------------------------------
# 2. Feature-Specific Heatmaps (optional)
# -------------------------------------
import math
cols = 3
rows = math.ceil(len(df.columns) / cols)
fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
axes = axes.flatten()

for i, feature in enumerate(df.columns):
    weights = som.get_weights()[:, :, i]
    norm_weights = (weights - weights.min()) / (weights.max() - weights.min())
    cmap = plt.cm.get_cmap('jet')

    ax = axes[i]
    ax.set_title(feature)
    for x in range(som_x):
        for y in range(som_y):
            color = cmap(norm_weights[x, y])
            circle = plt.Circle((x + 0.5, y + 0.5), 0.4, color=color, ec='black')
            ax.add_patch(circle)

    ax.set_xlim(0, som_x)
    ax.set_ylim(0, som_y)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])

    sm = plt.cm.ScalarMappable(cmap='jet', norm=plt.Normalize(vmin=weights.min(), vmax=weights.max()))
    sm.set_array([])
    fig.colorbar(sm, ax=ax)

# Hide unused plots
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

fig.suptitle("Fig. 2: Heatmaps for visualisation of possible Self-Organizing Maps", fontsize=16, y=1.02)
plt.figtext(0.5, 0.01,
            "The figure shows the visualization of SOM for each variable in the Liver dataset.\n"
            "Each plot is scaled to real values to see the true relationship. The color bar shows the range.",
            ha="center", fontsize=10)
plt.tight_layout(rect=[0, 0.05, 1, 0.98])
plt.show()

