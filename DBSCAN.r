# 1. Load required libraries
library(dbscan)       # For DBSCAN clustering
library(dplyr)        # For data manipulation
library(caret)        # For data standardization (centering & scaling)
library(readr)        # For reading CSV files
library(modeest)      # For computing the most frequent value (mode)
library(e1071)        # For computing classification accuracy

# # 2. Load the liver dataset
# df <- read_csv("liver.csv")   # Uncomment this line to use liver dataset
# # 3. Separate features and labels
# X <- df %>% select(-selector)    # Features (excluding the label column)
# y_true <- df$selector            # Ground truth labels

# 2. Load the heart dataset
df <- read_csv("heart_processed_data.csv")  # Use the heart dataset
# 3. Separate features and labels

X <- df %>% select(-target)    # Features (excluding the label column)
y_true <- df$target            # Ground truth labels

# 4. Standardize the features (mean = 0, std = 1)
preproc <- preProcess(X, method = c("center", "scale"))
X_scaled <- predict(preproc, X)

# 5. Apply DBSCAN
# db <- dbscan(X_scaled, eps = 1.1, minPts = 1)  # For liver dataset
db <- dbscan(X_scaled, eps = 3, minPts = 1)      # For heart dataset

y_pred <- db$cluster   # Cluster labels assigned by DBSCAN

# 6. Filter out noise points (cluster label 0 in R means noise)
mask <- y_pred != 0
y_pred_filtered <- y_pred[mask]
y_true_filtered <- y_true[mask]

# 7. Map DBSCAN cluster labels to actual labels using majority voting
mapped_preds <- numeric(length(y_pred_filtered))

unique_clusters <- unique(y_pred_filtered)
for (cluster_id in unique_clusters) {
  cluster_indices <- which(y_pred_filtered == cluster_id)
  true_labels <- y_true_filtered[cluster_indices]
  
  if (length(true_labels) > 0) {
    most_common <- mfv(true_labels)  # Most frequent true label
    mapped_preds[cluster_indices] <- most_common
  }
}

# 8. Calculate classification accuracy
accuracy <- mean(mapped_preds == y_true_filtered)
cat(sprintf("Classification Accuracy (DBSCAN): %.4f\n", accuracy))

## choose a good eps value for DBSCAN by plotting the k-distance graph and identifying the "elbow" point.
## Load required libraries
# library(FNN)     # For computing k-nearest neighbors
# library(ggplot2) # For plotting graphs

# # Step 1: Set the number of nearest neighbors to use in DBSCAN
# k <- 13  # Usually, MinPts = number of dimensions + 1. Here: 12 columns → try k = 13

# # Step 2: Compute the distance to the k-th nearest neighbor for each point
# # The get.knn function returns a matrix of distances to the nearest neighbors
# knn_dist <- get.knn(X_scaled, k = k)$nn.dist[, k]
# # We select the k-th column (distance to the k-th nearest neighbor)

# # Step 3: Sort the distances in ascending order
# knn_dist_sorted <- sort(knn_dist)

# # Step 4: Create a data.frame for ggplot with the sorted distances
# df <- data.frame(
#   point_index = 1:length(knn_dist_sorted),  # Index of points (X-axis)
#   dist = knn_dist_sorted                    # k-distance values (Y-axis)
# )

# # Step 5: Plot the k-distance graph using ggplot2
# p <- ggplot(df, aes(x = point_index, y = dist)) +  # Map variables to axes
#   geom_line() +                                    # Line plot
#   labs(
#     title = "K-distance plot for DBSCAN",
#     x = "Points sorted by distance",
#     y = paste0(k, "-nearest neighbor distance")    # Dynamic Y-axis label
#   ) +
#   theme_minimal()  # Minimal plot theme

# # Display the plot
# print(p)

