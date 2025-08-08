###############################   heart dataset  ###################################################

# ---- STEP 1: Load the data ----
data <- read.csv("C:/Users/rache/Desktop/seminar/Kclustering/heart_processed_data.csv")

# ---- STEP 2: Convert 'True'/'False' columns to numeric (1/0) ----
# Identify logical columns and convert them
data$thal_fixed <- as.numeric(data$thal_fixed == "True")
data$thal_normal <- as.numeric(data$thal_normal == "True")
data$thal_reversible <- as.numeric(data$thal_reversible == "True")

# ---- STEP 3: Extract target and features ----
target <- data$target  # Used only for evaluation
features <- data[, !colnames(data) %in% c("target")]

# ---- STEP 4: Scale numeric features ----
scaled_features <- scale(features)

# ---- STEP 5: Run K-Means ----
set.seed(123)
k <- 40
kmeans_result <- kmeans(scaled_features, centers = k)

# ---- STEP 6: Evaluate Clustering Accuracy ----
evaluate_accuracy <- function(clusters, true_labels) {
  sum(sapply(unique(clusters), function(cl) {
    idx <- which(clusters == cl)
    majority <- names(which.max(table(true_labels[idx])))
    sum(true_labels[idx] == majority)
  })) / length(true_labels)
}

accuracy <- evaluate_accuracy(kmeans_result$cluster, target)
cat("k = ",k ," Weighted Classification Accuracy:", round(accuracy * 100, 2), "%\n")
# # ---- STEP 7: Visualize Clusters ----
# library(ggplot2)
# library(cluster)
# # Create a data frame for plotting
# plot_data <- data.frame(scaled_features, cluster = as.factor(kmeans_result$cluster))
# ggplot(plot_data, aes(x = scaled_features[, 1], y = scaled_features[, 2], color = cluster)) +
#   geom_point() +
#   labs(title = "K-Means Clustering of Heart Disease Data",
#        x = "Feature 1 (scaled)",
#        y = "Feature 2 (scaled)",
#        color = "Cluster") +
#   theme_minimal()



###############################   liver dataset  ###################################################
# ---- STEP 1: Load the data ----
data <- read.csv("C:/Users/rache/Desktop/seminar/Kclustering/liver_cleaned.csv")

# ---- STEP 2: Extract target and features ----
target <- data$target  # Used only for evaluation
features <- data[, !colnames(data) %in% c("target")]

# ---- STEP 2: Scale numeric features ----
scaled_features <- scale(features)

# ---- STEP 2: Run K-Means ----
set.seed(123)
k <- 60
kmeans_result <- kmeans(scaled_features, centers = k)

# ---- STEP 2: Evaluate Clustering Accuracy ----
evaluate_accuracy <- function(clusters, true_labels) {
  sum(sapply(unique(clusters), function(cl) {
    idx <- which(clusters == cl)
    majority <- names(which.max(table(true_labels[idx])))
    sum(true_labels[idx] == majority)
  })) / length(true_labels)
}

accuracy <- evaluate_accuracy(kmeans_result$cluster, target)
cat("k = ",k ," Weighted Classification Accuracy:", round(accuracy * 100, 2), "%\n")

