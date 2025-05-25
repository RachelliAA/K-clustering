# ---- STEP 1: Load and Prepare the Data ----
data <- read.csv("heart_processed_data2.csv")

# Convert target to a factor (not used in clustering, only for evaluation)
target <- data$target

# Select features only (remove target)
features <- data[, !colnames(data) %in% c("target")]

# Optionally scale features (recommended for K-Means)
scaled_features <- scale(features)

# ---- STEP 2: Run K-Means ----
set.seed(123)
k <- 2  # You expect two groups: target 0 and 1
kmeans_result <- kmeans(scaled_features, centers = k, nstart = 10)

# ---- STEP 3: Evaluate Weighted Classification Accuracy ----

# Define accuracy evaluator
evaluate_accuracy <- function(clusters, true_labels) {
  sum(sapply(unique(clusters), function(cl) {
    idx <- which(clusters == cl)
    majority_class <- names(which.max(table(true_labels[idx])))
    sum(true_labels[idx] == majority_class)
  })) / length(true_labels)
}

# Compute weighted accuracy
accuracy <- evaluate_accuracy(kmeans_result$cluster, target)

cat("Weighted Classification Accuracy:", round(accuracy * 100, 2), "%\n")
