####################################plane kmeans on liver data
# # ---- STEP 1: Load the Data ----
# data <- read.csv("C:/Users/rache/Desktop/seminar/Kclustering/liver.csv")


# # ---- STEP 2: Create Binary Target from 'drinks' ----
# # Dichotomize: 0 = drinks <= 2, 1 = drinks > 2
# data$drinks_label <- ifelse(data$drinks > 2, 1, 0)

# # ---- STEP 3: Use Only Feature Columns (exclude selector, drinks, and drinks_label) ----
# features <- data[, c("mcv", "alkphos", "sgpt", "sgot", "gammagt")]

# # ---- STEP 4: Scale the Features ----
# scaled_features <- scale(features)

# # ---- STEP 5: Run K-Means ----
# set.seed(123)
# k <- 2  # Since the label is binary
# kmeans_result <- kmeans(scaled_features, centers = k, nstart = 10)

# # ---- STEP 6: Evaluate Clustering Accuracy Using 'drinks_label' ----
# true_labels <- data$drinks_label

# evaluate_accuracy <- function(clusters, labels) {
#   sum(sapply(unique(clusters), function(cl) {
#     idx <- which(clusters == cl)
#     majority <- names(which.max(table(labels[idx])))
#     sum(labels[idx] == majority)
#   })) / length(labels)
# }

# accuracy <- evaluate_accuracy(kmeans_result$cluster, true_labels)
# cat("Weighted Classification Accuracy (based on drinks dichotomy):", round(accuracy * 100, 2), "%\n")

########################## SOM and then kmeans
library(kohonen)

# ---- STEP 2: Load and Prepare Data ----
data <- read.csv("C:/Users/rache/Desktop/seminar/Kclustering/liver.csv")

# Create binary label from 'drinks' (used only for evaluation)
#data$drinks_label <- ifelse(data$drinks > 2, 1, 0)

# Select only features (exclude drinks, selector, drinks_label)
features <- data[, c("mcv","alkphos","sgpt","sgot","gammagt","drinks")]

# Scale features
scaled_features <- scale(features)

# ---- STEP 3: Train SOM ----
set.seed(123)
som_grid <- somgrid(xdim = 6, ydim = 6, topo = "hexagonal")  # 6x6 grid
som_model <- som(as.matrix(scaled_features), grid = som_grid, rlen = 100)

# ---- STEP 4: Estimate Number of Clusters (k) from SOM ----
bmu <- som_model$unit.classif
estimated_k <- length(unique(bmu))  # Number of unique BMUs
cat("Estimated number of clusters (k) from SOM:", estimated_k, "\n")

# ---- STEP 5: Run K-Means Using Estimated k ----
set.seed(123)
kmeans_result <- kmeans(scaled_features, centers = estimated_k, nstart = 10)

# ---- STEP 6: Evaluate Clustering Accuracy with 'drinks_label' ----
true_labels <- data$drinks_label

evaluate_accuracy <- function(clusters, labels) {
  sum(sapply(unique(clusters), function(cl) {
    idx <- which(clusters == cl)
    majority <- names(which.max(table(labels[idx])))
    sum(labels[idx] == majority)
  })) / length(labels)
}

accuracy <- evaluate_accuracy(kmeans_result$cluster, true_labels)
cat("Weighted Classification Accuracy (based on drinks dichotomy):", round(accuracy * 100, 2), "%\n")
#Estimated number of clusters (k) from SOM: 36 
#Weighted Classification Accuracy (based on drinks dichotomy): 64.35 %
