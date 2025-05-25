# ---- STEP 1: Load Libraries ----
install.packages(c("kohonen", "GA", "cluster", "factoextra", "dplyr", "ggplot2"))
library(kohonen)
library(GA)
library(factoextra)
library(cluster)
library(dplyr)
library(ggplot2)

# ---- STEP 2: Load and Prepare Data ----
data <- read.csv("C:/Users/rache/Desktop/liver.csv")
features <- scale(data[, 1:6]) # normalize features
labels <- data$selector        # 1 or 2

# ---- STEP 3: SOM to Estimate Optimal Clusters ----
som_grid <- somgrid(xdim = 6, ydim = 6, topo = "hexagonal")
som_model <- som(features, grid = som_grid, rlen = 100, alpha = c(0.05, 0.01))
plot(som_model, type = "codes")

# Estimate optimal number of clusters (rule of thumb: 1 neuron ~ 10 data points)
total_neurons <- 6 * 6
estimated_clusters <- 5  # You can adjust based on visual or heuristic

# ---- STEP 4: Genetic Algorithm to Find Optimal Centroids ----
k <- estimated_clusters
d <- ncol(features)

# Fitness Function
fitness_function <- function(centroids_vec) {
  centroids <- matrix(centroids_vec, nrow = k, byrow = TRUE)
  clusters <- apply(features, 1, function(x) which.min(colSums((t(centroids) - x)^2)))
  sum(sapply(1:k, function(i) sum(rowSums((features[clusters == i, , drop = FALSE] - centroids[i, ])^2))))
}

lower <- rep(apply(features, 2, min), k)
upper <- rep(apply(features, 2, max), k)

ga_result <- ga(type = "real-valued",
                fitness = function(x) -fitness_function(x),
                lower = lower, upper = upper,
                maxiter = 50, popSize = 50, run = 20)

best_centroids <- matrix(ga_result@solution, nrow = k, byrow = TRUE)

# ---- STEP 5: K-Means with GA-Initiated Centroids ----
set.seed(123)
kmeans_result <- kmeans(features, centers = best_centroids, iter.max = 100, nstart = 1)

# ---- STEP 6: Evaluate Clustering Accuracy ----
evaluate_accuracy <- function(cluster, true_labels) {
  sum(sapply(unique(cluster), function(cl) {
    idx <- which(cluster == cl)
    majority <- as.numeric(names(sort(table(true_labels[idx]), decreasing = TRUE))[1])
    sum(true_labels[idx] == majority)
  })) / length(true_labels)
}

accuracy <- evaluate_accuracy(kmeans_result$cluster, labels)
cat("Weighted Classification Accuracy:", round(accuracy * 100, 2), "%\n")

# ---- STEP 7: Visualize Clusters ----
fviz_cluster(list(data = features, cluster = kmeans_result$cluster),
             ellipse.type = "convex", geom = "point", ggtheme = theme_minimal())

# ---- STEP 8: Visualize Density Plots for Cluster 1 and 5 ----
plot_density <- function(cluster_num) {
  cluster_data <- as.data.frame(features)
  cluster_data$cluster <- kmeans_result$cluster
  cluster_data$in_cluster <- ifelse(cluster_data$cluster == cluster_num, paste("Cluster", cluster_num), "Others")

  cluster_data_long <- cluster_data %>%
    select(-cluster) %>%
    pivot_longer(-in_cluster, names_to = "Feature", values_to = "Value")

  ggplot(cluster_data_long, aes(x = Value, fill = in_cluster)) +
    geom_density(alpha = 0.5) +
    facet_wrap(~Feature, scales = "free", ncol = 3) +
    ggtitle(paste("Density Plot for Cluster", cluster_num)) +
    theme_minimal()
}

plot_density(1)  # like Figure 5 in article
plot_density(5)  # like Figure 4 in article (if 5 clusters)


#####################################from perplexity
# Install required packages if needed
if (!require("kohonen")) install.packages("kohonen")
if (!require("cluster")) install.packages("cluster")
if (!require("GA")) install.packages("GA")

library(kohonen)
library(cluster)
library(GA)

# Load and preprocess data
liver_data <- read.csv("liver_data.csv", header = TRUE)
numeric_data <- scale(liver_data[, c("mcv", "alkphos", "sgpt", "sgot", "gammagt", "drinks")])

# Flexible Parameter 1: SOM grid dimensions
som_grid <- somgrid(xdim = 6, ydim = 4, topo = "hexagonal") 

# Train SOM
set.seed(123)
som_model <- som(numeric_data, 
                 grid = som_grid, 
                 rlen = 1000,
                 alpha = c(0.05, 0.01),
                 keep.data = TRUE)

# Determine optimal clusters from SOM
# Flexible Parameter 2: Clustering method
som_clusters <- cutree(hclust(dist(som_model$codes[[1]])), k = 3) 

# Get suggested k from SOM
optimal_k <- length(unique(som_clusters))

# Genetic K-Means implementation
# Flexible Parameter 3: GA parameters
ga_fitness <- function(clusters) {
  centers <- matrix(clusters, ncol = ncol(numeric_data))
  sum(kmeans(numeric_data, centers = centers)$withinss)
}

ga_result <- ga(type = "real-valued",
                fitness = function(x) -ga_fitness(x),
                lower = rep(min(numeric_data), optimal_k * ncol(numeric_data)),
                upper = rep(max(numeric_data), optimal_k * ncol(numeric_data)),
                popSize = 50,
                maxiter = 500,
                run = 100,
                mutation = ga_pmutation)

# Extract best centers
best_centers <- matrix(ga_result@solution[1,], 
                       ncol = ncol(numeric_data), 
                       byrow = TRUE)

# Final clustering
final_clusters <- kmeans(numeric_data, centers = best_centers)

# Visualization
plot(som_model, type = "codes", main = "SOM Codebook Vectors")
plot(som_model, type = "mapping", bgcol = rainbow(optimal_k)[som_clusters])


#############################################################
# ---- STEP 1: Install and Load Minimal Packages ----
install.packages(c("kohonen", "GA", "cluster", "ggplot2"))
library(kohonen)
library(GA)
library(cluster)
library(ggplot2)

# ---- STEP 2: Load and Normalize Your Liver Data ----
data <- read.csv("C:/Users/rache/Desktop/liver.csv")
features <- scale(data[, 1:6])  # Normalize the numeric features
labels <- data$selector         # Save the class labels for evaluation

# ---- STEP 3: Train SOM to Get Topological Prototypes ----
som_grid <- somgrid(xdim = 6, ydim = 6, topo = "hexagonal")
set.seed(123)
som_model <- som(as.matrix(features), grid = som_grid, rlen = 100, alpha = c(0.05, 0.01))

# ---- STEP 4: Estimate Optimal Number of Clusters from SOM ----
bmu_mapping <- som_model$unit.classif
TData <- som_model$codes[[1]]  # The SOM prototypes (36 nodes in 6x6 grid)
k <- length(unique(bmu_mapping))  # Number of unique BMUs actually used

# ---- STEP 5: Genetic Algorithm to Optimize Initial Centroids ----
d <- ncol(features)  # number of features

fitness_function <- function(centroids_vec) {
  centroids <- matrix(centroids_vec, nrow = k, byrow = TRUE)
  clusters <- apply(features, 1, function(x) which.min(colSums((t(centroids) - x)^2)))
  sum(sapply(1:k, function(i) sum(rowSums((features[clusters == i, , drop = FALSE] - centroids[i, ])^2))))
}

lower <- rep(apply(features, 2, min), k)
upper <- rep(apply(features, 2, max), k)

set.seed(123)
ga_result <- ga(type = "real-valued",
                fitness = function(x) -fitness_function(x),
                lower = lower, upper = upper,
                maxiter = 50, popSize = 50, run = 20)

centroids <- matrix(ga_result@solution, nrow = k, byrow = TRUE)

# ---- STEP 6: K-Means Clustering Using GA-Optimized Centroids ----
set.seed(123)
kmeans_result <- kmeans(features, centers = centroids, iter.max = 100, nstart = 1)
clusters <- kmeans_result$cluster
data$cluster <- clusters  # Add cluster labels to original data

# ---- STEP 7: Evaluate Clustering Accuracy (Optional) ----
evaluate_accuracy <- function(cluster, true_labels) {
  sum(sapply(unique(cluster), function(cl) {
    idx <- which(cluster == cl)
    majority <- as.numeric(names(sort(table(true_labels[idx]), decreasing = TRUE))[1])
    sum(true_labels[idx] == majority)
  })) / length(true_labels)
}

cat("Weighted Classification Accuracy:",
    round(evaluate_accuracy(clusters, labels) * 100, 2), "%\n")

# ---- STEP 8: Basic Cluster Visualization Using First 2 Features ----
df <- as.data.frame(features)
df$cluster <- as.factor(clusters)

ggplot(df, aes(x = df[,1], y = df[,2], color = cluster)) +
  geom_point(size = 2) +
  labs(title = "K-Means Clusters (First 2 Features)") +
  theme_minimal()
##################################################################################3
library(kohonen)

# ---- STEP 2: Load and Clean Data ----
data <- read.csv("C:/Users/rache/Desktop/seminar/Kclustering/heart_processed_data.csv")

# Convert 'True'/'False' to numeric 1/0
data$thal_fixed <- as.numeric(data$thal_fixed == "True")
data$thal_normal <- as.numeric(data$thal_normal == "True")
data$thal_reversible <- as.numeric(data$thal_reversible == "True")

# Separate target
target <- data$target
features <- data[, !colnames(data) %in% c("target")]

# ---- STEP 3: Normalize Features ----
scaled_features <- scale(features)

# ---- STEP 4: Train SOM ----
set.seed(123)
som_grid <- somgrid(xdim = 6, ydim = 6, topo = "hexagonal")  # 6x6 grid = 36 nodes
som_model <- som(as.matrix(scaled_features), grid = som_grid, rlen = 100)

# ---- STEP 5: Estimate k from SOM ----
# Number of unique BMUs = estimated k
bmu <- som_model$unit.classif
estimated_k <- length(unique(bmu))

cat("Estimated number of clusters from SOM (k):", estimated_k, "\n")

# ---- STEP 6: Run K-Means Using Estimated k ----
set.seed(123)
kmeans_result <- kmeans(scaled_features, centers = estimated_k, nstart = 10)

# ---- STEP 7: Evaluate Weighted Accuracy ----
evaluate_accuracy <- function(clusters, true_labels) {
  sum(sapply(unique(clusters), function(cl) {
    idx <- which(clusters == cl)
    majority <- names(which.max(table(true_labels[idx])))
    sum(true_labels[idx] == majority)
  })) / length(true_labels)
}

accuracy <- evaluate_accuracy(kmeans_result$cluster, target)
cat("Weighted Classification Accuracy:", round(accuracy * 100, 2), "%\n")
# Estimated number of clusters from SOM (k): 33 
# Weighted Classification Accuracy: 84.72 %
> 
