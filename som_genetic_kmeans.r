library(kohonen)   # SOM
library(GA)        # Genetic Algorithm
library(cluster)   # kmeans
library(dplyr)    
library(readr)     
library(purrr) 
library(reshape2) # לציור FIG 2
library(ggplot2)
library(tidyr)
library(gridExtra)
library(grid)


# 1. Load and preprocess the dataset
df <- read_csv("liver_cleaned.csv")
X <- df %>% select(-target) %>% as.matrix()
y <- df$target

# # Use only the 5 blood tests as features
# X <- df %>% select(mcv, alkphos, sgpt, sgot, gammagt) %>% as.matrix() 
# # Create binary classification target from 'drinks'
# y_raw <- df$drinks
# y <- ifelse(y_raw >= 1, 1, 0)


# df <- read_csv("heart_processed_data.csv")
# X <- df %>% select(-target) %>% as.matrix()
# y <- df$target


# Normalize features (scale to mean=0 sd=1)
X_scaled <- scale(X)

# 2. Train SOM to estimate the number of clusters (k)
som_grid <- somgrid(xdim = 6, ydim = 6, topo = "hexagonal")
set.seed(123)

som_model <- som(X_scaled, grid = som_grid, rlen = 750, alpha = c(0.5, 0.01))
# som_model <- som(X, grid = som_grid, rlen = 750, alpha = c(0.5, 0.01))

# Get BMU indices for each sample
bmu_indices <- som_model$unit.classif

# Count how many data points mapped to each BMU (neuron)
bmu_table <- table(bmu_indices)

# Set threshold: only count BMUs with more than X% of data points
threshold_ratio <- 0.05
min_count <- threshold_ratio * nrow(X_scaled)

# Filter BMUs with sufficient density
dense_bmus <- bmu_table[bmu_table > min_count]

dense_codebook <- som_model$codes[[1]][as.integer(names(dense_bmus)), ]

# Count unique BMUs above the threshold = estimated k
k <- nrow(dense_codebook)
cat("Estimated number of dense clusters (k) from SOM:", k, "\n")


# # ========= FIG 2: Plot component planes =========
# jet.colors <- colorRampPalette(c("blue", "cyan", "green", "yellow", "red"))

# # Plot each component plane
# par(mfrow = c(3, 3))  

# for (i in 1:ncol(X)) {
#   var_name <- colnames(X)[i]
#   var_values <- som_model$codes[[1]][, i]
  
#   # Use zlim based on codebook values — not the original raw data
#   zlim_range <- range(var_values, na.rm = TRUE)
  
#   plot(som_model, 
#        type = "property", 
#        property = var_values,
#        main = var_name,
#        palette.name = jet.colors,
#        zlim = zlim_range)  # Use actual SOM scale
# }
# par(mfrow = c(1,1))  # reset to default single plot layout
# # ========= FIG 2: Plot component planes =========

# 3. Genetic Algorithm to find optimal centroids based on SOM codebook vectors
data_for_ga <- som_model$codes[[1]]  # SOM codebook vectors (centroids)

# Fitness function: minimize sum of distances from data points to closest centroid
fitness_function <- function(centroids_vec) {
  # Reshape vector to matrix k x n_features
  centroids <- matrix(centroids_vec, nrow = k, byrow = TRUE)
  
  # For each data point, find distance to closest centroid
  dists <- apply(X_scaled, 1, function(x) min(sqrt(rowSums((t(centroids) - x)^2))))
  # Return negative sum (because GA maximizes)
  return(-sum(dists))
}

# GA parameters
population_size <- 100
generations <- 150
n_features <- ncol(X_scaled)

suggestion <- matrix(
  rep(as.vector(t(dense_codebook)), population_size),
  nrow = population_size,
  byrow = TRUE
)


# Run GA
set.seed(123)
ga_result <- ga(
  type = "real-valued",
  fitness = fitness_function,
  lower = rep(apply(X_scaled, 2, min), k),
  upper = rep(apply(X_scaled, 2, max), k),
  popSize = population_size,
  maxiter = generations,
  run = 50,
  suggestions = suggestion
)

# Best centroids from GA (reshape vector to matrix)
best_centroids <- matrix(ga_result@solution, nrow = k, byrow = TRUE)

# 4. Run K-Means with GA-optimized centroids
set.seed(42)
kmeans_result <- kmeans(X_scaled, centers = best_centroids, iter.max = 300, nstart = 1)


# ========= FIG 3: Visualize SOM with clusters from KMeans =========

# # Step 1: Get SOM grid coordinates
# coords <- som_model$grid$pts  # positions of neurons in SOM grid

# # Step 2: Map each sample to its BMU (Best Matching Unit)
# unit_classif <- kohonen::map(som_model, newdata = X_scaled)$unit.classif

# # Step 3: Assign each SOM node to a cluster based on majority vote
# node_cluster <- sapply(1:nrow(coords), function(i) {
#   inds <- which(unit_classif == i)
#   if (length(inds) > 0) {
#     clust <- kmeans_result$cluster[inds]
#     as.integer(names(sort(table(clust), decreasing = TRUE))[1])  # dominant cluster
#   } else {
#     NA  # node has no mapped data
#   }
# })

# # Step 4: Prepare dataframe for plotting
# df_plot <- data.frame(
#   x = coords[, 1],
#   y = coords[, 2],
#   cluster = factor(node_cluster)
# )

# # Step 5: Define cluster colors (fixed colors as in article)
# cluster_colors <- c("1" = "blue", "2" = "orange", "3" = "red", "4" = "purple", "5" = "green")

# # Step 6: Plot SOM with fixed cluster colors
# ggplot(df_plot, aes(x = x, y = y, fill = cluster)) +
#   geom_point(shape = 21, color = "black", size = 8, stroke = 0.5) +
#   scale_fill_manual(values = cluster_colors, na.value = "white") +
#   ggtitle("FIG 3: Cluster Representation of Liver Disease Data") +
#   theme_minimal() +
#   theme(
#     plot.title = element_text(hjust = 0.5, size = 16, face = "bold"),
#     axis.title = element_blank(),
#     axis.text = element_blank(),
#     axis.ticks = element_blank(),
#     panel.grid = element_blank()
#   )


# ========= FIG 3:  =========


# 5. Evaluate clustering accuracy (weighted by cluster size)
weighted_accuracy <- function(clusters, true_labels) {
  total_points <- length(true_labels)
  weighted_sum <- 0
  
  for (cluster_id in unique(clusters)) {
    indices <- which(clusters == cluster_id)
    cluster_labels <- true_labels[indices]
    if (length(cluster_labels) == 0) next
    
    most_common <- names(sort(table(cluster_labels), decreasing = TRUE))[1]
    correct <- sum(cluster_labels == most_common)
    weighted_sum <- weighted_sum + correct
  }
  return(weighted_sum / total_points)
}

accuracy <- weighted_accuracy(kmeans_result$cluster, y)
cat(sprintf("Weighted Classification Accuracy: %.4f\n", accuracy))


