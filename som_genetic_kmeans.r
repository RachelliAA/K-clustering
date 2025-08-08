library(kohonen)   # SOM
library(GA)        # Genetic Algorithm
library(cluster)   # kmeans
library(dplyr)    
library(readr)     
library(purrr) 
library(reshape2)  
library(ggplot2)
library(ggforce)
library(tidyr)
library(gridExtra)
library(grid)

# 1. Load and preprocess the dataset
# df <- read_csv("liver_cleaned.csv")
df <- read_csv("heart_cleaned.csv")

X <- df %>% select(-target) %>% as.matrix()
y <- df$target

# Normalize features (scale to mean=0 sd=1)
X_scaled <- scale(X)

# 2. Train SOM to estimate the number of clusters (k)
som_grid <- somgrid(xdim = 6, ydim = 6, topo = "hexagonal")
set.seed(123)

som_model <- som(X_scaled, grid = som_grid, rlen = 750, alpha = c(0.5, 0.01))
#som_model <- som(X, grid = som_grid, rlen = 750, alpha = c(0.5, 0.01)) # For Fig 2

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
# par(mfrow = c(3, 3))  
# for (i in 1:ncol(X)) {
#   var_name <- colnames(X)[i]
#   var_values <- som_model$codes[[1]][, i]
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
  centroids <- matrix(centroids_vec, nrow = k, byrow = TRUE)
  dists <- apply(X_scaled, 1, function(x) 
    min(sqrt(rowSums((t(centroids) - x)^2))))
  return(-sum(dists))  # Return negative sum (because GA maximizes)
}

# GA parameters
population_size <- 100
generations <- 150

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


# # ========= FIG 3: Function (dynamic colors & k) =========
# plot_fig3 <- function(som_model, X_scaled, kmeans_result, title = "Clusters / Segments") {

#   coords <- som_model$grid$pts
#   unit_classif <- kohonen::map(som_model, newdata = X_scaled)$unit.classif

#   # Assign clusters to SOM nodes via majority vote
#   node_cluster <- sapply(1:nrow(coords), function(i) {
#     inds <- which(unit_classif == i)
#     if (length(inds) > 0) {
#       clust <- kmeans_result$cluster[inds]
#       as.integer(names(sort(table(clust), decreasing = TRUE))[1])
#     } else {
#       NA
#     }
#   })

#   df_plot <- data.frame(
#     x = coords[, 1],
#     y = -coords[, 2],  # flip Y-axis
#     cluster = factor(node_cluster)
#   )

#   # Define simple base colors
# base_colors <- c("#377eb8", "#ff7f00", "#4daf4a", "#e41a1c", "#984ea3",  "yellow", 
#                    "pink", "gray", "cyan", "darkgreen", "darkblue", "gold", "magenta", "darkred")


#   # Map as many colors as needed
#   unique_clusters <- sort(unique(na.omit(as.integer(levels(df_plot$cluster)))))
#   n_clusters <- length(unique_clusters)
#   color_map <- setNames(base_colors[1:n_clusters], as.character(unique_clusters))

#   # Add radius = 0.5 so diameter = 1 unit (touching)
#   df_plot$radius <- 0.5

#   p <- ggplot(df_plot) +
#     ggforce::geom_circle(aes(x0 = x, y0 = y, r = radius, fill = cluster), 
#                          color = "black", size = 0.8) +
#     scale_fill_manual(values = color_map, na.value = "white") +
#     coord_fixed(ratio = 1, expand = FALSE) +
#     ggtitle(title) +
#     theme_void() +
#     theme(
#       plot.title = element_text(hjust = 0.5, size = 16, face = "bold"),
#       legend.position = "none",
#       plot.background = element_rect(fill = "white", color = NA),
#       panel.background = element_rect(fill = "white", color = NA)
#     )

#   return(p)
# }


# # Show it in RStudio
# print(plot_fig3(som_model, X_scaled, kmeans_result))


# # ==== Save directly to file ====
# ggsave("FIG3_SOM_Clusters.pdf", plot = plot_fig3(som_model, X_scaled, kmeans_result), width = 6, height = 6)

# # ========= FIG 3: Visualize SOM with clusters from KMeans =========


# # --------------------- Fig 4,5 -------------------------

# # Reuse the GA + KMeans results
# data_clusters <- kmeans_result$cluster  

# # Combine numeric data with cluster assignments
# numeric_data <- as.data.frame(X)  # original numeric features (unscaled)
# colnames(numeric_data) <- colnames(X)
# numeric_data$Cluster <- as.factor(data_clusters)

# # Define feature order
# feature_order <- c("mcv", "sgpt", "gammagt", "alkphos", "sgot")

# # Function to plot density plots for one cluster
# plot_cluster_features <- function(df, target_cluster) {
#   df <- df %>%
#     mutate(group = ifelse(Cluster == target_cluster, "cluster", "population"))
  
#   df_long <- df %>%
#     select(all_of(feature_order), group) %>%
#     pivot_longer(cols = -group, names_to = "feature", values_to = "value")
  
#   df_long$feature <- factor(df_long$feature, levels = feature_order)  # keep order
  
#   p <- ggplot(df_long, aes(x = value, fill = group)) +
#     geom_density(alpha = 0.7) +
#     facet_wrap(~ feature, scales = "free", ncol = 3) +  # first row will have mcv, sgpt, gammagt
#     scale_fill_manual(values = c("cluster" = "#ecc4c2", "population" = "#a0dadb")) +
#     theme_minimal(base_size = 14) +
#     theme(
#       strip.text = element_text(face = "bold", size = 13),
#       legend.position = "right",
#       plot.title = element_text(hjust = 0.5, size = 16, face = "bold"),
#       panel.background = element_rect(fill = "#e5e5e5", color = NA),
#       panel.grid.major = element_line(color = "white"),
#       panel.grid.minor = element_blank()
#     ) +
#     labs(
#       fill = "Group",
#       title = paste("Density plot of Cluster", target_cluster)
#     )
#   return(p)
# }

# # Save all cluster density plots into a single PDF
# pdf("cluster_density_plots.pdf", width = 10, height = 8)

# for (cl in sort(unique(numeric_data$Cluster))) {
#   print(plot_cluster_features(numeric_data, target_cluster = cl))
# }

# dev.off()

# # --------------------- Fig 4,5 -------------------------


