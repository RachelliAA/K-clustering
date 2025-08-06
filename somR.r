
#--------- 1. Load Required Libraries
library(kohonen)  # For SOM
library(reshape2) # For reshaping data for plotting
library(ggplot2)  # For data visualization
library(dplyr)    # For data manipulation
library(tidyr)    # For tidying data
library(gridExtra)
library(grid)


#--------- 2. Load the Data
#liver_data <- read.csv("C:/Users/rache/Desktop/liver.csv")
liver_data <- read.csv("C:/Users/Esther Malka/OneDrive/Documents/year 4/semester 1/סמינר להנדסת תוכנה/K-clustering/liver.csv")

#--------- 3. Data Preprocessing (exclude selector)
numeric_data <- liver_data[sapply(liver_data, is.numeric) & names(liver_data) != "selector"]

#--------- 4. Train SOM with raw values
som_grid <- somgrid(xdim = 6, ydim = 6, topo = "hexagonal")

set.seed(123)
som_model <- som(as.matrix(numeric_data), grid = som_grid, rlen = 2000)

#--------- 5. Define jet color palette
jet.colors <- colorRampPalette(c("blue", "cyan", "green", "yellow", "red"))

#--------- 6. Plot each component plane
par(mfrow = c(3, 3))  # Adjust grid as needed

for (i in 1:ncol(numeric_data)) {
  var_name <- colnames(numeric_data)[i]
  var_values <- som_model$codes[[1]][, i]
  
  # Use zlim based on codebook values — not the original raw data
  zlim_range <- range(var_values, na.rm = TRUE)
  
  plot(som_model, 
       type = "property", 
       property = var_values,
       main = var_name,
       palette.name = jet.colors,
       zlim = zlim_range)  # Use actual SOM scale
}
par(mfrow = c(1,1))  # reset to default single plot layout

#---------------------Fig 4,5-------------------------
# 7. Cluster the SOM codebook vectors using K-Means
set.seed(123)
som_codes <- som_model$codes[[1]]
kmeans_result <- kmeans(som_codes, centers = 5)

# 8. Map each data sample to its cluster according to its winning neuron
unit_classif <- som_model$unit.classif
data_clusters <- kmeans_result$cluster[unit_classif]

# 9. Combine numeric data with cluster assignments
clustered_data <- numeric_data
clustered_data$Cluster <- as.factor(data_clusters)

# 10. Function to plot density plots comparing target cluster to all others
plot_cluster_features <- function(df, numeric_cols = NULL, target_cluster) {
  # Use all numeric columns except "Cluster" if not specified
  if (is.null(numeric_cols)) {
    numeric_cols <- names(df)[sapply(df, is.numeric) & names(df) != "Cluster"]
  }

  # Add group label: "cluster" for target cluster, "population" for all others
  df <- df %>%
    mutate(group = ifelse(Cluster == target_cluster, "cluster", "population"))

  # Convert to long format for ggplot
  df_long <- df %>%
    select(all_of(numeric_cols), group) %>%
    pivot_longer(cols = -group, names_to = "feature", values_to = "value")

  # Create density plots
  p <- ggplot(df_long, aes(x = value, fill = group)) +
    geom_density(alpha = 0.7) +
    facet_wrap(~ feature, scales = "free", ncol = 3) +
    scale_fill_manual(values = c("cluster" = "#ecc4c2", "population" = "#a0dadb")) +
    theme_minimal(base_size = 14) +
    theme(
      strip.text = element_text(face = "bold", size = 13),
      legend.position = "right",
      plot.caption = element_text(hjust = 0.5, face = "italic"),
      plot.title = element_text(hjust = 0.5, size = 16, face = "bold"),
      panel.background = element_rect(fill = "#e5e5e5", color = NA),  # light gray background
      panel.grid.major = element_line(color = "white"),
      panel.grid.minor = element_blank()
    ) +
    labs(
      fill = "Group",
      title = paste("Density plot of cluster", target_cluster)
      # caption = "The figure shows density plots for each feature comparing a specific cluster\nagainst the rest of the population using liver disease data."
    )

  return(p)
}

# 11. Plot density plots for cluster 5 and cluster 1
print(plot_cluster_features(clustered_data, target_cluster = 5))
#print(plot_cluster_features(clustered_data, target_cluster = 1))

