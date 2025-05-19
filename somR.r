#--------- 1. Load Required Libraries
#install.packages("kohonen")
#install.packages("ggplot2")  

library(kohonen)
library(ggplot2)


#--------- 2. Load the data
liver_data <- read.csv("C:/Users/rache/Desktop/liver.csv")

# View the first few rows
head(liver_data)

#--------- 3. Data Preprocessing
# Keep only numeric columns
numeric_data <- liver_data[sapply(liver_data, is.numeric)]

# Scale the numeric data
data_scaled <- scale(numeric_data)

#--------- 4. Train the Self-Organizing Map (SOM)
# Define SOM grid
som_grid <- somgrid(xdim = 6, ydim = 6, topo = "hexagonal")

# Train the SOM
set.seed(123)
som_model <- som(data_scaled, grid = som_grid, rlen = 100)

#--------- 5. Visualize the SOM
# Plot each variable's distribution across the map
par(mfrow = c(2, 2))  # Adjust grid if you have more/less variables

for (i in 1:ncol(data_scaled)) {
  plot(som_model, type = "property", property = som_model$codes[[1]][, i],
       main = colnames(data_scaled)[i], palette.name = terrain.colors)
}

#--------- 6. Cluster Analysis
# Hierarchical clustering of codebook vectors
som_codes <- som_model$codes[[1]]
dist_matrix <- dist(som_codes)
hc <- hclust(dist_matrix)
clusters <- cutree(hc, k = 4)  # Adjust 'k' as needed

# Plot with clusters
plot(som_model, type = "mapping", bgcol = rainbow(4)[clusters], main = "SOM Clusters")
add.cluster.boundaries(som_model, clusters)
