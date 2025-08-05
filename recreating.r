
#recreatinng the article from gpt

# Install if not already installed
#install.packages(c("kohonen", "GA", "cluster", "factoextra", "ggplot2", "reshape2"))

library(kohonen)      # For SOM
library(GA)           # For Genetic Algorithm
library(cluster)      # For clustering and visualization
library(factoextra)   # For clustering visualization
library(ggplot2)      # For plotting
library(reshape2)     # For heatmaps


#loade the csv safely
liver <- read.csv("C:/Users/rache/Desktop/seminar/Kclustering/liver.csv", header = TRUE, strip.white = TRUE)  # Removes leading/trailing spaces

# Assign column names
colnames(liver) <- c("mcv","alkphos","sgpt","sgot","gammagt","drinks","selector")

# Normalize data for SOM
#liver_scaled <- scale(liver[,1:6])



######
# Grid: 6x6 as in the paper (36 neurons)
som_grid <- somgrid(xdim = 6, ydim = 6, topo = "hexagonal")

# Train SOM
set.seed(123)
som_model <- som(liver_scaled, grid = som_grid, rlen = 100)

#--------- 5. Define jet color palette
jet.colors <- colorRampPalette(c("blue", "cyan", "green", "yellow", "red"))

# Visualize heatmaps (Figure 2)
par(mfrow=c(2,3))
for(i in 1:ncol(liver_scaled)) {
  plot(som_model, type="property", property=som_model$codes[[1]][,i],
       main=colnames(liver_scaled)[i], palette.name=jet.colors)
}


#**************************************************************************
library(kohonen)
library(ggplot2)

# Load CSV
liver <- read.csv("C:/Users/rache/Desktop/seminar/Kclustering/liver.csv", 
                  header = TRUE, strip.white = TRUE)

#colnames(liver) <- c("mcv","alkphos","sgpt","sgot","gammagt","drinks","selector")

# Convert first 6 columns to numeric and scale
liver[,1:6] <- lapply(liver[,1:6], function(x) as.numeric(gsub("[^0-9\\.]", "", x)))
#liver_scaled <- scale(liver[,1:6])

# SOM grid (6x6 rectangular)
som_grid <- somgrid(xdim = 6, ydim = 6, topo = "hexagonal")

# Train SOM
set.seed(123)
som_model <- som(liver, grid = som_grid, rlen = 100)

# Jet color palette
jet.colors <- colorRampPalette(c("blue", "cyan", "green", "yellow", "red"))

# Visualize SOM property heatmaps (Figure 2)
par(mfrow = c(2, 3))  # 6 features

for (i in 1:ncol(liver_numeric)) {
  feature_name <- colnames(liver_numeric)[i]
  feature_values <- som_model$codes[[1]][, i]
  
  # Force the color scale to original feature min/max
  original_range <- range(liver_numeric[, i], na.rm = TRUE)
  
  plot(som_model,
       type = "property",
       property = feature_values,
       main = feature_name,
       palette.name = jet.colors,
       zlim = original_range)  # <-- force scale to real values
}
