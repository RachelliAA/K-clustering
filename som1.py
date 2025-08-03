# import numpy as np
# import pandas as pd

# class SOM:
#     def __init__(self, grid_size, input_dim, learning_rate=0.3):
#         self.weights = np.random.rand(grid_size[0], grid_size[1], input_dim)
#         self.grid_size = grid_size
#         self.learning_rate = learning_rate

#     def find_bmu(self, x):
#         distances = np.linalg.norm(self.weights - x, axis=2)
#         return np.unravel_index(np.argmin(distances), distances.shape)

#     def update_weights(self, x, bmu_idx, radius=1):
#         for i in range(self.grid_size[0]):
#             for j in range(self.grid_size[1]):
#                 dist_to_bmu = np.linalg.norm([i - bmu_idx[0], j - bmu_idx[1]])
#                 if dist_to_bmu <= radius:
#                     influence = np.exp(-dist_to_bmu**2 / (2 * radius**2))
#                     self.weights[i, j] += self.learning_rate * influence * (x - self.weights[i, j])

#     def train(self, data, epochs=10):
#         for epoch in range(epochs):
#             for sample in data:
#                 bmu = self.find_bmu(sample)
#                 self.update_weights(sample, bmu, radius=max(self.grid_size) * (1 - epoch / epochs))


# def genetic_kmeans(data, population_size=50, generations=175):
#     som = SOM(grid_size=(5, 5), input_dim=data.shape[1])
#     som.train(data)
    
#     bmu_counts = np.zeros(som.grid_size)
#     for sample in data:
#         bmu = som.find_bmu(sample)
#         bmu_counts[bmu] += 1
#     k = np.count_nonzero(bmu_counts)
#     print(f"Estimated number of clusters (k) from SOM: {k}")

#     population = [np.random.choice(data.flatten(), size=(k, data.shape[1])) for _ in range(population_size)]

#     for _ in range(generations):
#         fitness = []
#         for individual in population:
#             distances = np.linalg.norm(data[:, np.newaxis] - individual, axis=2)
#             min_distances = np.min(distances, axis=1)
#             fitness.append(-np.sum(min_distances**2))

#         parents = []
#         for _ in range(population_size):
#             candidates = np.random.choice(range(population_size), size=3)
#             parents.append(population[candidates[np.argmax([fitness[c] for c in candidates])]])

#         new_population = []
#         for i in range(0, population_size, 2):
#             crossover_point = np.random.randint(1, k)
#             child1 = np.vstack((parents[i][:crossover_point], parents[i + 1][crossover_point:]))
#             child2 = np.vstack((parents[i + 1][:crossover_point], parents[i][crossover_point:]))
#             new_population.extend([child1, child2])

#         population = []
#         for individual in new_population:
#             if np.random.rand() < 0.1:
#                 mutation_point = np.random.randint(k)
#                 individual[mutation_point] += np.random.normal(0, 0.1, size=data.shape[1])
#             population.append(individual)

#     best_individual = population[np.argmax(fitness)]

#     # Assign each point to the nearest cluster center
#     distances = np.linalg.norm(data[:, np.newaxis] - best_individual, axis=2)
#     labels = np.argmin(distances, axis=1)

#     return labels, k


# # --- Main Execution ---
# # Load data from CSV
# df = pd.read_csv("liver.csv")

# # Use only features (exclude 'selector' column for clustering)
# features = df.drop(columns=["selector"]).values

# # Normalize the data (recommended for SOM and clustering)
# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# data_scaled = scaler.fit_transform(features)

# # Apply the clustering
# labels, k = genetic_kmeans(data_scaled)

# # --- Classification Accuracy Evaluation ---
# true_labels = df["selector"].values
# total_points = len(true_labels)
# weighted_accuracy_sum = 0

# for cluster_id in np.unique(labels):
#     indices = np.where(labels == cluster_id)[0]
#     cluster_labels = true_labels[indices]
#     majority_class = np.bincount(cluster_labels).argmax()
#     correct = np.sum(cluster_labels == majority_class)
#     cluster_accuracy = correct / len(indices)
#     weighted_accuracy_sum += len(indices) * cluster_accuracy

# weighted_avg_accuracy = weighted_accuracy_sum / total_points
# print(f"Weighted Average Classification Accuracy: {weighted_avg_accuracy:.4f}")


##################
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from collections import Counter
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

class SOM:
    def __init__(self, grid_size=(5, 5), input_dim=10, learning_rate=0.5):
        self.grid_size = grid_size
        self.input_dim = input_dim
        self.learning_rate = learning_rate
        self.weights = np.random.rand(grid_size[0], grid_size[1], input_dim)

    def find_bmu(self, x):
        distances = np.linalg.norm(self.weights - x, axis=2)
        return np.unravel_index(np.argmin(distances), distances.shape)

    def update_weights(self, x, bmu_idx, radius):
        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                distance_to_bmu = np.linalg.norm([i - bmu_idx[0], j - bmu_idx[1]])
                if distance_to_bmu <= radius:
                    influence = np.exp(-(distance_to_bmu**2) / (2 * (radius**2)))
                    self.weights[i, j] += self.learning_rate * influence * (x - self.weights[i, j])

    def train(self, data, epochs=10):
        for epoch in range(epochs):
            radius = max(self.grid_size) * (1 - epoch / epochs)
            for x in data:
                bmu_idx = self.find_bmu(x)
                self.update_weights(x, bmu_idx, radius)

def initialize_population(data, k, population_size):
    return [data[np.random.choice(len(data), k, replace=False)] for _ in range(population_size)]

def evaluate_fitness(individual, data):
    distances = np.linalg.norm(data[:, np.newaxis] - individual, axis=2)
    return -np.sum(np.min(distances, axis=1))

def tournament_selection(population, fitnesses, tournament_size=3):
    selected = []
    for _ in range(len(population)):
        indices = np.random.choice(len(population), tournament_size)
        best = indices[np.argmax([fitnesses[i] for i in indices])]
        selected.append(population[best])
    return selected

def crossover(parent1, parent2):
    point = np.random.randint(1, len(parent1))
    child1 = np.vstack((parent1[:point], parent2[point:]))
    child2 = np.vstack((parent2[:point], parent1[point:]))
    return child1, child2

def mutate(individual, mutation_rate, data_shape):
    if np.random.rand() < mutation_rate:
        idx = np.random.randint(len(individual))
        individual[idx] += np.random.normal(0, 0.1, data_shape[1])
    return individual

def genetic_kmeans(data, population_size=50, generations=175):
    som = SOM(grid_size=(5, 5), input_dim=data.shape[1])
    som.train(data)

    bmu_map = {}
    for x in data:
        bmu = som.find_bmu(x)
        if bmu not in bmu_map:
            bmu_map[bmu] = []
        bmu_map[bmu].append(x)

    k = len(bmu_map)
    print(f"Estimated number of clusters (k) from SOM: {k}")

    population = initialize_population(data, k, population_size)

    for _ in range(generations):
        fitnesses = [evaluate_fitness(ind, data) for ind in population]
        parents = tournament_selection(population, fitnesses)

        next_generation = []
        for i in range(0, population_size, 2):
            child1, child2 = crossover(parents[i], parents[i + 1])
            child1 = mutate(child1, 0.1, data.shape)
            child2 = mutate(child2, 0.1, data.shape)
            next_generation.extend([child1, child2])

        population = next_generation

    best_individual = population[np.argmax([evaluate_fitness(ind, data) for ind in population])]
    distances = np.linalg.norm(data[:, np.newaxis] - best_individual, axis=2)
    labels = np.argmin(distances, axis=1)

    return labels, k

# --- Main Execution ---

# Load dataset
df = pd.read_csv("liver.csv")

# Drop the 'selector' column for clustering
X = df.drop(columns=["selector"]).values
y = df["selector"].values

# Normalize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Run clustering
labels, k = genetic_kmeans(X_scaled)

# --- Classification Accuracy Evaluation ---
total_points = len(y)
weighted_sum = 0

for cluster_id in np.unique(labels):
    indices = np.where(labels == cluster_id)[0]
    cluster_labels = y[indices]
    if len(cluster_labels) == 0:
        continue
    most_common = Counter(cluster_labels).most_common(1)[0][0]
    correct = np.sum(cluster_labels == most_common)
    accuracy = correct / len(indices)
    weighted_sum += accuracy * len(indices)

weighted_accuracy = weighted_sum / total_points
print(f"Weighted Average Classification Accuracy: {weighted_accuracy:.4f}")
