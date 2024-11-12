import numpy as np
import matplotlib.pyplot as plt
from geopy.geocoders import Nominatim
import random
import math

# Parameters for Ant Colony Optimization
NUM_ANTS = 10
NUM_ITERATIONS = 10
ALPHA = 1  # Pheromone importance
BETA = 2   # Distance importance
EVAPORATION_RATE = 0.5
Q = 10  # Pheromone constant

# Function to calculate Euclidean distance
def distance(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

# Function to calculate total tour distance
def total_distance(tour, cities):
    return sum(distance(cities[tour[i]], cities[tour[(i + 1) % len(tour)]]) for i in range(len(tour)))

# Function to read city names and coordinates
def read_cities(filename):
    cities = []
    names = []
    geolocator = Nominatim(user_agent="TSP_App")
    with open(filename, 'r') as file:
        for line in file:
            city = line.strip()
            location = geolocator.geocode(city, timeout=10)
            if location:
                cities.append((round(location.longitude, 2), round(location.latitude, 2)))
                names.append(city)
            else:
                print(f"Could not geocode city: {city}")
    return cities, names

# Plot function for the current best tour
def plot_tour(tour, cities, dist, names, iteration):
    tour_points = [cities[i] for i in tour] + [cities[tour[0]]]
    tour_points = np.array(tour_points)
    plt.figure(figsize=(10, 8))
    plt.plot(tour_points[:, 0], tour_points[:, 1], 'o-')
    for i, name in enumerate(names):
        plt.annotate(name, (cities[i][0], cities[i][1]), textcoords="offset points", xytext=(0, 5), ha='center')
    plt.title(f'ACO Iteration {iteration} - Distance: {dist:.2f}')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.grid(True)
    plt.show()

# Ant Colony Optimization function
def ant_colony_optimization(cities, names):
    num_cities = len(cities)
    pheromones = np.ones((num_cities, num_cities))
    best_tour = None
    best_distance = float('inf')
    
    for iteration in range(NUM_ITERATIONS):
        all_tours = []
        all_distances = []
        
        for ant in range(NUM_ANTS):
            tour = [random.randint(0, num_cities - 1)]
            while len(tour) < num_cities:
                i = tour[-1]
                probabilities = []
                for j in range(num_cities):
                    if j not in tour:
                        prob = (pheromones[i][j] ** ALPHA) * ((1 / distance(cities[i], cities[j])) ** BETA)
                        probabilities.append((j, prob))
                
                # Choose next city based on probability distribution
                next_city = max(probabilities, key=lambda x: x[1])[0]
                tour.append(next_city)
                
            dist = total_distance(tour, cities)
            all_tours.append(tour)
            all_distances.append(dist)
            
            if dist < best_distance:
                best_tour = tour
                best_distance = dist
        
        # Update pheromones
        pheromones *= (1 - EVAPORATION_RATE)
        for i in range(NUM_ANTS):
            for j in range(num_cities):
                pheromones[all_tours[i][j]][all_tours[i][(j + 1) % num_cities]] += Q / all_distances[i]

        # Visualize the best tour at this iteration
        plot_tour(best_tour, cities, best_distance, names, iteration + 1)
    
    return best_tour, best_distance

def main():
    filename = 'India_cities.txt'
    cities, names = read_cities(filename)
    
    if len(cities) < 2:
        print("Not enough cities to compute the TSP.")
        return

    best_tour, best_distance = ant_colony_optimization(cities, names)
    
    print(f"Best distance found by ACO: {best_distance:.2f}")
    print("Best tour:", [names[i] for i in best_tour])
    plot_tour(best_tour, cities, best_distance, names, "Final Solution")

if __name__ == "__main__":
    main()
