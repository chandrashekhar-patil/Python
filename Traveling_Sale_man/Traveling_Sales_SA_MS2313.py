import numpy as np
import matplotlib.pyplot as plt
from geopy.geocoders import Nominatim
import random
import math

# Function to calculate Euclidean distance
def distance(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

# Function to calculate the total distance of a tour
def total_distance(tour, cities):
    dist = 0.0
    n = len(tour)
    for i in range(n):
        dist += distance(cities[tour[i]], cities[tour[(i + 1) % n]])
    return dist

# Function to read city names and get their coordinates
def read_cities(filename):
    cities = []
    names = []
    geolocator = Nominatim(user_agent="TSP_App")
    
    with open(filename, 'r') as file:
        for line in file:
            city = line.strip()
            if city:
                location = geolocator.geocode(city, timeout=10)
                if location:
                    x, y = round(location.longitude, 2), round(location.latitude, 2)
                    cities.append((x, y))
                    names.append(city)
                else:
                    print(f"Could not geocode city: {city}")
    return cities, names

# Function to plot the tour
def plot_tour(tour, cities, dist, names):
    tour_points = [cities[i] for i in tour] + [cities[tour[0]]]
    tour_points = np.array(tour_points)
    
    plt.figure(figsize=(12, 10))
    plt.plot(tour_points[:, 0], tour_points[:, 1], 'o-')
    for i, name in enumerate(names):
        plt.annotate(name, (cities[i][0], cities[i][1]), textcoords="offset points", xytext=(0, 5), ha='center')
    plt.title(f'TSP Solution with Total Distance: {dist:.2f}')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.grid(True)
    plt.show()

# Function to perform simulated annealing
def simulated_annealing(cities, names, temp=10000, cooling_rate=0.995, min_temp=1):
    n = len(cities)
    current_solution = list(range(n))
    best_solution = current_solution[:]
    current_distance = total_distance(current_solution, cities)
    best_distance = current_distance
    
    while temp > min_temp:
        new_solution = current_solution[:]
        i, j = random.sample(range(n), 2)
        new_solution[i], new_solution[j] = new_solution[j], new_solution[i]
        
        new_distance = total_distance(new_solution, cities)
        if new_distance < current_distance or random.uniform(0, 1) < math.exp((current_distance - new_distance) / temp):
            current_solution = new_solution[:]
            current_distance = new_distance
            if current_distance < best_distance:
                best_solution = current_solution[:]
                best_distance = current_distance
        
        temp *= cooling_rate
    
    return best_solution, best_distance

def main():
    filename = 'India_cities.txt'
    cities, names = read_cities(filename)
    
    if len(cities) < 2:
        print("Not enough cities to compute the TSP.")
        return

    best_tour, best_distance = simulated_annealing(cities, names)
    
    print(f"Best distance: {best_distance:.2f}")
    print(f"Best tour: {best_tour}")
    
    plot_tour(best_tour, cities, best_distance, names)

if __name__ == "__main__":
    main()
