import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import random
import math
from geopy.geocoders import Nominatim

def distance(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def total_distance(tour, cities):
    dist = 0.0
    n = len(tour)
    for i in range(n):
        dist += distance(cities[tour[i]], cities[tour[(i + 1) % n]])
    return dist

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

def plot_tour(tour, cities, dist, names, generation):
    tour_points = [cities[i] for i in tour] + [cities[tour[0]]]
    tour_points = np.array(tour_points)
    
    plt.figure(figsize=(12, 10))
    plt.plot(tour_points[:, 0], tour_points[:, 1], 'o-')
    for i, name in enumerate(names):
        plt.annotate(name, (cities[i][0], cities[i][1]), textcoords="offset points", xytext=(0, 5), ha='center')
    plt.title(f'Generation {generation} - TSP Solution with Total Distance: {dist:.2f}')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.grid(True)
    plt.show()

def init_population(pop_size, n_cities):
    population = []
    for _ in range(pop_size):
        population.append(random.sample(range(n_cities), n_cities))
    return population

def rank_population(population, cities):
    fitness_results = {}
    for i, tour in enumerate(population):
        fitness_results[i] = 1 / total_distance(tour, cities)  # Fitness is inverse of distance
    return sorted(fitness_results.items(), key=lambda x: x[1], reverse=True)

def selection(pop_ranked, elite_size):
    selection_results = []
    df = np.cumsum([x[1] for x in pop_ranked])
    df /= df[-1]
    
    for i in range(elite_size):
        selection_results.append(pop_ranked[i][0])
    for i in range(len(pop_ranked) - elite_size):
        pick = random.random()
        for j in range(len(pop_ranked)):
            if pick <= df[j]:
                selection_results.append(pop_ranked[j][0])
                break
    return selection_results

def mating_pool(population, selection_results):
    return [population[i] for i in selection_results]

def crossover(parent1, parent2):
    gene_a = int(random.random() * len(parent1))
    gene_b = int(random.random() * len(parent1))
    
    start_gene = min(gene_a, gene_b)
    end_gene = max(gene_a, gene_b)

    child_p1 = parent1[start_gene:end_gene]
    child_p2 = [item for item in parent2 if item not in child_p1]

    return child_p1 + child_p2

def crossover_population(matingpool, elite_size):
    children = []
    length = len(matingpool) - elite_size
    pool = random.sample(matingpool, len(matingpool))

    for i in range(elite_size):
        children.append(matingpool[i])

    for i in range(length):
        child = crossover(pool[i], pool[len(matingpool) - i - 1])
        children.append(child)
    return children

def mutate(individual, mutation_rate):
    for swapped in range(len(individual)):
        if random.random() < mutation_rate:
            swap_with = int(random.random() * len(individual))

            city1 = individual[swapped]
            city2 = individual[swap_with]

            individual[swapped] = city2
            individual[swap_with] = city1
    return individual

def mutate_population(population, mutation_rate):
    mutated_pop = []
    
    for individual in range(len(population)):
        mutated_ind = mutate(population[individual], mutation_rate)
        mutated_pop.append(mutated_ind)
    return mutated_pop

def next_generation(current_gen, elite_size, mutation_rate, cities):
    pop_ranked = rank_population(current_gen, cities)
    selection_results = selection(pop_ranked, elite_size)
    matingpool = mating_pool(current_gen, selection_results)
    children = crossover_population(matingpool, elite_size)
    next_gen = mutate_population(children, mutation_rate)
    return next_gen, pop_ranked
def genetic_algorithm(cities, names, pop_size=100, elite_size=20, mutation_rate=0.01, generations=500):
    population = init_population(pop_size, len(cities))
    avg_fitness_history = []

    for generation in range(generations):
        population, pop_ranked = next_generation(population, elite_size, mutation_rate, cities)
        best_tour_index = pop_ranked[0][0]  
        best_tour = population[best_tour_index] 
        best_distance = 1 / pop_ranked[0][1]
        avg_fitness = np.mean([1 / x[1] for x in pop_ranked])
        avg_fitness_history.append(avg_fitness)
        
        if generation % 50 == 0:
            plot_tour(best_tour, cities, best_distance, names, generation)

    plt.plot(avg_fitness_history)
    plt.title('Average Fitness Over Generations')
    plt.xlabel('Generation')
    plt.ylabel('Average Fitness')
    plt.show()

def main():
    filename = 'India_cities.txt' 
    cities, names = read_cities(filename)
    
    if len(cities) < 2:
        print("Not enough cities to compute the TSP.")
        return

    genetic_algorithm(cities, names)

if __name__ == "__main__":
    main()
