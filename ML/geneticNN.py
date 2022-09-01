import numpy as np
import random

def sigmoid(v: float) -> float:
    return 1.0 / (1.0 + np.exp(-v))

def tanh(v: float) -> float:
    return np.tanh(v)

class Layer(object):
    def __init__(self, input_size: int, output_size: int, activation: "function", weights: np.ndarray = np.array([])) -> None:
        self.bias = 0.0 # np.random.rand(1, output_size) - 0.5
        if weights.shape[0] > 0:
            self.weights = weights
        else:
            self.weights = np.random.rand(input_size, output_size) - 0.5
        self.activation = np.vectorize(activation)
    
    def forward(self, input: np.ndarray) -> np.ndarray:
        return self.activation(np.dot(input, self.weights) + self.bias)

class NeuralNetwork(object):
    def __init__(self, layers: "list[int]", activation: "function", weights: "list[np.ndarray]" = []) -> None:
        self.layers: list[Layer] = []
        
        _in = layers[0]
        if len(weights) == 0:
            for num_neurons in layers[1:]:
                self.layers.append(Layer(_in, num_neurons, activation))
                _in = num_neurons
        else:
            for num_neurons, layer_weights in zip(layers[1:], weights):
                self.layers.append(Layer(_in, num_neurons, activation, layer_weights))
                _in = num_neurons    
    
    def predict(self, input: np.ndarray) -> np.ndarray:
        output = input
        for layer in self.layers:
            output = layer.forward(output)
        
        return output
    
    def get_weights(self) -> "list[np.ndarray]":
        return [l.weights.copy() for l in self.layers]

class Generation(object):
    def __init__(self, networks: "list[NeuralNetwork]") -> None:
        self.networks = networks
        self.fitness: np.ndarray = np.zeros(len(networks))
        self.sorted_ids: np.ndarray = np.full(len(networks), -1)
    
    def add_fitness(self, fitness: np.ndarray):
        self.fitness = fitness
        self.sorted_ids = np.argsort(self.fitness)[::-1] # in descending order
    
    def get_best_NN(self) -> "tuple[int, NeuralNetwork, float]":
        return (self.sorted_ids[0], self.networks[self.sorted_ids[0]], self.fitness[self.sorted_ids[0]])
    
    def generate_new_generation(self, elitism: float, mutation_rate: float) -> "list[list[np.ndarray]]":
        ### rank-based selection ###
        population = len(self.networks)

        ### selection
        n = elitism
        new_generation = [self.networks[self.sorted_ids[i]].get_weights() for i in range(int(population * n))]
        
        # 10% from top with randomization
        n += 0.1
        while len(new_generation) < int(population * n):
            id = self.sorted_ids[random.randint(0, int(population * elitism))]
            w = self.networks[id].get_weights()
            
            self.mutation(w, 0.4)
            new_generation.append(w)

        ### crossover

        # 10% from top
        n += 0.1
        while len(new_generation) < int(population * n):
            p1 = self.sorted_ids[random.randint(0, int(population * elitism))]
            p2 = self.sorted_ids[random.randint(0, int(population * elitism))]
            if p1 == p2: pass

            offspring = self.crossover(p1, p2, mutation_rate)
            new_generation.append(offspring)

        # rest
        while len(new_generation) < population:
            p1 = random.randint(0, population-1)
            p2 = random.randint(0, population-1)
            if p1 == p2: pass

            offspring = self.crossover(p1, p2, mutation_rate)
            new_generation.append(offspring)

        return new_generation
    
    def crossover(self, id1: int, id2: int, mutation_rate: float) -> np.ndarray:
        w1 = self.networks[id1].get_weights()
        w2 = self.networks[id2].get_weights()

        for layer in range(len(w1)):
            split = np.random.randint(w1[layer].shape[1])
            t = w1[layer][:, split:].copy()
            w1[layer][:, split:] = w2[layer][:, split:]
            w2[layer][:, split:] = t
        
        w_res = w1 if np.random.rand() < 0.5 else w2
        self.mutation(w_res, mutation_rate)

        return w_res

    def mutation(self, weights: "list[np.ndarray]", mutation_rate: float):
        for w in weights:
            if np.random.rand() < mutation_rate:
                f = 1 + (np.random.rand(w.shape[0], w.shape[1]) - 0.5)*3 + np.random.rand(w.shape[0], w.shape[1]) - 0.5
                w *= f

class GeneticNN(object):
    def __init__(self, population: int, nn_params: list, max_generation: int = -1, max_fitness: float = -1) -> None:
        self.population = population
        self.max_generation = max_generation
        self.max_fitness = max_fitness

        self.elitism = 0.2 # fraction of chromosomes to keep unchanged
        self.mutation_rate = 0.1

        self.nn_params = nn_params
        self.current_generation = 0
        self.generations: list[Generation] = []
    
    def predict(self, id: int, input: np.ndarray) -> np.ndarray:
        return self.generations[self.current_generation-1].networks[id].predict(input)

    def get_best_NN(self) -> "tuple[int, NeuralNetwork, float]":
        return self.generations[self.current_generation-1].get_best_NN()

    def first_generation(self):
        self.current_generation = 1
        networks = [NeuralNetwork(self.nn_params[0], self.nn_params[1]) for _ in range(self.population)]
        self.generations.append(Generation(networks))

    def next_generation(self, fitness: np.ndarray) -> "tuple[bool, tuple[int, NeuralNetwork, float]]":
        last_gen = self.generations[self.current_generation-1]
        last_gen.add_fitness(fitness)

        sorted_ids = last_gen.sorted_ids
        max_score = last_gen.fitness[sorted_ids[0]]

        best = last_gen.get_best_NN()

        if self.max_generation != -1 and self.current_generation == self.max_generation:
            print("Max Generation reached")
            return False, best

        if self.max_fitness != -1 and max_score >= self.max_fitness:
            print("Max fitness achieved:", max_score)
            return False, best
        
        new_generation = last_gen.generate_new_generation(self.elitism, self.mutation_rate)
        
        new_networks = [NeuralNetwork(self.nn_params[0], self.nn_params[1], weights=w) for w in new_generation]

        self.current_generation += 1
        self.generations.append(Generation(new_networks))
        return True, best

            

if __name__ == "__main__":
    gnn_params = dict()
    gnn_params['population'] = 50
    gnn_params['max_generation'] = -1
    gnn_params['max_fitness'] = -1
    gnn_params['elitism'] = 0.2
    gnn_params['mutation_rate'] = 0.1
    gnn_params['checkpoint'] = 10

    nn_params = dict()
    nn_params['layers'] = [2,5,2,1]
    nn_params['activation'] = 'sigmoid'

    gnn_params['nn_params'] = nn_params

    net = NeuralNetwork([5,4,2,1], sigmoid)
    input = np.array([1,1,0,1,0])
    print(net.predict(input))
    weights = net.get_weights()
    net2 = NeuralNetwork([5,4,2,1], sigmoid)
    print(net2.predict(input))

    gnn = GeneticNN(10, [[5,4,2,1], sigmoid], max_generation=10, max_fitness=0.95)
    gnn.first_generation()
    end = False
    while not end:
        print("gen: ", gnn.current_generation, "/", 10)
        end = not gnn.next_generation(np.random.rand(10))

