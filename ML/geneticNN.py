import numpy as np
import random

from pathlib import Path
import json

def sigmoid(v: float) -> float:
    return 1.0 / (1.0 + np.exp(-v))

def tanh(v: float) -> float:
    return np.tanh(v)

class Layer(object):
    def __init__(self, input_size: int, output_size: int, activation: "function", weights: np.ndarray = np.array([])) -> None:
        self.bias = np.zeros(output_size)
        if weights.shape[0] > 0:
            self.weights = weights
        else:
            self.weights = np.random.rand(output_size, input_size) - 0.5
        self.activation = np.vectorize(activation)
    
    def forward(self, input: np.ndarray) -> np.ndarray:
        return self.activation(np.dot(self.weights, input) + self.bias)

class NeuralNetwork(object):
    def __init__(self, nn_params: dict) -> None:
        layers: list[int] = nn_params['layers']
        if 'weights' in nn_params:
            weights: list[np.ndarray] = nn_params['weights']
            if type(weights[0]) is not np.ndarray:
                weights = [np.array(w) for w in weights]
        else:
            weights: list[np.ndarray] = []

        if nn_params['activation'] == 'sigmoid':
            activation = sigmoid
        elif nn_params['activation'] == 'tanh':
            activation = tanh
        else:
            raise Exception("This activation function does not exists")

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
        self.score: np.ndarray = np.zeros(len(networks))
        self.sorted_ids: np.ndarray = np.full(len(networks), -1)
    
    def add_fitness(self, fitness: np.ndarray, score: np.ndarray):
        self.fitness = fitness
        self.score = score
        self.sorted_ids = np.argsort(self.fitness)[::-1] # in descending order
    
    def get_best_NN(self) -> "tuple[int, NeuralNetwork, float, float]":
        return (self.sorted_ids[0], self.networks[self.sorted_ids[0]], self.fitness[self.sorted_ids[0]], self.score[self.sorted_ids[0]])
    
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
            
            self.mutation(w, 0.2 + mutation_rate)
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
    def __init__(self, gnn_params: dict) -> None:
        self.population: int    =    gnn_params['population']
        self.max_generation: int =   gnn_params['max_generation']
        self.max_fitness: float =    gnn_params['max_fitness']

        self.checkpoint_dir: str = gnn_params['checkpoint_dir']
        self.checkpoint: int = gnn_params['checkpoint']

        self.elitism: float = gnn_params['elitism'] # fraction of chromosomes to keep unchanged
        self.mutation_rate: float = gnn_params['mutation_rate']
        self.discount_prev_res: float = gnn_params['discount_prev_res']

        self.nn_params: dict = gnn_params['nn_params']
        self.current_generation = 0
        self.generations: list[Generation] = []

        if "checkpoint_file" in gnn_params:
            f = gnn_params['checkpoint_file']
            if f != '':
                self.load_checkpoint()
    
    def predict(self, id: int, input: np.ndarray) -> np.ndarray:
        return self.generations[-1].networks[id].predict(input)

    def get_best_NN(self) -> "tuple[int, NeuralNetwork, float, float]":
        return self.generations[-1].get_best_NN()

    def fitness_elitism_generations(self):
        if len(self.generations) < 2:
            return None
        
        prev_gen = self.generations[-2]
        current_gen = self.generations[-1]
        n = int(self.population * self.elitism)
        fitness_diff = [(prev_gen.fitness[prev_gen.sorted_ids[idx]], current_gen.fitness[idx]) for idx in range(n)]

        return fitness_diff

    def first_generation(self):
        self.current_generation = 1
        networks = [NeuralNetwork(self.nn_params) for _ in range(self.population)]
        self.generations.append(Generation(networks))

    def next_generation(self, fitness: np.ndarray, score: np.ndarray) -> "tuple[bool, tuple[int, NeuralNetwork, float, float]]":
        last_gen = self.generations[-1]

        if self.discount_prev_res > 0.0 and len(self.generations) > 1:
            g_prev = self.generations[-2]
            prev_best = [g_prev.sorted_ids[i] for i in range(int(self.population * self.elitism))]
            curr_best = [i for i in range(int(self.population * self.elitism))]
            fitness[curr_best] += g_prev.fitness[prev_best] * self.discount_prev_res

        last_gen.add_fitness(fitness, score)

        print("Diff fitness (top genes)")
        print(self.fitness_elitism_generations())

        sorted_ids = last_gen.sorted_ids
        max_score = last_gen.fitness[sorted_ids[0]]

        best = last_gen.get_best_NN()

        if self.checkpoint_dir != '' and self.current_generation % self.checkpoint == 0:
            self.save_checkpoint()

        if self.max_generation != -1 and self.current_generation == self.max_generation:
            print("Max Generation reached")
            return False, best

        if self.max_fitness != -1 and max_score >= self.max_fitness:
            print("Max fitness achieved:", max_score)
            return False, best
        
        new_generation = last_gen.generate_new_generation(self.elitism, self.mutation_rate)
        
        new_networks = [NeuralNetwork(dict(self.nn_params, **{'weights':w})) for w in new_generation]

        self.current_generation += 1
        self.generations.append(Generation(new_networks))
        return True, best
    
    def save_checkpoint(self):
        filename = self.checkpoint_dir + '/checkpoint_' + str(int(self.current_generation / self.checkpoint))
        print("Saving checkpoint to " + filename)

        last_gen = self.generations[-1]

        data = dict()
        data['generation'] = self.current_generation
        data['fitness'] = last_gen.fitness.tolist()
        data['score'] = last_gen.score.tolist()
        data['nn_weights'] = dict()
        for i, nn in zip(range(self.population), last_gen.networks):
            weights = [w.tolist() for w in nn.get_weights()]
            data['nn_weights'][i] = weights

        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        with open(filename + '.json', 'w') as f:
            json.dump(data, f)
        
        self.save_best_nn()
    
    def save_best_nn(self):
        filename = self.checkpoint_dir + "/best_nn_G" + str(self.current_generation) + ".json"
        print("Saving best NN to " + filename)

        data = self.nn_params.copy()
        _, nn, fitness, score = self.get_best_NN()
        data['generation'] = self.current_generation
        data['fitness'] = fitness
        data['score'] = score
        data['weights'] = [w.tolist() for w in nn.get_weights()]
        with open(filename, 'w') as f:
            json.dump(data, f)

    def load_checkpoint(self, checkpoint_file: str):
        with open(checkpoint_file, 'r') as f:
            data: dict = json.load(f)
            self.current_generation = data['generation']
            
            nn_weights = data['nn_weights']
            weights = [[np.array(w) for w in net_w] for _, net_w in nn_weights.items()]
            networks = [NeuralNetwork(dict(self.nn_params, **{'weights':w})) for w in weights]
            
            fitness = data['fitness']
            score = data['score']
            self.generations.append(Generation(networks))

            temp = self.checkpoint_dir
            self.checkpoint_dir = ''
            self.next_generation(fitness, score)
            self.checkpoint_dir = temp
            

if __name__ == "__main__":
    gnn_params = dict()
    gnn_params['population'] = 5
    gnn_params['max_generation'] = 10
    gnn_params['max_fitness'] = -1
    gnn_params['elitism'] = 0.2
    gnn_params['mutation_rate'] = 0.1
    gnn_params['checkpoint'] = 2
    gnn_params['checkpoint_dir'] = '/home/nacho/Flappy-bird-ML/checkpoints/test'

    nn_params = dict()
    nn_params['layers'] = [2,5,2,1]
    nn_params['activation'] = 'sigmoid'
    nn_params['weights'] = []

    gnn_params['nn_params'] = nn_params

    gnn = GeneticNN(gnn_params)
    gnn.first_generation()
    next_gen = True
    while next_gen:
        print("gen: ", gnn.current_generation, "/", 10)
        fitness = np.random.rand(gnn.population)
        next_gen, best = gnn.next_generation(fitness)
    
    print("Test predict")
    print(gnn.generations[-1].fitness)
    input_test = np.array([0.5, -0.5])
    for i in range(gnn.population):
        print(i, gnn.predict(i, input_test))
    
    print("Load checkpoint")
    gnn_2 = GeneticNN(gnn_params)
    gnn_2.load_checkpoint(gnn.checkpoint_dir + '/checkpoint_5.json')
    print(gnn_2.generations[-1].fitness)
    for i in range(gnn_2.population):
        print(i, gnn_2.predict(i, input_test))

