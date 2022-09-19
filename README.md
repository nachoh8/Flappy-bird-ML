# Flappy Bird - Genetic Neural Network

Machine learning for a Flappy Bird controller.
The game is a modification of the [Flappy Bird clone](https://github.com/sourabhv/FlapPyBird) made by sourabhv.
The method for learning the controller is based on neural networks and genetic algorithms, Genetic Neural Network (GNN).
The GNN has been created from scratch with Numpy, the source code can be found in _ML/geneticNN.py_

## Genetic algorithm

1. Generate a new population of NNs with random weights.
2. The population plays the game N times simultaneously.
3. Calculate the average fitness (travaled distance) of each NN. Optional: sum previous fitness (multiplied by a factor) to the best NNs of the previous generation
4. Create a new generation from the current NNs:

    - The best K NNs are passed on to the next generation without modification.
    - A 20% of the new population is generated from randomly selected top NNs with mutations.
    - The rest of the new population is created from the crossover between 2 NNs chosen at random from the top NNs.
5. Go to 2.

## Experiments

### Common parameters

- Population: 100
- Elitism: 0.2
- Mutation rate: 0.2
- Number of games per generation: 3
- Max generation: 100
- Activation function: sigmoid

### NN models

- NN-1:
  - Architecture: 2-6-1
  - Iput:
    - normalized horizontal distance between the midpoint of the bird and the end of the nearest gap
    - normalized vertical distance between the midpoint of the bird and the center of the nearest gap
  - Output: if output > 0.5 then flap
- NN-2:
  - Architecture: 2-5-2-1
  - Iput:
    - normalized horizontal distance between the midpoint of the bird and the end of the nearest gap
    - normalized vertical distance between the midpoint of the bird and the center of the nearest gap
  - Output: if output > 0.5 then flap
- NN-3:
  - Architecture: 3-5-2-1
  - Input:
    - normalized horizontal distance between the midpoint of the bird and the end of the nearest gap
    - normalized horizontal distance between the midpoint of the bird and the beginning of the nearest gap
    - normalized vertical distance between the midpoint of the bird and the center of the nearest gap
  - Output: if output > 0.5 then flap

### Evaluation

Evaluation of the best NN of each model in 50 games

|  NN  | Average Score | Max Score |
|:----:|:-------------:|:---------:|
| NN-1 |       85.5       |    983    |
| NN-2 |       95.12       |    1598   |
| NN-3 |       72.9       |    976    |
