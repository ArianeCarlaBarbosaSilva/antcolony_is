import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

from typing import *
import random
import math

class InstanceSelection:

    def __init__(self):
        return

    # calcula a distancia par a par entre todas as instancias da base de dados
    def get_pairwise_distance(self, matrix: np.ndarray) -> np.ndarray:
        return euclidean_distances(matrix)

    # matriz de visibilidade que eh o inverso da distancia
    def get_visibility_rates_by_distances(self, distances: np.ndarray) -> np.ndarray:
        visibilities = np.zeros(distances.shape)
        for i in range(distances.shape[0]):
            for j in range(distances.shape[1]):
                if i != j:
                    if distances[i, j] == 0:
                        visibilities[i, j] = 0
                    else:
                        visibilities[i, j] = 1 / distances[i, j]

        return visibilities

    '''
    A estrutura de dados Colonia é uma matriz XxX sendo X o número de instâncias, composta de 0's e 1's.
    0 -> a formiga não visitou a instancia, 1 se foi ela foi adicionada em seu conjunto de solução.
    '''
    def create_colony(self, num_ants):
        return np.full((num_ants, num_ants), -1)


    def create_pheromone_trails(self, search_space: np.ndarray, initial_pheromone: float) -> np.ndarray:
        trails = np.full(search_space.shape, initial_pheromone, dtype=np.float64)
        np.fill_diagonal(trails, 0)
        return trails

    # ant_choices é uma lista das arestas que a formiga escolheu passar
    def get_pheromone_deposit(self, ant_choices: List[Tuple[int, int]], distances: np.ndarray, deposit_factor: float) -> float:
        tour_length = 0
        for path in ant_choices:
            tour_length += distances[path[0], path[1]]

        if tour_length == 0: # na primeira iteracao a formiga ainda nao tem tour_length
            return 0

        if math.isinf(tour_length): # retorna true se o numero for infinito
            print('deu muito ruim!')

        return deposit_factor / tour_length


    def get_probabilities_paths_ordered(self, ant: np.array, visibility_rates: np.array, phe_trails) \
            -> Tuple[Tuple[int, Any]]:
        available_instances = np.nonzero(ant < 0)[0] # pega as instancias que ainda nao foram visitadas -1, retorna a dimensao [0]
        # The pheromones over the available paths
        smell = np.sum(
            phe_trails[available_instances]
            * visibility_rates[available_instances])

        # Calculate the probalilty by available instance using
        # the sum of pheromones in rest of tour
        # instanciasPossiveis x 2 sendo [0] -> Instancia, [1] -> probabilidade
        probabilities = np.zeros((len(available_instances), 2))
        for i, available_instance in enumerate(available_instances):
            probabilities[i, 0] = available_instance
            path_smell = phe_trails[available_instance] * \
                         visibility_rates[available_instance]

            if path_smell == 0: #sera que precisa desse if? Nao poderia deixar so o de baixo ja que 0/smell = 0
                probabilities[i, 1] = 0
            else:
                probabilities[i, 1] = path_smell / smell
        # ordena por probabilidades sem alterar os dados da matriz
        sorted_probabilities = probabilities[probabilities[:, 1].argsort()][::-1] # Esse final [::-1] reverte a lista para retornar do maior para o menor
        # retorna uma lista de tuplas ordenadas de acordo com a probabilidade
        return tuple([(int(i[0]), i[1]) for i in sorted_probabilities])


    def get_best_solution(self, ant_solutions: np.ndarray, X, Y, X_valid, Y_valid) -> np.array:
        accuracies = np.zeros(ant_solutions.shape[0], dtype=np.float64)
        best_solution = 0
        for i, solution in enumerate(ant_solutions):
            instances_selected = np.nonzero(solution)[0]

            X_train = X[instances_selected, :] # retorna as linhas selecionadas e todas as colunas dessas linhas
            Y_train = Y[instances_selected]
            classifier_1nn = KNeighborsClassifier(n_neighbors=1).fit(X_train, Y_train)
            Y_pred = classifier_1nn.predict(X_valid)
            accuracy = accuracy_score(Y_valid, Y_pred)
            ##Y_pred = classifier_1nn.predict(X_valid)
            ##accuracy = accuracy_score(Y_valid, Y_pred)
            ##accuracy = recall_score(Y, Y_pred, average='macro')
            accuracies[i] = accuracy
            if accuracy > accuracies[best_solution]:
                best_solution = i

        # print(f"The winner is ant {best_solution} with accuracy {accuracies[best_solution]}")
        return ant_solutions[best_solution]


    def run_colony(self, X, Y, X_valid, Y_valid, initial_pheromone, evaporarion_rate, Q):
        distances = self.get_pairwise_distance(X)
        visibility_rates = self.get_visibility_rates_by_distances(distances)
        the_colony = self.create_colony(X.shape[0]) #passa a quantidade de linhas da base de dados
        for i in range(X.shape[0]):
            the_colony[i, i] = 1 # coloca 1 em toda diagonal principal da matriz Colonia indicando a instancia de onde a formiga saiu (a primeira instancia) pertence ao conjunto solucao da formiga

        ant_choices = [[(i, i)] for i in range(the_colony.shape[0])]
        pheromone_trails = self.create_pheromone_trails(distances, initial_pheromone)

        # esse while dura enquanto ainda tiver formiga para escolher o caminho. Será?
        while -1 in the_colony:
            # Each ant will choose their next instance
            for i, ant in enumerate(the_colony):
                if -1 in ant: # se existe -1 em ant significa que aquela formiga tem instancia ainda nao avaliada (0 ou 1)
                    last_choice = ant_choices[i][-1]
                    ant_pos = last_choice[1]
                    choices = self.get_probabilities_paths_ordered(
                        ant,
                        visibility_rates[ant_pos, :], # pega todas as colunas que representam as instancias naquela linha da formiga
                        pheromone_trails[ant_pos, :]) # pega todas as colunas na linha da formiga em analise representando a qtde de feromonio

                    for choice in choices:
                        next_instance = choice[0]
                        probability = choice[1]

                        ajk = random.randint(0, 1)

                        final_probability = probability * ajk
                        if final_probability != 0:
                            ant_choices[i].append((ant_pos, next_instance))
                            the_colony[i, next_instance] = 1 # escreve na matriz Colonia que a formiga i escolheu a instancia next_instance para visitar
                            break
                        else:
                            the_colony[i, next_instance] = 0 # escreve 0 na matriz Colonia informando que a formiga i nao escolheu next_instance para visitar

            # Ants deposit the pheromones
            for i in range(the_colony.shape[0]):
                ant_deposit = self.get_pheromone_deposit(ant_choices[i], distances, Q)
                for path in ant_choices[i][1:]:  # Never deposit in pheromone on i == j!
                    pheromone_trails[path[0], path[1]] += ant_deposit

            # Pheromones evaporation
            for i in range(pheromone_trails.shape[0]):
                for j in range(pheromone_trails.shape[1]):
                    pheromone_trails[i, j] = (1 - evaporarion_rate) * pheromone_trails[i, j]

        # np.nonzero retorna os indices que não são zero indicando quais instancias foram visitadas pela formiga
        instances_selected = np.nonzero(self.get_best_solution(the_colony, X, Y, X_valid, Y_valid))[0]
        # o retorno do algoritmo Ant-IS sao os indices das instancias selecionadas
        return instances_selected

