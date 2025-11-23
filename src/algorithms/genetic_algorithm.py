import numpy as np
from typing import Callable, Tuple, Optional


class GeneticAlgorithm:
    """
    Algoritmo Genético para seleção de features.
    
    Utiliza encoding binário onde cada gene representa presença (1) ou ausência (0) de uma feature.
    """
    
    def __init__(
        self,
        population_size: int = 100,
        generations: int = 50,
        crossover_rate: float = 0.8,
        mutation_rate: float = 0.01,
        tournament_size: int = 3,
        elitism: int = 2,
        random_state: Optional[int] = None
    ):
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.elitism = elitism
        self.random_state = random_state
        
        self.population = None
        self.fitness_scores = None
        self.best_chromosome = None
        self.best_fitness = -np.inf
        self.fitness_history = []
        
        if random_state is not None:
            np.random.seed(random_state)
    
    def fit(
        self,
        n_features: int,
        fitness_function: Callable[[np.ndarray], float],
        verbose: bool = True
    ) -> 'GeneticAlgorithm':
        """
        Executa o algoritmo genético.
        
        Args:
            n_features: Número total de features disponíveis
            fitness_function: Função que avalia fitness de um chromosome
            verbose: Se True, imprime progresso
        """
        self.n_features = n_features
        self._initialize_population()
        
        for generation in range(self.generations):
            self.fitness_scores = self._evaluate_fitness(fitness_function)
            
            current_best_idx = np.argmax(self.fitness_scores)
            current_best_fitness = self.fitness_scores[current_best_idx]
            
            if current_best_fitness > self.best_fitness:
                self.best_fitness = current_best_fitness
                self.best_chromosome = self.population[current_best_idx].copy()
            
            self.fitness_history.append({
                'generation': generation,
                'best_fitness': self.best_fitness,
                'mean_fitness': np.mean(self.fitness_scores),
                'std_fitness': np.std(self.fitness_scores),
                'n_features_best': np.sum(self.best_chromosome)
            })
            
            if verbose and generation % 10 == 0:
                n_features_selected = np.sum(self.best_chromosome)
                print(f"Geração {generation}: "
                      f"Fitness={self.best_fitness:.4f}, "
                      f"Features={n_features_selected}/{n_features}")
            
            new_population = []
            
            if self.elitism > 0:
                elite_indices = np.argsort(self.fitness_scores)[-self.elitism:]
                for idx in elite_indices:
                    new_population.append(self.population[idx].copy())
            
            while len(new_population) < self.population_size:
                parent1 = self._selection()
                parent2 = self._selection()
                
                if np.random.random() < self.crossover_rate:
                    offspring1, offspring2 = self._crossover(parent1, parent2)
                else:
                    offspring1, offspring2 = parent1.copy(), parent2.copy()
                
                offspring1 = self._mutation(offspring1)
                offspring2 = self._mutation(offspring2)
                
                new_population.append(offspring1)
                if len(new_population) < self.population_size:
                    new_population.append(offspring2)
            
            self.population = np.array(new_population[:self.population_size])
        
        if verbose:
            n_features_selected = np.sum(self.best_chromosome)
            print(f"\nFinalizado: Fitness={self.best_fitness:.4f}, "
                  f"Features={n_features_selected}/{n_features}")
        
        return self
    
