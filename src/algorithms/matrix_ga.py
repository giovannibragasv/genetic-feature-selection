import numpy as np
from typing import Callable, Tuple, Optional


class MatrixGeneticAlgorithm:
    """
    Matrix Genetic Algorithm para seleção de features.
    
    Implementa estrutura populacional 2D conforme artigo (seção 4.1).
    População organizada como matriz onde linhas/colunas facilitam operações genéticas.
    """
    
    def __init__(
        self,
        population_size: int = 100,
        generations: int = 50,
        crossover_rate: float = 0.8,
        mutation_rate: float = 0.01,
        tournament_size: int = 3,
        elitism: int = 2,
        matrix_rows: Optional[int] = None,
        random_state: Optional[int] = None
    ):
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.elitism = elitism
        self.random_state = random_state
        
        if matrix_rows is None:
            self.matrix_rows = int(np.sqrt(population_size))
            self.matrix_cols = population_size // self.matrix_rows
        else:
            self.matrix_rows = matrix_rows
            self.matrix_cols = population_size // matrix_rows
        
        actual_pop_size = self.matrix_rows * self.matrix_cols
        if actual_pop_size != population_size:
            print(f"Ajustando population_size: {population_size} → {actual_pop_size} "
                  f"(matriz {self.matrix_rows}×{self.matrix_cols})")
            self.population_size = actual_pop_size
        
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
    ) -> 'MatrixGeneticAlgorithm':
        """
        Executa o algoritmo genético matricial.
        
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
            
            new_population = self._evolve_population()
            self.population = new_population
        
        if verbose:
            n_features_selected = np.sum(self.best_chromosome)
            print(f"\nFinalizado: Fitness={self.best_fitness:.4f}, "
                  f"Features={n_features_selected}/{n_features}")
        
        return self
    
