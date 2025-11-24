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
    
