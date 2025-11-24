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
    
    def _initialize_population(self):
        """Inicializa população como matriz de chromosomes binários."""
        self.population = np.random.randint(
            0, 2, 
            size=(self.population_size, self.n_features)
        )
        
        for i in range(self.population_size):
            if np.sum(self.population[i]) == 0:
                random_indices = np.random.choice(
                    self.n_features, 
                    size=max(1, self.n_features // 10),
                    replace=False
                )
                self.population[i][random_indices] = 1
    
    def _evaluate_fitness(self, fitness_function: Callable[[np.ndarray], float]) -> np.ndarray:
        """Avalia fitness de todos chromosomes."""
        fitness_scores = np.zeros(self.population_size)
        
        for i, chromosome in enumerate(self.population):
            if np.sum(chromosome) == 0:
                fitness_scores[i] = 0.0
            else:
                fitness_scores[i] = fitness_function(chromosome)
        
        return fitness_scores
    
    def _evolve_population(self) -> np.ndarray:
        """
        Evolui população usando operadores matriciais.
        Combina elitismo, seleção, crossover e mutação.
        """
        new_population = []
        
        if self.elitism > 0:
            elite_indices = np.argsort(self.fitness_scores)[-self.elitism:]
            for idx in elite_indices:
                new_population.append(self.population[idx].copy())
        
        population_matrix = self.population.reshape(
            self.matrix_rows, 
            self.matrix_cols, 
            self.n_features
        )
        
        while len(new_population) < self.population_size:
            if np.random.random() < 0.5:
                offspring1, offspring2 = self._row_crossover(population_matrix)
            else:
                offspring1, offspring2 = self._column_crossover(population_matrix)
            
            offspring1 = self._mutation(offspring1)
            offspring2 = self._mutation(offspring2)
            
            new_population.append(offspring1)
            if len(new_population) < self.population_size:
                new_population.append(offspring2)
        
        return np.array(new_population[:self.population_size])
    
    def _row_crossover(self, population_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Crossover baseado em linhas da matriz populacional."""
        parent1 = self._selection()
        parent2 = self._selection()
        
        if np.random.random() < self.crossover_rate:
            crossover_point = np.random.randint(1, self.n_features)
            offspring1 = np.concatenate([
                parent1[:crossover_point],
                parent2[crossover_point:]
            ])
            offspring2 = np.concatenate([
                parent2[:crossover_point],
                parent1[crossover_point:]
            ])
        else:
            offspring1 = parent1.copy()
            offspring2 = parent2.copy()
        
        return offspring1, offspring2
    
    def _column_crossover(self, population_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Crossover baseado em colunas da matriz populacional."""
        parent1 = self._selection()
        parent2 = self._selection()
        
        if np.random.random() < self.crossover_rate:
            mask = np.random.random(self.n_features) < 0.5
            offspring1 = np.where(mask, parent1, parent2)
            offspring2 = np.where(mask, parent2, parent1)
        else:
            offspring1 = parent1.copy()
            offspring2 = parent2.copy()
        
        return offspring1, offspring2
    
    def _selection(self) -> np.ndarray:
        """Seleção por torneio."""
        tournament_indices = np.random.choice(
            self.population_size,
            size=self.tournament_size,
            replace=False
        )
        tournament_fitness = self.fitness_scores[tournament_indices]
        winner_idx = tournament_indices[np.argmax(tournament_fitness)]
        
        return self.population[winner_idx].copy()
    
    def _mutation(self, chromosome: np.ndarray) -> np.ndarray:
        """Mutação bit-flip."""
        mutated = chromosome.copy()
        
        for i in range(self.n_features):
            if np.random.random() < self.mutation_rate:
                mutated[i] = 1 - mutated[i]
        
        if np.sum(mutated) == 0:
            random_idx = np.random.randint(0, self.n_features)
            mutated[random_idx] = 1
        
        return mutated
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Aplica seleção de features usando melhor chromosome encontrado."""
        if self.best_chromosome is None:
            raise ValueError("MGA não foi executado. Chame fit() primeiro.")
        
        selected_features = self.best_chromosome == 1
        return X[:, selected_features]
    
    def get_selected_features(self) -> np.ndarray:
        """Retorna índices das features selecionadas."""
        if self.best_chromosome is None:
            raise ValueError("MGA não foi executado. Chame fit() primeiro.")
        
        return np.where(self.best_chromosome == 1)[0]
    
    def get_fitness_history(self) -> list:
        """Retorna histórico de fitness por geração."""
        return self.fitness_history
    
    def get_population_matrix_shape(self) -> Tuple[int, int]:
        """Retorna dimensões da matriz populacional."""
        return (self.matrix_rows, self.matrix_cols)