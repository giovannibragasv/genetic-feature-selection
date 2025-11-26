import numpy as np
from typing import Callable, Tuple, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..encoding import BaseEncoding


class MatrixGeneticAlgorithm:
    """
    Matrix Genetic Algorithm para seleção de features.
    
    Implementa estrutura populacional 2D conforme artigo (seção 4.1).
    População organizada como matriz onde linhas/colunas facilitam operações genéticas.
    Suporta múltiplos encodings (binary, decimal, real, gaussian, adaptive).
    """
    
    def __init__(
        self,
        population_size: int = 600,
        generations: int = 55,
        crossover_rate: float = 1.0,
        mutation_rate: float = 0.2,
        tournament_size: int = 3,
        elitism: int = 2,
        matrix_rows: Optional[int] = None,
        encoding: Optional['BaseEncoding'] = None,
        random_state: Optional[int] = None
    ):
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.elitism = elitism
        self.encoding = encoding
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
        
        if self.encoding is not None and self.encoding.n_features != n_features:
            raise ValueError(
                f"Encoding n_features ({self.encoding.n_features}) != "
                f"n_features fornecido ({n_features})"
            )
        
        self._initialize_population()
        
        for generation in range(self.generations):
            self.fitness_scores = self._evaluate_fitness(fitness_function)
            
            current_best_idx = np.argmax(self.fitness_scores)
            current_best_fitness = self.fitness_scores[current_best_idx]
            
            if current_best_fitness > self.best_fitness:
                self.best_fitness = current_best_fitness
                self.best_chromosome = self.population[current_best_idx].copy()
            
            if self._is_adaptive_encoding():
                self.encoding.update_threshold(current_best_fitness)
            
            n_features_selected = self._get_n_selected(self.best_chromosome)
            
            self.fitness_history.append({
                'generation': generation,
                'best_fitness': self.best_fitness,
                'mean_fitness': np.mean(self.fitness_scores),
                'std_fitness': np.std(self.fitness_scores),
                'n_features_best': n_features_selected
            })
            
            if verbose and generation % 10 == 0:
                extra_info = ""
                if self._is_adaptive_encoding():
                    extra_info = f", Threshold={self.encoding.threshold:.3f}"
                print(f"Geração {generation}: "
                      f"Fitness={self.best_fitness:.4f}, "
                      f"Features={n_features_selected}/{n_features}{extra_info}")
            
            new_population = self._evolve_population()
            self.population = new_population
        
        if verbose:
            n_features_selected = self._get_n_selected(self.best_chromosome)
            print(f"\nFinalizado: Fitness={self.best_fitness:.4f}, "
                  f"Features={n_features_selected}/{n_features}")
        
        return self
    
    def _is_adaptive_encoding(self) -> bool:
        """Verifica se encoding é adaptativo."""
        if self.encoding is None:
            return False
        return hasattr(self.encoding, 'update_threshold')
    
    def _get_n_selected(self, chromosome: np.ndarray) -> int:
        """Retorna número de features selecionadas."""
        if self.encoding is not None:
            return self.encoding.get_n_selected_features(chromosome)
        return int(np.sum(chromosome))
    
    def _initialize_population(self):
        """Inicializa população."""
        if self.encoding is not None:
            self.population = np.array([
                self.encoding.initialize_chromosome()
                for _ in range(self.population_size)
            ])
        else:
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
            n_selected = self._get_n_selected(chromosome)
            if n_selected == 0:
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
            if self.encoding is not None:
                return self.encoding.crossover(parent1, parent2)
            
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
            if self.encoding is not None:
                return self.encoding.crossover(parent1, parent2)
            
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
        """Mutação usando encoding ou bit-flip default."""
        if self.encoding is not None:
            return self.encoding.mutate(chromosome, self.mutation_rate)
        
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
        
        selected_features = self.get_selected_features()
        return X[:, selected_features]
    
    def get_selected_features(self) -> np.ndarray:
        """Retorna índices das features selecionadas."""
        if self.best_chromosome is None:
            raise ValueError("MGA não foi executado. Chame fit() primeiro.")
        
        if self.encoding is not None:
            binary = self.encoding.decode(self.best_chromosome)
        else:
            binary = self.best_chromosome
        
        return np.where(binary == 1)[0]
    
    def get_fitness_history(self) -> list:
        """Retorna histórico de fitness por geração."""
        return self.fitness_history
    
    def get_population_matrix_shape(self) -> Tuple[int, int]:
        """Retorna dimensões da matriz populacional."""
        return (self.matrix_rows, self.matrix_cols)
    
    def get_encoding_stats(self) -> Optional[dict]:
        """Retorna estatísticas do encoding (se adaptive)."""
        if self._is_adaptive_encoding():
            return self.encoding.get_stats()
        return None
