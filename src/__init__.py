from .data import DataLoader, DataPreprocessor, preprocess_dataset
from .algorithms import KNNClassifier, find_optimal_k, GeneticAlgorithm, MatrixGeneticAlgorithm
from .fitness import AccuracyFitness, create_fitness_function
from .encoding import (
    BaseEncoding,
    BinaryEncoding,
    DecimalEncoding,
    RealEncoding,
    GaussianEncoding,
    AdaptiveEncoding
)

__all__ = [
    'DataLoader',
    'DataPreprocessor',
    'preprocess_dataset',
    'KNNClassifier',
    'find_optimal_k',
    'GeneticAlgorithm',
    'MatrixGeneticAlgorithm',
    'AccuracyFitness',
    'create_fitness_function',
    'BaseEncoding',
    'BinaryEncoding',
    'DecimalEncoding',
    'RealEncoding',
    'GaussianEncoding',
    'AdaptiveEncoding'
]
