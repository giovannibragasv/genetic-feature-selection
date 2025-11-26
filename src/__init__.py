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
from .utils import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    print_classification_report,
    compute_statistics,
    format_result,
    plot_convergence,
    plot_multi_encoding_convergence,
    plot_k_value_analysis,
    plot_feature_comparison,
    plot_accuracy_comparison,
    plot_boxplot_results,
    plot_threshold_evolution,
    plot_dataset_comparison,
    create_results_table
)

__all__ = [
    # Data
    'DataLoader',
    'DataPreprocessor',
    'preprocess_dataset',
    # Algorithms
    'KNNClassifier',
    'find_optimal_k',
    'GeneticAlgorithm',
    'MatrixGeneticAlgorithm',
    # Fitness
    'AccuracyFitness',
    'create_fitness_function',
    # Encodings
    'BaseEncoding',
    'BinaryEncoding',
    'DecimalEncoding',
    'RealEncoding',
    'GaussianEncoding',
    'AdaptiveEncoding',
    # Metrics
    'accuracy_score',
    'precision_score',
    'recall_score',
    'f1_score',
    'confusion_matrix',
    'classification_report',
    'print_classification_report',
    'compute_statistics',
    'format_result',
    # Visualization
    'plot_convergence',
    'plot_multi_encoding_convergence',
    'plot_k_value_analysis',
    'plot_feature_comparison',
    'plot_accuracy_comparison',
    'plot_boxplot_results',
    'plot_threshold_evolution',
    'plot_dataset_comparison',
    'create_results_table'
]
