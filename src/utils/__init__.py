"""
Utilitários para experimentos de feature selection.
"""

from .metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    print_classification_report,
    compute_statistics,
    format_result
)

from .visualization import (
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
    # Métricas
    'accuracy_score',
    'precision_score',
    'recall_score',
    'f1_score',
    'confusion_matrix',
    'classification_report',
    'print_classification_report',
    'compute_statistics',
    'format_result',
    # Visualização
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
