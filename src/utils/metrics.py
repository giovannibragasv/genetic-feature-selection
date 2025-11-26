"""
Métricas de avaliação para classificação.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calcula acurácia conforme Eq 12 do paper.
    
    Accuracy = (TP + TN) / (TP + TN + FP + FN)
    
    Args:
        y_true: Labels verdadeiras
        y_pred: Labels preditas
        
    Returns:
        Acurácia (0 a 1)
    """
    return np.mean(y_true == y_pred)


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Calcula matriz de confusão.
    
    Args:
        y_true: Labels verdadeiras
        y_pred: Labels preditas
        
    Returns:
        Matriz de confusão (n_classes x n_classes)
    """
    classes = np.unique(np.concatenate([y_true, y_pred]))
    n_classes = len(classes)
    
    class_to_idx = {c: i for i, c in enumerate(classes)}
    
    cm = np.zeros((n_classes, n_classes), dtype=int)
    
    for true, pred in zip(y_true, y_pred):
        cm[class_to_idx[true], class_to_idx[pred]] += 1
    
    return cm


def precision_score(y_true: np.ndarray, y_pred: np.ndarray, 
                    average: str = 'macro') -> float:
    """
    Calcula precisão.
    
    Precision = TP / (TP + FP)
    
    Args:
        y_true: Labels verdadeiras
        y_pred: Labels preditas
        average: 'macro', 'micro', 'weighted' ou None
        
    Returns:
        Precisão
    """
    cm = confusion_matrix(y_true, y_pred)
    n_classes = cm.shape[0]
    
    precisions = []
    supports = []
    
    for i in range(n_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        
        if tp + fp > 0:
            precisions.append(tp / (tp + fp))
        else:
            precisions.append(0.0)
        
        supports.append(cm[i, :].sum())
    
    if average == 'macro':
        return np.mean(precisions)
    elif average == 'micro':
        tp_sum = np.trace(cm)
        total = cm.sum()
        return tp_sum / total if total > 0 else 0.0
    elif average == 'weighted':
        total_support = sum(supports)
        if total_support > 0:
            return sum(p * s for p, s in zip(precisions, supports)) / total_support
        return 0.0
    else:
        return np.array(precisions)


def recall_score(y_true: np.ndarray, y_pred: np.ndarray,
                 average: str = 'macro') -> float:
    """
    Calcula recall (sensibilidade).
    
    Recall = TP / (TP + FN)
    
    Args:
        y_true: Labels verdadeiras
        y_pred: Labels preditas
        average: 'macro', 'micro', 'weighted' ou None
        
    Returns:
        Recall
    """
    cm = confusion_matrix(y_true, y_pred)
    n_classes = cm.shape[0]
    
    recalls = []
    supports = []
    
    for i in range(n_classes):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        
        if tp + fn > 0:
            recalls.append(tp / (tp + fn))
        else:
            recalls.append(0.0)
        
        supports.append(cm[i, :].sum())
    
    if average == 'macro':
        return np.mean(recalls)
    elif average == 'micro':
        tp_sum = np.trace(cm)
        total = cm.sum()
        return tp_sum / total if total > 0 else 0.0
    elif average == 'weighted':
        total_support = sum(supports)
        if total_support > 0:
            return sum(r * s for r, s in zip(recalls, supports)) / total_support
        return 0.0
    else:
        return np.array(recalls)


def f1_score(y_true: np.ndarray, y_pred: np.ndarray,
             average: str = 'macro') -> float:
    """
    Calcula F1-score.
    
    F1 = 2 * (precision * recall) / (precision + recall)
    
    Args:
        y_true: Labels verdadeiras
        y_pred: Labels preditas
        average: 'macro', 'micro', 'weighted' ou None
        
    Returns:
        F1-score
    """
    prec = precision_score(y_true, y_pred, average=None)
    rec = recall_score(y_true, y_pred, average=None)
    
    f1s = []
    for p, r in zip(prec, rec):
        if p + r > 0:
            f1s.append(2 * p * r / (p + r))
        else:
            f1s.append(0.0)
    
    cm = confusion_matrix(y_true, y_pred)
    supports = [cm[i, :].sum() for i in range(cm.shape[0])]
    
    if average == 'macro':
        return np.mean(f1s)
    elif average == 'micro':
        return accuracy_score(y_true, y_pred)
    elif average == 'weighted':
        total_support = sum(supports)
        if total_support > 0:
            return sum(f * s for f, s in zip(f1s, supports)) / total_support
        return 0.0
    else:
        return np.array(f1s)


def classification_report(y_true: np.ndarray, y_pred: np.ndarray,
                          class_names: Optional[List[str]] = None) -> Dict:
    """
    Gera relatório completo de classificação.
    
    Args:
        y_true: Labels verdadeiras
        y_pred: Labels preditas
        class_names: Nomes das classes (opcional)
        
    Returns:
        Dicionário com métricas por classe e médias
    """
    cm = confusion_matrix(y_true, y_pred)
    classes = np.unique(np.concatenate([y_true, y_pred]))
    n_classes = len(classes)
    
    if class_names is None:
        class_names = [str(c) for c in classes]
    
    prec = precision_score(y_true, y_pred, average=None)
    rec = recall_score(y_true, y_pred, average=None)
    f1 = f1_score(y_true, y_pred, average=None)
    
    report = {
        'classes': {},
        'accuracy': accuracy_score(y_true, y_pred),
        'macro_avg': {
            'precision': np.mean(prec),
            'recall': np.mean(rec),
            'f1-score': np.mean(f1)
        },
        'weighted_avg': {
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1-score': f1_score(y_true, y_pred, average='weighted')
        }
    }
    
    for i, name in enumerate(class_names):
        support = cm[i, :].sum()
        report['classes'][name] = {
            'precision': prec[i],
            'recall': rec[i],
            'f1-score': f1[i],
            'support': int(support)
        }
    
    return report


def print_classification_report(y_true: np.ndarray, y_pred: np.ndarray,
                                 class_names: Optional[List[str]] = None) -> str:
    """
    Imprime relatório de classificação formatado.
    
    Args:
        y_true: Labels verdadeiras
        y_pred: Labels preditas
        class_names: Nomes das classes (opcional)
        
    Returns:
        String formatada do relatório
    """
    report = classification_report(y_true, y_pred, class_names)
    
    lines = []
    lines.append(f"{'':>15} {'precision':>10} {'recall':>10} {'f1-score':>10} {'support':>10}")
    lines.append("")
    
    for name, metrics in report['classes'].items():
        lines.append(
            f"{name:>15} {metrics['precision']:>10.4f} {metrics['recall']:>10.4f} "
            f"{metrics['f1-score']:>10.4f} {metrics['support']:>10d}"
        )
    
    lines.append("")
    lines.append(f"{'accuracy':>15} {'':>10} {'':>10} {report['accuracy']:>10.4f} {sum(m['support'] for m in report['classes'].values()):>10d}")
    
    macro = report['macro_avg']
    lines.append(
        f"{'macro avg':>15} {macro['precision']:>10.4f} {macro['recall']:>10.4f} "
        f"{macro['f1-score']:>10.4f}"
    )
    
    weighted = report['weighted_avg']
    lines.append(
        f"{'weighted avg':>15} {weighted['precision']:>10.4f} {weighted['recall']:>10.4f} "
        f"{weighted['f1-score']:>10.4f}"
    )
    
    return "\n".join(lines)


def compute_statistics(values: List[float]) -> Dict[str, float]:
    """
    Calcula estatísticas descritivas para múltiplos runs.
    
    Args:
        values: Lista de valores (ex: accuracies de 50 runs)
        
    Returns:
        Dicionário com mean, std, min, max, median
    """
    values = np.array(values)
    
    return {
        'mean': np.mean(values),
        'std': np.std(values),
        'min': np.min(values),
        'max': np.max(values),
        'median': np.median(values),
        'n_runs': len(values)
    }


def format_result(mean: float, std: float, precision: int = 4) -> str:
    """
    Formata resultado como "mean ± std" (padrão do paper).
    
    Args:
        mean: Média
        std: Desvio padrão
        precision: Casas decimais
        
    Returns:
        String formatada
    """
    return f"{mean:.{precision}f} ± {std:.{precision}f}"
