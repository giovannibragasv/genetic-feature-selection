"""
Funções de visualização para experimentos de feature selection.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
from pathlib import Path


plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.figsize': (10, 6),
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})


def plot_convergence(
    fitness_history: List[Dict],
    title: str = "Convergence Plot",
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plota curva de convergência do GA.
    
    Args:
        fitness_history: Lista de dicts com 'generation', 'best_fitness', 'mean_fitness'
        title: Título do gráfico
        save_path: Caminho para salvar figura
        show: Se True, exibe a figura
        
    Returns:
        Objeto Figure do matplotlib
    """
    generations = [h['generation'] for h in fitness_history]
    best_fitness = [h['best_fitness'] for h in fitness_history]
    mean_fitness = [h['mean_fitness'] for h in fitness_history]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(generations, best_fitness, 'b-', linewidth=2, label='Best Fitness')
    ax.plot(generations, mean_fitness, 'r--', linewidth=1.5, alpha=0.7, label='Mean Fitness')
    
    ax.set_xlabel('Generation')
    ax.set_ylabel('Fitness (Accuracy)')
    ax.set_title(title)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    ax.set_ylim([0, 1.05])
    
    if save_path:
        fig.savefig(save_path)
        print(f"Figura salva em: {save_path}")
    
    if show:
        plt.show()
    
    return fig


def plot_multi_encoding_convergence(
    results: Dict[str, List[Dict]],
    title: str = "Convergence Comparison",
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plota curvas de convergência para múltiplos encodings.
    Similar à Figure 8 do paper.
    
    Args:
        results: Dict com {encoding_name: fitness_history}
        title: Título do gráfico
        save_path: Caminho para salvar
        show: Se True, exibe
        
    Returns:
        Objeto Figure
    """
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    linestyles = ['-', '--', '-.', ':', '-']
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    for i, (name, history) in enumerate(results.items()):
        generations = [h['generation'] for h in history]
        best_fitness = [h['best_fitness'] for h in history]
        
        ax.plot(generations, best_fitness, 
                color=colors[i % len(colors)],
                linestyle=linestyles[i % len(linestyles)],
                linewidth=2, 
                label=name)
    
    ax.set_xlabel('Generation')
    ax.set_ylabel('Best Fitness (Accuracy)')
    ax.set_title(title)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])
    
    if save_path:
        fig.savefig(save_path)
        print(f"Figura salva em: {save_path}")
    
    if show:
        plt.show()
    
    return fig


def plot_k_value_analysis(
    k_values: List[int],
    accuracies: List[float],
    optimal_k: int = 7,
    title: str = "K-Value Analysis (Figure 9)",
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plota análise do valor K conforme Figure 9 do paper.
    
    Args:
        k_values: Lista de valores de K testados
        accuracies: Lista de acurácias correspondentes
        optimal_k: Valor ótimo de K (default 7 conforme paper)
        title: Título do gráfico
        save_path: Caminho para salvar
        show: Se True, exibe
        
    Returns:
        Objeto Figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(k_values, accuracies, 'b-o', linewidth=2, markersize=8)
    
    if optimal_k in k_values:
        idx = k_values.index(optimal_k)
        ax.axvline(x=optimal_k, color='red', linestyle='--', alpha=0.7, label=f'Optimal K={optimal_k}')
        ax.scatter([optimal_k], [accuracies[idx]], color='red', s=150, zorder=5)
        ax.annotate(f'{accuracies[idx]:.4f}', 
                   (optimal_k, accuracies[idx]),
                   textcoords="offset points",
                   xytext=(10, 10),
                   fontsize=10)
    
    ax.set_xlabel('K Value')
    ax.set_ylabel('Recognition Rate (Accuracy)')
    ax.set_title(title)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(k_values)
    
    if save_path:
        fig.savefig(save_path)
        print(f"Figura salva em: {save_path}")
    
    if show:
        plt.show()
    
    return fig


def plot_feature_comparison(
    encoding_names: List[str],
    features_before: List[int],
    features_after: List[int],
    title: str = "Feature Selection Comparison",
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plota comparação de features antes/depois da seleção.
    Similar à Table 3 do paper em forma visual.
    
    Args:
        encoding_names: Nomes dos encodings
        features_before: Número de features antes
        features_after: Número de features depois
        title: Título
        save_path: Caminho para salvar
        show: Se True, exibe
        
    Returns:
        Objeto Figure
    """
    x = np.arange(len(encoding_names))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bars1 = ax.bar(x - width/2, features_before, width, label='Before Selection', color='lightblue')
    bars2 = ax.bar(x + width/2, features_after, width, label='After Selection', color='coral')
    
    ax.set_xlabel('Encoding Method')
    ax.set_ylabel('Number of Features')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(encoding_names, rotation=45, ha='right')
    ax.legend()
    
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{int(height)}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3),
                   textcoords="offset points",
                   ha='center', va='bottom', fontsize=9)
    
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{int(height)}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3),
                   textcoords="offset points",
                   ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path)
        print(f"Figura salva em: {save_path}")
    
    if show:
        plt.show()
    
    return fig


def plot_accuracy_comparison(
    encoding_names: List[str],
    accuracies: List[float],
    stds: Optional[List[float]] = None,
    title: str = "Accuracy Comparison by Encoding",
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plota comparação de acurácia entre encodings com barras de erro.
    
    Args:
        encoding_names: Nomes dos encodings
        accuracies: Acurácias médias
        stds: Desvios padrão (opcional)
        title: Título
        save_path: Caminho para salvar
        show: Se True, exibe
        
    Returns:
        Objeto Figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(encoding_names))
    colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c', '#f39c12']
    
    if stds is not None:
        bars = ax.bar(x, accuracies, yerr=stds, capsize=5,
                     color=colors[:len(encoding_names)], edgecolor='black')
    else:
        bars = ax.bar(x, accuracies, color=colors[:len(encoding_names)], edgecolor='black')
    
    ax.set_xlabel('Encoding Method')
    ax.set_ylabel('Accuracy')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(encoding_names, rotation=45, ha='right')
    ax.set_ylim([0, 1.1])
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar, acc in zip(bars, accuracies):
        ax.annotate(f'{acc:.4f}',
                   xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                   xytext=(0, 5),
                   textcoords="offset points",
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path)
        print(f"Figura salva em: {save_path}")
    
    if show:
        plt.show()
    
    return fig


def plot_boxplot_results(
    results: Dict[str, List[float]],
    title: str = "Distribution of Results (50 Runs)",
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plota boxplot dos resultados de múltiplos runs.
    
    Args:
        results: Dict com {encoding_name: lista_de_accuracies}
        title: Título
        save_path: Caminho para salvar
        show: Se True, exibe
        
    Returns:
        Objeto Figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    data = [results[name] for name in results.keys()]
    labels = list(results.keys())
    
    bp = ax.boxplot(data, labels=labels, patch_artist=True)
    
    colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c', '#f39c12']
    for patch, color in zip(bp['boxes'], colors[:len(labels)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_xlabel('Encoding Method')
    ax.set_ylabel('Accuracy')
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path)
        print(f"Figura salva em: {save_path}")
    
    if show:
        plt.show()
    
    return fig


def plot_threshold_evolution(
    threshold_history: List[float],
    fitness_history: List[float],
    title: str = "Adaptive Threshold Evolution",
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plota evolução do threshold adaptativo junto com fitness.
    
    Args:
        threshold_history: Histórico de thresholds
        fitness_history: Histórico de fitness
        title: Título
        save_path: Caminho para salvar
        show: Se True, exibe
        
    Returns:
        Objeto Figure
    """
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    generations = range(len(threshold_history))
    
    color1 = 'tab:blue'
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Threshold', color=color1)
    ax1.plot(generations, threshold_history, color=color1, linewidth=2, label='Threshold')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_ylim([0, 1])
    
    ax2 = ax1.twinx()
    color2 = 'tab:red'
    ax2.set_ylabel('Fitness (Accuracy)', color=color2)
    
    if len(fitness_history) < len(threshold_history):
        fitness_plot = fitness_history + [fitness_history[-1]] * (len(threshold_history) - len(fitness_history))
    else:
        fitness_plot = fitness_history[:len(threshold_history)]
    
    ax2.plot(generations, fitness_plot, color=color2, linewidth=2, linestyle='--', label='Best Fitness')
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.set_ylim([0, 1.05])
    
    ax1.set_title(title)
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower right')
    
    ax1.grid(True, alpha=0.3)
    
    if save_path:
        fig.savefig(save_path)
        print(f"Figura salva em: {save_path}")
    
    if show:
        plt.show()
    
    return fig


def plot_dataset_comparison(
    datasets: List[str],
    results: Dict[str, Dict[str, float]],
    metric: str = 'accuracy',
    title: str = "Dataset Comparison",
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plota comparação entre datasets para cada encoding.
    Similar à Figure 10 do paper.
    
    Args:
        datasets: Lista de nomes dos datasets
        results: Dict com {encoding: {dataset: valor}}
        metric: Nome da métrica
        title: Título
        save_path: Caminho para salvar
        show: Se True, exibe
        
    Returns:
        Objeto Figure
    """
    fig, ax = plt.subplots(figsize=(14, 7))
    
    x = np.arange(len(datasets))
    width = 0.15
    n_encodings = len(results)
    
    colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c', '#f39c12']
    
    for i, (encoding, data) in enumerate(results.items()):
        values = [data.get(d, 0) for d in datasets]
        offset = (i - n_encodings/2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=encoding, color=colors[i % len(colors)])
    
    ax.set_xlabel('Dataset')
    ax.set_ylabel(metric.capitalize())
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=45, ha='right')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path)
        print(f"Figura salva em: {save_path}")
    
    if show:
        plt.show()
    
    return fig


def create_results_table(
    results: Dict[str, Dict[str, Tuple[float, float]]],
    datasets: List[str],
    encodings: List[str],
    save_path: Optional[str] = None
) -> str:
    """
    Cria tabela de resultados no formato do paper (Table 5).
    
    Args:
        results: Dict com {dataset: {encoding: (mean, std)}}
        datasets: Lista de datasets
        encodings: Lista de encodings
        save_path: Caminho para salvar CSV
        
    Returns:
        String com tabela formatada
    """
    header = f"{'Dataset':<15}" + "".join(f"{enc:<20}" for enc in encodings)
    lines = [header, "-" * len(header)]
    
    for dataset in datasets:
        row = f"{dataset:<15}"
        for encoding in encodings:
            if dataset in results and encoding in results[dataset]:
                mean, std = results[dataset][encoding]
                row += f"{mean:.4f} ± {std:.4f}    "
            else:
                row += f"{'N/A':<20}"
        lines.append(row)
    
    table = "\n".join(lines)
    
    if save_path:
        import csv
        with open(save_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Dataset'] + encodings)
            for dataset in datasets:
                row = [dataset]
                for encoding in encodings:
                    if dataset in results and encoding in results[dataset]:
                        mean, std = results[dataset][encoding]
                        row.append(f"{mean:.4f} ± {std:.4f}")
                    else:
                        row.append("N/A")
                writer.writerow(row)
        print(f"Tabela salva em: {save_path}")
    
    return table
