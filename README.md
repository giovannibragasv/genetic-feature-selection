# Feature Selection com Algoritmos Geneticos para Dados de Alta Dimensionalidade

Replicacao do experimento descrito no artigo "Feature selection algorithm based on optimized genetic algorithm and the application in high-dimensional data processing" (Feng, 2024), publicado na PLOS ONE.

## Equipe

- Giovanni Braga Soares Vasconcelos
- Carlos Eduardo Ferreira Teixeira
- Felipe Claudino Cruz
- Gabriel Mattos Teixeira dos Santos

Centro Universitario do Estado do Para (CESUPA)  
Curso de Ciencia da Computacao  
Disciplina: Inteligencia Artificial  
Professora: Polyana Santos Fonseca Nascimento  
2025

## Descricao do Experimento

Este projeto implementa e replica a metodologia proposta por Feng (2024) para selecao de features em dados de alta dimensionalidade utilizando algoritmos geneticos otimizados. O objetivo e comparar cinco diferentes estrategias de codificacao genetica (encodings) aplicadas ao problema de selecao de features em datasets de microarray para diagnostico de cancer.

### Problema

Dados de microarray apresentam o desafio da alta dimensionalidade: milhares de features (genes) para poucas amostras (pacientes). Isso causa problemas como overfitting, alto custo computacional e dificuldade de interpretacao. A selecao de features busca identificar um subconjunto relevante de genes que maximize a acuracia de classificacao.

### Metodologia

O algoritmo genetico evolui uma populacao de solucoes candidatas, onde cada individuo representa uma selecao de features. A qualidade de cada solucao e avaliada pela acuracia de um classificador KNN treinado apenas com as features selecionadas.

#### Encodings Implementados

1. **Binary**: Representacao binaria direta (0/1) para cada feature
2. **Decimal**: Valores inteiros com threshold para selecao
3. **Real**: Valores em ponto flutuante no intervalo [0,1]
4. **Gaussian**: Valores amostrados de distribuicao normal
5. **Adaptive**: Threshold dinamico que se ajusta baseado na evolucao do fitness

#### Parametros do Algoritmo Genetico

Conforme especificado na Table 4 do artigo original:

| Parametro            | Valor         |
| -------------------- | ------------- |
| Tamanho da populacao | 600           |
| Numero de geracoes   | 55            |
| Taxa de crossover    | 1.0           |
| Taxa de mutacao      | 0.2           |
| Metodo de selecao    | Torneio (k=3) |
| Elitismo             | 2 individuos  |

#### Classificador

- K-Nearest Neighbors (KNN)
- k = 7 (valor otimo conforme Figure 9 do artigo)
- Metrica de distancia: Euclidiana

#### Validacao

- Divisao treino/teste: 70%/30%
- Normalizacao: Z-score (media=0, desvio padrao=1)
- 10 execucoes independentes por configuracao
- Seeds fixas para reproducibilidade

## Datasets

Cinco datasets de microarray para classificacao de cancer foram utilizados:

| Dataset  | Features | Amostras | Classes | Dominio                 |
| -------- | -------- | -------- | ------- | ----------------------- |
| Colon    | 2.000    | 62       | 2       | Tumor de colon          |
| Leukemia | 7.129    | 34       | 2       | Leucemia ALL/AML        |
| CNS      | 7.129    | 60       | 2       | Sistema nervoso central |
| MLL      | 12.582   | 15       | 3       | Leucemia MLL            |
| Ovarian  | 15.154   | 253      | 2       | Cancer ovariano         |

Os datasets estao em formato Elvira (.dbc) e foram obtidos do Kent Ridge Bio-medical Dataset Repository.

## Estrutura do Repositorio

```
genetic-feature-selection/
├── src/
│   ├── data/                 # Carregamento e preprocessamento
│   │   ├── loader.py         # Parser para formato Elvira
│   │   ├── preprocessor.py   # Normalizacao z-score
│   │   └── fashion_mnist_loader.py
│   ├── algorithms/           # Algoritmos principais
│   │   ├── genetic_algorithm.py
│   │   ├── knn_classifier.py # KNN implementado do zero
│   │   └── matrix_ga.py
│   ├── encoding/             # Cinco estrategias de codificacao
│   │   ├── binary_encoding.py
│   │   ├── decimal_encoding.py
│   │   ├── real_encoding.py
│   │   ├── gaussian_encoding.py
│   │   └── adaptive_encoding.py
│   ├── fitness/              # Funcao de avaliacao
│   │   └── accuracy_fitness.py
│   └── utils/                # Metricas e visualizacao
│       ├── metrics.py
│       └── visualization.py
├── experiments/
│   ├── config/               # Configuracoes YAML
│   ├── run_experiments.py    # Script principal
│   ├── run_fashion_mnist.py  # Experimento Fashion-MNIST
│   ├── generate_tables.py    # Geracao de tabelas
│   └── generate_figures.py   # Geracao de graficos
├── data/
│   └── raw/                  # Datasets originais
├── results/
│   ├── tables/               # Resultados em CSV
│   └── figures/              # Graficos gerados
└── docs/
    └── article.pdf           # Artigo original
```

## Instalacao

### Requisitos

- Python 3.10 ou superior
- NumPy
- Pandas
- Matplotlib
- PyYAML

### Configuracao do Ambiente

```bash
# Clonar repositorio
git clone https://github.com/giovannibragasv/genetic-feature-selection.git
cd genetic-feature-selection

# Criar ambiente virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate     # Windows

# Instalar dependencias
pip install numpy pandas matplotlib pyyaml scikit-learn
```

## Execucao

### Teste de Validacao

Verificar se todos os componentes estao funcionando:

```bash
cd experiments
python test_system.py
```

### Experimento Completo

Executar todos os datasets com 10 runs por configuracao:

```bash
# Execucao sequencial
python run_experiments.py

# Execucao paralela (recomendado)
python run_experiments.py -d colon -n 10 &
python run_experiments.py -d leukemia -n 10 &
python run_experiments.py -d cns -n 10 &
python run_experiments.py -d mll -n 10 &
python run_experiments.py -d ovarian -n 10 &
```

### Experimento Fashion-MNIST

Para replicar a Table 3 do artigo original:

```bash
# Teste rapido
python run_fashion_mnist.py --quick

# Experimento completo
python run_fashion_mnist.py --samples 10000 --runs 10
```

### Geracao de Resultados

Apos a execucao dos experimentos:

```bash
python generate_tables.py    # Gera tabelas CSV e resumos
python generate_figures.py   # Gera graficos de convergencia e comparacao
```

## Resultados

### Comparacao de Acurácia por Encoding (Table 5)

| Dataset  | Binary      | Decimal     | Real        | Gaussian    | Adaptive    |
| -------- | ----------- | ----------- | ----------- | ----------- | ----------- |
| Colon    | 0.9158 ± 0.0444 | 0.9632 ± 0.0355 | 0.9632 ± 0.0355 | 1.0000 ± 0.0000 | 0.9947 ± 0.0166 |
| Leukemia | 0.9636 ± 0.0636 | 0.9909 ± 0.0287 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 |
| CNS      | 0.9000 ± 0.0438 | 0.9444 ± 0.0586 | 0.9333 ± 0.0511 | 0.9778 ± 0.0287 | 0.9833 ± 0.0268 |
| MLL      | 0.8000 ± 0.0000 | 0.8000 ± 0.0000 | 0.8000 ± 0.0000 | 0.8000 ± 0.0000 | 0.8000 ± 0.0000 |
| Ovarian  | 0.9526 ± 0.0257 | 0.9763 ± 0.0204 | 0.9829 ± 0.0206 | 0.9987 ± 0.0042 | 1.0000 ± 0.0000 |

Valores apresentados como média com desvio padrãoo de 10 execuções independentes.

### Principais Achados



## Implementação

### Decisões de Projeto

1. **Implementacao from scratch**: KNN, algoritmo genetico e todos os encodings foram implementados sem uso de bibliotecas prontas (exceto NumPy para operacoes matriciais), conforme requisito da disciplina.

2. **Otimizacao do KNN**: Versao vetorizada utilizando broadcasting do NumPy para calculo eficiente de distancias, reduzindo significativamente o tempo de execucao.

3. **Parser Elvira**: Desenvolvimento de parser customizado para leitura de arquivos .dbc no formato Elvira, utilizado pelos datasets de microarray.

4. **Reproducibilidade**: Uso de seeds fixas em todas as operacoes estocasticas para garantir reproducibilidade dos resultados.

### Diferencas em Relacao ao Artigo Original

- **Versoes dos datasets**: Os datasets utilizados apresentam pequenas diferencas no numero de features e amostras em relacao aos reportados no artigo, devido a diferentes versoes disponiveis nos repositorios.

- **Numero de execucoes**: Utilizamos 10 execucoes independentes (vs. 50 no artigo) devido a limitacoes de tempo computacional. Ainda assim, os resultados sao estatisticamente validos.

- **Tamanho dos conjuntos de teste**: Alguns datasets possuem conjuntos de teste muito pequenos (5-19 amostras), o que pode resultar em alta variancia e ocasionais acuracias de 100%.

## Referencias

FENG, Guilian (2024). Feature selection algorithm based on optimized genetic algorithm and the application in high-dimensional data processing. **PLOS ONE** 19(5): e0303088.. [DOI](https://doi.org/10.1371/journal.pone.0303088)

## Licença

Este projeto foi desenvolvido para fins academicos como parte da disciplina de Inteligencia Artificial do curso de Ciencia da Computacao do CESUPA.
