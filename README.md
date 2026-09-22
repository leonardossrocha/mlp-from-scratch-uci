# Implementação de Perceptron Multicamadas (MLP) com Backpropagation

[Português](#implementação-de-perceptron-multicamadas-mlp-com-backpropagation) | [English](#multilayer-perceptron-mlp-implementation-with-backpropagation)

---

## Versão em Português

### 1. Descrição do Trabalho Acadêmico
Este repositório consiste no desenvolvimento e documentação do **Trabalho Prático Avaliativo (1ª Avaliação Parcial)** da disciplina **Tópicos Avançados em Inteligência Computacional (Código: DIN4101)**, ministrada pelos professores **Dr. Rodrigo Clemente Thom de Souza** e **Dr. Rodrigo Calvo** no âmbito do **Programa de Pós-Graduação em Ciência da Computação (PCC)** do Centro de Tecnologia / Departamento de Informática da **Universidade Estadual de Maringá (UEM)**.

#### 1.1 Objetivo e Escopo da Avaliação
O objetivo principal da atividade é consolidar o domínio prático e formal sobre as arquiteturas fundamentais de Redes Neurais Artificiais (RNAs), especificamente o modelo de **Perceptron Multicamadas (Multilayer Perceptron - MLP)** treinado por meio do algoritmo de **Retropropagação do Erro (*Backpropagation*)**.

A tarefa requer:
1. **Seleção de Base de Dados:** Escolha de um conjunto de dados do repositório *[UCI Machine Learning Repository](https://archive.ics.uci.edu/datasets)*.
2. **Implementação em Baixo Nível (NumPy Puro):** Evoluir o script base fornecido em aula (`rede_multicamada.py`) para reconhecer os padrões do dataset selecionado, operando diretamente sobre matrizes e derivadas analíticas. É expressamente vedado o uso de frameworks de alto nível de Deep Learning (como Keras, PyTorch, TensorFlow ou Scikit-Learn MLPClassifier).
3. **Flexibilidade e Otimização Arquitetural:** Ajustar livremente dimensões de entrada, camadas ocultas, número de neurônios, taxa de aprendizado, termo de momento e funções de ativação (Sigmoide, ReLU, Tanh).
4. **Mecanismos de Generalização e Regularização:** Implementação de rotinas manuais de *Early Stopping* e técnicas de sintonia fina (*hyperparameter tuning*).
5. **Métricas de Avaliação:** Avaliação no conjunto de teste empregando métricas analíticas de erro (MAE, MSE, RMSE) e Acurácia percentual.

---

### 2. Base de Dados Selecionada: Breast Cancer Wisconsin (Diagnostic)

Para a validação e treinamento do modelo, foi selecionado o dataset **[Breast Cancer Wisconsin (Diagnostic)](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)**, mantido pelo *UCI Machine Learning Repository*.

#### 2.1 Características do Dataset
- **Domínio:** Diagnóstico médico / Oncologia computacional.
- **Número de Instâncias:** 569 amostras.
- **Número de Atributos Preditores:** 30 atributos contínuos (calculados a partir de imagens digitalizadas de punção aspirativa por agulha fina - FNA de massas mamárias).
- **Dados Faltantes (*Missing Values*):** Nenhum.
- **Atributos:** Medições de raio, textura, perímetro, área, suavidade, compacidade, concavidade, pontos côncavos, simetria e dimensão fractal (divididos em valores médios, erro padrão e "pior" valor para cada núcleo celular).
- **Variável Alvo (*Target*):** Diagnóstico binário:
  - **Maligno (M):** Mapeado para classe positiva ($1$).
  - **Benigno (B):** Mapeado para classe negativa ($0$).
- **Distribuição de Classes:** 357 benignos (62,7%) e 212 malignos (37,3%).

#### 2.2 Pré-processamento Aplicado
- **Remoção de Identificadores:** Eliminação da coluna de identificação única do paciente (`ID`).
- **Codificação de Classes:** Mapeamento de `M` $\rightarrow 1$ e `B` $\rightarrow 0$.
- **Normalização Min-Max:** Reescalonamento linear de todos os 30 preditores para o intervalo $[0, 1]$ a fim de evitar a saturação precoce das funções de ativação e instabilidade nos gradientes:  
  $$x_{\text{norm}} = \frac{x - x_{\min}}{x_{\max} - x_{\min}}$$
- **Particionamento:** Divisão em subconjuntos de **Treino (70%)**, **Validação (15%)** e **Teste (15%)**, mantendo semente pseudoaleatória controlada para assegurar reprodutibilidade.

---

### 3. Fundamentação Teórica e Formulação Matemática

#### 3.1 Modelo do Neurônio Artificial
Cada neurônio $k$ processa o sinal de entrada $x \in \mathbb{R}^{m+1}$ através de uma combinação linear seguida por uma função de ativação não linear $\varphi(\cdot)$:
$$v_k = \sum_{j=0}^{m} w_{kj} x_j = \mathbf{w}_k^T \mathbf{x}$$
$$y_k = \varphi(v_k)$$
Onde $x_0 = +1$ representa a entrada do *bias* ($w_{k0} = b_k$).

#### 3.2 Algoritmo Backpropagation
O treinamento supervisionado minimiza a função de custo baseada no erro quadrático instantâneo:  
$$\xi(n) = \frac{1}{2} \sum_{k \in C_{\text{saída}}} e_k^2(n) = \frac{1}{2} \sum_{k \in C_{\text{saída}}} (d_k(n) - y_k(n))^2$$

Pela regra da cadeia, o gradiente local ($\delta$) é calculado:
- **Para a camada de saída ($k$):**
  $$\delta_k(n) = e_k(n) \cdot \varphi'(v_k(n)) = (d_k(n) - y_k(n)) \cdot \varphi'(v_k(n))$$
- **Para a camada oculta ($j$):**
  $$\delta_j(n) = \varphi'(v_j(n)) \sum_{k} \delta_k(n) w_{kj}(n)$$

A atualização dos pesos com termo de momento ($\alpha$) e taxa de aprendizado ($\eta$) é dada por:
$$\Delta w_{ji}(n) = \alpha \Delta w_{ji}(n-1) + \eta \delta_j(n) y_i(n)$$
$$w_{ji}(n+1) = w_{ji}(n) + \Delta w_{ji}(n)$$

#### 3.3 Funções de Ativação e Derivadas Analíticas
- **Sigmoide:**  
  $$\sigma(z) = \frac{1}{1 + e^{-z}}, \quad \sigma'(z) = \sigma(z)(1 - \sigma(z))$$
- **ReLU (Rectified Linear Unit):**  
  $$f(z) = \max(0, z), \quad f'(z) = \begin{cases} 1, & \text{se } z > 0 \\ 0, & \text{se } z \le 0 \end{cases}$$
- **Tangente Hiperbólica ($\tanh$):**  
  $$\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}, \quad \tanh'(z) = 1 - \tanh^2(z)$$

#### 3.4 Métricas de Avaliação
O modelo é avaliado no conjunto de teste através das seguintes métricas analíticas:
- **Erro Médio Absoluto (MAE):**   
$\text{MAE} = \frac{1}{N} \sum_{i=1}^N \vert{}d_i - y_i\vert{}$
- **Erro Quadrático Médio (MSE):**  
$\text{MSE} = \frac{1}{N} \sum_{i=1}^N (d_i - y_i)^2$
- **Raiz do Erro Quadrático Médio (RMSE):**  
$\text{RMSE} = \sqrt{\text{MSE}}$
- **Acurácia:**  
$\text{Acurácia} = \frac{\text{Previsões Corretas}}{N} \times 100\%$

---

### 4. Configuração do Ambiente (Linux Debian)

#### 4.1 Pré-requisitos
Certifique-se de que os pacotes essenciais do sistema estejam instalados:
```bash
sudo apt update && sudo apt install -y python3 python3-pip python3-venv git
```

#### 4.2 Clonando o Repositório e Criando o Ambiente Virtual
```bash
git clone https://github.com/leonardossrocha/mlp-from-scratch-uci.git
cd mlp-from-scratch-uci

python3 -m venv venv
source venv/bin/activate
```

#### 4.3 Instalando Dependências
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

### 5. Estrutura do Repositório
```text
mlp-from-scratch-uci/
├── data/                  # Conjunto de dados (Breast Cancer Wisconsin - Diagnostic)
│   └── wdbc.data
├── notebooks/             # Desenvolvimento interativo no JupyterLab
│   └── 01_mlp_breast_cancer.ipynb
├── src/                   # Código-fonte da implementação pura em NumPy
│   ├── __init__.py
│   ├── activation.py      # Funções de ativação (Sigmoid, ReLU, Tanh) e derivadas
│   ├── metrics.py         # Métricas analíticas (MAE, MSE, RMSE, Acurácia)
│   ├── mlp.py             # Classe do Perceptron Multicamadas e Backpropagation
│   └── utils.py           # Normalização Min-Max e split treino/validação/teste
├── main.py                # Script de execução direta via linha de comando
├── requirements.txt       # Dependências mínimas do projeto
├── LICENSE                # Licença MIT
└── README.md              # Documentação técnica bilíngue do projeto
```

---

### 6. Execução do Projeto
Para executar o treinamento via linha de comando:
```bash
python3 main.py
```

Para abrir o ambiente interativo de experimentação:
```bash
jupyter lab
```

---
---

# Multilayer Perceptron (MLP) Implementation with Backpropagation

## English Version

### 1. Academic Assignment Description
This repository contains the codebase, experiments, and technical documentation developed for the **Partial Practical Assessment (1st Evaluation Term)** of the course **Advanced Topics in Computational Intelligence (Code: DIN4101)**, taught by Prof. Dr. Rodrigo Clemente Thom de Souza and Prof. Dr. Rodrigo Calvo within the **Graduate Program in Computer Science (PCC)**, Department of Informatics, **State University of Maringá (UEM)**.

#### 1.1 Scope and Technical Requirements
The core purpose of this evaluation is to demonstrate mastery over the formal and practical mechanics of connectionist models, focusing on the **Multilayer Perceptron (MLP)** trained using the classic **Backpropagation** algorithm.

Key project guidelines include:
1. **Benchmark Selection:** Selection of a real-world tabular dataset from the *[UCI Machine Learning Repository](https://archive.ics.uci.edu/datasets)*.
2. **Low-Level Native Implementation (Pure NumPy):** Extending the baseline script provided in class (`rede_multicamada.py`) to classify the chosen dataset using native matrix algebra and analytical differentiation. High-level deep learning frameworks (e.g., PyTorch, TensorFlow, Keras, Scikit-Learn MLPClassifier) are strictly prohibited.
3. **Topology and Hyperparameter Tuning:** Custom design of input dimensions, hidden dense layers, neuron counts, learning rates, momentum factors, and activation functions (Sigmoid, ReLU, Tanh).
4. **Regularization & Training Stability:** Development of a native *Early Stopping* mechanism to prevent overfitting on validation loss.
5. **Analytical Metrics Evaluation:** Comprehensive model evaluation on an unseen test set using standard error metrics (MAE, MSE, RMSE) and classification accuracy.

---

### 2. Selected Benchmark Dataset: Breast Cancer Wisconsin (Diagnostic)

Model evaluation and training are carried out on the **[Breast Cancer Wisconsin (Diagnostic)](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)** dataset, hosted on the *UCI Machine Learning Repository*.

#### 2.1 Dataset Summary
- **Domain:** Medical Diagnosis / Computational Oncology.
- **Instances:** 569 samples.
- **Predictor Attributes:** 30 continuous numeric features (computed from digitized images of a fine needle aspirate - FNA of breast masses).
- **Missing Values:** None.
- **Features:** Radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, and fractal dimension (divided into mean, standard error, and "worst" values for each cell nucleus).
- **Target Variable:** Binary classification:
  - **Malignant (M):** Mapped to positive class ($1$).
  - **Benign (B):** Mapped to negative class ($0$).
- **Class Distribution:** 357 benign (62.7%) and 212 malignant (37.3%).

#### 2.2 Preprocessing Pipeline
- **ID Removal:** Dropping the non-informative patient identifier column (`ID`).
- **Target Encoding:** Mapping labels `M` $\rightarrow 1$ and `B` $\rightarrow 0$.
- **Min-Max Scaling:** Linearly normalizing all 30 predictors into the $[0, 1]$ interval to prevent early activation saturation and numerical instabilities:
  $$x_{\text{norm}} = \frac{x - x_{\min}}{x_{\max} - x_{\min}}$$
- **Dataset Partitioning:** Partitioned into **Train (70%)**, **Validation (15%)**, and **Test (15%)** sets using fixed random seeds for strict reproducibility.

---

### 3. Theoretical Background and Mathematical Formulation

#### 3.1 Artificial Neuron Model
Each neuron $k$ computes an affine transformation followed by an element-wise non-linear activation function $\varphi(\cdot)$:  
$$v_k = \sum_{j=0}^{m} w_{kj} x_j = \mathbf{w}_k^T \mathbf{x}$$  
$$y_k = \varphi(v_k)$$  
Where $x_0 = +1$ denotes the fixed bias input ($w_{k0} = b_k$).

#### 3.2 Backpropagation Algorithm
Supervised training minimizes the instantaneous sum of squared errors cost function:  
$$\xi(n) = \frac{1}{2} \sum_{k \in C_{\text{output}}} e_k^2(n) = \frac{1}{2} \sum_{k \in C_{\text{output}}} (d_k(n) - y_k(n))^2$$

Applying the chain rule yields the local gradient ($\delta$):  
- **For Output Layer Neurons ($k$):**  
  $$\delta_k(n) = e_k(n) \cdot \varphi'(v_k(n)) = (d_k(n) - y_k(n)) \cdot \varphi'(v_k(n))$$
- **For Hidden Layer Neurons ($j$):**  
  $$\delta_j(n) = \varphi'(v_j(n)) \sum_{k} \delta_k(n) w_{kj}(n)$$

Weight updates incorporating momentum ($\alpha$) and learning rate ($\eta$) are defined as:  
$$\Delta w_{ji}(n) = \alpha \Delta w_{ji}(n-1) + \eta \delta_j(n) y_i(n)$$  
$$w_{ji}(n+1) = w_{ji}(n) + \Delta w_{ji}(n)$$

#### 3.3 Activation Functions and Derivatives
- **Sigmoid:**  
  $$\sigma(z) = \frac{1}{1 + e^{-z}}, \quad \sigma'(z) = \sigma(z)(1 - \sigma(z))$$
- **ReLU (Rectified Linear Unit):**  
  $$f(z) = \max(0, z), \quad f'(z) = \begin{cases} 1, & \text{if } z > 0 \\ 0, & \text{if } z \le 0 \end{cases}$$  
- **Hyperbolic Tangent ($\tanh$):**  
  $$\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}, \quad \tanh'(z) = 1 - \tanh^2(z)$$

#### 3.4 Evaluation Metrics
Model performance on the test set is assessed using:
- **Mean Absolute Error (MAE):** $\text{MAE} = \frac{1}{N} \sum_{i=1}^N \vert{}d_i - y_i\vert{}$
- **Mean Squared Error (MSE):** $\text{MSE} = \frac{1}{N} \sum_{i=1}^N (d_i - y_i)^2$
- **Root Mean Squared Error (RMSE):** $\text{RMSE} = \sqrt{\text{MSE}}$
- **Accuracy:** $\text{Accuracy} = \frac{\text{Correct Predictions}}{N} \times 100\%$

---

### 4. Environment Setup (Linux Debian)

#### 4.1 Prerequisites
Ensure base packages are installed via `apt`:
```bash
sudo apt update && sudo apt install -y python3 python3-pip python3-venv git
```

#### 4.2 Repository Setup & Virtual Environment
```bash
git clone https://github.com/leonardossrocha/mlp-from-scratch-uci.git
cd mlp-from-scratch-uci

python3 -m venv venv
source venv/bin/activate
```

#### 4.3 Dependencies Installation
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

### 5. Repository Structure
```text
mlp-from-scratch-uci/
├── data/                  # Datasets retrieved from the UCI Repository
│   └── wdbc.data
├── notebooks/             # Exploratory analysis and visual experiments
│   └── 01_mlp_breast_cancer.ipynb
├── src/                   # Pure NumPy source implementation
│   ├── __init__.py
│   ├── activation.py      # Activation functions (Sigmoid, ReLU, Tanh) and derivatives
│   ├── metrics.py         # Analytical metrics (MAE, MSE, RMSE, Accuracy)
│   ├── mlp.py             # Multilayer Perceptron & Backpropagation engine
│   └── utils.py           # Min-Max scaler and train/val/test splitting
├── main.py                # Main execution script and evaluation benchmark
├── requirements.txt       # Minimal project requirements
├── LICENSE                # MIT License
└── README.md              # Bilingual technical documentation
```

---

### 6. Running the Code
To train and evaluate the network via CLI:
```bash
python3 main.py
```

To run interactive experiments in JupyterLab:
```bash
jupyter lab
```