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
2. **Implementação em Baixo Nível (NumPy Puro):** Evoluir o script base fornecido em aula (`rede_multicamada.py`) para reconhecer os padrões do dataset escolhido, operando diretamente sobre matrizes e derivadas analíticas. É expressamente vedado o uso de frameworks de alto nível de Deep Learning (como Keras, PyTorch, TensorFlow ou Scikit-Learn MLPClassifier).
3. **Flexibilidade e Otimização Arquitetural:** Ajustar livremente dimensões de entrada, camadas ocultas, número de neurônios, taxa de aprendizado, termo de momento e funções de ativação (Sigmoide, ReLU, Tanh).
4. **Mecanismos de Generalização e Regularização:** Implementação de rotinas manuais de *Early Stopping* e técnicas de sintonia fina (*hyperparameter tuning*).
5. **Métricas de Avaliação:** Avaliação rigorosa no conjunto de teste empregando métricas analíticas de erro (MAE, MSE, RMSE) e Acurácia percentual.

---

### 2. Fundamentação Teórica e Formulação Matemática

#### 2.1 Modelo do Neurônio Artificial
Cada neurônio $k$ processa o sinal de entrada $x \in \mathbb{R}^{m+1}$ através de uma combinação linear seguida por uma função de ativação não linear $\varphi(\cdot)$:
$$v_k = \sum_{j=0}^{m} w_{kj} x_j = \mathbf{w}_k^T \mathbf{x}$$
$$y_k = \varphi(v_k)$$
Onde $x_0 = +1$ representa a entrada do *bias* ($w_{k0} = b_k$).

#### 2.2 Algoritmo Backpropagation
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

#### 2.3 Métricas de Avaliação
O modelo é avaliado no conjunto de teste através das seguintes métricas analíticas:
- **Erro Médio Absoluto (MAE):** $\text{MAE} = \frac{1}{N} \sum_{i=1}^N |d_i - y_i|$
- **Erro Quadrático Médio (MSE):** $\text{MSE} = \frac{1}{N} \sum_{i=1}^N (d_i - y_i)^2$
- **Raiz do Erro Quadrático Médio (RMSE):** $\text{RMSE} = \sqrt{\text{MSE}}$
- **Acurácia:** $\text{Acurácia} = \frac{\text{Previsões Corretas}}{N} \times 100\%$

---

### 3. Configuração do Ambiente (Linux Debian)

#### 3.1 Pré-requisitos
Certifique-se de que os pacotes essenciais do sistema estejam instalados:
```bash
sudo apt update && sudo apt install -y python3 python3-pip python3-venv git
```

#### 3.2 Clonando o Repositório e Criando o Ambiente Virtual
```bash
git clone https://github.com/SEU_USUARIO/SEU_REPOSITORIO.git
cd SEU_REPOSITORIO

python3 -m venv venv
source venv/bin/activate
```

#### 3.3 Instalando Dependências
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

### 4. Estrutura do Repositório
```text
├── data/                  # Conjuntos de dados obtidos do UCI Repository
├── notebooks/             # Análise exploratória e experimentos em Jupyter
│   └── exploratory.ipynb
├── src/                   # Código-fonte da implementação pura em NumPy
│   ├── __init__.py
│   ├── activation.py      # Funções de ativação (Sigmoid, ReLU, Tanh) e derivadas
│   ├── metrics.py         # Métricas analíticas (MAE, MSE, RMSE, Acurácia)
│   ├── mlp.py             # Classe do Perceptron Multicamadas e Backpropagation
│   └── utils.py           # Normalização Min-Max e split treino/validação/teste
├── main.py                # Script de execução principal e benchmark
├── requirements.txt       # Dependências mínimas do projeto
└── README.md              # Documentação técnica do projeto
```

---

### 5. Execução do Projeto
Para executar o treinamento e a avaliação com os hiperparâmetros configurados:
```bash
python3 main.py
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

### 2. Theoretical Background and Mathematical Formulation

#### 2.1 Artificial Neuron Model
Each neuron $k$ computes an affine transformation followed by an element-wise non-linear activation function $\varphi(\cdot)$:
$$v_k = \sum_{j=0}^{m} w_{kj} x_j = \mathbf{w}_k^T \mathbf{x}$$
$$y_k = \varphi(v_k)$$
Where $x_0 = +1$ denotes the fixed bias input ($w_{k0} = b_k$).

#### 2.2 Backpropagation Algorithm
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

#### 2.3 Evaluation Metrics
Model performance on the test set is assessed using:
- **Mean Absolute Error (MAE):** $\text{MAE} = \frac{1}{N} \sum_{i=1}^N |d_i - y_i|$
- **Mean Squared Error (MSE):** $\text{MSE} = \frac{1}{N} \sum_{i=1}^N (d_i - y_i)^2$
- **Root Mean Squared Error (RMSE):** $\text{RMSE} = \sqrt{\text{MSE}}$
- **Accuracy:** $\text{Accuracy} = \frac{\text{Correct Predictions}}{N} \times 100\%$

---

### 3. Environment Setup (Linux Debian)

#### 3.1 Prerequisites
Ensure base packages are installed via `apt`:
```bash
sudo apt update && sudo apt install -y python3 python3-pip python3-venv git
```

#### 3.2 Repository Setup & Virtual Environment
```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
cd YOUR_REPOSITORY

python3 -m venv venv
source venv/bin/activate
```

#### 3.3 Dependencies Installation
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

### 4. Repository Structure
```text
├── data/                  # Datasets retrieved from the UCI Repository
├── notebooks/             # Exploratory analysis and visual experiments
│   └── exploratory.ipynb
├── src/                   # Pure NumPy source implementation
│   ├── __init__.py
│   ├── activation.py      # Activation functions (Sigmoid, ReLU, Tanh) and derivatives
│   ├── metrics.py         # Analytical metrics (MAE, MSE, RMSE, Accuracy)
│   ├── mlp.py             # Multilayer Perceptron & Backpropagation engine
│   └── utils.py           # Min-Max scaler and train/val/test splitting
├── main.py                # Main execution script and evaluation benchmark
├── requirements.txt       # Minimal project requirements
└── README.md              # Technical documentation
```

---

### 5. Running the Code
To train and evaluate the network with the selected dataset:
```bash
python3 main.py
```