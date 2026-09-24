"""
=====================================================================
Universidade Estadual de Maringá - UEM
Programa de Pós-Graduação em Ciência da Computação - PCC
Disciplina: Tópicos Avançados em Inteligência Computacional (DIN4101)
Docentes: Dr. Rodrigo Clemente Thom de Souza e Dr. Rodrigo Calvo
Discente: Leonardo S. S. da Rocha

Script Principal de Execução (main.py)
Benchmark da Rede Perceptron Multicamadas (MLP) com NumPy Puro
=====================================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo

# =====================================================================
# 1. FUNÇÕES DE ATIVAÇÃO, MÉTRICAS E REDE MLP
# =====================================================================

class Ativacoes:
    @staticmethod
    def sigmoid(z):
        z_clipped = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z_clipped))
    
    @staticmethod
    def sigmoid_derivada(a):
        return a * (1.0 - a)
    
    @staticmethod
    def relu(z):
        return np.maximum(0.0, z)
    
    @staticmethod
    def relu_derivada(a):
        return np.where(a > 0.0, 1.0, 0.0)
    
    @staticmethod
    def tanh(z):
        return np.tanh(z)
    
    @staticmethod
    def tanh_derivada(a):
        return 1.0 - (a ** 2)


class Metricas:
    @staticmethod
    def mae(y_true, y_pred):
        return np.mean(np.abs(y_true - y_pred))
    
    @staticmethod
    def mse(y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)
    
    @staticmethod
    def rmse(y_true, y_pred):
        return np.sqrt(Metricas.mse(y_true, y_pred))
    
    @staticmethod
    def acuracia(y_true, y_pred, limiar=0.5):
        y_pred_binario = np.where(y_pred >= limiar, 1.0, 0.0)
        return np.mean(y_pred_binario == y_true) * 100.0


class CamadaDensa:
    def __init__(self, n_entradas, n_neuronios, ativacao='sigmoid', seed=42):
        np.random.seed(seed)
        if ativacao == 'relu':
            limite = np.sqrt(2.0 / n_entradas)
        else:
            limite = np.sqrt(6.0 / (n_entradas + n_neuronios))
            
        self.pesos = np.random.uniform(-limite, limite, (n_entradas, n_neuronios))
        self.bias = np.zeros((1, n_neuronios))
        self.v_pesos = np.zeros_like(self.pesos)
        self.v_bias = np.zeros_like(self.bias)
        
        self.tipo_ativacao = ativacao
        if ativacao == 'sigmoid':
            self.ativar = Ativacoes.sigmoid
            self.derivar = Ativacoes.sigmoid_derivada
        elif ativacao == 'relu':
            self.ativar = Ativacoes.relu
            self.derivar = Ativacoes.relu_derivada
        elif ativacao == 'tanh':
            self.ativar = Ativacoes.tanh
            self.derivar = Ativacoes.tanh_derivada
        else:
            raise ValueError(f"Ativação '{ativacao}' não suportada.")


class RedeMLP:
    def __init__(self, camadas_arquitetura, ativacoes_camadas, taxa_aprendizado=0.1, momento=0.5, seed=42):
        self.taxa_aprendizado = taxa_aprendizado
        self.momento = momento
        self.camadas = []
        
        for i in range(len(camadas_arquitetura) - 1):
            n_in = camadas_arquitetura[i]
            n_out = camadas_arquitetura[i + 1]
            ativ = ativacoes_camadas[i]
            self.camadas.append(CamadaDensa(n_in, n_out, ativacao=ativ, seed=seed + i))

    def forward(self, X):
        ativacoes = [X]
        for camada in self.camadas:
            entrada_atual = ativacoes[-1]
            soma_sinapse = np.dot(entrada_atual, camada.pesos) + camada.bias
            saida_ativada = camada.ativar(soma_sinapse)
            ativacoes.append(saida_ativada)
        return ativacoes

    def backward(self, ativacoes, y_true):
        y_pred = ativacoes[-1]
        erro = y_true - y_pred
        delta = erro * self.camadas[-1].derivar(y_pred)
        deltas = [delta]
        
        for i in reversed(range(len(self.camadas) - 1)):
            camada_atual = self.camadas[i]
            camada_seguinte = self.camadas[i + 1]
            delta_seguinte = deltas[-1]
            delta_oculto = np.dot(delta_seguinte, camada_seguinte.pesos.T) * camada_atual.derivar(ativacoes[i + 1])
            deltas.append(delta_oculto)
            
        deltas.reverse()
        
        for i, camada in enumerate(self.camadas):
            entrada_camada = ativacoes[i]
            gradiente_pesos = np.dot(entrada_camada.T, deltas[i])
            gradiente_bias = np.sum(deltas[i], axis=0, keepdims=True)
            
            camada.v_pesos = (self.momento * camada.v_pesos) + (self.taxa_aprendizado * gradiente_pesos)
            camada.pesos += camada.v_pesos
            
            camada.v_bias = (self.momento * camada.v_bias) + (self.taxa_aprendizado * gradiente_bias)
            camada.bias += camada.v_bias

    def treinar(self, X_train, y_train, X_val, y_val, epocas=4000, early_stopping=True, paciencia=300):
        historico = {'treino_mse': [], 'treino_mae': [], 'val_mse': [], 'val_mae': []}
        melhor_loss_val = np.inf
        contador_paciencia = 0
        melhores_pesos = None
        melhores_biases = None
        
        for epoca in range(epocas):
            ativacoes_treino = self.forward(X_train)
            self.backward(ativacoes_treino, y_train)
            
            y_pred_treino = ativacoes_treino[-1]
            y_pred_val = self.forward(X_val)[-1]
            
            t_mse = Metricas.mse(y_train, y_pred_treino)
            t_mae = Metricas.mae(y_train, y_pred_treino)
            v_mse = Metricas.mse(y_val, y_pred_val)
            v_mae = Metricas.mae(y_val, y_pred_val)
            
            historico['treino_mse'].append(t_mse)
            historico['treino_mae'].append(t_mae)
            historico['val_mse'].append(v_mse)
            historico['val_mae'].append(v_mae)
            
            if early_stopping:
                if v_mse < melhor_loss_val:
                    melhor_loss_val = v_mse
                    contador_paciencia = 0
                    melhores_pesos = [np.copy(c.pesos) for c in self.camadas]
                    melhores_biases = [np.copy(c.bias) for c in self.camadas]
                else:
                    contador_paciencia += 1
                    if contador_paciencia >= paciencia:
                        for i, c in enumerate(self.camadas):
                            c.pesos = melhores_pesos[i]
                            c.bias = melhores_biases[i]
                        break
        return historico

    def prever(self, X):
        return self.forward(X)[-1]


# =====================================================================
# 2. PIPELINE DE CARGA, PRÉ-PROCESSAMENTO E BENCHMARK
# =====================================================================

def main():
    print("=" * 75)
    print("Executando Pipeline MLP - Breast Cancer Wisconsin Diagnostic")
    print("=" * 75)

    # 1. Carregamento dos dados
    try:
        dados = fetch_ucirepo(id=17)
        X_raw = dados.data.features.to_numpy()
        y_raw = dados.data.targets.to_numpy().ravel()
        print("[INFO] Dataset carregado com sucesso via UCI Repository API.")
    except Exception:
        caminho_local = os.path.join("data", "wdbc.data")
        print(f"[INFO] Carregando a partir do arquivo local: {caminho_local}")
        df = pd.read_csv(caminho_local, header=None)
        y_raw = df.iloc[:, 1].to_numpy()
        X_raw = df.iloc[:, 2:].to_numpy()

    # 2. Codificação binária e Normalização Min-Max
    y = np.where(y_raw == 'M', 1.0, 0.0).reshape(-1, 1)
    min_val = np.min(X_raw, axis=0)
    max_val = np.max(X_raw, axis=0)
    X = (X_raw - min_val) / (max_val - min_val)

    # 3. Particionamento (70% Treino, 15% Validação, 15% Teste)
    np.random.seed(42)
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)

    n_total = len(indices)
    n_train = int(0.70 * n_total)
    n_val = int(0.15 * n_total)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    print(f"[DADOS] Total: {n_total} | Treino: {len(train_idx)} | Validação: {len(val_idx)} | Teste: {len(test_idx)}")

    # 4. Configuração dos experimentos
    experimentos = [
        {"nome": "Exp 1: [30-8-1] Sigmoid", "camadas": [30, 8, 1], "ativ": ['sigmoid', 'sigmoid'], "eta": 0.1, "momento": 0.5},
        {"nome": "Exp 2: [30-16-1] ReLU + Sigmoid", "camadas": [30, 16, 1], "ativ": ['relu', 'sigmoid'], "eta": 0.05, "momento": 0.7},
        {"nome": "Exp 3: [30-16-8-1] Profunda (ReLU)", "camadas": [30, 16, 8, 1], "ativ": ['relu', 'relu', 'sigmoid'], "eta": 0.01, "momento": 0.8},
        {"nome": "Exp 4: [30-32-16-1] Alta Capacidade (Tanh)", "camadas": [30, 32, 16, 1], "ativ": ['tanh', 'tanh', 'sigmoid'], "eta": 0.05, "momento": 0.9}
    ]

    resultados = []
    historicos = {}

    for exp in experimentos:
        print(f"\nTreinando: {exp['nome']}...")
        rede = RedeMLP(exp['camadas'], exp['ativ'], taxa_aprendizado=exp['eta'], momento=exp['momento'], seed=42)
        hist = rede.treinar(X_train, y_train, X_val, y_val, epocas=4000, early_stopping=True, paciencia=300)
        historicos[exp['nome']] = hist

        # Métricas no treino
        p_train = rede.prever(X_train)
        mse_tr = Metricas.mse(y_train, p_train)
        mae_tr = Metricas.mae(y_train, p_train)
        rmse_tr = Metricas.rmse(y_train, p_train)
        acc_tr = Metricas.acuracia(y_train, p_train)

        # Métricas no teste
        p_test = rede.prever(X_test)
        mse_ts = Metricas.mse(y_test, p_test)
        acc_ts = Metricas.acuracia(y_test, p_test)

        resultados.append({
            "Configuração": exp['nome'],
            "Épocas": len(hist['treino_mse']),
            "Treino MSE": round(mse_tr, 6),
            "Treino MAE": round(mae_tr, 6),
            "Treino RMSE": round(rmse_tr, 6),
            "Treino Acc (%)": round(acc_tr, 2),
            "Teste MSE": round(mse_ts, 6),
            "Teste Acc (%)": round(acc_ts, 2)
        })

    # 5. Exibição da tabela consolidada
    df_res = pd.DataFrame(resultados)
    print("\n" + "=" * 80)
    print("TABELA DE RESULTADOS CONSOLIDADA:")
    print("=" * 80)
    print(df_res.to_string(index=False))

    # 6. Salvar curvas de convergência
    os.makedirs("notebooks", exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for nome, hist in historicos.items():
        axes[0].plot(hist['treino_mse'], label=f"{nome} ({hist['treino_mse'][-1]:.4f})")
        axes[1].plot(hist['treino_mae'], label=f"{nome} ({hist['treino_mae'][-1]:.4f})")

    axes[0].set_title("Evolução do Erro de Treinamento (MSE)", fontweight='bold')
    axes[0].set_xlabel("Épocas")
    axes[0].set_ylabel("MSE")
    axes[0].grid(True, linestyle='--', alpha=0.6)
    axes[0].legend(fontsize=8)

    axes[1].set_title("Evolução do Erro Médio Absoluto (MAE)", fontweight='bold')
    axes[1].set_xlabel("Épocas")
    axes[1].set_ylabel("MAE")
    axes[1].grid(True, linestyle='--', alpha=0.6)
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    caminho_figura = os.path.join("notebooks", "curvas_convergencia_erro.png")
    plt.savefig(caminho_figura, dpi=300)
    print(f"\n[OK] Gráficos de convergência salvos em: {caminho_figura}")


if __name__ == "__main__":
    main()