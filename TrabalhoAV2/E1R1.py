# Etapa 1 Regressão 1 (E1R1)

# 1. Faça a organização do conjunto de dados para serem apresentados às redes neurais.

import numpy as np


# ================================================
# ============= Carregando os dados: =============
# ================================================
dados = np.loadtxt('TrabalhoAV2/dados/aerogerador.dat')

# Separa features (X) e labels (y)
X = dados[:, :1] # Primeira coluna: Velocidade do vento (m/s)
X = X.T # Transpõe para ficar na forma (1, N)
X = np.vstack((
    -np.ones(X.shape[1]), 
    X
)) # Adiciona coluna de bias (-1) na primeira linha 
y = dados[:, 1] # Segunda coluna: Potência gerada pelo aerogerador (kW)


p = X.shape[0] - 1 # Número de amostras (p) p = 1
N = X.shape[1] # Número de features (N) N = 2250


# =================================================
# ============= Normalização dos dados: ===========
# =================================================

# Normalização Min-Max para X (excluindo o bias)
X_min = X[1:].min()
X_max = X[1:].max()
X[1:] = (X[1:] - X_min) / (X_max - X_min)

# Normalização Min-Max para y
y_min = y.min()
y_max = y.max()
y = (y - y_min) / (y_max - y_min)

# ==========================================
# ======== Partição dos dados: =============
# ==========================================
def data_partition(m, train_size=0.8):
    # Embaralhamento dos dados para partição
    idx = np.random.permutation(X.shape[1])
    X_shuffled = X[:, idx]
    y_shuffled = y[idx]
    
    # Separa os dados para treinamento e teste (80% treino, 20% teste) para X e y
    X_train, X_test = X_shuffled[:, :int(train_size * N)], X_shuffled[:, int(train_size * N):] # X = [p x N]

    # Ajustando a dimensão de y para [m x N]
    Y = y_shuffled.reshape(m, N) # y = [m x N]
    # Separa os dados para treinamento e teste (80% treino, 20% teste) para y
    y_train, y_test = Y[:, :int(train_size * N)], Y[:, int(train_size * N):]
    
    return X_train, X_test, y_train, y_test


# ================================================
# ========== Apresentação dos dados: =============
# ================================================
print("\n=====================================\n")

print(f"Conjunto de dados: {dados.shape}")
print(f"Número de amostras (N): {N}")
print(f"Número de features (p): {p}")

print("\n=====================================\n")

print(f"X.shape (bias incluso): {X.shape}")
print(f"y.shape: {y.shape}")

print("\n=====================================\n")

# Exibir algumas amostras
print(f"dados[10:20]: \n{dados[10:20]}")



bp = 1