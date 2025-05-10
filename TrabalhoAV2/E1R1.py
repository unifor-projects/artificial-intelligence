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
)) # Adiciona coluna de bias (-1) na primeira coluna
y = dados[:, 1] # Segunda coluna: Potência gerada pelo aerogerador (kW)


p = X.shape[0] - 1 # Número de amostras (p) p = 1
N = X.shape[1] # Número de features (N) N = 2250


# ================================================
# ============= Normalização dos dados: ===========
# ================================================

# Normalização Min-Max para X (excluindo o bias)
X_min = X[1:].min()
X_max = X[1:].max()
X[1:] = (X[1:] - X_min) / (X_max - X_min)

# Normalização Min-Max para y
y_min = y.min()
y_max = y.max()
y = (y - y_min) / (y_max - y_min)


# Apresentação dos dados
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