# Etapa 1 Classificação 1 (E1C1)

# 1. Faça a organização do conjunto de dados para serem apresentados às redes neurais.

import numpy as np

# ================================================
# ============= Carregando os dados: =============
# ================================================
dados = np.loadtxt(
    "TrabalhoAV2/dados/Spiral3d.csv",
    delimiter=",",
)

# Separa features (X) e labels (y)
X = dados[:, :3] # Primeira 3 colunas definindo 3 variáveis independentes
X = X.T # Transpõe para ficar na forma (3, N)
X = np.vstack((
    -np.ones(X.shape[1]), 
    X
)) # Adiciona coluna de bias (-1) na primeira linha
y = dados[:, 3] # 4º Coluna definindo o rótulo

p = X.shape[0] - 1 # Número de amostras (p) p = 1
N = X.shape[1] # Número de features (N) N = 2250

# ================================================
# =========== Normalizando os dados: =============
# ================================================

# Normalização Min-Max para X (excluindo o bias)
X_min = X[1:].min()
X_max = X[1:].max()
X[1:] = (X[1:] - X_min) / (X_max - X_min)

# Normalização Min-Max para y
y_min = y.min()
y_max = y.max()
y = (y - y_min) / (y_max - y_min)

#==========================================
#======== Partição dos dados: =============
#==========================================

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

y = dados[:, -1]  # Pega a última coluna
