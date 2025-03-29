import numpy as np
import matplotlib.pyplot as plt

dados = np.loadtxt("TrabalhoAV1/dados/EMGsDataset.csv", delimiter=",")

# Extração das características (sensores) e rótulos:
sensor1 = dados[0, :] # Corrugador do Supercílio (1ª linha)
sensor2 = dados[1, :] # Zigomático Maior (2ª linha)
rotulos = dados[2, :].astype(int)  # Classes (3ª linha) 

N = dados.shape[1] # Quantidade de amostras
P = 2 # Quantidade de sensores
C = 5 # Número de classes

# --------------------------------------------
# 1. Formato para MQO (Least Square Method)
# --------------------------------------------
X_mqo = np.column_stack((sensor1, sensor2)) # Dimensão (N x p), p = 2

# Converter Y para o one-hot (N x C)
Y_mqo = np.zeros((N, C))
Y_mqo[np.arange(N), rotulos - 1] = 1 # Subtrai 1 para ajustar o índice que era de 1 a 5 para 0 a 4

# --------------------------------------------
# 2. Formato para Modelos Gaussianos Bayesianos
# --------------------------------------------
X_bayes = np.vstack((sensor1, sensor2)) # Dimensão (p x N), p = 2

# Converter Y para one-hot (C × N)
Y_bayes = np.zeros((C, N))
Y_bayes[rotulos - 1, np.arange(N)] = 1


# --------------------------------------------
# Verificação das dimensões
# --------------------------------------------
print("Formato MQO:")
print("X_mqo shape:", X_mqo.shape)  # Deve ser (50000, 2)
print("Y_mqo shape:", Y_mqo.shape)  # Deve ser (50000, 5)

print("\nFormato Bayesiano:")
print("X_bayes shape:", X_bayes.shape)  # Deve ser (2, 50000)
print("Y_bayes shape:", Y_bayes.shape)  # Deve ser (5, 50000)


bp=1