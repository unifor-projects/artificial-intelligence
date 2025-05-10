# Etapa 1 Classificação 1 (E1C1)

# 1. Faça a organização do conjunto de dados para serem apresentados às redes neurais.

import numpy as np


# ================================================
# ============= Carregando os dados: =============
# ================================================
dados = np.loadtxt('dados/Spiral3d.csv', delimiter=',')

# Separa features (X) e labels (y)
X = dados[:, :3] # Primeiras três colunas: coordenadas 3D
y = dados[:, 3] # Quarta coluna: classe (1 ou -1)


# ================================================
# =========== Normalizando os dados: =============
# ================================================
media = X.mean(axis=0)
desvio_padrao = X.std(axis=0)
X_normalizado = (X - media) / desvio_padrao