from regressao01 import np, X, Y, Z

# Matriz de variáveis regressoras (Temperatura e pH)
X = np.column_stack((X, Y)) # Dimensão (N x p), p = 2

# Vetor da variável dependente (Atividade Enzimática)
y = Z.reshape(-1, 1) # Dimensão (N x 1)

print(X.shape, y.shape)

bp=1