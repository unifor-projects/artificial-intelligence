import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Carregar os dados
df = pd.read_csv("/home/mari/college/artificial-intelligence/TrabalhoAV2/dados/Spiral3d.csv", header=None)
df.columns = ['x1', 'x2', 'x3', 'label']

# Separar dados e rótulos
X = df[['x1', 'x2', 'x3']].values
y = df['label'].values

# Adicionar coluna de bias
X_bias = np.hstack((np.ones((X.shape[0], 1)), X))  # [N, 4]

# Inicializar pesos
pesos = np.zeros(X_bias.shape[1])

# Hiperparâmetros
learning_rate = 0.01
max_epochs = 1000
erros_por_epoca = []

# Treinamento
for epoca in range(max_epochs):
    erros = 0
    for i in range(X_bias.shape[0]):
        xi = X_bias[i]
        yi = y[i]
        pred = 1 if np.dot(pesos, xi) >= 0 else -1
        erro = yi - pred
        if erro != 0:
            pesos += learning_rate * erro * xi
            erros += 1
    erros_por_epoca.append(erros)
    if erros == 0:
        break

# Predições finais
y_pred = []
for i in range(X_bias.shape[0]):
    pred = 1 if np.dot(pesos, X_bias[i]) >= 0 else -1
    y_pred.append(pred)
y_pred = np.array(y_pred)

# Cálculo manual da acurácia
acertos = np.sum(y_pred == y)
acuracia = acertos / len(y)

# Exibir resultados
print(f"Acurácia: {acuracia * 100:.2f}%")
print(f"Pesos finais: {pesos}")

# Gráfico 3D da distribuição dos dados
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
cores = ['red' if label == -1 else 'blue' for label in y]
ax.scatter(df['x1'], df['x2'], df['x3'], c=cores, alpha=0.6)
ax.set_title('Distribuição 3D dos Dados - Spiral3d')
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('x3')
plt.tight_layout()
plt.show()
