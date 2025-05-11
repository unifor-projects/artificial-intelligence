import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Carregar dados
df = pd.read_csv("/home/mari/college/artificial-intelligence/TrabalhoAV2/dados/Spiral3d.csv", header=None)
df.columns = ['x1', 'x2', 'x3', 'label']
X = df[['x1', 'x2', 'x3']].values
y = np.where(df['label'].values == -1, 0, 1)

# Funções
def treinar_perceptron(X, y, epocas=1000, lr=0.01):
    X_bias = np.hstack((np.ones((X.shape[0], 1)), X))
    pesos = np.zeros(X_bias.shape[1])
    for _ in range(epocas):
        for xi, yi in zip(X_bias, y):
            pred = 1 if np.dot(pesos, xi) >= 0 else 0
            erro = yi - pred
            if erro != 0:
                pesos += lr * erro * xi
    return pesos

def prever(X, pesos):
    X_bias = np.hstack((np.ones((X.shape[0], 1)), X))
    return (np.dot(X_bias, pesos) >= 0).astype(int)

def acuracia(y_true, y_pred):
    return np.mean(y_true == y_pred)

def matriz_confusao(y_true, y_pred):
    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    return np.array([[TN, FP], [FN, TP]])

# Monte Carlo
R = 250
accs = []
predicoes = []

for _ in tqdm(range(R), desc="Perceptron Simples"):
    idx = np.random.permutation(len(X))
    split = int(0.8 * len(X))
    X_train, X_test = X[idx[:split]], X[idx[split:]]
    y_train, y_test = y[idx[:split]], y[idx[split:]]

    pesos = treinar_perceptron(X_train, y_train)
    y_pred = prever(X_test, pesos)

    accs.append(acuracia(y_test, y_pred))
    predicoes.append((y_test, y_pred))

accs = np.array(accs)

# Estatísticas
media = np.mean(accs)
desvio = np.std(accs)
maior = np.max(accs)
menor = np.min(accs)

print("\nResumo da Acurácia - Perceptron Simples")
print(f"Média: {media:.4f}")
print(f"Desvio-Padrão: {desvio:.4f}")
print(f"Maior Valor: {maior:.4f}")
print(f"Menor Valor: {menor:.4f}")

# Matriz de confusão
idx_max = np.argmax(accs)
idx_min = np.argmin(accs)

sns.heatmap(matriz_confusao(*predicoes[idx_max]), annot=True, fmt='d', cmap="Blues")
plt.title("Perceptron - Melhor Rodada")
plt.show()

sns.heatmap(matriz_confusao(*predicoes[idx_min]), annot=True, fmt='d', cmap="Reds")
plt.title("Perceptron - Pior Rodada")
plt.show()

# Boxplot
plt.figure(figsize=(6, 4))
sns.boxplot(data=[accs], palette="Set2")
plt.xticks([0], ["Perceptron Simples"])
plt.title("Boxplot da Acurácia - Perceptron")
plt.ylabel("Acurácia")
plt.grid(True)
plt.tight_layout()
plt.show()
