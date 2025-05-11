# ======= Perceptron Simples - Monte Carlo com curva de aprendizado =======
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Carregar dados
df = pd.read_csv("TrabalhoAV2/dados/Spiral3d.csv", header=None)
df.columns = ["x1", "x2", "x3", "label"]
X = df[["x1", "x2", "x3"]].values
y = np.where(df["label"].values == -1, 0, 1)


# Funcoes auxiliares
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
curvas = []

for _ in tqdm(range(R), desc="Perceptron Simples"):
    idx = np.random.permutation(len(X))
    split = int(0.8 * len(X))
    X_train, X_test = X[idx[:split]], X[idx[split:]]
    y_train, y_test = y[idx[:split]], y[idx[split:]]

    X_bias = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
    pesos = np.zeros(X_bias.shape[1])
    erros_por_epoca = []

    for _ in range(100):
        erros = 0
        for xi, yi in zip(X_bias, y_train):
            pred = 1 if np.dot(pesos, xi) >= 0 else 0
            erro = yi - pred
            if erro != 0:
                pesos += 0.01 * erro * xi
                erros += 1
        erros_por_epoca.append(erros)
        if erros == 0:
            break

    y_pred = prever(X_test, pesos)
    accs.append(acuracia(y_test, y_pred))
    predicoes.append((y_test, y_pred))
    curvas.append(erros_por_epoca)

# Estatísticas finais
accs = np.array(accs)
print("\nResumo da Acurácia - Perceptron Simples")
print(f"Média: {np.mean(accs):.4f}")
print(f"Desvio-Padrão: {np.std(accs):.4f}")
print(f"Maior Valor: {np.max(accs):.4f}")
print(f"Menor Valor: {np.min(accs):.4f}")

# Matrizes de confusão
idx_max = np.argmax(accs)
idx_min = np.argmin(accs)

sns.heatmap(matriz_confusao(*predicoes[idx_max]), annot=True, fmt="d", cmap="Blues")
plt.title("Perceptron - Melhor Rodada")
plt.show()

sns.heatmap(matriz_confusao(*predicoes[idx_min]), annot=True, fmt="d", cmap="Reds")
plt.title("Perceptron - Pior Rodada")
plt.show()

# Curvas de aprendizado
plt.figure(figsize=(10, 4))
plt.plot(curvas[idx_max], label="Melhor Rodada")
plt.plot(curvas[idx_min], label="Pior Rodada")
plt.title("Curva de Aprendizado (Erros por Época) - Perceptron")
plt.xlabel("Época")
plt.ylabel("Número de Erros")
plt.legend()
plt.grid(True)
plt.tight_layout()
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
