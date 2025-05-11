# ======= MLP - Monte Carlo com curva de aprendizado =======
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


# MLP
class MLP:
    def __init__(self, input_dim, hidden=[10, 10], lr=0.1, epochs=300):
        self.lr = lr
        self.epochs = epochs
        self.W = []
        self.loss_history = []
        self.init_weights(input_dim, hidden)

    def init_weights(self, input_dim, hidden):
        layers = [input_dim] + hidden + [1]
        for i in range(len(layers) - 1):
            self.W.append(np.random.uniform(-0.5, 0.5, (layers[i + 1], layers[i] + 1)))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_deriv(self, x):
        return x * (1 - x)

    def forward(self, x):
        self.y = []
        for w in self.W:
            x = np.insert(x, 0, -1)
            x = self.sigmoid(np.dot(w, x))
            self.y.append(x)
        return self.y[-1]

    def backward(self, x, d):
        deltas = [None] * len(self.W)
        deltas[-1] = self.sigmoid_deriv(self.y[-1]) * (d - self.y[-1])
        for l in reversed(range(len(self.W) - 1)):
            W_no_bias = self.W[l + 1][:, 1:]
            deltas[l] = self.sigmoid_deriv(self.y[l]) * np.dot(
                W_no_bias.T, deltas[l + 1]
            )

        input_vals = []
        x_bias = np.insert(x, 0, -1)
        input_vals.append(x_bias)
        for out in self.y[:-1]:
            input_vals.append(np.insert(out, 0, -1))

        for l in range(len(self.W)):
            self.W[l] += self.lr * np.outer(deltas[l], input_vals[l])

    def fit(self, X, Y):
        N = X.shape[0]
        for _ in range(self.epochs):
            eqm = 0
            for x, d in zip(X, Y):
                y = self.forward(x)
                self.backward(x, d)
                eqm += np.mean((d - y) ** 2)
            self.loss_history.append(eqm / N)

    def predict(self, X):
        return np.array([self.forward(x) for x in X]).squeeze()


# Funcoes auxiliares
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

for _ in tqdm(range(R), desc="MLP"):
    idx = np.random.permutation(len(X))
    split = int(0.8 * len(X))
    X_train, X_test = X[idx[:split]], X[idx[split:]]
    y_train, y_test = y[idx[:split]], y[idx[split:]]

    mlp = MLP(input_dim=3)
    mlp.fit(X_train, y_train.reshape(-1, 1))
    y_pred = (mlp.predict(X_test) >= 0.5).astype(int)

    accs.append(acuracia(y_test, y_pred))
    predicoes.append((y_test, y_pred))
    curvas.append(mlp.loss_history)

# Estatísticas finais
accs = np.array(accs)
print("\nResumo da Acurácia - MLP")
print(f"Média: {np.mean(accs):.4f}")
print(f"Desvio-Padrão: {np.std(accs):.4f}")
print(f"Maior Valor: {np.max(accs):.4f}")
print(f"Menor Valor: {np.min(accs):.4f}")

# Matrizes de confusão
idx_max = np.argmax(accs)
idx_min = np.argmin(accs)

sns.heatmap(matriz_confusao(*predicoes[idx_max]), annot=True, fmt="d", cmap="Blues")
plt.title("MLP - Melhor Rodada")
plt.show()

sns.heatmap(matriz_confusao(*predicoes[idx_min]), annot=True, fmt="d", cmap="Reds")
plt.title("MLP - Pior Rodada")
plt.show()

# Curvas de aprendizado
plt.figure(figsize=(10, 4))
plt.plot(curvas[idx_max], label="Melhor Rodada")
plt.plot(curvas[idx_min], label="Pior Rodada")
plt.title("Curva de Aprendizado (EQM por Época) - MLP")
plt.xlabel("Época")
plt.ylabel("Erro Quadrático Médio")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Boxplot
plt.figure(figsize=(6, 4))
sns.boxplot(data=[accs], palette="Set2")
plt.xticks([0], ["MLP"])
plt.title("Boxplot da Acurácia - MLP")
plt.ylabel("Acurácia")
plt.grid(True)
plt.tight_layout()
plt.show()
