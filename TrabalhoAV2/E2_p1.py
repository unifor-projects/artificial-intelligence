import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# =====================================================
# ============== Implementação Adaline: ===============
# =====================================================
class Adaline:
    def __init__(self, learning_rate=0.01, n_epochs=1000, pr=1e-12):
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.pr = pr
        self.weights = None
        self.cost_history = []

    def __calc_eqm__(self, X, y):
        p, N = X.shape
        eq = 0
        for k in range(N):
            x_k = X[:, k].reshape(p, 1)
            u_k = (self.weights.T @ x_k)[0, 0]
            eq += np.sum((y[:, k] - u_k)**2)
        eqm = eq / (2 * N)
        return eqm

    def fit(self, X, y):
        p, N = X.shape
        m = y.shape[0]  # Número de classes
        self.weights = np.random.random_sample((p, m)) - 0.5
        self.cost_history = []

        EQM1 = 0
        EQM2 = 1
        for epoch in tqdm(range(self.n_epochs), desc="Adaline Training", colour="red", ncols=100):
            if np.abs(EQM1 - EQM2) < self.pr:
                break
            
            EQM1 = self.__calc_eqm__(X, y)
            self.cost_history.append(EQM1)
            
            for k in range(N):
                x_k = X[:, k].reshape(p, 1)
                u_k = (self.weights.T @ x_k)
                e_k = y[:, k].reshape(m, 1) - u_k
                self.weights += self.learning_rate * x_k @ e_k.T
            
            EQM2 = self.__calc_eqm__(X, y)

    def predict(self, X):
        linear_output = self.weights.T @ X
        # Convert to one-hot encoding
        predictions = np.zeros_like(linear_output)
        for i in range(linear_output.shape[1]):
            max_idx = np.argmax(linear_output[:, i])
            predictions[max_idx, i] = 1
        return predictions

# =====================================================
# ================ Implementação MLP: =================
# =====================================================
class MLP:
    def __init__(self, L=2, Q=[4, 4], m=3, lr=0.01, n_epochs=1000, pr=1e-12):
        self.L = L
        self.Q = Q
        self.m = m
        self.n_epochs = n_epochs
        self.lr = lr
        self.pr = pr
        self.W = []
        self.i = []
        self.y = []
        self.δ = []
        self.loss_history = []
    
    def __restart__(self):
        self.W = []
        self.i = []
        self.y = []
        self.δ = []
        self.loss_history = []
        
    def __init_weights__(self, p):
        self.__restart__()
        self.W.append(np.random.random_sample((self.Q[0], p + 1)) - 0.5)
        for i in range(1, self.L):
            self.W.append(np.random.random_sample((self.Q[i], self.Q[i-1] + 1)) - 0.5)
        self.W.append(np.random.random_sample((self.m, self.Q[-1] + 1)) - 0.5)
        self.i = [None] * (self.L + 1)
        self.y = [None] * (self.L + 1)
        self.δ = [None] * (self.L + 1)
        
    def __sigmoid__(self, x):
        return 1 / (1 + np.exp(-x))
    
    def __sigmoid_derivative__(self, x):
        return x * (1 - x)
    
    def forward(self, x_sample):
        for j in range(self.L + 1):
            if j == 0: 
                self.i[j] = np.dot(self.W[j], x_sample)
                self.y[j] = self.__sigmoid__(self.i[j])
            else: 
                y_prev_bias = np.insert(self.y[j-1], 0, -1)
                self.i[j] = np.dot(self.W[j], y_prev_bias)
                self.y[j] = self.__sigmoid__(self.i[j])
                
    def backward(self, x_sample, d):
        last_layer = len(self.W) - 1
        for j in range(last_layer, -1, -1):
            if j + 1 == len(self.W):
                self.δ[j] = self.__sigmoid_derivative__(self.y[j]) * (d - self.y[j])
                y_prev_bias = np.insert(self.y[j-1], 0, -1)
                self.W[j] += self.lr * np.outer(self.δ[j], y_prev_bias)
            elif j == 0:
                W_b = self.W[j+1][:, 1:]
                self.δ[j] = self.__sigmoid_derivative__(self.y[j]) * np.dot(W_b.T, self.δ[j+1])
                self.W[j] += self.lr * np.outer(self.δ[j], x_sample)
            else:
                W_b = self.W[j+1][:, 1:]
                self.δ[j] = self.__sigmoid_derivative__(self.y[j]) * np.dot(W_b.T, self.δ[j+1])
                y_prev_bias = np.insert(self.y[j-1], 0, -1)
                self.W[j] += self.lr * np.outer(self.δ[j], y_prev_bias)

    def __calc_eqm__(self, X, Y):
        EQM = 0
        N = X.shape[1]
        for k in range(N):
            x_k = X[:, k]
            d_k = Y[:, k]
            self.forward(x_k)
            y_k = self.y[-1]
            EQI = np.sum((d_k - y_k) ** 2)
            EQM += EQI
        EQM /= (2 * N)
        return EQM
        
    def fit(self, X, Y):
        p = X.shape[0] - 1
        self.__init_weights__(p)
        
        for epoch in tqdm(range(self.n_epochs), desc="MLP Training", colour="blue", ncols=100):
            idx = np.random.permutation(X.shape[1])
            X_shuffled = X[:, idx]
            Y_shuffled = Y[:, idx]
            
            for n in range(X_shuffled.shape[1]):
                x_sample = X_shuffled[:, n]
                d = Y_shuffled[:, n]
                self.forward(x_sample)
                self.backward(x_sample, d)
            
            EQM = self.__calc_eqm__(X, Y)
            self.loss_history.append(EQM)
            if EQM < self.pr:
                break
    
    def predict(self, X):
        N = X.shape[1]
        Y = np.zeros((self.m, N))
        for k in range(N):
            x_sample = X[:, k]
            self.forward(x_sample)
            Y[:, k] = self.y[-1]
        # Convert to one-hot encoding
        predictions = np.zeros_like(Y)
        for i in range(Y.shape[1]):
            max_idx = np.argmax(Y[:, i])
            predictions[max_idx, i] = 1
        return predictions

# =====================================================
# ================ Funções Auxiliares: ================
# =====================================================
def normalize_data(X):
    """Normaliza os dados para o intervalo [0, 1] exceto a coluna de bias"""
    # Não normaliza a coluna de bias (primeira linha)
    X_normalized = X.copy()
    for i in range(1, X.shape[0]):
        min_val = np.min(X[i, :])
        max_val = np.max(X[i, :])
        if max_val != min_val:  # Evita divisão por zero
            X_normalized[i, :] = (X[i, :] - min_val) / (max_val - min_val)
    return X_normalized

def load_data(filename):
    data = np.genfromtxt(filename, delimiter=',', dtype=str)
    X = data[:, :-1].astype(float).T
    y_str = data[:, -1]
    
    # One-hot encoding (0 a 1)
    y = np.zeros((3, len(y_str)))
    for i, label in enumerate(y_str):
        if label == 'NO':
            y[:, i] = [1, 0, 0]
        elif label == 'DH':
            y[:, i] = [0, 1, 0]
        elif label == 'SL':
            y[:, i] = [0, 0, 1]
    
    # Add bias to X
    X = np.vstack((-np.ones(X.shape[1]), X))
    return X, y

def data_partition(X, y, train_size=0.8):
    N = X.shape[1]
    idx = np.random.permutation(N)
    train_idx = idx[:int(train_size * N)]
    test_idx = idx[int(train_size * N):]
    
    X_train = X[:, train_idx]
    y_train = y[:, train_idx]
    X_test = X[:, test_idx]
    y_test = y[:, test_idx]
    
    return X_train, X_test, y_train, y_test

def accuracy(y_true, y_pred):
    correct = np.sum(np.all(y_true == y_pred, axis=0))
    return correct / y_true.shape[1]

def confusion_matrix(y_true, y_pred):
    cm = np.zeros((3, 3))
    for i in range(y_true.shape[1]):
        true_class = np.argmax(y_true[:, i])
        pred_class = np.argmax(y_pred[:, i])
        cm[true_class, pred_class] += 1
    return cm

def plot_confusion_matrix(cm, title):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', 
                xticklabels=['NO', 'DH', 'SL'], 
                yticklabels=['NO', 'DH', 'SL'])
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(f"TrabalhoAV2/results-multiclass/confusion_matrix_{title}.png")
    plt.close()

def plot_learning_curve(history, title):
    plt.figure(figsize=(10, 6))
    plt.plot(history, label=title)
    plt.title("Curva de Aprendizado")
    plt.xlabel('Época')
    plt.ylabel('EQM')
    plt.legend()
    plt.grid()
    plt.savefig(f"TrabalhoAV2/results-multiclass/learning_curve_{title}.png")
    plt.close()

def monte_carlo_simulation(X, y, R=100):
    adaline_accs = []
    mlp_accs = []
    
    adaline_best = {'acc': 0, 'model': None, 'cm': None}
    adaline_worst = {'acc': 1, 'model': None, 'cm': None}
    mlp_best = {'acc': 0, 'model': None, 'cm': None}
    mlp_worst = {'acc': 1, 'model': None, 'cm': None}
    
    X_normalized = normalize_data(X)
    
    for _round_ in tqdm(range(R), desc="Monte Carlo Simulation", colour="green", ncols=100):
        print()
        print("--------------------------------")
        print("Monte Carlo Simulation - Rodada: ", _round_)
        print("--------------------------------")
        X_train, X_test, y_train, y_test = data_partition(X_normalized, y)
        
        # Adaline
        adaline = Adaline(learning_rate=0.01, n_epochs=1000, pr=1e-5)
        adaline.fit(X_train, y_train)
        y_pred = adaline.predict(X_test)
        acc = accuracy(y_test, y_pred)
        adaline_accs.append(acc)
        
        if acc > adaline_best['acc']:
            adaline_best['acc'] = acc
            adaline_best['model'] = adaline
            adaline_best['cm'] = confusion_matrix(y_test, y_pred)
        if acc < adaline_worst['acc']:
            adaline_worst['acc'] = acc
            adaline_worst['model'] = adaline
            adaline_worst['cm'] = confusion_matrix(y_test, y_pred)
        
        # MLP
        mlp = MLP(L=2, Q=[10, 5], m=3, lr=0.01, n_epochs=1000, pr=1e-2)
        mlp.fit(X_train, y_train)
        y_pred = mlp.predict(X_test)
        acc = accuracy(y_test, y_pred)
        mlp_accs.append(acc)
        
        if acc > mlp_best['acc']:
            mlp_best['acc'] = acc
            mlp_best['model'] = mlp
            mlp_best['cm'] = confusion_matrix(y_test, y_pred)
        if acc < mlp_worst['acc']:
            mlp_worst['acc'] = acc
            mlp_worst['model'] = mlp
            mlp_worst['cm'] = confusion_matrix(y_test, y_pred)
    
    # Plot results
    plot_confusion_matrix(adaline_best['cm'], 'Adaline - Melhor Acurácia')
    plot_confusion_matrix(adaline_worst['cm'], 'Adaline - Pior Acurácia')
    plot_confusion_matrix(mlp_best['cm'], 'MLP - Melhor Acurácia')
    plot_confusion_matrix(mlp_worst['cm'], 'MLP - Pior Acurácia')
    
    # Learning curves
    plot_learning_curve(adaline_best['model'].cost_history, 'Adaline (Melhor)')
    plot_learning_curve(adaline_worst['model'].cost_history, 'Adaline (Pior)')
    plot_learning_curve(mlp_best['model'].loss_history, 'MLP (Melhor)')
    plot_learning_curve(mlp_worst['model'].loss_history, 'MLP (Pior)')
    
    # Results table
    print("\nResultados das 100 rodadas de Monte Carlo:")
    print("| Modelo                  | Média   | Desvio-Padrão | Maior Valor | Menor Valor |")
    print("|-------------------------|---------|---------------|-------------|-------------|")
    print(f"| Adaline                | {np.mean(adaline_accs):.4f} | {np.std(adaline_accs):.4f}     | {np.max(adaline_accs):.4f}    | {np.min(adaline_accs):.4f}    |")
    # print(f"| Perceptron Multicamadas | {np.mean(mlp_accs):.4f} | {np.std(mlp_accs):.4f}     | {np.max(mlp_accs):.4f}    | {np.min(mlp_accs):.4f}    |")
    
    summary_lines = [
        "\nResultados das 100 rodadas de Monte Carlo:",
        "| Modelo                  | Média   | Desvio-Padrão | Maior Valor | Menor Valor |",
        "|-------------------------|---------|---------------|-------------|-------------|"
    ]
    summary_lines.append(f"| Adaline                | {np.mean(adaline_accs):.4f} | {np.std(adaline_accs):.4f}     | {np.max(adaline_accs):.4f}    | {np.min(adaline_accs):.4f}    |")
    summary_lines.append(f"| Perceptron Multicamadas | {np.mean(mlp_accs):.4f} | {np.std(mlp_accs):.4f}     | {np.max(mlp_accs):.4f}    | {np.min(mlp_accs):.4f}    |")
    summary_text = "\n".join(summary_lines)
    print(summary_text)
    with open("TrabalhoAV2/results-multiclass/summary_report.txt", 'w') as f:
        f.write(summary_text)
    
    
    # Boxplot
    plt.figure(figsize=(8, 6))
    plt.boxplot([adaline_accs, mlp_accs], labels=['Adaline', 'MLP'])
    plt.title('Distribuição das Acurácias')
    plt.ylabel('Acurácia')
    plt.grid()
    plt.savefig("TrabalhoAV2/results-multiclass/boxplot.png")
    plt.close()
    
    bp = 1

# =====================================================
# ================ Execução Principal: ================
# =====================================================
if __name__ == "__main__":
    # Load data
    X, y = load_data("TrabalhoAV2/dados/coluna_vertebral.csv")
    
    # Run Monte Carlo simulation
    monte_carlo_simulation(X, y, R=100)