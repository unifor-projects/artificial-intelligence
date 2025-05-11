# Etapa 1 Regressão 3 (E1R3)

# 3. Os modelos de RNA a serem implementados nessa etapa serão: ADALINE e Perceptron de Múltiplas Camadas (MLP). 
# Para cada modelo, deve-se discutir como os hiperparâmetros foram escolhidos.

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import E1R1 as r1


# =====================================================
# ============== Implementação Adaline: ===============
# =====================================================
class AdalineRegressor:
    def __init__(self, learning_rate=0.01, n_epochs=1000, pr= 1e-12):
        self.learning_rate = learning_rate # eta
        self.n_epochs = n_epochs
        self.pr = pr
        self.weights = None
        self.cost_history = []

    def __calc_eqm__(self, X, y):
        
        p,N = X.shape
        eq = 0
        for k in range(N):
            x_k = X[:,k].reshape(p,1)
            u_k = (self.weights.T@x_k)[0,0]
            eq += (float(y[k])-u_k)**2

        eqm = eq/(2*N)
        return eqm

    def fit(self, X, y, show_eqm=False):
        p, N = X.shape
        self.weights = np.random.random_sample((p,1))-.5
        self.cost_history = []
        
        y = y.reshape(N,1)

        EQM1 = 0
        EQM2 = 1
        for epoch in tqdm(range(self.n_epochs), desc='Treinando Adaline', ncols=100):
            if np.abs(EQM1 - EQM2) < self.pr:
                break
            
            EQM1 = self.__calc_eqm__(X, y)
            self.cost_history.append(EQM1)
            
            for k in range(N):
                x_k = X[:, k].reshape(p,1)
                u_k = (self.weights.T@x_k)[0,0]
                e_k = y[k] - u_k
                self.weights += self.learning_rate * e_k * x_k
        
            EQM2 = self.__calc_eqm__(X, y)
            
            if show_eqm and epoch % 25 == 0:
                print()
                print(f"Época {epoch}, EQM1: {EQM1:.6f}, EQM2: {EQM2:.6f}")

    def predict(self, X):
        linear_output = np.dot(self.weights.T, X)
        return linear_output

if __name__ == '__main__':
    # Teste de implementação
    adaline = AdalineRegressor(learning_rate=0.01, n_epochs=300, pr= 1e-12)
    X_train, X_test, y_train, y_test = r1.data_partition(m=1, train_size=0.8)
    adaline.fit(X_train, y_train, show_eqm=True)
    predict = adaline.predict(X_test)

    # Gera pontos para a linha de regressão
    x_min, x_max = r1.X[1:2, :].min(), r1.X[1:2, :].max()
    x_reg = np.linspace(x_min, x_max, r1.N).reshape(-1, 1)
    X_reg_b = np.c_[-np.ones((x_reg.shape[0], 1)), x_reg]
    y_reg = adaline.predict(X_reg_b.T)

    print(f"Predição: \n{predict} \n")
    print(f"Pesos: \n{adaline.weights} \n")
    print(f"EQM: {adaline.cost_history[-1]}")
    
    # Plota os dados de treinamento e a linha de regressão
    plt.figure(figsize=(10, 6))
    plt.scatter(r1.X[1:2, :], r1.y, color='blue', s=10, alpha=0.5)
    plt.scatter(X_test[1:2, :], predict.T[:, -1], color='green', s=10, alpha=0.5)
    plt.plot(x_reg, y_reg.T[:, -1], color='red', label='Regressão Linear (Adaline)')
    plt.title('Gráfico de Espalhamento: Velocidade do vento x Potência gerada')
    plt.xlabel('Velocidade do vento (m/s)')
    plt.ylabel('Potência gerada (kW)')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    bp = 1


# =====================================================
# ================ Implementação MLP: =================
# =====================================================

class MLPRegressor:
    def __init__(self, L=3, Q=[3, 4, 3], m=3, lr=0.01, n_epochs=1000, pr=1e-12):
        """
        Inicializa a rede MLP com os parâmetros fornecidos.
        
        Args:
            L (int): Número de camadas ocultas.
            Q (list): Lista com o número de neurônios em cada camada oculta.
            m (int): Número de neurônios na camada de saída.
            lr (float): Taxa de aprendizado (η).
            n_epochs (int): Número máximo de épocas para treinamento.
            pr (float): Precisão (ε) para critério de parada do treinamento.
        """
        
        self.L = L # Quantidade de Camadas Ocultas
        self.Q = Q # Lista da quantidade de Neurônios por Camada Oculta
        self.m = m # Quantidade de Neurônios na Camada de Saída
        self.n_epochs = n_epochs # Quantidade de Epocas
        self.lr = lr # Learning Rate (η) - eta
        self.pr = pr # Precisão (ε) para parada de treinamento (Stop Criteria)
        
        # Inicializar estruturas para armazenar pesos, entradas, saídas e deltas
        # Cada lista possui L+1 elementos, onde o primeiro elemento é a camada de entrada
        # e os demais elementos correspondem às camadas ocultas e de saída.
        self.W = []  # Lista de matrizes de pesos
        self.i = []  # Lista de vetores de entrada líquida
        self.y = []  # Lista de vetores de saída
        self.δ = []  # Lista de vetores de delta
        self.loss_history = [] # Lista de erros quadráticos médios (EQM)
    
    def __restart__(self):
        self.W = []
        self.i = []
        self.y = []
        self.δ = []
        self.loss_history = []
        
    def __init_weights__(self, p):
        """
        Inicializa os pesos da rede MLP com valores aleatórios entre -0.5 e 0.5.
        
        Args:
            p (int): Número de atributos de entrada.
        """
        # Reinicia os pesos, entradas, saídas e deltas
        self.__restart__()
        
        # Inicializa os pesos da camada de entrada para a primeira camada oculta
        # W possui dimensão [q0 por p+1]
        self.W.append(np.random.random_sample((self.Q[0], p + 1)) - 0.5)
        
        # Inicializa os pesos entre as camadas ocultas
        for i in range(1, self.L):
            # W possui dimensão [qi por qi-1+1]
            self.W.append(np.random.random_sample((self.Q[i], self.Q[i-1] + 1)) - 0.5)
        
        # Inicializa os pesos da última camada oculta para a camada de saída
        self.W.append(np.random.random_sample((self.m, self.Q[-1] + 1)) - 0.5) # Camada de saída

        # Inicializar listas com None para i, y, δ
        self.i = [None] * (self.L + 1)
        self.y = [None] * (self.L + 1)
        self.δ = [None] * (self.L + 1)
        
    def __sigmoid__(self, x):
        """
        Função de ativação sigmoide.
        
        Args:
            x (np.ndarray): Vetor de entrada.
        
        Returns:
            np.ndarray: Vetor de saída da função de ativação sigmoide.
        """
        
        return 1 / (1 + np.exp(-x))
    
    def __sigmoid_derivative__(self, x):
        """
        Derivada da função de ativação sigmoide.
        
        Args:
            x (np.ndarray): Vetor de entrada após a função de ativação sigmoide.
        
        Returns:
            np.ndarray: Vetor de saída da derivada da função de ativação sigmoide.
        """
        
        # g(u) = 1 / (1 + exp(-u))
        # g'(u) = g(u) * (1 - g(u)) 
        #   # essa multiplicação gera a seguinte expressão: exp(-u) / (1 + exp(-u))^2
        #   # que é equivalente a: g(u) * (1 - g(u))
        # Portanto, a derivada da função sigmoide é dada por:
        # g'(u) = x * (1 - x)
        # onde x = g(u)
        
        return x * (1 - x)
    
    def forward(self, x_sample):
        """
        Realiza a propagação para frente (forward propagation) na rede MLP.
        
        Args:
            x_sample (np.ndarray): Vetor de entrada da amostra. (com o bias já incluído) 
        """
        
        for j in range(self.L + 1):
            
            if j == 0: 
                # Camdada de entrada -> primeira camada oculta
                self.i[j] = np.dot(self.W[j], x_sample) # i0 = w0 * x
                # como resultado obtemos uma matriz [q0 x 1]
                # a função de ativação é aplicada
                self.y[j] = self.__sigmoid__(self.i[j]) # y0 = g(i0)
                
            else: 
                # Adiciona o bias (-1) à saida da camada anterior
                y_prev_bias = np.insert(self.y[j-1], 0, -1) # y_prev_bias = [-1, y_prev]
                self.i[j] = np.dot(self.W[j], y_prev_bias) # i_j = w_j * y_prev_bias
                self.y[j] = self.__sigmoid__(self.i[j]) # y_j = g(i_j)
                
    def backward(self, x_sample, d):
        """
        Realiza a retropropagação (backward propagation) na rede MLP.
        
        Args:
            x_sample (np.ndarray): Vetor de entrada da amostra. (com o bias já incluído) 
            d (np.ndarray): Vetor de rótulo desejada para a amostra.
        """
        
        # Começamos pela última camada (saída)
        # e retrocedemos até a primeira camada oculta
        last_layer = len(self.W) - 1 # len(self.W) = camada oculta (L) + camada de saída(1) = self.L + 1
        
        for j in range(last_layer, -1, -1):
            
            if j + 1 == len(self.W): # Camada de saída
                # Formula do delta: δm = g'(U_m) * (d(k) - y_m(k))
                self.δ[j] = self.__sigmoid_derivative__(self.y[j]) * (d - self.y[j])
                
                # Adiciona o bias (-1) à saida da camada anterior
                y_prev_bias = np.insert(self.y[j-1], 0, -1) # y_prev_bias = [-1, y_prev]
                
                # Formula do gradiente: W_j = W_j + η * δ_m ⊗ y_prev_bias
                self.W[j] += self.lr * np.outer(self.δ[j], y_prev_bias) # Atualiza os pesos da camada de saída
            
            elif j == 0: # Primera camada oculta
                # Remove o bias da camada à direita (segunda camada oculta)
                W_b = self.W[j+1][:,1:]
                
                # Formula do delta: δ0 = g'(U_0) * W_1.T * δ1
                self.δ[j] = self.__sigmoid_derivative__(self.y[j]) * np.dot(W_b.T, self.δ[j+1])
                
                # Formula do gradiente: W_0 = W_0 + η * δ_0 ⊗ x
                self.W[j] += self.lr * np.outer(self.δ[j], x_sample) # Atualiza os pesos da camada de entrada
            
            else: # Camada oculta intermediária
                # Remove o bias da camada à direita (camada de saída)
                W_b = self.W[j+1][:,1:]
                
                # Formula do delta: δj = g'(U_j) * W_(j+1).T * δ_(j+1)
                self.δ[j] = self.__sigmoid_derivative__(self.y[j]) * np.dot(W_b.T, self.δ[j+1])
                
                # Adiciona o bias (-1) à saida da camada anterior
                y_prev_bias = np.insert(self.y[j-1], 0, -1) # y_prev_bias = [-1, y_prev]
                
                # Formula do gradiente: W_j = W_j + η * δ_j ⊗ y_prev_bias
                self.W[j] += self.lr * np.outer(self.δ[j], y_prev_bias) # Atualiza os pesos da camada oculta

    def __calc_eqm__(self, X, Y):
        """
        Calcula o erro quadrático médio (EQM) entre as saídas da rede e os rótulos desejados.
        
        Args:
            X (np.ndarray): Matriz de dados de entrada (p+1 x N).
            Y (np.ndarray): Matriz de Rótulos de treinamento (m x N). m é igual a c — cada rótulo é uma saída da rede.
        
        Returns:
            float: Valor do erro quadrático médio (EQM).
        """
        EQM = 0
        N = X.shape[1] # Número de amostras
        
        for k in range(N):
            x_k = X[:, k] # Amostra de entrada (p+1 x 1)
            d_k = Y[:, k] # Rótulo de saída (m x 1)
            
            self.forward(x_k) # Propagação para frente
            y_k = self.y[-1] # Saída da camada de saída (m)
            
            # formula do EQI: EQI = Σ1^m (d_k - y_k)^2
            EQI = 0
            for n in range(self.m): # n = quantidade de neurônios na camada de saída — Basicamente a quantidade de classes
                EQI += (d_k[n] - y_k[n]) ** 2
            
            EQM += EQI
            
        # formula do EQM: EQM = 1/(2N) * { Σ1^N [EQI] }
        EQM /= (2*N)
        return EQM
        
        
    
    def fit(self, X, Y, show_eqm=False):
        """
        Treina a rede MLP com os dados de entrada e saída fornecidos.
        
        Args:
            X (np.ndarray): Matriz de dados de treinamento (p x N).
            Y (np.ndarray): Matriz de Rótulos de treinamento (m x N).
        """
        
        self.__init_weights__(r1.p) # Inicializa os pesos da rede MLP
        
        EQM = 1
        for epoch in tqdm(range(self.n_epochs), desc="Treinamento MLP", ncols=100):
            
            if EQM < self.pr:
                break
            
            # Embaralhamento dos dados para cada época
            idx = np.random.permutation(X.shape[1])
            X_shuffled = X[:, idx]
            Y_shuffled = Y[:, idx]
            
            
            # Treinamento para cada amostra
            for n in range(X_shuffled.shape[1]):
                x_sample = X_shuffled[:, n] # Amostra de entrada (p+1 x 1)
                d = Y_shuffled[:, n] # Rótulo de saída (m x 1)
                
                self.forward(x_sample) # Propagação para frente
                self.backward(x_sample, d) # Propagação para trás
            
            # Calcula o EQM para cada epoca
            EQM = self.__calc_eqm__(X, Y)
            self.loss_history.append(EQM)
            
            # Opcional: imprime o EQM a cada 100 épocas
            if show_eqm and epoch % 25 == 0:
                print()
                print(f"Época {epoch}, EQM: {EQM:.6f}")
    
    
    def predict(self, X):
        """
        Realiza a predição da saída da rede MLP para os dados de entrada fornecidos.
        
        Args:
            X (np.ndarray): Matriz de dados de entrada (p+1 x N).
        
        Returns:
            np.ndarray: Matriz de saídas da rede MLP (m x N).
        """
        
        N = X.shape[1] # Número de amostras
        Y = np.zeros((self.m, N)) # Inicializa a matriz de saídas (m x N)
        
        for k in range(N):
            x_sample = X[:, k] # Amostra de entrada (p+1 x 1)
            self.forward(x_sample) # Propagação para frente
            Y[:, k] = self.y[-1] # Saída da camada de saída (m)
        
        return Y
        

if (__name__ == "__main__"):
    m=1 # Número de neurônios na camada de saída
    mlp = MLPRegressor(L=2, Q=[150, 150], m=m, lr=0.1, n_epochs=1000, pr=1e-5) 
    
    X_train, X_test, y_train, y_test = r1.data_partition(m=m, train_size=0.8)
    
    # Treina a rede MLP com os dados de treinamento
    mlp.fit(X_train, y_train)
    
    # Testa a rede MLP com os dados de teste
    Y = mlp.predict(X_test)
    print(Y)
    

bp = 1