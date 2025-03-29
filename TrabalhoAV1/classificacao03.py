import classificacao01 as c1, classificacao02 as c2

# -------------------------------- Obs: --------------------------------
# Apesar desse arquivo se referir ao terceiro passo, devida organização 
# do trabalho, estarei adicionando algumas coisas de outras questões, 
# que achei interessante ja colocar aqui, porém será replicado nos 
# outros arquivos de suas respectivas questões.
# ---------------------------------------------------------------------

np, plt = c1.np, c1.plt

X = c2.X # Formato (N x p)
y = c2.y # Classes (1 a 5)

# Normalização (importante para MQO e modelos Gaussianos)
mu = np.mean(X, axis=0)
X = (X - mu) / np.std(X, axis=0)

# One-hot encoding para Y (MQO)
Y_onehot = c1.Y_mqo

# Embaralhamento para divisão treino/teste
np.random.seed(42)
indices = np.random.permutation(X.shape[0])
train_size = int(0.8 * X.shape[0])

# Divisão treino/teste para Y (MQO)
X_train, y_train = X[indices[:train_size]], y[indices[:train_size]]
X_test, y_test = X[indices[train_size:]], y[indices[train_size:]]
Y_onehot_train = Y_onehot[indices[:train_size]]


# ------------------------------------------------------------------------------
# ================== MQO Tradicional (Regressão Multinomial) ===================
# ------------------------------------------------------------------------------
class MQOMultinomial:
    def fit(self, X, Y):
        self.W = np.linalg.inv(X.T @ X) @ X.T @ Y  # Solução fechada: W = (X^T X)^-1 X^T Y

    def predict(self, X):
        scores = X @ self.W
        return np.argmax(scores, axis=1) + 1 # +1 para corrigir o zero-index (Classes 0-1)
    



# ------------------------------------------------------------------------------
# ================= Classificador Gaussiano Tradicional (GDA) ==================
# ------------------------------------------------------------------------------
class GDA:
    def fit (self, X, y):
        self.classes = np.unique(y) # separa em um vetor unico de classes y
        self.means = {}
        self.covs = {}
        self.priors = {}
        
        for c in self.classes:
            X_c = X[y == c] # seleciona as amostras(X) pertencentes a classe (c) = X_C
            self.means[c] = np.mean(X_c, axis=0)
            self.covs[c] = np.cov(X_c.T) # Matriz de covariancia para a classe c
            self.priors[c] = len(X_c) / len(X) # probabilidade da classe c
            
    def predict(self, X):
        posteriors = []
        for c in self.classes:
            diff = X - self.means[c]
            if (np.linalg.det(self.covs[c]) == 0):
                posteriors.append(np.zeros(X.shape[0]))
                print()
                print(" ============================================== ")
                print("Matriz de covariancia singular para a classe", c)
                print(" ============================================== ")
                print()
                continue
            inv_cov = np.linalg.inv(self.covs[c])
            exponent = -0.5 * np.sum(diff @ inv_cov * diff, axis=1)
            norm = 1 / np.sqrt((2 * np.pi)**2 * np.linalg.det(self.covs[c]))
            likelihood = norm * np.exp(exponent)
            posteriors.append(likelihood * self.priors[c])
            
        return self.classes[np.argmax(np.array(posteriors), axis=0)]


# ------------------------------------------------------------------------------
# =========== Classificador Gaussiano com Covariâncias Iguais (LDA) ============
# ------------------------------------------------------------------------------
class LDA:
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.means = {}
        self.priors = {}
        self.shared_cov = np.zeros((X.shape[1], X.shape[1]))
        
        for c in self.classes:
            X_c = X[y == c]
            self.means[c] = np.mean(X_c, axis=0)
            self.priors[c] = len(X_c) / len(X)
            self.shared_cov += (len(X_c) - 1) * np.cov(X_c.T)

        self.shared_cov /= (len(X) - len(self.classes))  # Média ponderada

    def predict(self, X):
        posteriors = []
        inv_cov = np.linalg.inv(self.shared_cov)
        for c in self.classes:
            diff = X - self.means[c]
            exponent = -0.5 * np.sum(diff @ inv_cov * diff, axis=1)
            prior = np.log(self.priors[c])
            posteriors.append(prior + exponent)
        
        return self.classes[np.argmax(np.array(posteriors), axis=0)]


# ------------------------------------------------------------------------------
# ======== Classificador Gaussiano com Matriz Agregada (GDA com MA) ============
# ------------------------------------------------------------------------------
class GDAAggregated:
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.means = {}
        self.priors = {}
        covs = []

        for c in self.classes:
            X_c = X[y == c]
            self.means[c] = np.mean(X_c, axis=0)
            self.priors[c] = len(X_c) / len(X)
            covs.append(np.cov(X_c.T))
            
        self.aggregated_cov = np.mean(covs, axis=0)
    
    def predict(self, X):
        posteriors = []
        inv_cov = np.linalg.inv(self.aggregated_cov)
        for c in self.classes:
            diff = X - self.means[c]
            exponent = -0.5 * np.sum(diff @ inv_cov * diff, axis=1)
            prior = np.log(self.priors[c])
            posteriors.append(prior + exponent)
        
        return self.classes[np.argmax(np.array(posteriors), axis=0)]

# ------------------------------------------------------------------------------
# ============== Classificador Gaussiano Regularizado (Friedman) ===============
# ------------------------------------------------------------------------------
class GDARegularized:
    def __init__(self, lambda_=0.5):
        self.lambda_ = lambda_
    
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.means = {}
        self.priors = {}
        self.covs = {}
        global_cov = np.cov(X.T)
        
        for c in self.classes:
            X_c = X[y == c]
            self.means[c] = np.mean(X_c, axis=0)
            self.priors[c] = len(X_c) / len(X)
            self.covs[c] = (1 - self.lambda_) * np.cov(X_c.T) + self.lambda_ * global_cov
        
    def predict(self, X):
        posteriors = []
        for c in self.classes:
            diff = X - self.means[c]
            inv_cov = np.linalg.inv(self.covs[c])
            exponent = -0.5 * np.sum(diff @ inv_cov * diff, axis=1)
            norm = 1 / np.sqrt((2 * np.pi)**2 * np.linalg.det(self.covs[c]))
            posteriors.append(norm * np.exp(exponent) * self.priors[c])
            
        return self.classes[np.argmax(np.array(posteriors), axis=0)]
    
    
# ------------------------------------------------------------------------------
# ================ Classificador de Bayes Ingênuo (Naive Bayes) ================
# ------------------------------------------------------------------------------
class NaiveBayes:
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.means = {}
        self.vars = {}
        self.priors = {}
        
        for c in self.classes:
            X_c = X[y == c]
            self.means[c] = np.mean(X_c, axis=0)
            self.vars[c] = np.var(X_c, axis=0)
            self.priors[c] = len(X_c) / len(X)
            
    def predict(self, X):
        posteriors = []
        for c in self.classes:
            diff = X - self.means[c]
            exponent = -0.5 * np.sum(diff ** 2 / self.vars[c], axis=1)
            norm = 1 / np.sqrt((2 * np.pi) * np.prod(self.vars[c]))
            posteriors.append(norm * np.exp(exponent) * self.priors[c])
            
        return self.classes[np.argmax(np.array(posteriors), axis=0)]


# ------------------------------------------------------------------------------
# =========================== Avaliação de Modelos =============================
# ------------------------------------------------------------------------------
def evaluate_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = np.mean(y_pred == y_test)
    print(f"Acurácia: {accuracy:.2%}")
    
    # Matriz de confusão
    cm = np.zeros((5, 5), dtype=int)
    for true, pred in zip(y_test, y_pred):
        cm[true-1, pred-1] += 1
    
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, cmap='Blues')
    plt.title("Matriz de Confusão: " + model.__class__.__name__)
    plt.colorbar()
    plt.xticks(np.arange(5), ['1', '2', '3', '4', '5'])
    plt.yticks(np.arange(5), ['1', '2', '3', '4', '5'])
    plt.xlabel("Predito")
    plt.ylabel("Verdadeiro")
    # plt.show() # Descomentar para exibir a matriz de confusão para cada modelo


print()
print("=== MQO ===")
evaluate_model(MQOMultinomial(), X_train, Y_onehot_train, X_test, y_test)
print()
print("=== GDA ===")
evaluate_model(GDA(), X_train, y_train, X_test, y_test)
print()
print("=== LDA ===")
evaluate_model(LDA(), X_train, y_train, X_test, y_test)
print()
print("=== GDA com MA ===")
evaluate_model(GDAAggregated(), X_train, y_train, X_test, y_test)
print()
print("=== GDA Regularizado ===")
evaluate_model(GDARegularized(), X_train, y_train, X_test, y_test)
print()
print("=== Naive Bayes ===")
evaluate_model(NaiveBayes(), X_train, y_train, X_test, y_test)

plt.show()

bp = 1