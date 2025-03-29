from regressao01 import np, plt
from regressao02 import X, y
from regressao04 import calc_ridge_regression, lambdas

# Número de rodadas de Monte Carlo
R = 500

# Listas para armazenar os RSS de cada modelo
rss_ridge_lambda0 = []  # Ridge com lambda = 0 (MQO Tradicional)
rss_ridge_lambda025 = []  # Ridge com lambda = 0.25
rss_ridge_lambda05 = []  # Ridge com lambda = 0.5
rss_ridge_lambda075 = []  # Ridge com lambda = 0.75
rss_ridge_lambda1 = []  # Ridge com lambda = 1
rss_mean = []  # Média dos Valores Observados

# Loop de Monte Carlo

for r in range(R):
    # Embaralhar os dados
    np.random.seed(r) # Definir uma semente para reprodutibilidade
    indices = np.random.permutation(len(X))
    X_shuffled = X[indices]
    y_shuffled = y[indices]
    
    
    # Particionar os dados em treino (80%) e teste (20%)
    split_idx = int(0.8 * len(X_shuffled))
    X_train, X_test = X_shuffled[:split_idx], X_shuffled[split_idx:]
    y_train, y_test = y_shuffled[:split_idx], y_shuffled[split_idx:]
    
    
    # Adicionar uma coluna de 1s para o intercepto
    X_train_b = np.c_[np.ones((X_train.shape[0], 1)), X_train]
    X_test_b = np.c_[np.ones((X_test.shape[0], 1)), X_test]
    
    # Loop para calculo dos RSS(Residual Sum of Squares)
    for i, lambda_val in enumerate(lambdas):
        # calculo do coeficiênte(βridge​) com a matriz de treinamento
        beta_ridge = calc_ridge_regression(lambda_val=lambda_val, X_b=X_train_b, y=y_train)
        
        # Calulo das previsões de teste, sendo testes o valor real.
        y_pred_ridge = X_test_b.dot(beta_ridge)
        
        # Calculo das somas dos quadrados dos residuos(RSS) para cada valor de lambda
        if lambda_val == 0:
            rss_ridge_lambda0.append(np.sum((y_test - y_pred_ridge) ** 2))
        elif lambda_val == 0.25:
            rss_ridge_lambda025.append(np.sum((y_test - y_pred_ridge) ** 2))
        elif lambda_val == 0.5:
            rss_ridge_lambda05.append(np.sum((y_test - y_pred_ridge) ** 2))
        elif lambda_val == 0.75:
            rss_ridge_lambda075.append(np.sum((y_test - y_pred_ridge) ** 2))
        elif lambda_val == 1:
            rss_ridge_lambda1.append(np.sum((y_test - y_pred_ridge) ** 2))
        
        
    # Agora fazer o calulo para a Média dos Valores Observados
    y_mean = np.mean(y_train)
    # predição para a média dos valores observados com a variável de teste
    y_pred_mean = np.full_like(y_test, y_mean)
    # Calculo do RSS para a média
    rss_mean.append(np.sum((y_test - y_pred_mean) ** 2))
    
    

# Exibindo os resultados médios de RSS para cada modelo
print("RSS Médio para cada modelo:")
print(f"Ridge (lambda = 0) | mesmo que MQO tradicional: {np.mean(rss_ridge_lambda0)}")
print(f"Ridge (lambda = 0.25): {np.mean(rss_ridge_lambda025)}")
print(f"Ridge (lambda = 0.5): {np.mean(rss_ridge_lambda05)}")
print(f"Ridge (lambda = 0.75): {np.mean(rss_ridge_lambda075)}")
print(f"Ridge (lambda = 1): {np.mean(rss_ridge_lambda1)}")
print(f"Média dos Valores Observados: {np.mean(rss_mean)}")


# Listas de RSS para cada modelo
rss_data = [rss_ridge_lambda0, rss_ridge_lambda025, rss_ridge_lambda05, rss_ridge_lambda075, rss_ridge_lambda1, rss_mean]
labels = ["MQO Tradicional | Ridge (λ=0)", "Ridge (λ=0.25)", "Ridge (λ=0.5)", "Ridge (λ=0.75)", "Ridge (λ=1)", "Média Observada"]

# Plotando o boxplot
plt.figure(figsize=(12, 6))
plt.boxplot(rss_data, labels=labels)
plt.xlabel("Modelos")
plt.ylabel("RSS")
plt.title("Distribuição do RSS para Diferentes Modelos")
plt.xticks(rotation=45)
plt.grid(True)
plt.show()

bp=1