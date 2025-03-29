from regressao01 import plt, np
from regressao02 import X, y

# MQO tradicional (Least Square Method)

# Adicionando uma coluna de 1s para o termo de bias (intercepto)
X_b = np.c_[np.ones((X.shape[0], 1)), X]

# Calculando os coeficientes usando a equação normal
theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

# Calculando previsões
y_pred_linear = X_b.dot(theta)





# MQO Regularizado (Ridge Regression - Tikhonov)
# Utilizado para evitar problemas de sobreajuste (overfitting) e colinearidade entre as variáveis regressoras.
# Formula para os coeficiêntes: θ=(X^T . X + λI)^(−1) . X^T . y

# Definidindo parametro de regularização lambda
lambda_ridge = 0.5

# Repaproveitando variavel X_b (Adicionando uma coluna de 1s para o termo de bias (intercepto))
X_b = X_b

# Calculando os coeficientes usando a equação normal com regularização Ridge
I = np.eye(X_b.shape[1])  # Matriz identidade
I[0, 0] = 0  # Não regularizamos o intercepto (não aplicamos a regularização ao termo de intercepto (β_0))
theta_ridge = np.linalg.inv(X_b.T.dot(X_b) + lambda_ridge * I).dot(X_b.T).dot(y)

# Calculando previsões:
y_pred_ridge = X_b.dot(theta_ridge)




# Média dos Valores Observados
# calculando a média dos valores observados
y_mean = np.mean(y)

# Fazendo previsões (todos os valores previstos são iguais à média)
y_pred_mean = np.full_like(y, y_mean)





# Plotando os resultados
plt.figure(figsize=(12, 5))

# Loop para criar os 8 gráficos
for i in range(2):
    for j in range(4):
        # Calculando o índice do subplot
        idx = i * 4 + j + 1

        # Plotando
        plt.subplot(2, 4, idx)

        if idx == 1:
            plt.scatter(X[:, 0], y, color='blue', alpha=0.6, label='Dados reais')
        elif idx == 2:
            plt.plot(X[:, 0], y_pred_linear, color='red', label='MQO Tradicional')
        elif idx == 3:
            plt.plot(X[:, 0], y_pred_ridge, color='green', label='MQO Regularizado (Ridge)')
        elif idx == 4:
            plt.plot(X[:, 0], y_pred_mean, color='purple', label='Média dos Valores Observados')
        elif idx == 5:
            plt.scatter(X[:, 1], y, color='red', alpha=0.6, label='Dados reais')
        elif idx == 6:
            plt.plot(X[:, 1], y_pred_linear, color='blue', label='MQO Tradicional')
        elif idx == 7:
            plt.plot(X[:, 1], y_pred_ridge, color='green', label='MQO Regularizado (Ridge)')
        elif idx == 8:
            plt.plot(X[:, 1], y_pred_mean, color='purple', label='Média dos Valores Observados')

        plt.xlabel("Temperatura (°C)" if idx <= 4 else "pH")
        plt.ylabel("Atividade Enzimática")
        plt.title("Temperatura vs Atividade Enzimática" if idx <= 4 else "pH vs Atividade Enzimática")
        plt.legend()

plt.tight_layout()
plt.show()

bp=1