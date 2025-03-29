from regressao01 import np, plt
from regressao02 import y
from regressao03 import X_b, theta

# valores lambdas para serem testados
lambdas = [0, 0.25, 0.5, 0.75, 1]

# lista para armazenar os coeficiêntes betas para cada lambda
beta_list = []

# Repaproveitando variavel X_b (Adicionando uma coluna de 1s para o termo de bias (intercepto))
X_b = X_b

# Calcula o MQO regularizado (Ridge)
def calc_ridge_regression(lambda_val, X_b, y):
    I = np.eye(X_b.shape[1])  # Matriz identidade
    I[0, 0] = 0  # Não regularizamos o intercepto (não aplicamos a regularização ao termo de intercepto (β_0))
    
    # Calculando os coeficientes usando a equação normal com regularização Ridge
    beta = np.linalg.inv(X_b.T.dot(X_b) + lambda_val * I).dot(X_b.T).dot(y)
    return beta

# Loop para calcular o coeficiênte para cada lambda
for lambda_val in lambdas:
    
    # Adicionando valor de beta a lista
    beta = calc_ridge_regression(lambda_val, X_b, y)
    beta_list.append(beta)
    
    # Exibindo os coeficientes para o valor atual de lambda
    print(f"Coeficientes beta para lambda = {lambda_val}:")
    print(beta)
    print()
    
    
    
    
# Plotando os coeficientes para cada lambda
plt.figure(figsize=(10, 6))

for i, lambda_val in enumerate(lambdas):  # Incluindo lambda = 0 (MQO tradicional)
    plt.plot(beta_list[i], label=f"lambda = {lambda_val}")

plt.xlabel("Índice do Coeficiente")
plt.ylabel("Valor do Coeficiente")
plt.title("Coeficientes Beta para Diferentes Valores de Lambda")
plt.legend()
plt.grid(True)
plt.show()