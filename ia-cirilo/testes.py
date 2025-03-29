import matplotlib.pyplot as plt
import matplotlib as mps
import numpy as np 

#TAREFA DE REGRESSÃO
# questão 1

data = np.loadtxt('ia-cirilo/atividade_enzimatica.csv', delimiter=",")

temperature = data[:, 0]
ph = data[:, 1]
enzymatic_activities = data[:, 2]

plt.figure(figsize=(8, 5))
plt.scatter(temperature, enzymatic_activities, color="blue", alpha=0.6)
plt.xlabel("Temperatura")
plt.ylabel("Atividade Enzimática")
plt.title("Temperatura vs Atividade Enzimática")
plt.show()

plt.scatter(ph, enzymatic_activities, color="blue", alpha=0.6)
plt.xlabel("pH")
plt.ylabel("Atividade Enzimática")
plt.title("pH vs Atividade Enzimática")
plt.show()

"""
Nos gráficos de dispersão, observamos que a relação entre a temperatura e a atividade enzimática é não linear, 
tornando evidente que a atividade enzimática pode aumentar até uma certa temperatura e depois diminuir. Por outro 
lado,quando analisamos a relação entre pH e atividade enzimática é mais simples, mostrando um comportamento 
linear, onde a atividade aumenta conforme o pH sobe. Para essa relação, um modelo de regressão linear seria 
adequado.
"""

# questão 2
matrix_x = np.column_stack((temperature, ph))
vector_y = enzymatic_activities.reshape(-1, 1)

print(matrix_x)


print("Matriz X (variáveis regressoras):")
print(matrix_x)
print("Dimensão de X:", matrix_x.shape)  

print("\nVetor y (variável dependente):")
print(vector_y)
print("Dimensão de y:", vector_y.shape)  

# questão 3
X_intercept = np.column_stack((np.ones(matrix_x.shape[0]), matrix_x))

mqo_traditional = np.linalg.inv(X_intercept.T @ X_intercept) @ X_intercept.T @ vector_y

lambda_ = 1
I = np.eye(X_intercept.shape[1])  # Matriz identidade
I[0, 0] = 0  # Não regularizamos o intercepto (não aplicamos a regularização ao termo de intercepto (β_0))
mqo_regularized = np.linalg.inv(X_intercept.T @ X_intercept + lambda_ * I) @ X_intercept.T @ vector_y

media_y = np.mean(vector_y)

print("Coeficientes do MQO tradicional:", mqo_traditional)
print("Coeficientes do MQO regularizado:", mqo_regularized)
print("Média dos valores observáveis:", media_y)

# questão 4
lambdas = [0, 0.25, 0.5, 0.75, 1]
regularized_coefficients = {}

for lambda_count in lambdas: 
    mqo_regularized = np.linalg.inv(X_intercept.T @ X_intercept + lambda_count * np.eye(X_intercept.shape[1])) @ X_intercept.T @ vector_y
    regularized_coefficients[lambda_count] = mqo_regularized

for lambda_count, coef in regularized_coefficients.items():
    print(f"Coeficientes para lambda = {lambda_count}: {coef}")

# questão 5 
def calc_rss(y_true, y_pred):
    return np.sum((y_true - y_pred) ** 2)

r = 500

rss_traditional = []
rss_regularized_0 = [] 
rss_regularized_025 = []
rss_regularized_05 = []  
rss_regularized_075 = [] 
rss_regularized_1 = []  
rss_mean = []

for _ in range(r):
    n = len(matrix_x)
    indices = np.random.permutation(n)
    train_size = int(0.8 * n)
    
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    X_train = matrix_x[train_indices]
    y_train = vector_y[train_indices]
    X_test = matrix_x[test_indices]
    y_test = vector_y[test_indices]
    
    # Adicionar uma coluna de 1s para o intercepto
    X_train_b = np.c_[np.ones((X_train.shape[0], 1)), X_train]
    X_test_b = np.c_[np.ones((X_test.shape[0], 1)), X_test]

    mqo_traditional_question_5 = np.linalg.inv(X_train_b.T @ X_train_b) @ X_train_b.T @ y_train
    y_pred_traditional = X_test_b @ mqo_traditional_question_5
    rss_traditional.append(calc_rss(y_test, y_pred_traditional))

    mqo_regularized_0 = np.linalg.inv(X_train_b.T @ X_train_b + 0 * np.eye(X_train_b.shape[1])) @ X_train_b.T @ y_train
    y_pred_regularized_0 = X_test_b @ mqo_regularized_0
    rss_regularized_0.append(calc_rss(y_test, y_pred_regularized_0))
    
    # lambda = 0.25
    mqo_regularized_025 = np.linalg.inv(X_train_b.T @ X_train_b + 0.25 * np.eye(X_train_b.shape[1])) @ X_train_b.T @ y_train
    y_pred_regularized_025 = X_test_b @ mqo_regularized_025
    rss_regularized_025.append(calc_rss(y_test, y_pred_regularized_025))
    # lambda = 0.5
    mqo_regularized_05 = np.linalg.inv(X_train_b.T @ X_train_b + 0.5 * np.eye(X_train_b.shape[1])) @ X_train_b.T @ y_train
    y_pred_regularized_05 = X_test_b @ mqo_regularized_05
    rss_regularized_05.append(calc_rss(y_test, y_pred_regularized_05))
    # lambda = 0.75
    mqo_regularized_075 = np.linalg.inv(X_train_b.T @ X_train_b + 0.75 * np.eye(X_train_b.shape[1])) @ X_train_b.T @ y_train
    y_pred_regularized_075 = X_test_b @ mqo_regularized_075
    rss_regularized_075.append(calc_rss(y_test, y_pred_regularized_075))
    # lambda = 1
    mqo_regularized_1 = np.linalg.inv(X_train_b.T @ X_train_b + 1 * np.eye(X_train_b.shape[1])) @ X_train_b.T @ y_train
    y_pred_regularized_1 = X_test_b @ mqo_regularized_1
    rss_regularized_1.append(calc_rss(y_test, y_pred_regularized_1))

    y_pred_mean = np.full_like(y_test, np.mean(y_train))
    rss_mean.append(calc_rss(y_test, y_pred_mean))

print("RSS do MQO tradicional:", rss_traditional)
print("\nRSS do MQO regularizado (lambda = 0):", rss_regularized_0)
print("\nRSS do MQO regularizado (lambda = 0.25):", rss_regularized_025)
print("\nRSS do MQO regularizado (lambda = 0.5):", rss_regularized_05)
print("\nRSS do MQO regularizado (lambda = 0.75):", rss_regularized_075)
print("\nRSS do MQO regularizado (lambda = 1):", rss_regularized_1)
print("\nRSS da média dos valores observáveis:", rss_mean)

# questão 6
def calc_statistics(rss_list):
    return {
        'Média': np.mean(rss_list),
        'Desvio-Padrão': np.std(rss_list),
        'Valor Máximo': np.max(rss_list),
        'Valor Mínimo': np.min(rss_list)
    }


stats_traditional = calc_statistics(rss_traditional)

stats_regularized_0 = calc_statistics(rss_regularized_0)

stats_regularized_025 = calc_statistics(rss_regularized_025)

stats_regularized_05 = calc_statistics(rss_regularized_05)

stats_regularized_075 = calc_statistics(rss_regularized_075)

stats_regularized_1 = calc_statistics(rss_regularized_1)

stats_mean = calc_statistics(rss_mean)

# Exibindo os resultados de forma organizada
print("Resultados dos RSS para cada modelo (média, desvio-padrão, valor máximo, valor mínimo):\n")

# Cabeçalho da tabela
print(f"{'Modelo':<40}{'Média':<15}{'Desvio-Padrão':<15}{'Valor Máximo':<15}{'Valor Mínimo':<15}")

# Exibindo as estatísticas para cada modelo
print(f"{'MQO Tradicional':<40}{stats_traditional['Média']:<15.4f}{stats_traditional['Desvio-Padrão']:<15.4f}{stats_traditional['Valor Máximo']:<15.4f}{stats_traditional['Valor Mínimo']:<15.4f}")
print(f"{'MQO Regularizado (λ=0)':<40}{stats_regularized_0['Média']:<15.4f}{stats_regularized_0['Desvio-Padrão']:<15.4f}{stats_regularized_0['Valor Máximo']:<15.4f}{stats_regularized_0['Valor Mínimo']:<15.4f}")
print(f"{'MQO Regularizado (λ=0.25)':<40}{stats_regularized_025['Média']:<15.4f}{stats_regularized_025['Desvio-Padrão']:<15.4f}{stats_regularized_025['Valor Máximo']:<15.4f}{stats_regularized_025['Valor Mínimo']:<15.4f}")
print(f"{'MQO Regularizado (λ=0.5)':<40}{stats_regularized_05['Média']:<15.4f}{stats_regularized_05['Desvio-Padrão']:<15.4f}{stats_regularized_05['Valor Máximo']:<15.4f}{stats_regularized_05['Valor Mínimo']:<15.4f}")
print(f"{'MQO Regularizado (λ=0.75)':<40}{stats_regularized_075['Média']:<15.4f}{stats_regularized_075['Desvio-Padrão']:<15.4f}{stats_regularized_075['Valor Máximo']:<15.4f}{stats_regularized_075['Valor Mínimo']:<15.4f}")
print(f"{'MQO Regularizado (λ=1)':<40}{stats_regularized_1['Média']:<15.4f}{stats_regularized_1['Desvio-Padrão']:<15.4f}{stats_regularized_1['Valor Máximo']:<15.4f}{stats_regularized_1['Valor Mínimo']:<15.4f}")
print(f"{'Média dos Valores Observáveis':<40}{stats_mean['Média']:<15.4f}{stats_mean['Desvio-Padrão']:<15.4f}{stats_mean['Valor Máximo']:<15.4f}{stats_mean['Valor Mínimo']:<15.4f}")


