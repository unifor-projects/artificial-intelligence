import matplotlib.pyplot as plt
import matplotlib as mps
import numpy as np 
emg_s = np.loadtxt(fname='atividade_enzimatica.csv', delimiter=',')

X = emg_s[:, :1] # temperatura
Y = emg_s[:, 1:2] # pH
Z = emg_s[:, 2:3] # Atividade Enzimática

# Criação do Grafico de disperção
plt.figure(figsize=(12,5))


# Grafico 1: Temperatura vs Atividade Enzimática
plt.subplot(1,2,1)
plt.scatter(X,Z, color='blue', alpha=0.6)
plt.xlabel("Temperatura (°C)")
plt.ylabel("Atividade Enzimática")
plt.title("Temperatura vs Atividade Enzimática")

# O comportamento da atividade enzimática em relação à temperatura sugere um padrão não linear. 
# A atividade tende a aumentar até um certo ponto e depois decair, indicando um possível pico 
# de atividade em uma temperatura ideal. Esse comportamento é típico de reações enzimáticas, onde 
# o calor pode aumentar a eficiência da enzima até um limite, após o qual a desnaturação 
# ocorre e a atividade diminui.


# Grafico 2: pH vs Atividade Enzimática
plt.subplot(1, 2, 2)
plt.scatter(Y, Z, color='red', alpha=0.6)
plt.xlabel("pH")
plt.ylabel("Atividade Enzimática")
plt.title("pH vs Atividade Enzimática")

# De forma semelhante, a relação entre pH e atividade enzimática parece seguir um padrão curvilíneo, 
# onde a atividade pode ser maior dentro de uma faixa específica de pH e diminuir em valores muito 
# ácidos ou básicos. Isso ocorre porque cada enzima possui um pH ótimo, fora do qual sua estrutura 
# pode ser afetada, reduzindo sua eficiência.

#
# Para capturar corretamente esse padrão, um modelo preditivo eficientes deve possuir as seguintes características
# 1. Capacidade de modelar relações não lineares
#   - Modelos de regressão linear simples podem não ser adequados, pois a relação entre 
#   as variáveis não parece ser estritamente linear.
#   - Modelos polinomiais (exemplo: regressão polinomial de segundo ou terceiro grau) podem 
#   ser mais apropriados para capturar o formato curvo dos dados.
#  
# 2. Generalização sem overfitting
#   - O modelo deve capturar o comportamento real da atividade enzimática sem superajustar 
#   aos dados específicos do conjunto de treino.
#   - Isso pode ser feito ajustando o grau do polinômio ou utilizando regularização 
#   para evitar oscilações excessivas na curva predita.
#   
# 3. Interpretabilidade
#   - O modelo escolhido deve permitir interpretar a influência de temperatura e pH na atividade enzimática, 
#   ajudando a identificar os valores ótimos para maximizar a eficiência enzimática.
#


# * Overfitting é quando o modelo "decora" os dados de treinamento em vez de aprender a generalizar. 
#   Para evitá-lo, é importante controlar a complexidade do modelo, seja ajustando o grau do polinômio 
#   ou utilizando técnicas de regularização. O objetivo é garantir que o modelo funcione bem tanto nos dados 
#   de treinamento quanto em dados novos.

plt.tight_layout()
plt.show()

# Matriz de variáveis regressoras (Temperatura e pH)
X = np.column_stack((X, Y)) # Dimensão (N x p), p = 2

# Vetor da variável dependente (Atividade Enzimática)
y = Z.reshape(-1, 1) # Dimensão (N x 1)

print(X.shape, y.shape)


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
lambda_ridge = 1.0

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



print("Coeficientes do MQO tradicional:", y_pred_linear)




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

