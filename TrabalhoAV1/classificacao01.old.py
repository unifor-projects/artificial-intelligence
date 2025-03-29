import numpy as np
import matplotlib.pyplot as plt

dados = np.loadtxt("TrabalhoAV1/dados/EMGsDataset.csv", delimiter=",")


# Questão 1
X = dados[:2, :].T
Y_labels = dados[2, :].astype(int)

# Número de classes C e amostras N
N = X.shape[0]
C = 5  # Sabemos que há 5 classes

Y_mqo = np.zeros((N, C))
Y_mqo[np.arange(N), Y_labels - 1] = 1  # Ajuste para índice zero

# Criar as versões transpostas
X_bayes = X.T  # (p, N)
Y_bayes = Y_mqo.T  # (C, N)

# Exibir dimensões finais
print("X (MQO):", X.shape)  # (N, p)
print("Y (MQO):", Y_mqo.shape)  # (N, C)
print("X (Bayes):", X_bayes.shape)  # (p, N)
print("Y (Bayes):", Y_bayes.shape)  # (C, N)

# 2ª Questão

""" 
    Análise do Gráfico:
    
    Com base na distribuição dos pontos, podemos levantar algumas hipóteses sobre
    a separabilidade das classes:
    
    1. Linearmente Separáveis ou não?
    
        Algumas classes aparentam estar bem separadas espacialmente. Por exemplo
        a classe 'Sorriso' (verde) está localizada em uma região isolada do 
        gráfico.
        
        As classes 'Surpreso' e 'Rabugento', possuem uma distribuição mais ampla
        e podem apresentar sobrepoisção, indicando que não são facilmente separáveis
        por um modelo linear.
        
        A classe 'Neutro' e 'Sobrancelhas levantadas' estão muito próximas ao eixo
        inferior do gráfico, possivelmente indicando menor variação em uma das dimensões,
        o que pode dificultar a separação.
    
    2. Características esperadas de um modelo eficaz

        Um modelo linear simples pode ser suficiente para distinguir algumas classes,
        como 'Sorriso'.
        
        Para outras classes que possuem sobreposição, pode ser necessário um modelo mais
        complexo, como uma rede neural ou um classificador baseado em kernels.
        
        A escolha de sensores X e Z pode impactar a eficácia do modelo. Caso a separação
        ainda não seja clara, pode ser útil adicionar mais variáveis ou transformar os
        dados.

"""


X1 = dados[0, :]  # Sensor 1 (Corrugador do Supercílio)
X2 = dados[1, :]  # Sensor 2 (Zigomático Maior)
Y = dados[2, :].astype(int)  # Classes (1 a 5)

# Definir cores para cada classe
cores = ["blue", "green", "red", "purple", "orange"]
labels = ["Neutro", "Sorriso", "Sobrancelhas levantadas", "Surpreso", "Rabugento"]

# Criar figura e eixo 3D
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection="3d")

# Criar gráfico de dispersão 3D
for classe in range(1, 6):
    mask = Y == classe  # Filtra amostras da classe específica
    ax.scatter(
        X1[mask],
        X2[mask],
        Y[mask],
        label=labels[classe - 1],
        color=cores[classe - 1],
        alpha=0.6,
    )

# Configurar rótulos dos eixos
ax.set_xlabel("Sensor 1 (Corrugador do Supercílio)")
ax.set_ylabel("Sensor 2 (Zigomático Maior)")
ax.set_zlabel("Classe da Expressão")
ax.set_title("Gráfico de Dispersão 3D das Classes de Expressões Faciais")

# Exibir legenda e mostrar gráfico
ax.legend()
plt.show()