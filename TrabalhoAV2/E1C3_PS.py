import matplotlib.pyplot as plt
import numpy as np
from E1C1 import X_normalizado, y

# ================================================
# ========= Perceptron Simples (Genérico) ========
# ================================================

# Convertendo os rótulos para -1 e 1
y_binario = np.where(y == 1, 1, -1)

# Hiperparâmetros
taxa_aprendizado = 0.01
max_epocas = 1000

# Inicialização dos pesos e bias
n_amostras, n_caracteristicas = X_normalizado.shape
pesos = np.zeros(n_caracteristicas)
bias = 0

# Treinamento do Perceptron
for epoca in range(max_epocas):
    erros = 0
    for i in range(n_amostras):
        xi = X_normalizado[i]
        yi = y_binario[i]

        # Cálculo da ativação
        ativacao = np.dot(xi, pesos) + bias

        # Função de ativação (degrau)
        predicao = 1 if ativacao >= 0 else -1

        # Atualização se erro
        if predicao != yi:
            atualizacao = taxa_aprendizado * (yi - predicao)
            pesos += atualizacao * xi
            bias += atualizacao
            erros += 1

    # Critério de parada: convergência
    if erros == 0:
        print(f"Convergiu na época {epoca + 1}")
        break

print(f"Pesos finais: {pesos}")
print(f"Bias final: {bias}")

# Avaliação no conjunto de treino
ativacoes = np.dot(X_normalizado, pesos) + bias
predicoes = np.where(ativacoes >= 0, 1, -1)
acuracia = np.mean(predicoes == y_binario)

print(f"Acurácia no conjunto de treino: {acuracia * 100:.2f}%")

# ================================================
# ===== Representação Gráfica do Resultado =======
# ================================================

# Gerar os pontos do plano
xx, yy = np.meshgrid(
    np.linspace(X_normalizado[:, 0].min(), X_normalizado[:, 0].max(), 10),
    np.linspace(X_normalizado[:, 1].min(), X_normalizado[:, 1].max(), 10)
)

zz = -(pesos[0] * xx + pesos[1] * yy + bias) / pesos[2]

# Plot
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection="3d")

# Cores: verde para acertos, preto para erros
ativacoes = np.dot(X_normalizado, pesos) + bias
predicoes = np.where(ativacoes >= 0, 1, -1)
y_binario = np.where(y == 1, 1, -1)
cores = ["green" if p == t else "black" for p, t in zip(predicoes, y_binario)]

ax.scatter(X_normalizado[:, 0], X_normalizado[:, 1], X_normalizado[:, 2], c=cores, s=15)
ax.plot_surface(xx, yy, zz, alpha=0.3, color='gray', rstride=1, cstride=1, edgecolor='none')

ax.set_title("Plano de Decisão do Perceptron Simples")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

plt.show()


"""

    Considerações Finais:

    Acurácia = 50.90%
    
    Os dados fornecidos não são linearmente separáveis, logo a acurácia
    apontada indica que o Perceptron está quase o mesmo que um chute 
    aleatório, pois o Perceptron Simples não consegue aprender
    fronteiras de decisão não lineares, então mesmo com a aplicação
    correta do modelo ele é simples demais para o padrão complexo de 
    dados fornecido.

    Além disso, os dados são em formato espiral e o Perceptron tenta
    encontrar um hiperplano que separe os dados em duas classes, e 
    isso funciona apenas para dados linearmente separáveis.

    Outro ponto do resultado da acurácia seria que o Perceptron é como
    se estivesse "cortando" o espaço tridimensional de forma grosseira,
    logo ele irá acertar parte dos exemplos que por sorte caem do lado
    certo do plano e erra o restante que está do outro lado, mesmo que
    pertecem a mesma classe, por isso a classificação ficou próximo a 
    50%, demonstrando a aleatoriedade na aplicação problema.

"""