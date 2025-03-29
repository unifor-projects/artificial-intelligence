import numpy as np
import matplotlib.pyplot as plt


atvd_eximatica = np.loadtxt(fname='TrabalhoAV1/dados/atividade_enzimatica.csv', delimiter=',') # Regreção Linear

X = atvd_eximatica[:, :1] # temperatura
Y = atvd_eximatica[:, 1:2] # pH
Z = atvd_eximatica[:, 2:3] # Atividade Enzimática

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

bp=1
