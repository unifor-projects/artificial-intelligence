# Etapa 1 Regressão 2 (E1R2)

# 2. Faça uma visualização inicial dos dados através do gráfico de espalhamento. 

import matplotlib.pyplot as plt
import E1R1 as r1


# ======================================================
# ========= Criando o gráfico de espalhamento: =========
# ======================================================
plt.figure(figsize=(10, 5))
plt.scatter(r1.X[1:2, :], r1.y, color='blue', s=10, alpha=0.5)
plt.title('Gráfico de Espalhamento: Velocidade do vento x Potência gerada')
plt.xlabel('Velocidade do vento (m/s)')
plt.ylabel('Potência gerada (kW)')
plt.grid(True)
plt.show()



bp = 1