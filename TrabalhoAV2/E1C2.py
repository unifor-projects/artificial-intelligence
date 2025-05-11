import matplotlib.pyplot as plt
from E1C1 import X, y

# Extrair apenas as features (ignorando a linha do bias)
X_vis = X[1:, :]  # Agora X_vis está em [3, N]

# Se y estiver com shape (1, N), aplaine:
if y.ndim > 1:
    y = y.flatten()

# Cores para cada classe
cores = ["purple" if label >= 0.5 else "cyan" for label in y]

# Plotar gráfico de dispersão 3D
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection="3d")

ax.scatter(X_vis[0], X_vis[1], X_vis[2], c=cores, s=20)
ax.set_title("Visualização Inicial - Dados Normalizados (Espalhamento 3D)")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
plt.tight_layout()
plt.show()
