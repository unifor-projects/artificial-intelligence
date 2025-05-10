import matplotlib.pyplot as plt

from E1C1 import X_normalizado, y

# ================================================
# =========== Plotar o Gráfico: ==================
# ================================================
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection="3d")

cores = ["purple" if label == 1 else "cyan" for label in y]

ax.scatter(X_normalizado[:, 0], X_normalizado[:, 1], X_normalizado[:, 2], c=cores, s=20)
ax.set_title("Visualização Inicial - Dados Normalizados (Espalhamento 3D)")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
plt.show()
