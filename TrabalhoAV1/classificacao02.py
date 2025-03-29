import classificacao01 as c1

np, plt = c1.np, c1.plt

X = c1.X_mqo # Formato (N x p) p = 2
y = c1.rotulos # Classes (1 a 5)

# --------------------------------------
# Mapeando classes para nomes e cores
# --------------------------------------
classes = {
    1: "Neutro",
    2: "Sorriso",
    3: "Sobrancelhas levantadas",
    4: "Surpreso",
    5: "Rabugento"
}
colors = ["blue", "green", "red", "purple", "orange"]


# --------------------------------------
# Criação do gráfico de dispersão
# --------------------------------------
plt.figure(figsize=(10, 8))

for class_id in np.unique(y):
    mask = y == class_id
    plt.scatter(
        X[mask, 0], X[mask, 1],
        label=classes[class_id],
        color=colors[class_id - 1],
        alpha=0.6,
        edgecolors='w'
    )

    
plt.title("Gráfico de Dispersão das Classes de Expressões Faciais")
plt.xlabel("Sensor 1 (Corrugador do Supercílio)")
plt.ylabel("Sensor 2 (Zigomático Maior)")
plt.legend()
plt.tight_layout()
plt.show()


# ------------------------------------------------------------------------------
# =============================== Observações ==================================
# ------------------------------------------------------------------------------
# Após a analise do gráfico, podemos concluir que as classes são parcialmentemente 
# linearmente separáveis. a classe "Sorriso" e "Rabugento" podem ser identificadas 
# por modelos de regressão linear, enquanto as demais exigem abordagens não-lineares 
# ou extração de features. 
# 
# Algumas recomendações: 
# 1. Pré-processamento: 
#    - Normalizar dados e adicionar features estatísticas 
#    (ex.: desvio padrão em janelas temporais). 
#
# 2. MGB (Modelos Gaussianos Bayesianos):
#    - este modelo pode ser uma boa escolha, especialmente devido à natureza contínua 
#    dos sinais dos sensores e à possível distribuição normal dos dados. 
#
# 3. Outros:
#    - Outras abordagens, como classificação com KNN, podem ser vantajosas 
#    em alguns casos. (Ainda não estudado na disciplina)
# ------------------------------------------------------------------------------
# ==============================================================================
# ------------------------------------------------------------------------------


bp=1