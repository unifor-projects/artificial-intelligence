import matplotlib.pyplot as plt
import E1C3_PSM as c3
import E1C1 as c1
import numpy as np


def plot_model_results(X_train, y_train, X_test, y_pred, model, title_prefix):
    """
    Plota os resultados do modelo MLP para classificação 3D, incluindo o gráfico de dispersão 3D
    e a curva de aprendizado.

    Args:
        X_train: Dados de treino (features)
        y_train: Dados de treino (target)
        X_test: Dados de teste (features)
        y_pred: Predições do modelo
        model: Modelo MLP treinado
        title_prefix: Prefixo para os títulos dos gráficos
    """
    fig = plt.figure(figsize=(15, 5))

    # Criando subplots: um para visualização 3D e outro para curva de aprendizado
    ax1 = fig.add_subplot(121, projection="3d")
    ax2 = fig.add_subplot(122)

    # Removendo BIAS de X_train e X_test
    X_train_no_bias = X_train[1:, :].T  # Transpondo para ter formato correto
    X_test_no_bias = X_test[1:, :].T  # Transpondo para ter formato correto

    # ========================= Gráfico de dispersão 3D =========================
    # Plotando dados de treino
    cores1 = ["purple" if label == 1 else "cyan" for label in y_train[0]]
    ax1.scatter(
        X_train_no_bias[:, 0],  # Primeira coluna após transposição
        X_train_no_bias[:, 1],  # Segunda coluna após transposição
        X_train_no_bias[:, 2],  # Terceira coluna após transposição
        c=cores1,
        alpha=0.1,
        label="Dados de treino",
    )

    # Plotando dados de teste
    cores2 = ["red" if label == 1 else "blue" for label in y_pred[0]]
    ax1.scatter(
        X_test_no_bias[:, 0],  # Primeira coluna após transposição
        X_test_no_bias[:, 1],  # Segunda coluna após transposição
        X_test_no_bias[:, 2],  # Terceira coluna após transposição
        c=cores2,
        marker="^",
        alpha=0.8,
        label="Predições de teste",
    )

    ax1.set_title(f"Modelo {title_prefix}")
    ax1.set_xlabel("X1")
    ax1.set_ylabel("X2")
    ax1.set_zlabel("X3")
    ax1.legend()

    # ========================= Curva de aprendizado =========================
    ax2.plot(model.loss_history, label="Treino")
    ax2.set_title(f"Curva de Aprendizado - Modelo {title_prefix}")
    ax2.set_xlabel("Época")
    ax2.set_ylabel("MSE (Erro Quadrático Médio)")
    ax2.legend()

    plt.tight_layout()
    plt.show()

    print("teste plot")


X_train, X_test, y_train, y_test = c1.data_partition(m=1, train_size=0.8)

# =========================================================================
# ========= Modelo muito simples (subdimensionado - underfitting) =========
# =========================================================================
# generate Q for L = 100 using random numbers between 50 and 200
underfit_model = c3.MLP(L=1, Q=[5], m=1, lr=0.005, n_epochs=300, pr=1e-1)
underfit_model.fit(X_train, y_train)
underfit_pred = underfit_model.predict(X_test)

# Plotando resultados do modelo subdimensionado
plot_model_results(
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_pred=underfit_pred,
    model=underfit_model,
    title_prefix="Subdimensionado (Underfitting)",
)


# =========================================================================
# ======== Modelo muito complexo (superdimensionado - overfitting) ========
# =========================================================================
overfit_model = c3.MLP(
    L=6, Q=[300, 250, 200, 200, 250, 300], m=1, lr=0.01, n_epochs=1500, pr=1e-5
)
overfit_model.fit(X_train, y_train)
overfit_pred = overfit_model.predict(X_test)

# Plotando resultados do modelo superdimensionado
plot_model_results(
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_pred=overfit_pred,
    model=overfit_model,
    title_prefix="Superdimensionado (Overfitting)",
)


# =========================================================================
# ================== Modelo bem dimensionado (adequado) ===================
# =========================================================================
adequate_model = c3.MLP(L=2, Q=[30, 20], m=1, lr=0.005, n_epochs=1000, pr=1e-2)
adequate_model.fit(X_train, y_train)
adequate_pred = adequate_model.predict(X_test)

# Plotando resultados do modelo adequado
plot_model_results(
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_pred=adequate_pred,
    model=adequate_model,
    title_prefix="Adequado",
)


bp = 1
