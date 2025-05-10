# Etapa 1 Regressão 4 (E1R4)

# Para o modelo MLP, pede-se que identifique o underfitting e overfitting. Neste caso, sua equipe deve
# explorar diferentes topologias da rede, ao modificar os hiperparâmetros associados à quantidade de
# neurônios e camadas (realizando um subdimensionamento e superdimensionamento). Faça a composição
# do resultado ao descrever as topologias encontradas e exiba a curva de aprendizado para os dois casos
# (superdimensionado e subdimensionado).

import E1R1 as r1
import E1R2 as r2
import E1R3 as r3
import matplotlib.pyplot as plt

def plot_model_results(X_train, y_train, X_test, y_pred, model, title_prefix):
    """
    Plota os resultados do modelo MLP, incluindo o gráfico de dispersão e a curva de aprendizado.
    
    Args:
        X_train: Dados de treino (features)
        y_train: Dados de treino (target)
        X_test: Dados de teste (features) 
        y_pred: Predições do modelo
        model: Modelo MLP treinado
        title_prefix: Prefixo para os títulos dos gráficos (ex: "Subdimensionado", "Superdimensionado")
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Removendo BIAS de X_train e X_test
    X_train = X_train[1:, :]
    X_test = X_test[1:, :]
    
    # ========================= Gráfico de dispersão =========================
    ax1.scatter(X_train, y_train, alpha=0.3, label='Dados reais')
    ax1.scatter(X_test, y_pred, label='Predição teste', color='red')
    ax1.set_title(f'Modelo {title_prefix}')
    ax1.set_xlabel('Velocidade do vento (normalizada)')
    ax1.set_ylabel('Potência gerada (normalizada)')
    ax1.legend()
    
    
    # ========================= Curva de aprendizado =========================
    ax2.plot(model.loss_history, label='Treino')
    ax2.set_title(f'Curva de Aprendizado - Modelo {title_prefix}')
    ax2.set_xlabel('Época')
    ax2.set_ylabel('MSE (Erro Quadrático Médio)')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

X_train, X_test, y_train, y_test = r1.data_partition(m=1, train_size=0.8)


# =========================================================================
# ========= Modelo muito simples (subdimensionado - underfitting) =========
# =========================================================================
underfit_model = r3.MLPRegressor(L=2, Q=[15, 30], m=1, lr=0.01 , n_epochs=100, pr=1e-3)
underfit_model.fit(X_train, y_train)
underfit_pred = underfit_model.predict(X_test)

# Plotando resultados do modelo subdimensionado
plot_model_results(
    X_train=X_train,
    y_train=y_train, 
    X_test=X_test, 
    y_pred=underfit_pred, 
    model=underfit_model, 
    title_prefix="Subdimensionado (Underfitting)"
)


# =========================================================================
# ======== Modelo muito complexo (superdimensionado - overfitting) ========
# =========================================================================
overfit_model = r3.MLPRegressor(L=5, Q=[100,150,200,150,100], m=1, lr=0.01 , n_epochs=100, pr=1e-3)
overfit_model.fit(X_train, y_train)
overfit_pred = overfit_model.predict(X_test)

# Plotando resultados do modelo superdimensionado
plot_model_results(
    X_train=X_train,
    y_train=y_train, 
    X_test=X_test, 
    y_pred=overfit_pred, 
    model=overfit_model, 
    title_prefix="Superdimensionado (Overfitting)"
)


# =========================================================================
# ================== Modelo bem dimensionado (adequado) ===================
# =========================================================================
adequate_model = r3.MLPRegressor(L=3, Q=[100,150,100], m=1, lr=0.01 , n_epochs=100, pr=1e-3)
adequate_model.fit(X_train, y_train)
adequate_pred = adequate_model.predict(X_test)

# Plotando resultados do modelo adequado
plot_model_results(
    X_train=X_train,
    y_train=y_train, 
    X_test=X_test, 
    y_pred=adequate_pred,
    model=adequate_model, 
    title_prefix="Adequado"
)


bp = 1