import pandas as pd
from regressao01 import plt, np
from regressao04 import beta_list
from regressao05 import (
    rss_ridge_lambda0, 
    rss_ridge_lambda025, 
    rss_ridge_lambda05, 
    rss_ridge_lambda075, 
    rss_ridge_lambda1, 
    rss_mean
)

rss_data = {
    "Ridge (λ=0) | MQO Tradicional": rss_ridge_lambda0,
    "Ridge (λ=0.25)": rss_ridge_lambda025,
    "Ridge (λ=0.5)": rss_ridge_lambda05,
    "Ridge (λ=0.75)": rss_ridge_lambda075,
    "Ridge (λ=1)": rss_ridge_lambda1,
    "Média Observada": rss_mean
}

results = []
for model_name, rss_list in rss_data.items():
    mean_rss = np.mean(rss_list)  # Média aritmética
    std_rss = np.std(rss_list)    # Desvio padrão
    max_rss = np.max(rss_list)    # Maior valor
    min_rss = np.min(rss_list)    # Menor valor
    results.append([model_name, mean_rss, std_rss, max_rss, min_rss])


# Criando um DataFrame para exibir os resultados
results_df = pd.DataFrame(results, columns=["Modelo", "Média RSS", "Desvio Padrão RSS", "Maior RSS", "Menor RSS"])

# Exibindo a tabela
print(results_df)


# teste de previsão usando o modelo "Ridge (λ=1)"
temperatura = 4.32
pH = 3.4

X_novo = np.array([[1, temperatura, pH]])
beta_ridge = beta_list[-1]

y_predito = X_novo.dot(beta_ridge)

print(f"Atividade Enzimática Prevista: {y_predito[0][0]:.2f}")
print()