import classificacao01 as c1
import classificacao03 as c3

np, plt = c1.np, c1.plt

# Definição dos lambdas a serem testados
lambdas = [0, 0.25, 0.5, 0.75, 1]

accuracy_list = []

for lambda_ in lambdas:
    # Usando GDARegularized definido na classificacao03 (Questão 3 do trabalho)
    model = c3.GDARegularized(lambda_=lambda_)
    model.fit(X=c3.X_train, y=c3.y_train)
    
    # Avaliação do modelo
    y_pred = model.predict(X=c3.X_test)
    accuracy = np.mean(y_pred == c3.y_test)
    accuracy_list.append(accuracy)
    
    # Exibição dos resultados
    print(f"Lambda: {lambda_:2f} | Acuracia: {accuracy:.2%}")
    cov = model.covs[1]  # Covariância da classe 1
    plt.figure(figsize=(6, 6))
    plt.imshow(cov, cmap='viridis')
    plt.title(f"Matriz de Covariância (Classe 1) para λ = {lambda_}")
    plt.colorbar()
    plt.show()

# Gráfico de acurácia vs lambda (Analise de desempenho por comparação)
plt.figure(figsize=(8, 4))
plt.plot(lambdas, accuracy_list, marker='o', linestyle='--')
plt.xlabel("Valor de λ")
plt.ylabel("Acurácia no Teste")
plt.title("Desempenho do Modelo vs Hiperparâmetro λ")
plt.grid(True)
plt.show()

bp=1