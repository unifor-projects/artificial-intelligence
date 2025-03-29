from tqdm import tqdm
import classificacao01 as c1
import classificacao03 as c3

np, plt = c1.np, c1.plt

R = 500  # Número de rodadas de Monte Carlo
models = {
    "MQO Tradicional": c3.MQOMultinomial(), # MQO tradicional
    "GDA (Covariâncias diferentes)": c3.GDA(), # Classificador Gaussiano Tradicional
    "LDA (Covariâncias iguais)": c3.LDA(), # Classificador Gaussiano (Cov. de todo cj. treino)
    "GDA Matriz Agregada": c3.GDAAggregated(), # Classificador Gaussiano (Cov. Agregada)
    "Naive Bayes": c3.NaiveBayes(), # Classificador de Bayes Ingˆenuo (Naive Bayes Classifier)
    "GDA Regularizado 025": c3.GDARegularized(lambda_=0.25), # Classificador Gaussiano Regularizado (Friedman λ =0,25)
    "GDA Regularizado 050": c3.GDARegularized(lambda_=0.50), # Classificador Gaussiano Regularizado (Friedman λ =0,50)
    "GDA Regularizado 075": c3.GDARegularized(lambda_=0.75), # Classificador Gaussiano Regularizado (Friedman λ =0,75)
}

# Dicionário para armazenar as acuracias
accuracy_list = {name: [] for name in models.keys()}

for r in tqdm(range(R)):
    # Particionamento Aleatório dos Dados (80/20)
    np.random.seed(r)
    indices = np.random.permutation(c3.X.shape[0])
    train_size = int(0.8 * c3.X.shape[0])
    
    X_train, y_train = c3.X[indices[:train_size]], c3.y[indices[:train_size]]
    X_test, y_test = c3.X[indices[train_size:]], c3.y[indices[train_size:]]
    
    # # One-hot encoding para MQO (apenas nesta rodada)
    N = len(y_train)
    C = 5
    Y_train_onehot = np.zeros((N, 5))
    Y_train_onehot[np.arange(N), y_train-1] = 1
    
    # Avaliação para cada modelo
    for name, model in models.items():
        if name == "MQO Tradicional":
            model.fit(X_train, Y_train_onehot)
        else:
            model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy_list[name].append(np.mean(y_pred == y_test))


# Estatísticas finais
print("\n=== Acurácia Média e Desvio Padrão ===")
for name, acc_list in accuracy_list.items():
    print(f"{name}: {np.mean(acc_list):.2%} ± {np.std(acc_list):.2%}")


plt.figure(figsize=(10, 6))
plt.boxplot(accuracy_list.values(), labels=accuracy_list.keys())
plt.title("Distribuição das Acurácias (R = 500 rodadas)")
plt.ylabel("Acurácia")
plt.xticks(rotation=45)
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()


bp=1