# Etapa 1 Regressão 5 (E1R5)

# Para validar os modelos (com suas topologias definidas) utilizados na tarefa, sua equipe deve projetar
# a validação utilizando as simulações por Monte Carlo. Nessa etapa, defina a quantidade de rodadas da
# simulação igual a R=250. Em cada rodada, deve-se realizar o particionamento em 80% dos dados
# para treinamento e 20% para teste. A medida de desempenho utilizada é a média de desvios quadráticos
# (MSE). Para produzir um resultado em tabela, calcule a média, desvio-padrão, maior valor e menor valor
# de MSE para as 250 rodadas.

from tabulate import tabulate
import tqdm
import E1R1 as r1
# import E1R2 as r2
import E1R3 as r3
# import E1R4 as r4
import numpy as np

def monte_carlo(R, train_size, model):
    """
    Realiza o particionamento dos dados em 80% para treinamento e 20% para teste,
    e calcula a média, desvio-padrão, maior valor e menor valor de MSE para as R rodadas.
    
    Args:
        R (int): Número de rodadas da simulação
        train_size (float): Percentual de dados para treinamento
        model (object): Modelo a ser utilizado
        
    Returns:
        object: {
            "mean": float,
            "std": float,
            "max": float,
            "min": float,
            "all": list
        }
    """
    
    all_MSE = []
    
    for _ in tqdm.tqdm(range(R), desc="Executando Monte Carlo"):
        # Realiza o particionamento dos dados em 80% para treinamento e 20% para teste
        X_train, X_test, y_train, y_test = r1.data_partition(m=1, train_size=train_size)
        
        # Treina o modelo com os dados de treinamento
        model.fit(X_train, y_train, show_eqm=False)
        
        # Realiza a predição com os dados de teste
        y_pred = model.predict(X_test)
        
        # Calcula o MSE para a rodada atual
        MSE = np.mean((y_test - y_pred) ** 2)
        all_MSE.append(MSE)
    
    # Calcula a média, desvio-padrão, maior valor e menor valor de MSE
    all_MSE = np.array(all_MSE)
    return {
        "mean": np.mean(all_MSE),
        "std": np.std(all_MSE),
        "min": np.min(all_MSE),
        "max": np.max(all_MSE),
        "all": all_MSE
    }


if __name__ == "__main__":
    R = 250
    train_size = 0.8
    
    model = r3.AdalineRegressor(learning_rate=0.01, n_epochs=1000, pr=1e-8)
    Adaline_result = monte_carlo(R, train_size, model)
    
    model = r3.MLPRegressor(L=3, Q=[100,150,100], m=1, lr=0.01 , n_epochs=1000, pr=1e-3)
    MLP_result = monte_carlo(R, train_size, model)
    
    print(f"MLP: {MLP_result}")
    print(f"Adaline: {Adaline_result}")
    
    # Criando tabela com os resultados
    table = [
        ["Modelo", "Média", "Desvio Padrão", "Mínimo", "Máximo"],
        ["MLP", MLP_result["mean"], MLP_result["std"], MLP_result["min"], MLP_result["max"]],
        ["Adaline", Adaline_result["mean"], Adaline_result["std"], Adaline_result["min"], Adaline_result["max"]]
    ]
    
    # Exibindo a tabela
    print(tabulate(table, headers="firstrow", tablefmt="grid"))
    
    print("\n=====================================\n")