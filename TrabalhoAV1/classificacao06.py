import pandas as pd
import classificacao01 as c1
import classificacao05 as c5

np, plt = c1.np, c1.plt

accuracy_list = c5.accuracy_list

results = []
for name, acc_list in accuracy_list.items():
    results.append({
        "Modelo": name,
        "Média": f"{np.mean(acc_list):.2%}",
        "Desvio-Padrão": f"{np.std(acc_list):.4f}",
        "Maior Valor": f"{np.max(acc_list):.2%}",
        "Menor Valor": f"{np.min(acc_list):.2%}",
        "Média": np.mean(acc_list),
    })
    
    
df = pd.DataFrame(results)
print(df.to_markdown(index=False, tablefmt="grid"))

bp=1