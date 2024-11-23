import matplotlib.pyplot as plt

def pred_graph(x, y, titolo):
    plt.figure(figsize=(12, 6))
    plt.title(titolo)
    plt.plot(x, label='Prezzi reali')
    plt.plot(y, label='Prezzo previsto')
    plt.grid(True)
    plt.xlabel('Tempo')
    plt.ylabel('Prezzi')
    plt.legend()
    plt.show()