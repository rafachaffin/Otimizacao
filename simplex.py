import numpy as np

def getTableu(z, R, b):
    # Negativar z para o tableau
    z_row = -z
    # Adiciona b como última coluna de R
    R_ext = np.hstack([R, b.reshape(-1, 1)])
    # Adiciona linha do z e valor 0 na última coluna
    last_row = np.append(z_row, 0)
    tableau = np.vstack([R_ext, last_row])

    return tableau

def getRowPivot(tableau, col_pivot):
    # Calcula a razão entre última coluna (b) e coluna pivô, apenas para valores positivos na coluna pivô
    b = tableau[:-1, -1]
    col = tableau[:-1, col_pivot]
    with np.errstate(divide='ignore', invalid='ignore'):
        ratios = np.where(col > 0, b / col, np.inf)
    # Retorna o índice da menor razão
    return np.argmin(ratios)

def getColumnPivot(tableau):
    # Pega a última linha, exceto a última coluna (que é o termo independente)
    last_row = tableau[-1, :-1]
    # Encontra o índice do valor mais negativo
    col_pivot = np.argmin(last_row)
    if last_row[col_pivot] >= 0:
        return -1  # Não há coluna pivô (ótimo)
    return col_pivot

def runSimplex(tableau):
    while True:
        col_pivot = getColumnPivot(tableau)
        if col_pivot == -1:
            break  # Ótimo encontrado
        row_pivot = getRowPivot(tableau, col_pivot)
        print(f"Coluna pivô: {col_pivot}, Linha pivô: {row_pivot}")
        # Pivoteamento
        pivot_element = tableau[row_pivot, col_pivot]
        tableau[row_pivot, :] = tableau[row_pivot, :] / pivot_element
        for i in range(tableau.shape[0]):
            if i != row_pivot:
                tableau[i, :] = tableau[i, :] - tableau[i, col_pivot] * tableau[row_pivot, :]
        print("Tableau atualizado:")
        print(tableau)
    # Solução ótima: última coluna, exceto última linha
    solution = np.zeros(tableau.shape[1] - 1)
    for j in range(tableau.shape[1] - 1):
        col = tableau[:-1, j]
        if np.count_nonzero(col) == 1 and np.sum(col) == 1:
            row = np.where(col == 1)[0][0]
            solution[j] = tableau[row, -1]
    print("Solução ótima:", solution)
    print("Valor ótimo:", tableau[-1, -1])
    return solution, tableau[-1, -1]

def main():
    # Exemplo simples para teste
    z = np.array([3, 2])
    R = np.array([[1, 2], [1, 1]])
    b = np.array([4, 2])
    
    tableau = getTableu(z, R, b)
    print("Tableau inicial:")
    print(tableau)

    runSimplex(tableau)

if (__name__) == "__main__":
    main()
    