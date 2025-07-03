import numpy as np
from typing import Tuple

def get_tableau(z: np.ndarray, R: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Cria o tableau Simplex inicial para um problema de maximização na forma padrão.
    Adiciona automaticamente as variáveis de folga.
    """
    # Número de variáveis de folga
    
    num_slack_vars = len(b)
    
    # Cria a matriz identidade para as variáveis de folga
    I = np.identity(num_slack_vars)
    
    # Combina a matriz de restrições com a matriz identidade
    R_ext = np.hstack([R, I])
    
    # Adiciona o vetor b como a última coluna
    constraints_part = np.hstack([R_ext, b.reshape(-1, 1)])
    
    # Prepara a linha da função objetivo (z)
    z_padded = np.pad(z, (0, num_slack_vars), 'constant')
    z_row = np.append(-z_padded, 0) # Negativar para maximização e adicionar 0 para o valor de z
    
    # Combina tudo no tableau final
    tableau = np.vstack([constraints_part, z_row])
    
    return tableau

def get_pivot_column(tableau: np.ndarray) -> int:
    """Encontra o índice da coluna pivô (valor mais negativo na linha z)."""
    z_row = tableau[-1, :-1]
    if np.all(z_row >= 0):
        return -1  # Solução ótima encontrada
    return int(np.argmin(z_row))

def get_pivot_row(tableau: np.ndarray, col_pivot: int) -> int:
    """Encontra o índice da linha pivô usando o teste da razão mínima."""
    b_col = tableau[:-1, -1]
    pivot_col = tableau[:-1, col_pivot]
    
    # Previne divisão por zero ou por números negativos
    with np.errstate(divide='ignore', invalid='ignore'):
        ratios = np.where(pivot_col > 1e-10, b_col / pivot_col, np.inf)

    if np.all(ratios == np.inf):
        return -1 # Indica solução ilimitada
        
    return int(np.argmin(ratios))

def pivot(tableau: np.ndarray, row_pivot: int, col_pivot: int):
    """Realiza a operação de pivoteamento no tableau."""
    pivot_element = tableau[row_pivot, col_pivot]
    
    # Normaliza a linha pivô
    tableau[row_pivot, :] /= pivot_element
    
    # Zera os outros elementos da coluna pivô
    for i in range(tableau.shape[0]):
        if i != row_pivot:
            multiplier = tableau[i, col_pivot]
            tableau[i, :] -= multiplier * tableau[row_pivot, :]

def run_simplex(tableau: np.ndarray) -> Tuple[np.ndarray, float, str]:
    """Executa o algoritmo Simplex."""
    np.set_printoptions(precision=4, suppress=True, 
        formatter={'float_kind': lambda x: f"{x:,.4f}"})
    num_vars = tableau.shape[1] - 1 - tableau.shape[0] + 1
    iteration = 0

    while True:
       
        print("Tableau atual , interação " + str(iteration) + " :")
        print(np.round(tableau, 4))
        print("-" * 50)
        
        input()


        col_pivot = get_pivot_column(tableau)
        if col_pivot == -1:
            status = "Solução ótima encontrada"
            break

        row_pivot = get_pivot_row(tableau, col_pivot)
        if row_pivot == -1:
            status = "Problema com solução ilimitada"
            return None, np.inf, status
        
        print(f"Coluna pivô: {col_pivot}")
        print(f"Linha pivô: {row_pivot}")
        print(f"Elemento pivô: {tableau[row_pivot, col_pivot]:.4f}")
        
        pivot(tableau, row_pivot, col_pivot)
        iteration += 1
    
    
    solution = np.zeros(num_vars)
    for j in range(num_vars):
        col = tableau[:-1, j]
        # Verifica se a coluna é de uma variável básica (um valor próximo a 1, resto próximo a 0)
        is_basic = np.isclose(col, 1).sum() == 1 and np.isclose(col, 0).sum() == len(col) - 1
        if is_basic:
            row_index = np.where(np.isclose(col, 1))[0][0]
            solution[j] = tableau[row_index, -1]
            
    optimal_value = tableau[-1, -1]
    
    return solution, optimal_value, status

def main():
    option = 0
    while True:
        option = int(input("Qual problema deseja?\n 1) Exemplo 1 \n 2) Exemplo 2\n 3) Sair \n"))
        if (option == 1):
            print("Problema é : ")
            print("max z = 8x + 6y")
            print("5x + 3y <= 30\n" +
                     "2x + 3y <= 24\n" +
                     "1x + 3y <= 18\n")
            
            input()

            z = np.array([8, 6])
            R = np.array([[5, 3], 
                        [2, 3],
                        [1, 3]])
            b = np.array([30, 24, 18])

            tableau = get_tableau(z, R, b)
            print("Tableau Inicial:")
            print(np.round(tableau, 2))
            print("-" * 30)

            input()

            solution, optimal_value, status = run_simplex(tableau)
            print(status)
            if solution is not None:
                print(f"Valor ótimo de z: {optimal_value:.2f}")
                for i, val in enumerate(solution):
                    print(f"x{i+1} = {val:.2f}\n")
            
        elif (option == 2):
            print("Problema é : ")
            print("max z = 300r + 725s + 200t + 450u")
            print("r + 3s + t + 2u <= 60\n" +
                     "2r + 8s + 2t + 3u <= 140\n" +
                     "3r + 5s + 6t + 3u <= 100\n")
            
            input()

            z = np.array([300, 725, 200, 450])
            R = np.array([[1, 3, 1, 2], 
                        [2, 8, 2, 3],
                        [3, 5, 6, 3]])
            b = np.array([60, 140, 100])

            tableau = get_tableau(z, R, b)
            print("Tableau Inicial:")
            print(np.round(tableau, 2))
            print("-" * 30)

            input()
            
            solution, optimal_value, status = run_simplex(tableau)
            print(status)
            if solution is not None:
                print(f"Valor ótimo de z: {optimal_value:.2f}")
                for i, val in enumerate(solution):
                    print(f"x{i+1} = {val:.2f}")
        else:
            break

if __name__ == "__main__":
    main()