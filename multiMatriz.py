from numba import cuda
import numpy as np

@cuda.jit
def my_kernel( matriz_a, matriz_b, matriz_c, width):

    l = cuda.threadIdx.x
    c = cuda.blockIdx.x

    k = 0
    soma = 0

    if l < width and c < width:
      for k in range(width):
        soma = soma + matriz_a[l][k] * matriz_b[k][c]
      matriz_c[l][c] = soma

# vetor com números
matriz_a = np.array([[2, 3],[1, 0]])
matriz_b = np.array([[3, 1],[2, 4]])
matriz_c = np.array([[0, 0],[0, 0]])

# números de threads por bloco
threads_per_block = 32

# número de blocos por grid
blocks_per_grid = (2 + (threads_per_block - 1) )

# iniciando o kernel
my_kernel[blocks_per_grid, threads_per_block](matriz_a, matriz_b, matriz_c, 2)
print( 'Resultado Esperado')
print( np.matmul(matriz_a, matriz_b) )
print("\n")
print( 'Resultado Obtido')
print(matriz_c)
