# (EXERCÍCIOS EM RESOLUÇÃO - ATUALIZAÇÃO A MEDIDA DO POSSÍVEL:))
#Exercícios Numpy (Respostas na Semana #3)
                 
#1- Importe numpy como ‘np’ e imprima o número da versão.
import numpy as np
# print("versão: ", np.version.version)  

# 2- Crie uma matriz 1D com números de 0 a 9
# Saída desejada:
# array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
matrix = np.arange(10)
# print(matrix)

#3- Crie uma matriz booleana numpy 3×3 com ‘True’
#Saída desejada:
#array([[ True, True, True],
#[ True, True, True],
#[ True, True, True]], dtype=bool)

boolean = np.array([[True, True, True], [True, True, True], [True, True, True]])
#print(boolean)

#4- Extraia todos os números ímpares de ‘arr’
#arr = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
#Saída desejada: array([1, 3, 5, 7, 9])


#5- Substitua todos os números ímpares arr por -1
#arr = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
#Saída desejada: array([ 0, -1, 2, -1, 4, -1, 6, -1, 8, -1])
arr = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
arr[arr%2==1] = -1
#print(arr)

                        #➡ Médio

#1- Substitua todos os números ímpares em arr com -1 sem alterar arr
#arr = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
#Saída desejada: array([ 0, -1, 2, -1, 4, -1, 6, -1, 8, -1])
#arr == array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
arr = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
arr2 = np.copy(arr)
arr2[arr%2==1] = -1
#print(arr)
#print(arr2)

#2- Converta uma matriz 1D para uma matriz 2D com 2 linhas:
#np.arange(10)
#array([ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
#Saída desejada:
#array([[0, 1, 2, 3, 4],
#[5, 6, 7, 8, 9]])
a1 = np.arange(10)
half = len(a1) // 2
a2 = np.array([a1[0:half], a1[half:len(a1)]])
#print(a2)

#3- Empilhe matrizes verticalmente:
#a = np.arange(10).reshape(2,-1)
#b = np.repeat(1, 10).reshape(2,-1)
#Saída desejada:
#array([[0, 1, 2, 3, 4],
#[5, 6, 7, 8, 9],
#[1, 1, 1, 1, 1],
#[1, 1, 1, 1, 1]])
a = np.arange(10).reshape(2,-1)
b = np.repeat(1, 10).reshape(2,-1)
vertical = np.append(a, b, axis=0)
#print(vertical)

#4- Empilhe as matrizes horizontalmente:
#a = np.arange(10).reshape(2,-1)
#b = np.repeat(1, 10).reshape(2,-1)
#Saída desejada:
#array([[0, 1, 2, 3, 4, 1, 1, 1, 1, 1],
#[5, 6, 7, 8, 9, 1, 1, 1, 1, 1]])
a = np.arange(10).reshape(2,-1)
b = np.repeat(1, 10).reshape(2,-1)
horizontal = np.append(a, b, axis=1)
#print(horizontal)

#5- Crie o seguinte padrão sem codificação, usando apenas funções numpy e a matriz de entrada abaixo ‘a’.
#a = np.array([1,2,3])
#Saída desejada: array([1, 1, 1, 2, 2, 2, 3, 3, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3])

                      
#1- Calcule a pontuação softmax de ‘sepal length’:
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
sepallength = np.genfromtxt(url, delimiter=',', dtype='float', usecols=[0])
softmax = np.exp(sepallength) / np.sum(np.exp(sepallength))
#print(softmax)


#2- Filtre as linhas de iris_2d que possuem petallength (coluna 3) > 1.5 e sepallength (coluna 1) < 5.0
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris_2d = np.genfromtxt(url, delimiter=',', dtype='float', usecols=[0,1,2,3])
#print(iris_2d[(iris_2d[:,3] > 1.5)])

#3- Selecione as linhas de iris_2d que não têm nenhum valor ‘nan’
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris_2d = np.genfromtxt(url, delimiter=',', dtype='float', usecols=[0,1,2,3])
#print(iris_2d[~np.isnan(iris_2d).any(axis=1)])

#Exercícios Pandas (Respostas na Semana #3)

# 1- Importe Pandas e printe a versão
import pandas as pd
#print("Versão: ", pd.__version__)

# 2- Crie uma série panda de cada um dos ítens abaixo: uma lista, numpy e um dicionário
import numpy as np
minhalista = list('abcedfghijklmnopqrstuvwxyz')
meuarr = np.arange(26)
meudict = dict(zip(minhalista, meuarr))
serie1 = pd.Series(minhalista, index=np.arange(26))
#print(serie1)
serie2 = pd.Series(meuarr, index=np.arange(26))
#print(serie2)
serie3 = pd.Series(meudict)
#print(serie3)

# 3- Converta a série “ser” em um dataframe com seu índice como outra coluna no dataframe.
minhalista = list('abcedfghijklmnopqrstuvwxyz')
meuarr = np.arange(26)
meudict = dict(zip(minhalista, meuarr))
ser = pd.Series(meudict)
d = {'dados' : ser, 'indices' : ser.keys()}
df = pd.DataFrame(d) 
#print(df)

# 4- Combine ‘ser1’ e ‘ser2’ para formar um dataframe
import numpy as np
ser1 = pd.Series(list('abcedfghijklmnopqrstuvwxyz'))
ser2 = pd.Series(np.arange(26))
#print(pd.DataFrame(ser1, ser2))

#5- Atribua um nome a série “ser” chamando-a de ‘alfabeto’.
ser = pd.Series(list('abcedfghijklmnopqrstuvwxyz'))
ser = ser.rename('alfabeto')
#print(ser)
