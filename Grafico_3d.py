'''
import plotly.plotly as py
from .grid_objs import Grid, Column
from plotly.tools import FigureFactory as FF

import pandas as pd
import time

tabela = pd.read_csv('WDO.CSV', index_col=0)
'''
'''
#resprova = 1
listanumeros = [1,2,3,4,2,10]
def lista(list):
    print(list)
    soma = sum(list)
    qtd = len(list)
    media = (soma/qtd)
    return media
res =lista(listanumeros)
print(res)

'''
'''#resp prova2
a = [[1, 7, ],[-3, 6, ],[ 2, 0],[ -8, 9]]
todos_negativos = []
for dados in a:
    negativos = map(lambda x : x if x > 0 else 0, dados )
    for x in negativos:
        todos_negativos.append(x)
print(str(todos_negativos))
print(f'Os Numeros Negativos São: {str(todos_negativos)}.'.replace('[', '').replace(']', ''))
'''
'''

matriz = []
for c in range(0,3):
    print(c)
    for i in range(0,3):
        print(i)
        if (matriz[c][i]< 0):
            matriz [c][i] = 0
'''
'''
X = 1
Y = 'aula'
print(X+Y)
'''

#processo em paralelo
#bibliotecas
import threading
import numpy as np

'''
#1
def frase():
    print('tarefas executadas')#variavel texto
    return

t = threading.Thread(target=frase)
t.start()
t.join()
'''
'''
#2
def frase(texto):
    print(texto)#variavel texto
    return

t = threading.Thread(target=frase, args=('tarefas executadas',))
t.start()
t.join()
'''
'''
#3
def frase(texto):
    print(texto)#variavel texto
    return

for i in range(5):
    t = threading.Thread(target=frase, args=('tarefas executadas',))
    t.start()
    t.join()
'''
'''
#4
def squad(num):
    s=num*2
    print(s)
    return

k = np.array([1,2,9])
t = threading.Thread(target=squad, args=(k,))
t.start()
t.join()
'''
'''
#5
def squad(num):
    s = num*2
    print(s)
    return

k1 = np.array([1,2,9])
k2 = np.array([10,12,19])
t1= threading.Thread(target=squad, args=(k1,))
t2= threading.Thread(target=squad, args=(k2,))
#+++++++++++ dispara o processo 1 ++++++++++
t1.start()
#+++++++++++ dispara o processo 2 ++++++++++
t2.start()
#+++++++++++ ordem para parar o processo1 ++++++++++
t1.join()
#+++++++++++ ordem para parar o processo1 ++++++++++
t2.join()
'''
'''
#6
def squad(num):
    s = num*2
    print(s)
    return

k1 = np.array([1,2,9])
k2 = np.array([10,12,19])
t1= threading.Thread(target=squad, args=(k1,))
t2= threading.Thread(target=squad, args=(k2,))
#+++++++++++ dispara o processo 2 ++++++++++
t2.start()
#+++++++++++ dispara o processo 1 ++++++++++
t1.start()
#+++++++++++ ordem para parar o processo1 ++++++++++
t1.join()
#+++++++++++ ordem para parar o processo1 ++++++++++
t2.join()
'''
#RETORNANDO VALORES CALCULADOS NO THREAD
'''
#6
def squad(num, res):
    s = num*2
    res.append(s)
    return

res=[]
for i in range(5):
    t = threading.Thread(target=squad, args=(i,res))
    t.start()
    t.join()
print(res)

s = 0 
n =1
'''
#################
posicao = 1
price_open = 5215
price_current = 5215
gatilho = 10.0
anda = 0.5

def BreackEvan(posicao, gatilho):
    if posicao == 0:
        g_t = 1
        print('gatilho acionado')
        disparo = price_open + gatilho
        print(f'compra=', disparo)
        return g_t
        #request = {'action': mt5.TRADE_ACTION_SLTP, 'position': ticket, 'sl': disparo, "symbol": symbol}
        #result = mt5.order_send(request)
        #print(result)
    else:
        print('gatilho acionado')
        disparo = price_open - gatilho
        print(f'venda=', disparo)
        g_t = 1
        return
        #request = {'action': mt5.TRADE_ACTION_SLTP, 'position': ticket, 'sl': disparo, "symbol": symbol}
        #result = mt5.order_send(request)


t = threading.Thread(target=BreackEvan, args=(posicao, gatilho))
t.start()
t.join()
