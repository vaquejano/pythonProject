import pandas as pd
import numpy as np

'''
Crie um programa em python que analise open, hight, low, close, analise e mostre a melhor entrada de compra e venda, 
defenda com stoploss, defenda com breakevan e monitore a operação com tralingstop, para um lucro de dez mil reais
Aqui está um exemplo de um programa em Python que executa as tarefas que
'''
def buy_sell_entry(df, close='close'):
    buy_entry = 4898.0000
    sell_entry = None
    stop_loss = None
    trailing_stop = None
    breakeven = None
    profit = 10000
    for i in range(1, len(df)):
        if df['close'][i] > df['close'][i-1] and df['close'][i] > df['open'][i]:
            buy_entry = df['close'][i]
            stop_loss = buy_entry * 0.9
            breakeven = buy_entry + (buy_entry * 0.01)
            break
    for i in range(i, len(df)):
        if df['close'][i] < breakeven or df['close'][i] < stop_loss:
            sell_entry = df['close'][i]
            break
        if df[(close)][i] > buy_entry + profit:
            sell_entry = df[close][i]
            break
        if trailing_stop is None:
            trailing_stop = df[close][i] * 0.95
        elif df[close][i] < trailing_stop:
            sell_entry = trailing_stop
            break
    return buy_entry, sell_entry

df = pd.read_csv("WDO.CSV")
buy_entry, sell_entry = buy_sell_entry(df)

if buy_entry is not None and sell_entry is not None:
    print("Best buy entry:", buy_entry)
    print("Best sell entry:", sell_entry)
else:
    print("No profitable trade found.")

'''
Este programa usa um forloop para iterar sobre os dados e determinar os melhores pontos de entrada de compra e venda 
com base nas condições especificadas (vela verde, stop loss, breakeven, trailing stop e lucro de R$ 10.000). 
A buy_sell_entryfunção retorna as entradas de compra e venda e o restante do código gera os resultados. 
Observe que você precisará substituir "stock_data.csv" pelo caminho do arquivo para seus dados de estoque reais.
'''