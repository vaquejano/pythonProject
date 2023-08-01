import numpy as np
import pandas as pd
import MetaTrader5 as mt5
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn.ensemble
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, classification_report, confusion_matrix

## Configurar colunas do DataFrame
pd.set_option('display.max_columns', 400)  # número de colunas mostradas
pd.set_option('display.width', 1500)  # largura máxima da tabela exibida

## Função para obter dados OHLC
def get_ohlc(ativo, timeframe, n=10):
    ativo = mt5.copy_rates_from_pos(ativo, timeframe, 0, n)
    ativo = pd.DataFrame(ativo)
    ativo['time'] = pd.to_datetime(ativo['time'], unit='s')
    ativo.set_index('time', inplace=True)
    return ativo

if not mt5.initialize():
    print("initialize() failed")
    mt5.shutdown()
    if not mt5.initialize():
        print("initialize() failed")

# mt5.shutdown()
print('******************')
print('*   conectado    *')
print('******************')

#**************** Fim data atual**************************************************
hoje = datetime.now()
print(hoje)

## Informações do ativo
ativo = 'WDO$'
symbol = "WDOQ23"

## Obtendo dados OHLC
dados = get_ohlc(ativo, mt5.TIMEFRAME_M1, n=200)  # Aqui estou usando M5 como exemplo, mas você pode alterar para o período desejado.

## Definindo as médias móveis
ma_curta_periodos = 7
ma_longa_periodos = 21

tabela1 = dados
tabela1['MA_curta'] = tabela1['close'].rolling(window=ma_curta_periodos).mean()
tabela1['MA_longa'] = tabela1['close'].rolling(window=ma_longa_periodos).mean()

## Sinais de compra e venda
tabela1['sinal_compra'] = np.where(tabela1['MA_curta'] > tabela1['MA_longa'], 1, 0)
tabela1['sinal_venda'] = np.where(tabela1['MA_curta'] < tabela1['MA_longa'], -1, 0)

## Plota os gráficos
plt.figure(figsize=(12, 6))
plt.plot(tabela1['close'], label='Preço de Fechamento', color='blue')
#plt.plot(dados.index, dados['MA_curta'], label=f'Média Móvel {ma_curta_periodos} períodos', color='orange')
#plt.plot(dados.index, dados['MA_longa'], label=f'Média Móvel {ma_longa_periodos} períodos', color='red')

## Marca os pontos de compra e venda no gráfico
plt.plot(tabela1[tabela1['sinal_compra'] == 1].index, tabela1['close'][tabela1['sinal_compra'] == 1], '^', markersize=10, color='g', label='Sinal de Compra')
plt.plot(tabela1[tabela1['sinal_venda'] == -1].index, tabela1['close'][tabela1['sinal_venda'] == -1], 'v', markersize=10, color='r', label='Sinal de Venda')

plt.legend()
plt.title('Estratégia de Médias Móveis')
plt.xlabel('Data')
plt.ylabel('Preço de Fechamento')
plt.show()