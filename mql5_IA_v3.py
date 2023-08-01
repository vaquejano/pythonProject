import pandas as pd
import numpy as np
import MetaTrader5 as mt5
import seaborn as sns
import matplotlib.pyplot as plt
import time
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Configurações iniciais
pd.set_option('display.max_columns', 400)
pd.set_option('display.width', 1500)

# Função para obter dados OHLC
def get_ohlc(ativo, timeframe, n=10):
    ativo = mt5.copy_rates_from_pos(ativo, timeframe, 0, n)
    ativo = pd.DataFrame(ativo)
    ativo['time'] = pd.to_datetime(ativo['time'], unit='s')
    ativo.set_index('time', inplace=True)
    return ativo

# Inicializar API
if not mt5.initialize():
    print("initialize() falhou")
    mt5.shutdown()
    if not mt5.initialize():
        print("initialize() falhou")
        mt5.shutdown()
print('******************')
print('*   conectado    *')
print('******************')

# Data atual
hoje = datetime.now()
print('data_Hora_atual ->', hoje)

# Informações do ativo
ativo = "WDO$"
symbol = "WDON23"


tabela = get_ohlc(ativo, mt5.TIMEFRAME_M1, 99999)
tabela.to_csv('WDO.CSV')
print(tabela)

# Indicadores técnicos
def sma(data, window):
    return data['close'].rolling(window=window).mean()

def ema(data, window):
    return data['close'].ewm(span=window, adjust=False).mean()

def rsi(data, window):
    delta = data['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

indicadores = pd.DataFrame()
indicadores['SMA'] = sma(tabela, window=10)
indicadores['EMA'] = ema(tabela, window=10)
indicadores['RSI'] = rsi(tabela, window=14)
indicadores['Sinal'] = np.where(indicadores['SMA'] > indicadores['EMA'], 1, -1)

# Dividir os dados em treinamento e teste
X = indicadores.iloc[:-1, :-1]
y = indicadores.iloc[:-1, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Treinar o modelo Random Forest
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Lógica de ordem de compra e venda
def execute_buy_order():
    print("Executando ordem de compra")
    # Implemente a lógica para executar uma ordem de compra
    pass

def execute_sell_order():
    print("Executando ordem de venda")
    # Implemente a lógica para executar uma ordem de venda
