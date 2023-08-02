import pandas as pd
import numpy as np
import MetaTrader5 as mt5
import seaborn as sns
import matplotlib.pyplot as plt
import time
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report, mean_squared_error

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
symbol = 'WDOU23'
symbol_info = mt5.symbol_info(symbol)
print('simbolo:',symbol_info)
#posicao = (len(symbol_info))
last = symbol_info.last
print('last:',last)
#tabela = get_ohlc(ativo, mt5.TIMEFRAME_M1, 99999)
#tabela.to_csv('WDO.CSV')
tabela = pd.read_csv('WDO.CSV', index_col=0)
#print(tabela)
'''Setando as colunas das planilhas'''
pd.set_option('display.max_columns', 400)  # número de colunas mostradas
pd.set_option('display.width', 1500)  # max. largura máxima da tabela exibida
'''tabela de manipulacao de medias'''
tabela.loc[:, 'media2'] = tabela['close'].ewm(span=2, min_periods=2).mean()  # media exponencial
tabela.loc[:, 'media3'] = tabela['close'].ewm(span=3, min_periods=3).mean()  # media exponencial
tabela.loc[:, 'media5'] = tabela['close'].ewm(span=5, min_periods=5).mean()  # media exponencial
tabela.loc[:, 'media8'] = tabela['close'].ewm(span=8, min_periods=8).mean()  # media exponencial
#print(tabela.head())
tabela = tabela.drop(["spread", "tick_volume", "real_volume"], axis=1)
tabela.dropna(inplace=True)
#print(tabela)
#tabela.iloc[-10:]
#print(tabela.shape)
'''Verifica a correlacao'''
# tabela.corr()
'''grafico'''
## Preparar dados para o modelo
tabela['signal'] = np.where(
    (tabela['media2'] > tabela['close']) &
    (tabela['media3'] > tabela['close']) &
    (tabela['media5'] > tabela['close']) &
    (tabela['media8'] > tabela['close']),  # Sell signal condition
    -1,  # Sell signal
np.where(
    (tabela['media2'] < tabela['close']) &
    (tabela['media3'] < tabela['close']) &
    (tabela['media5'] < tabela['close']) &
    (tabela['media8'] < tabela['close']),  # Buy signal condition
    1,  # Buy signal
    0  # No signal
    )
)
#tabela.reset_index('time', inplace=True)#reseta index
tabela1 = tabela.iloc[-500:]
print(tabela1.index)
print(tabela1)
figure = plt.figure(figsize=(15, 8))
plt.plot(tabela1.index, tabela1['close'], label='Preço de Fechamento', color='blue')
plt.plot(tabela1.index, tabela1['media2'], label=f'media2', color='black')
plt.plot(tabela1.index, tabela1['media3'], label=f'media3', color='orange')
plt.plot(tabela1.index, tabela1['media5'], label=f'media5', color='red')
plt.plot(tabela1.index, tabela1['media8'], label=f'media8', color='magenta')
## Marca os pontos de compra e venda no gráfico
plt.plot(tabela1[tabela1['signal'] == 1].index, tabela1['close'][tabela1['signal'] == 1], '^', markersize=10, color='g', label='Sinal de Compra')
plt.plot(tabela1[tabela1['signal'] == -1].index, tabela1['close'][tabela1['signal'] == -1], 'v', markersize=10, color='r', label='Sinal de Venda')
'''titulo grafico'''
plt.title('Grafico Media')
plt.ylabel('preco')
plt.xlabel("time")
plt.legend()
plt.show()
'''Grafico correlacao '''
#sns.heatmap(tabela.corr(), annot=True, vmin=-1, vmax=1, cmap='Reds')
#plt.show()
'''Separa dados de X e y '''
X = tabela.drop(columns=["close", "signal"], axis=1) .fillna(0)
y = tabela['close'].values
#****************Grafico correlacao de X *********************************************
#sns.heatmap(X.corr(), annot=True, vmin=-1, vmax=1, cmap='Reds')
#plt.show()
'''Separa dados de trino e teste '''
X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.618, random_state=42)
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
n_stimators = [int(x) for x in np.linspace(start=1, stop=1000, num=1)]#'''Numero de Arvores[int(x) for x in np.linspace(start=1, stop=180, num=10)]'''
max_samples = [None]
'''Número de feições a serem consideradas em cada divisão'''
max_features = ['sqrt',
                'log2']  # número máximo de recursos (variáveis independentes) a serem considerados ao dividir um nó
max_features.append(1.0)
'''Número máximo de níveis na árvore'''
max_depth = [int(x) for x in np.linspace(2, 1000, num=2)]  # 11-[None, 30] 25
max_depth.append(None)
'''Número mínimo de amostras necessárias para dividir um nó'''
min_samples_split = [int(x) for x in np.linspace(2.0,450, num=5)]  # numero 2 e default--float usa ponto(número mínimo de amostras em um subconjunto (também conhecido como nó) para dividi-lo em mais dois subconjuntos)2, 3, 5, 8, 13, 21, 34, 55, 89, 134
min_samples_split.append(2)
'''Número mínimo de amostras necessárias em cada nó folha'''
min_samples_leaf = [int(x) for x in np.linspace(2.0, 150, num=15)]  # 8 '76' 1 = default-float usa ponto 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 134
min_samples_leaf.append(1)
'''Número mínimo de amostras necessárias em cada nó folha'''
bootstrap = [True]
oob_score = [False]
'''oob_score.append(bool)'''
random_state = [42]
n_jobs = [-1]
''' Grid search'''
param_grid = {
    'n_estimators': n_stimators,  # número de árvores no foreset
    'max_samples': max_samples,
    'max_features': max_features,  # número máximo de recursos considerados para dividir um nó
    'max_depth': max_depth,  # número máximo de níveis em cada árvore de decisão
    'min_samples_split': min_samples_split,
    # número mínimo de pontos de dados colocados em um nó antes que o nó seja dividido
    'min_samples_leaf': min_samples_leaf,  # número mínimo de pontos de dados permitidos em um nó folha
    'bootstrap': bootstrap,  # método para amostragem de pontos de dados (com ou sem reposição)
    'oob_score': oob_score, #estimar a pontuação de generalização
    'random_state': random_state,
    'n_jobs': n_jobs,
}
# print(param_grid)
'''Random Frorest'''
floresta = RandomForestRegressor()
# %timeit
rf_RandomGrid = RandomizedSearchCV(estimator=floresta,
                                   param_distributions=param_grid,
                                   n_iter=10,
                                   cv=3,
                                   verbose=0,
                                   random_state=42,
                                   n_jobs=-1
                                   )
'Treina o modelo com os Parametros da IA Grid Saerch'
rf_RandomGrid.fit(X_treino, y_treino)
'''Melhores Parametros da IA Grid Saerch*'''
k = rf_RandomGrid.best_params_
# print(k["max_depth"])
rf_RandomGrid.best_score_
rf_RandomGrid.cv_results_
df = pd.DataFrame(rf_RandomGrid.cv_results_)
# print(df)
# print(rf_RandomGrid.cv_results_)
y_pred_rf_rg = rf_RandomGrid.predict(X_teste)
print("y_pred->>>>", y_pred_rf_rg[-1])
rf_RandomGrid.score(X_teste, y_teste)
#print(rf_RandomGrid.score(X_teste, y_teste))
#accuracy_score(y_teste, y_pred_rf_rg)
print("estimetor", rf_RandomGrid.best_estimator_)
#****************Treino minha IA**********************************************
floresta = rf_RandomGrid.best_estimator_
floresta.fit(X_treino, y_treino)
#****************Faco minha predicao **********************************************
floresta_predic = floresta.predict(X_teste)
print(f"floresta_predic,{floresta_predic}")
dif = floresta_predic[-1]-last
print(dif)
#floresta_rmse = mean_squared_error(X_teste, p)
#print(floresta_rmse)
#fl = p[-1]
#print(fl)
#print('dif:', fl-last)
