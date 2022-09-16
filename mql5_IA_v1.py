#****************Bibliotecas*************************************
import math
import MetaTrader5 as mt5
from datetime import datetime
import time
import pandas as pd
import numpy as np
#****************Modulo sklearn*************************************
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score


#****************funcao criar data frame *************************************
def get_ohlc(ativo, timeframe, n=10):
    ativo = mt5.copy_rates_from_pos(ativo, timeframe, 0, n)
    ativo = pd.DataFrame(ativo)
    ativo['time'] = pd.to_datetime(ativo['time'], unit='s')
    ativo.set_index('time', inplace=True)
    return ativo

def bilhetagem(Symbol):
    symbol_info = mt5.symbol_info(symbol)
    posicao=(len(symbol_info))
    last = symbol_info.last
    ativo = symbol_info.name
    return symbol_info, last, ativo, posicao

def status(last, ativo, posicao):
    print(f'Ativo-------->> {ativo}')
    print(f'qtde_Ifo----->> {posicao}')
    print(f'ultimo_Preco->> {last:.3f}\n')

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    if not mt5.initialize():
        print("initialize() failed")
        mt5.shutdown()
        if not mt5.initialize():
            print("initialize() failed")

    #mt5.shutdown()
    print('******************')
    print('*   conectado    *')
    print('******************')

    #****************Data atual**************************************************
    #hoje = datetime.now()
    #print(hoje)    #****************Data atual**************************************************
    hoje = datetime.now()
    print(hoje)

    #****************Informacoes do ativo***********************************
    ativo = 'WDO$'
    symbol = "WDOV22"

    #****************funcao criar data frame *************************************
    wdo_m1 = get_ohlc(ativo, mt5.TIMEFRAME_M1, 99999)
    wdo_m1.to_csv('WDO.CSV')

    simbolo = bilhetagem(symbol)
    print('#####' * 2, 'Informacao do ativo', '#####' * 2)
    last = simbolo[1]
    ativo = simbolo[2]
    posicao = simbolo[3]
    status(last, ativo, posicao)

    #***************Busca dos dados ************************************
    # Setando as colunas das planilhas
    pd.set_option('display.max_columns', 400)  # número de colunas mostradas
    pd.set_option('display.width', 1500)  # max. largura máxima da tabela exibida

    #***************Busca os dados Historicos************************************
    tabela = pd.read_csv('WDO.CSV', index_col=0)
    #tabela.reset_index('time', inplace=True)#reseta index
    print(tabela)
    # ****************Tabela manipulaco**********************************************
    #tabela = tabela.drop(["open", "high", "low", "spread", "tick_volume", "real_volume"], axis=1)

    tabela.loc[:, 'media7(s)'] = (tabela['close'].rolling(7).mean())
    tabela.loc[:, 'media7(md)'] = (tabela['close'].rolling(7).median())
    tabela.loc[:, 'media7'] = (tabela['close'].ewm(span=7, min_periods=7).mean())  # media exponencial
    tabela.loc[:, 'media21'] = (tabela['close'].ewm(span=21, min_periods=21).mean())
    tabela.loc[:, 'media36'] = (tabela['close'].ewm(span=36, min_periods=36).mean())
    tabela.loc[:, 'momento'] = (tabela['close'] - tabela['close'].rolling(6).mean())

    tabela = tabela.drop(["high", "low","spread", "tick_volume", "media7(s)", "media7(md)"], axis=1)
    #"high", "low", "spread", "tick_volume", "real_volume", "media7(s)",'media7(md)', "media7", "media21", "media36"

    # ****************Preenche o valor vazio com o valor 0**********************************************
    tabela = tabela.fillna(0)  # preenche o valor vazio com o valor 0]
    #print(f'>', tabela[-20:])

    # ****************Separa dados de X e y*********************************************
    X = tabela.drop(["close"], axis=1)
    X = X.fillna(0)
    y = tabela['close'].values
    print(X[-25:])
    #print(y)
    # ****************Separa dados de trino e dados de teste*********************************************
    X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=24, random_state=42)
    # ****************Separa dados de trino e dados de teste*********************************************
    # print(X_teste)

    # para teste(10, 20, 30, 40, 50, 60, 70, 80, 90, 100)
    # ****************Parametros da IA Grid Saerch**********************************************
    # Número de árvores na floresta aleatória
    n_stimators = [int(x) for x in np.linspace(start=1, stop=100,
                                               num=1)]  # Numero de Arvores[int(x) for x in np.linspace(start=1, stop=180, num=10)]
    max_samples = [None]
    # Número de feições a serem consideradas em cada divisão
    max_features = ['sqrt','log2']  # número máximo de recursos (variáveis independentes) a serem considerados ao dividir um nó
    max_features.append(1.0)
    # Número máximo de níveis na árvore
    max_depth = [int(x) for x in np.linspace(10, 110, num=11)]  # [None, 30]
    max_depth.append(None)
    # Número mínimo de amostras necessárias para dividir um nó
    min_samples_split = [2, 5, 10,
                         20]  # numero 2 e default--float usa ponto(número mínimo de amostras em um subconjunto (também conhecido como nó) para dividi-lo em mais dois subconjuntos)2, 3, 5, 8, 13, 21, 34, 55, 89, 134
    # Número mínimo de amostras necessárias em cada nó folha
    min_samples_leaf = [1, 2, 4]  # 1 = default-float usa ponto 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 134
    # Método de seleção de amostras para treinamento de cada árvore
    bootstrap = [True, False]
    random_state = [42]
    n_jobs = [-1]

    param_grid = {
        'n_estimators': n_stimators,  # número de árvores no foreset
        'max_samples': max_samples,
        'max_features': max_features,  # número máximo de recursos considerados para dividir um nó
        'max_depth': max_depth,  # número máximo de níveis em cada árvore de decisão
        'min_samples_split': min_samples_split,
        # número mínimo de pontos de dados colocados em um nó antes que o nó seja dividido
        'min_samples_leaf': min_samples_leaf,  # número mínimo de pontos de dados permitidos em um nó folha
        'bootstrap': bootstrap,  # método para amostragem de pontos de dados (com ou sem reposição)
        'random_state': random_state,
        'n_jobs': n_jobs,

    }
    # print(param_grid)

    floresta = RandomForestRegressor()
    # %timeit
    rf_RandomGrid = RandomizedSearchCV(estimator=floresta, param_distributions=param_grid, n_iter=100, cv=3,
                                       verbose=2, random_state=42, n_jobs=-1)
    # print(RandomizedSearchCV)

    # ****************Treina o modelo com os Parametros da IA Grid Saerch**********************************************
    rf_RandomGrid.fit(X_treino, y_treino)

    # ****************Melhores Parametros da IA Grid Saerch**********************************************
    rf_RandomGrid.best_params_
    rf_RandomGrid.best_estimator_

    # ****************Treino minha IA**********************************************
    floresta = rf_RandomGrid.best_estimator_
    floresta.fit(X_treino, y_treino)

    # ****************Faco minha predicao **********************************************
    p = floresta.predict(X_teste)

    # ****************Decisao***********************************************
    fl = mean_squared_error(y_teste, p)
    print(fl)
    fl1 = np.sqrt(mean_squared_error(y_teste, p))
    # *2***************Printa os dados**********************************************
    print('resultado')
    print('***melhor parametro***')
    print(rf_RandomGrid.best_params_)
    print('***melhor estimador***')
    print(rf_RandomGrid.best_estimator_)
    # print('***melhor resultado***')
    # print(rf_RandomGrid.cv_results_)
    print('*$*' * 30)
    print(f'Treino Accuracy (r) - : {rf_RandomGrid.score(X_treino, y_treino):.3f}')
    print(f'Teste Accuracy - : {rf_RandomGrid.score(X_teste, y_teste):.3f}')
    print(f'Teste Accuracy Floresta- : {floresta.score(X_teste, y_teste):.3f}')
    #######################################################################################