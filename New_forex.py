#****************Bibliotecas*************************************
import math
import MetaTrader5 as mt5
from datetime import datetime
import time
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
#****************Modulo sklearn*************************************
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
#from BreackEv import price_open
#****************Modulo sched*************************************
import sched
import time
import threading
from datetime import datetime
#****************Fim Biblioteca*************************************
def repeat_at_interval(scheduler, event, interval=60, add_n=10, start_t=None):
    """Adds 'add_n' more calls to "event" at each "interval" seconds"""
    # Unix timestamp
    if start_t is None:
        t = time.time()
        # round to next interval -
        t = t - (t % interval) + interval
    else:
        t = start_t

    for i in range(add_n):
        scheduler.enterabs(t, 0, event)
        t += interval

    # Schedule call to self, to add "add_n" extra events
    # when these are over:
    scheduler.enterabs(t - interval, 0, repeat_at_interval, kwargs={
        "scheduler": scheduler,
        "event": event,
        "interval": interval,
        "add_n": add_n,
        "start_t": t
        })


def test():
    print (datetime.now())

def main():
    scheduler  = sched.scheduler(time.time, time.sleep)
    repeat_at_interval(scheduler, test, interval=60)
    thread = threading.Thread(target=scheduler.run)
    thread.start()
    while True:

        time.sleep(10)

        # ****************Cria minhas funcoes*************************************

        # ****************funcao criar data frame *************************************
        def get_ohlc(ativo, timeframe, n=10):
            ativo = mt5.copy_rates_from_pos(ativo, timeframe, 0, n)
            ativo = pd.DataFrame(ativo)
            ativo['time'] = pd.to_datetime(ativo['time'], unit='s')
            ativo.set_index('time', inplace=True)
            return ativo

        def bilhetagem(Symbol):
            symbol_info = mt5.symbol_info(symbol)
            posicao = (len(symbol_info))
            last = symbol_info.last
            ativo = symbol_info.name
            spread = symbol_info.spread
            return symbol_info, last, ativo, posicao, spread

        def status(last, ativo, posicao, spread):
            print(f'Ativo-------->> {ativo}')
            print(f'qtde_Ifo----->> {posicao}')
            print(f'ultimo_Preco->> {last:.1f}\n')
            print(f'ask->> {spread:.1f}\n')
            #print(f'bid->> {bid:.3f}\n')
# ****************Final da funcoes*************************************
        #if __name__ == '__main__':

        if not mt5.initialize():
            print("initialize() failed")
            mt5.shutdown()
            if not mt5.initialize():
                print("initialize() failed")

        # mt5.shutdown()
        print('******************')
        print('*   conectado    *')
        print('******************')

        # ****************Data atual**************************************************
        # hoje = datetime.now()
    # print(hoje)    #****************Data atual**************************************************
        hoje = datetime.now()
        print(hoje)

        # ****************Informacao do Ativo*************************************
        ativo = 'EURUSD'
        symbol = "EURUSD"
        spread = 15
        volume = 1.0

        # ****************funcao criar data frame *************************************
        forex_m1 = get_ohlc(ativo, mt5.TIMEFRAME_M1, 99999)
        # ****************Transforma em arquivo CSV***********************************
        forex_m1.to_csv('forex.CSV')

        simbolo = bilhetagem(symbol)
        print(simbolo)
        print('#####' * 2, 'Informacao do ativo', '#####' * 2)
        last = simbolo[1]
        print(f'last',
              last)
        ativo = simbolo[2]
        posicao = simbolo[3]
        spread = simbolo[3]
        status(last, ativo, posicao, spread)

        # ***************Seta a planilha dos dados ************************************
        '''Setando as colunas das planilhas'''
        pd.set_option('display.max_columns', 400)  # número de colunas mostradas
        pd.set_option('display.width', 1500)  # max. largura máxima da tabela exibida

        # ***************Busca os dados Historicos************************************
        tabela = pd.read_csv('forex.CSV', index_col=0)
        # tabela.reset_index('time', inplace=True)#reseta index
        # print(tabela)
        # ****************Tabela manipulaco**********************************************
        # tabela = tabela.drop(["open", "high", "low", "spread", "tick_volume", "real_volume"], axis=1)
        '''tabela de manipulacao de medias'''
        tabela.loc[:, 'media7(s)'] = (tabela['close'].rolling(7).mean())
        tabela.loc[:, 'media7(md)'] = (tabela['close'].rolling(7).median())
        tabela.loc[:, 'media7'] = (tabela['close'].ewm(span=7, min_periods=7).mean())  # media exponencial
        tabela.loc[:, 'media21'] = (tabela['close'].ewm(span=21, min_periods=21).mean())
        tabela.loc[:, 'media36'] = (tabela['close'].ewm(span=36, min_periods=36).mean())
        tabela.loc[:, 'momento'] = (tabela['close'] - tabela['close'].rolling(6).mean())
        tabela["OBV"] = (np.sign(tabela["close"].diff()) * tabela["real_volume"]).fillna(0).cumsum()

        tabela = tabela.drop(["high", "low", "spread"], axis=1)
        # "high", "low", "spread", "tick_volume", "real_volume", "media7(s)",'media7(md)', "media7", "media21", "media36"
        #tabela["OBV"] = (np.sign(tabela["close"].diff()) * tabela["real_volume"]).fillna(0).cumsum()

        # ****************Preenche o valor vazio com o valor 0**********************************************
        tabela = tabela.fillna(0)  # preenche o valor vazio com o valor 0]
        # print(f'>', tabela)
        # print(tabela.head())
        # print(tabela.shape)

        # ****************Verifica a correlacao**********************************************
        tabela.corr()
        #****************Grafico correlacao*********************************************
        # sns.heatmap(tabela.corr(), annot=True, vmin=-1, vmax=1, cmap='Reds')
        # sns.plt.show()
        # plt.show()

        # ****************Separa dados de X e y*********************************************
        X = tabela.drop(["close"], axis=1)
        X = X.fillna(0)
        # ****************Grafico correlacao de X *********************************************
        # sns.heatmap(X.corr(), annot=True, vmin=-1, vmax=1, cmap='Reds')
        # plt.show()
        y = tabela['close'].values

        # ****************Separa dados de trino e dados de teste*********************************************
        X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.618, random_state=42)
        # ****************Separa dados de trino e dados de teste*********************************************
        # print(X_teste)

        # para teste(10, 20, 30, 40, 50, 60, 70, 80, 90, 100)
        # ****************Parametros da IA Grid Saerch**********************************************
        # Número de árvores na floresta aleatória
        n_stimators = [int(x) for x in np.linspace(start=1, stop=1000,
                                                   num=1)]  # Numero de Arvores[int(x) for x in np.linspace(start=1, stop=180, num=10)]
        max_samples = [None]
        # Número de feições a serem consideradas em cada divisão
        max_features = ['sqrt',
                        'log2']  # número máximo de recursos (variáveis independentes) a serem considerados ao dividir um nó
        max_features.append(1.0)
        # Número máximo de níveis na árvore
        max_depth = [int(x) for x in np.linspace(10, 200, num=11)]  # 11-[None, 30]
        max_depth.append(None)
        # Número mínimo de amostras necessárias para dividir um nó
        min_samples_split = [int(x) for x in np.linspace(3.5, 200,
                                                         num=11)]  # numero 2 e default--float usa ponto(número mínimo de amostras em um subconjunto (também conhecido como nó) para dividi-lo em mais dois subconjuntos)2, 3, 5, 8, 13, 21, 34, 55, 89, 134
        min_samples_split.append(2)
        # Número mínimo de amostras necessárias em cada nó folha
        min_samples_leaf = [int(x) for x in np.linspace(2, 75.5,
                                                        num=10)]  # '76' 1 = default-float usa ponto 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 134
        min_samples_leaf.append(1)
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
        rf_RandomGrid = RandomizedSearchCV(estimator=floresta,
                                           param_distributions=param_grid,
                                           n_iter=10,
                                           cv=3,
                                           verbose=0,
                                           random_state=42,
                                           n_jobs=-1
                                           )
        # print(RandomizedSearchCV)

        # ****************Treina o modelo com os Parametros da IA Grid Saerch**********************************************
        rf_RandomGrid.fit(X_treino, y_treino)

        # ****************Melhores Parametros da IA Grid Saerch**********************************************
        k = rf_RandomGrid.best_params_
        # print(k["max_depth"])
        rf_RandomGrid.best_score_
        rf_RandomGrid.cv_results_
        df = pd.DataFrame(rf_RandomGrid.cv_results_)
        # print(df)
        # print(rf_RandomGrid.cv_results_)
        y_pred_rf_rg = rf_RandomGrid.predict(X_teste)
        rf_RandomGrid.score(X_teste, y_teste)
        rf_RandomGrid.score(X_teste, y_teste)

        # print(rf_RandomGrid.score(X_teste, y_teste))
        # accuracy_score(y_teste, y_pred_rf_rg)

        rf_RandomGrid.best_estimator_

        # ****************Treino minha IA**********************************************
        floresta = rf_RandomGrid.best_estimator_
        floresta.fit(X_treino, y_treino)

        # ****************Faco minha predicao **********************************************
        p = floresta.predict(X_teste)
        # accuracy_score(p, X_treino)

        # ****************Decisao***********************************************
        fl = (mean_squared_error(y_teste, p) * 1000)
        print(f'floresta--~~->>{fl}')
        fl1 = np.sqrt(mean_squared_error(y_teste, p))
        # *2***************Printa os dados**********************************************
        print('resultado')
        # print('***melhor parametro***')
        # print(rf_RandomGrid.best_params_)
        print('***melhor scor_params***')
        print(f'{rf_RandomGrid.best_score_:.2f},"%"')
        # print('***melhor estimador***')
        # print(rf_RandomGrid.best_estimator_)
        # print('***melhor resultado***')
        # print(rf_RandomGrid.cv_results_)
        print('*$*' * 30)
        # print(f'Treino Accuracy (r) - : {rf_RandomGrid.score(X_treino, y_treino):.3f}')
        # print(f'Teste Accuracy - : {rf_RandomGrid.score(X_teste, y_teste):.3f}')
        # print(f'Teste Accuracy Floresta- : {floresta.score(X_teste, y_teste):.3f}')
        #######################################################################################

    # ****************verificamos a presença de posições abertas*****************************
        positions_total = mt5.positions_total()


        if positions_total > 0:
            print("Total positions=", positions_total)
            print('<<Nao pode Negociar tem posicao aberta')
            if (positions_total):
                beAtivo = False
                print('BE ativo')
                if (positions_total & beAtivo):
                    breakEvan = (last)
                    print('verdade')
        else:
            print("Positions not found")
            print("<<>>Nao tem posicao aberta, pode Negociar!!!")

            # preparamos a solicitação
            # ****************funcao simples compra e venda**********************************************
            def neg(f):
                if (f >= ult_valor):
                    print('Compra ^')
                    volume = 1.0
                    stp = 5000
                    tkp = 10000
                    point = mt5.symbol_info(symbol).point
                    request = {
                        "action": mt5.TRADE_ACTION_DEAL,
                        "symbol": symbol,
                        "volume": 1.0,
                        "price": mt5.symbol_info_tick(symbol).ask,
                        "sl": mt5.symbol_info_tick(symbol).ask - stp * point,
                        "tp": mt5.symbol_info_tick(symbol).ask + tkp * point,
                        "deviation": 10,
                        "spreed": 5,
                        "type": mt5.ORDER_TYPE_BUY,
                        "type_filling": mt5.ORDER_FILLING_RETURN,
                        "comment": "Boa pega no brewwww V@",
                        "magic": 234000,
                    }
                    # enviamos a solicitação de negociação
                    result = mt5.order_check(request)
                    resultOrdby = mt5.order_send(request)
                    print(result);
                    print(resultOrdby);
                    # verificamos o resultado da execução
                    # print("1. order_send(): by {} {} lots at {} with deviation={} points".format(symbol, lot, price,
                    #
                    #                                                                                deviation))

                elif (f <= ult_valor):
                    print('Venda V')
                    volume = 1.0
                    stp = 5000
                    tkp = 10000
                    point = mt5.symbol_info(symbol).point
                    request = {
                        "action": mt5.TRADE_ACTION_DEAL,
                        "symbol": symbol,
                        "volume": 1.0,
                        "price": mt5.symbol_info_tick(symbol).bid,
                        "sl": mt5.symbol_info_tick(symbol).bid + stp * point,
                        "tp": mt5.symbol_info_tick(symbol).bid - tkp * point,
                        "deviation": 10,
                        "spreed": 5,
                        "type": mt5.ORDER_TYPE_SELL,
                        "type_filling": mt5.ORDER_FILLING_RETURN,
                        "comment": "Boa V@",
                        "magic": 234000,
                    }
                    # enviamos a solicitação de negociação
                    result = mt5.order_send(request)
                    print(result)
                    # verificamos o resultado da execução
                    # print("1. order_send(): by {} {} lots at {} with deviation={} points".format(symbol, volume, price,
                    #                                                                                 deviation))
                else:
                    print('Ordem pendente')

            flor = (fl)

            # ****************ultimo valor do tick**********************************************
            ult_valor = last

            # ****************Valor da condicao BUY e SELL**********************************************
            a1 = neg(flor)
            print(a1)

        ticket = mt5.positions_get(symbol=symbol)[0][0]
        TICKET = ticket
        dif = fl - last
        print(f'Ticket da posicao: {str(TICKET)}')
        print(f'fl {fl} - ult_valor {last} =--~~->>{dif} dif. pontos')
        print("controle")


if __name__ == "__main__":
    main()