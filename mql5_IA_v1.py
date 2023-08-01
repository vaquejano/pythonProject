'''
import MetaTrader5 as mt5
import time
mt5.initialize()
TICKET = 1295098706
TRAIL_AMOUNT =0.5 # stop
MAX_DIST_SL = 50.0  #max distancai
DEFAULT_SL = 3.0 # if posicao

def trail_sl(TICKET, MAX_DIST_SL, TRAIL_AMOUNT, DEFAULT_SL):
    #pega a posicao no ticket
    position = mt5.positions_get(ticket=TICKET)[0]
    print(position)
    symbol = position.symbol
    order_type = position.type
    price_current = position.price_current
    price_open = position.price_open
    profit = position.profit
    sl = position.sl
    #calculo distancia para sl
    dist_from_sl = round(profit, 6)
    print('dist_from_sl ', dist_from_sl)
    #calcula o trailing
    if dist_from_sl > MAX_DIST_SL:
        #calculo novo sl
        if sl !=0.0:
            print('1')
            if order_type == 0:
                new_sl = sl + TRAIL_AMOUNT
                print(new_sl)
            elif order_type == 1:
                new_sl = sl - TRAIL_AMOUNT
                print(new_sl)
        else:
            print('2')
            new_sl = price_open - DEFAULT_SL if order_type == 0 else price_open + DEFAULT_SL #seting
            result = trail_sl(TICKET, MAX_DIST_SL, TRAIL_AMOUNT, DEFAULT_SL)
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": TICKET,
            "sl": new_sl,
            "comment": "V@ -~~>> chicote estrala",
            "symbol": symbol,
        }
        # verificamos e exibimos o resultado como está
        result = mt5.order_check(request)
        resultOrd = mt5.order_send(request)
        print(result)
        print(f'res', resultOrd)
        return (result)
    print('fim')

if __name__ == '__main__':
    print('Start trailing Stoploss..')
    print(f'Position: {str(TICKET)}')

    while True:
        result = trail_sl(TICKET, MAX_DIST_SL, TRAIL_AMOUNT, DEFAULT_SL)

        if result:
            print(result)

        time.sleep(1)
'''
'''
spread = 15
custo = 5272.000# inclui todos os custos
c_fix = 2505#custo producao de um produto(unidade)
lucro = 200000#preco de venda do produto(unidade)
def equilibrio(custo, spread, c_fix, lucro):
    p_equilibrio = ((custo+spread) / (lucro - c_fix))# quantidade de equilibrio
    entrada = custo+p_equilibrio
    return entrada
gatilho = equilibrio(custo, spread, c_fix, lucro)
print(f'Ponto Equilibrio->> {gatilho:.3f}\n')
'''
#****************Bibliotecas*************************************
#****************biblioteca matematica*************************************
import math
import pandas as pd
import numpy as np
#****************Modulo sched*************************************
import sched
import time
import threading
#****************Modulo sched*************************************
import MetaTrader5 as mt5
#****************Modulo sched*************************************
import seaborn as sns
import matplotlib.pyplot as plt
#****************Modulo sklearn*************************************
from datetime import datetime
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score

#****************Modulo sched*************************************
import sched
import time
import threading
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
    print(datetime.now())

def main():
    scheduler = sched.scheduler(time.time, time.sleep)
    repeat_at_interval(scheduler, test, interval=60)
    thread = threading.Thread(target=scheduler.run)
    thread.start()
    while True:

        #time.sleep(10)
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
            print(f'ultimo_Preco->> {last:.3f}')
            print(f'spread->> {spread:.3f}')

            #print(f'bid->> {bid:.3f}\n')
# ****************Final da funcoes*************************************

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

        #****************Informacoes do ativo***********************************
        ativo = 'WDO$'
        symbol = 'WDOU23'
        #****************funcao criar data frame *************************************
        wdo_m1 = get_ohlc(ativo, mt5.TIMEFRAME_M1, 99999)
        wdo_m1.to_csv('WDO.CSV')
        #****************Informacao do bilhete *************************************
        simbolo = bilhetagem(symbol)
        point = mt5.symbol_info(symbol).point
        simbolo1 = mt5.symbol_info(symbol).name
        print('symb=>', simbolo1)
        print(point)
        print(f'symboloxx', simbolo)
        print('#####' * 2, 'Informacao do ativo', '#####' * 2)
        last = simbolo[1]
        ativo = simbolo[2]
        posicao = simbolo[3]
        spread = simbolo[3]
        status(last, ativo, posicao, spread)

        #***************Seta a planilha dos dados ************************************
        '''Setando as colunas das planilhas'''
        pd.set_option('display.max_columns', 400)  # número de colunas mostradas
        pd.set_option('display.width', 1500)  # max. largura máxima da tabela exibida

        #***************Busca os dados Historicos************************************
        tabela = pd.read_csv('WDO.CSV', index_col=0)
        # tabela.reset_index('time', inplace=True)#reseta index
        #print(tabela)
        #****************Tabela manipulaco**********************************************
        # tabela = tabela.drop(["open", "high", "low", "spread", "tick_volume", "real_volume"], axis=1)
        # Calculando o Estocástico Rápido
        #tabela = tabela.drop(["open","spread", "tick_volume", "real_volume"], axis=1)
        #print(tabela)
        '''tabela de manipulacao de medias'''
        tabela.loc[:, 'media(s)'] = tabela['close'].rolling(7).mean()
        tabela.loc[:, 'media(md)'] = tabela['close'].rolling(21).median()
        tabela.loc[:, 'media7'] = tabela['close'].ewm(span=7, min_periods=7).mean()  # media exponencial
        tabela.loc[:, 'media20'] = tabela['close'].ewm(span=21, min_periods=21).mean()
        tabela.loc[:, 'media36'] = tabela['close'].ewm(span=36, min_periods=36).mean()
        tabela.loc[:, 'sma_50'] = tabela["close"].rolling(window=50).mean()
        tabela.loc[:, 'momento'] = tabela['close'] - tabela['close'].rolling(20).mean()
        tabela.loc[:, 'OBV'] = (np.sign(tabela["close"].diff()) * tabela["real_volume"]).fillna(0).cumsum()
        tabela.loc[:, 'rsi_c'] = 100 - 100 / (1 + (tabela["close"].diff().fillna(0).rolling(window=7).mean() /
                                                 tabela["close"].diff().fillna(0).rolling(window=7).std()))
        tabela.loc[:, 'rsi_m'] = 100 - 100 / (1 + (tabela["close"].diff().fillna(0).rolling(window=14).mean() /
                                                 tabela["close"].diff().fillna(0).rolling(window=14).std()))
        tabela.loc[:, 'rsi_l'] = 100 - 100 / (1 + (tabela["close"].diff().fillna(0).rolling(window=21).mean() /
                                                 tabela["close"].diff().fillna(0).rolling(window=21).std()))
        # Calculando o Estocástico Rápido
        tabela.loc[:, 'n_highest_high'] = tabela["high"].rolling(21).max()
        tabela.loc[:, 'n_lowest_low'] = tabela["low"].rolling(21).min()
        tabela.loc[:, '%K'] = (tabela["close"] - tabela["n_lowest_low"]) / (
                tabela["n_highest_high"] - tabela["n_lowest_low"]) * 100
        tabela["%D"] = tabela['%K'].rolling(21).mean()
        tabela.dropna(inplace=True)
        tabela["Slow %K"] = (tabela["%D"])
        tabela["Slow %D"] = (tabela["Slow %K"].rolling(21).mean()) * 100
        # Calcular indicadores de análise técnica
        tabela["returns"] = (tabela["close"] - tabela["open"]) / tabela["open"]
        tabela["ma20"] = tabela["close"].rolling(window=20).mean()
        tabela["upper_band"] = tabela["ma20"] + 2 * tabela["close"].rolling(window=20).std()
        tabela["lower_band"] = tabela["ma20"] - 2 * tabela["close"].rolling(window=20).std()
        # Análise de entrada de compra
        tabela["buy_signal"] = 0
        tabela.loc[(tabela["returns"] > 0) & (tabela["close"] > tabela["ma20"]), "buy_signal"] = 1
        # Análise de saída de venda
        tabela["sell_signal"] = 0
        tabela.loc[(tabela["returns"] < 0) & (tabela["close"] < tabela["ma20"]), "sell_signal"] = 1
        # Análise de stop loss
        tabela["stop_loss"] = tabela["ma20"] - tabela["returns"].abs().rolling(window=20).mean()
        # Análise de break even
        tabela["break_even"] = tabela["open"] + tabela["returns"].abs().rolling(window=20).mean()
        # Análise de trailing stop
        tabela["trailing_stop"] = tabela["close"].rolling(window=20).max() - tabela["returns"].abs().rolling(
            window=20).mean()
        tabela.dropna(inplace=True)
        tabela.head()
        tabela = tabela.drop(["spread", "tick_volume", "n_highest_high", "n_lowest_low", ], axis=1)
        #print(tabela)
        #print(tabela.shape)
        '''Verifica a correlacao'''
        #        tabela.corr()
        '''Grafico correlacao '''
        #sns.heatmap(tabela.corr(), annot=True, vmin=-1, vmax=1, cmap='Reds')
        #plt.show()
        '''Separa dados de X e y '''
        X = tabela.drop(columns=["close"], axis=1).fillna(0)
        y = tabela['close'].values
        #****************Grafico correlacao de X *********************************************
        # sns.heatmap(X.corr(), annot=True, vmin=-1, vmax=1, cmap='Reds')
        # plt.show()
        '''Separa dados de trino e teste '''
        X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.618, random_state=42)
        '''Parametros da IA Grid Saerch '''
        '''Número de árvores na floresta aleatória '''
        n_stimators = [int(x) for x in np.linspace(start=1, stop=1000,
                                                   num=1)]#'''Numero de Arvores[int(x) for x in np.linspace(start=1, stop=180, num=10)]'''
        max_samples = [None]
        '''Número de feições a serem consideradas em cada divisão'''
        max_features = ['sqrt',
                        'log2']  # número máximo de recursos (variáveis independentes) a serem considerados ao dividir um nó
        max_features.append(1.0)
        '''Número máximo de níveis na árvore'''
        max_depth = [int(x) for x in np.linspace(20, 1000, num=2)]  # 11-[None, 30] 25
        max_depth.append(None)
        '''Número mínimo de amostras necessárias para dividir um nó'''
        min_samples_split = [int(x) for x in np.linspace(2.0, 450, num=1)]  # numero 2 e default--float usa ponto(número mínimo de amostras em um subconjunto (também conhecido como nó) para dividi-lo em mais dois subconjuntos)2, 3, 5, 8, 13, 21, 34, 55, 89, 134
        min_samples_split.append(2)
        '''Número mínimo de amostras necessárias em cada nó folha'''
        min_samples_leaf = [int(x) for x in np.linspace(2.0, 150, num=1)]  # 8 '76' 1 = default-float usa ponto 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 134
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
        p = floresta.predict(X_teste)
        fl = p[-1]
        dif = (last - fl)
        print('last->', last)
        print('fl->', fl)
        print('dif->', dif, 'last', '-', 'fl')
               #****************Printa os dados**********************************************
        print("*" * 100)
        print('***melhor scor_params***')
        print(f'{rf_RandomGrid.best_score_:.2f},"%"')
        print('*$*' * 30)
        #****************verificamos a presença de posições abertas*****************************
        positions_total = mt5.positions_total()
        if positions_total > 0:
            print("Total positions=", positions_total)
            print('<<Nao pode Negociar tem posicao aberta>>')
            bilhete = mt5.positions_get(symbol=symbol)[0]
            ticket = mt5.positions_get(symbol=symbol)[0][0]
            print(bilhete)
            bilhete_neg = bilhete.ticket
            price_open = bilhete.price_open
            sl = bilhete.sl
            profit = bilhete.profit
            type = bilhete.type
            print("t", type)
            price_open = bilhete.price_open
            if (type == 0):
                if (profit == 30.0):
                    print('BE ativo')
                    gatilho = 0.5
                    def BreackEvan(gatilho, price_open):
                        print('gatilho acionado')
                        disparo = price_open + gatilho
                        print(f'venda=', disparo)
                        request = {'action': mt5.TRADE_ACTION_SLTP,
                                   'position': ticket,
                                   'sl': disparo,
                                   "symbol": symbol}
                        resultOrd = mt5.order_send(request)
                        result = mt5.order_check(request)
                        print(result)
                        print(resultOrd)
                    BreackEvan(gatilho, price_open)
                    print('breack vaquero')
                elif profit >= 31.0:
                    print('traling Stop v@quejano')
                    TICKET = ticket
                    MAX_DIST_SL = 2.0  # max distancai
                    TRAIL_AMOUNT = 0.5  # amaont
                    def trail_sl(TICKET, MAX_DIST_SL, TRAIL_AMOUNT):
                        # pega a posicao no ticket
                        position = mt5.positions_get(ticket=TICKET)[0]
                        print(position)
                        symbol = position.symbol
                        order_type = position.type
                        price_current = position.price_current
                        price_open = position.price_open
                        sl = position.sl
                        proft = position.profit
                        dist_from_sl = round(sl + price_current, 6)  # proft
                        print('dist_from_sl ', dist_from_sl)
                        # Calcula o trailing
                        if dist_from_sl >= MAX_DIST_SL:
                            print('sobe')
                            new_sl = sl + TRAIL_AMOUNT
                            print(f'sell', new_sl)
                            request = {
                                "action": mt5.TRADE_ACTION_SLTP,
                                "symbol": symbol,
                                "position": TICKET,
                                "sl": new_sl,
                                "comment": "V@-~~~~~>>chicote estrala",
                            }
                            # verificamos e exibimos o resultado como está
                            result = mt5.order_check(request)
                            resultOrd = mt5.order_send(request)
                            print(result)
                            print(resultOrd)
                            return (result)
                    trail_sl(TICKET, MAX_DIST_SL, TRAIL_AMOUNT)
                    print('traling Stop v@quejano')
                elif profit < -10:
                    print('calanguiada')
                    offset = 50
            else:
                if (type == 1):
                    if (profit == 30.0 ):
                        print('BE ativo')
                        gatilho = 0.5
                        def BreackEvan(gatilho, price_open):
                            print('gatilho acionado')
                            disparo = price_open - gatilho
                            print(f'venda=', disparo)
                            request = {'action': mt5.TRADE_ACTION_SLTP, 'position': ticket, 'sl': disparo,
                                        "symbol": symbol}
                            result = mt5.order_check(request)
                            resultOrd = mt5.order_send(request)
                            print(result)
                            print(resultOrd)
                        BreackEvan(gatilho, price_open)
                    elif profit >=31.0:
                        print('traling Stop v@quejano')
                        TICKET = ticket
                        MAX_DIST_SL = 2.0  # max distancai
                        TRAIL_AMOUNT = 0.5  # amaont
                        def trail_sl(TICKET, MAX_DIST_SL, TRAIL_AMOUNT):
                            # pega a posicao no ticket
                            position = mt5.positions_get(ticket=TICKET)[0]
                            print(position)
                            symbol = position.symbol
                            order_type = position.type
                            price_current = position.price_current
                            price_open = position.price_open
                            sl = position.sl
                            proft = position.profit
                            dist_from_sl = round(sl - price_current, 6)  # proft
                            print('dist_from_sl ', dist_from_sl)
                            # Calcula o trailing
                            if dist_from_sl >= MAX_DIST_SL:
                                print('desce')
                                new_sl = sl - TRAIL_AMOUNT
                                print(f'sell', new_sl)
                                request = {
                                    "action": mt5.TRADE_ACTION_SLTP,
                                    "symbol": symbol,
                                    "position": TICKET,
                                    "sl": new_sl,
                                    "comment": "V@-~~~~~>>chicote estrala",
                                }
                                # verificamos e exibimos o resultado como está
                                result = mt5.order_check(request)
                                resultOrd = mt5.order_send(request)
                                print(result)
                                print(resultOrd)
                                return (result)
                        trail_sl(TICKET, MAX_DIST_SL, TRAIL_AMOUNT)
                    elif profit < -5.0:
                        print('calanguiada')
                        '''parametro da posicao'''

        else:
            print("<<>>Nao tem posicao aberta, pode Negociar!!!")
            #Preparamos a solicitação
        #****************funcao simples compra e venda**********************************************
            def neg(fl):
                print('-->',fl)
                if (fl >= last):
                    print('Compra ^')
                    volume = 1.0
                    stp = 10000
                    tkp = 20000
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
                        "magic": 171,
                    }
                    # enviamos a solicitação de negociação
                    result = mt5.order_check(request)
                    resultOrdby = mt5.order_send(request)
                    print(result)
                    print(resultOrdby)
                elif (fl <= last):
                    print('Venda V')
                    volume = 1.0
                    stp = 10000
                    tkp = 20000
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
                        "magic": 171,
                    }
                    # Enviamos a solicitação de negociação
                    result = mt5.order_send(request)
                    print(result)
                else:
                   print('Ordem pendente')
            # ****************Valor da condicao BUY e SELL**********************************************
            a1=neg(fl)
            #Envio_ordem_C_V.neg(fl)
            print(a1)
            # ****************Ultimo valor do tick**********************************************
            ult_valor = last
            print("V@quejano IA, aqui o sistema e sertanejo...")
            print("V@quejada IA valeu o Boiii..."
                  " Se voce tem um sonho nao deixe ninguem deboxar dele nao..."
                  "Seu sonho e grande..."
                  " Foi Deus quem botou no seu Coração..."
                  "Meu Deus eu confio no sonho que o Senhor tem p minha vida..."
                  " Levanta cedo pra labuta que eu to pronto...By: Joao Gomes")
            time.sleep(10)

if __name__ == "__main__":
    main()




