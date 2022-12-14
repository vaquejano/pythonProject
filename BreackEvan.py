__name__ == '__BreackEvan__'
#****************Bibliotecas*************************************
import MetaTrader5
import MetaTrader5 as mt5
import time
import threading

mt5.initialize()
symbol = "WDOV22"
bilhete = mt5.positions_get(symbol=symbol)
ticket = mt5.positions_get(symbol=symbol)[0][0]
posicao = mt5.positions_get(symbol=symbol)[0][5]
price_open = mt5.positions_get(symbol=symbol)[0][10]
print(bilhete)
print(f'ticket {ticket}, posicao {posicao}, price_open {price_open},')

#****************Bibliotecas*************************************
gatilho = 0.5

def BreackEvan(posicao, gatilho):
    while price_open>gatilho:
        if posicao == 0:
            g_t = 1
            print('gatilho acionado')
            disparo = price_open + gatilho
            print(f'compra=', disparo)
            request = {'action': mt5.TRADE_ACTION_SLTP, 'position': ticket, 'sl': disparo, "symbol": symbol}
            result = mt5.order_send(request)
            print(result)
            return
        else:
            print('gatilho acionado')
            disparo = price_open - gatilho
            print(f'venda=', disparo)
            g_t = 1
            request = {'action': mt5.TRADE_ACTION_SLTP, 'position': ticket, 'sl': disparo, "symbol": symbol}
            result = mt5.order_send(request)
            print(result)
            return
        time.sleep(2)
        digits = mt5.symbol_info(symbol)
TICKET = ticket
MAX_DIST_SL = 2.7  # max distancai
TRAIL_AMOUNT = 0.5  # amaont
DEFAULT_SL = 1.01  # if posicao


def trail_sl(TICKET, MAX_DIST_SL, TRAIL_AMOUNT, DEFAULT_SL):
    # pega a posicao no ticket
    position = mt5.positions_get(ticket=TICKET)[0]
    print(position)
    symbol = position.symbol
    order_type = position.type
    price_current = position.price_current
    price_open = position.price_open
    sl = position.sl
    proft = position.profit

    # calculo distancia para sl
    dist_from_sl = round(sl - price_current, 6) #proft
    print('dist_from_sl ', dist_from_sl)

    # calcula o trailing
    if dist_from_sl >= MAX_DIST_SL:
        # calculo novo sl
        if sl != 0.0:
            print('1 order')
            if order_type == 0:
                print('pos')
                new_sl = sl + TRAIL_AMOUNT
                print(f'buy',new_sl)
            elif order_type == 1:
                new_sl = sl - TRAIL_AMOUNT
                print(f'sell',new_sl)

        else:
            print(2)
            if order_type == 0:
                new_sl = price_open + DEFAULT_SL if order_type == 0 else price_open + DEFAULT_SL  # seting
                result = trail_sl(TICKET, MAX_DIST_SL, TRAIL_AMOUNT, DEFAULT_SL)
            else:
                new_sl = price_open - DEFAULT_SL if order_type == 1 else price_open - DEFAULT_SL  # seting
                result = trail_sl(TICKET, MAX_DIST_SL, TRAIL_AMOUNT, DEFAULT_SL)

        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "symbol": symbol,
            "position": TICKET,
            "sl": new_sl,
            "comment": "V@ -~~~~~>> chicote estrala",
        }
        # verificamos e exibimos o resultado como est??
        result = mt5.order_check(request)
        resultOrd = mt5.order_send(request)
        print(result);
        # print(resultOrd);
        return (result)

#if __name__ == '__main__':
#    print('Start trailing Stoploss..')
#    print(f'Position: {str(TICKET)}')

#    while True:
#        result = trail_sl(TICKET, MAX_DIST_SL, TRAIL_AMOUNT, DEFAULT_SL)

#        if result:
#            print(result)
#            # print(resultOrd);

#        time.sleep(1)
        
        
        
        
t1 = threading.Thread(target=BreackEvan, args=(posicao, gatilho))
t2 = threading.Thread(target=BreackEvan, args=(posicao, gatilho))
t1.start()
t2.start()
t1.join()
t2.join()








'''
def BreackEvan(posicao, gatilho):
    if gatilho == price_open:
        print('gatilho acionado')
        if posicao == 0:
            disparo = price_open + gatilho
            request = {'action': mt5.TRADE_ACTION_SLTP, 'position': ticket, 'sl': disparo, "symbol": symbol}
            result = mt5.order_send(request)
            print(result)


        else:
            disparo = price_open - gatilho
            request = {'action': mt5.TRADE_ACTION_SLTP, 'position': ticket, 'sl': disparo, "symbol": symbol}
            result = mt5.order_send(request)
            print(result)
    else:
        print('fim BreackEvan')

BreackEvan(posicao, gatilho)
if __name__ == '__main__':
    print('Start BreackEvan..')
    #print(f'Position: {str(TICKET)}')

    while price_open != gatilho:
        result = BreackEvan(posicao, gatilho)

        if result:
            print(result)

        time.sleep(1)
'''
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
        # verificamos e exibimos o resultado como est??
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
'''
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

            # mt5.shutdown()
            print('******************')
            print('*   conectado    *')
            print('******************')

            # ****************Data atual**************************************************
            # hoje = datetime.now()
            # print(hoje)    #****************Data atual**************************************************
            hoje = datetime.now()
            print(hoje)

            # ****************Informacoes do ativo***********************************
            ativo = 'WDO$'
            symbol = "WDOV22"

            # ****************funcao criar data frame *************************************
            wdo_m1 = get_ohlc(ativo, mt5.TIMEFRAME_M1, 99999)
            wdo_m1.to_csv('WDO.CSV')

            simbolo = bilhetagem(symbol)
            print('#####' * 2, 'Informacao do ativo', '#####' * 2)
            last = simbolo[1]
            ativo = simbolo[2]
            posicao = simbolo[3]
            status(last, ativo, posicao)

            # ***************Busca dos dados ************************************
            # Setando as colunas das planilhas
            pd.set_option('display.max_columns', 400)  # n??mero de colunas mostradas
            pd.set_option('display.width', 1500)  # max. largura m??xima da tabela exibida

            # ***************Busca os dados Historicos************************************
            tabela = pd.read_csv('WDO.CSV', index_col=0)
            # tabela.reset_index('time', inplace=True)#reseta index
            # print(tabela)
            # ****************Tabela manipulaco**********************************************
            # tabela = tabela.drop(["open", "high", "low", "spread", "tick_volume", "real_volume"], axis=1)

            tabela.loc[:, 'media7(s)'] = (tabela['close'].rolling(7).mean())
            tabela.loc[:, 'media7(md)'] = (tabela['close'].rolling(7).median())
            tabela.loc[:, 'media7'] = (tabela['close'].ewm(span=7, min_periods=7).mean())  # media exponencial
            tabela.loc[:, 'media21'] = (tabela['close'].ewm(span=21, min_periods=21).mean())
            tabela.loc[:, 'media36'] = (tabela['close'].ewm(span=36, min_periods=36).mean())
            tabela.loc[:, 'momento'] = (tabela['close'] - tabela['close'].rolling(6).mean())

            tabela = tabela.drop(["high", "low", "spread"], axis=1)
            # "high", "low", "spread", "tick_volume", "real_volume", "media7(s)",'media7(md)', "media7", "media21", "media36"
            tabela["OBV"] = (np.sign(tabela["close"].diff()) * tabela["real_volume"]).fillna(0).cumsum()

            # ****************Preenche o valor vazio com o valor 0**********************************************
            tabela = tabela.fillna(0)  # preenche o valor vazio com o valor 0]
            # print(f'>', tabela)
            # print(tabela.head())
            # print(tabela.shape)

            # ****************Verifica a correlacao**********************************************
            tabela.corr()
            # sns.heatmap(tabela.corr(), annot=True, vmin=-1, vmax=1, cmap='Reds')
            # sns.plt.show()
            # plt.show()

            # ****************Separa dados de X e y*********************************************
            X = tabela.drop(["close"], axis=1)
            X = X.fillna(0)
            # sns.heatmap(X.corr(), annot=True, vmin=-1, vmax=1, cmap='Reds')
            # plt.show()
            y = tabela['close'].values
            # print(X[-25:])
            # print(y)
            # ****************Separa dados de trino e dados de teste*********************************************
            X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.618, random_state=42)
            # ****************Separa dados de trino e dados de teste*********************************************
            # print(X_teste)

            # para teste(10, 20, 30, 40, 50, 60, 70, 80, 90, 100)
            # ****************Parametros da IA Grid Saerch**********************************************
            # N??mero de ??rvores na floresta aleat??ria
            n_stimators = [int(x) for x in np.linspace(start=1, stop=1000,
                                                       num=1)]  # Numero de Arvores[int(x) for x in np.linspace(start=1, stop=180, num=10)]
            max_samples = [None]
            # N??mero de fei????es a serem consideradas em cada divis??o
            max_features = ['sqrt',
                            'log2']  # n??mero m??ximo de recursos (vari??veis independentes) a serem considerados ao dividir um n??
            max_features.append(1.0)
            # N??mero m??ximo de n??veis na ??rvore
            max_depth = [int(x) for x in np.linspace(10, 100, num=10)]  # 11-[None, 30]
            max_depth.append(None)
            # N??mero m??nimo de amostras necess??rias para dividir um n??
            min_samples_split = [int(x) for x in np.linspace(3, 200,
                                                             num=11)]  # numero 2 e default--float usa ponto(n??mero m??nimo de amostras em um subconjunto (tamb??m conhecido como n??) para dividi-lo em mais dois subconjuntos)2, 3, 5, 8, 13, 21, 34, 55, 89, 134
            min_samples_split.append(2)
            # N??mero m??nimo de amostras necess??rias em cada n?? folha
            min_samples_leaf = [int(x) for x in np.linspace(2, 75.5,
                                                            num=10)]  # '76' 1 = default-float usa ponto 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 134
            min_samples_leaf.append(1)
            # M??todo de sele????o de amostras para treinamento de cada ??rvore
            bootstrap = [True, False]
            random_state = [42]
            n_jobs = [-1]

            param_grid = {
                'n_estimators': n_stimators,  # n??mero de ??rvores no foreset
                'max_samples': max_samples,
                'max_features': max_features,  # n??mero m??ximo de recursos considerados para dividir um n??
                'max_depth': max_depth,  # n??mero m??ximo de n??veis em cada ??rvore de decis??o
                'min_samples_split': min_samples_split,
                # n??mero m??nimo de pontos de dados colocados em um n?? antes que o n?? seja dividido
                'min_samples_leaf': min_samples_leaf,  # n??mero m??nimo de pontos de dados permitidos em um n?? folha
                'bootstrap': bootstrap,  # m??todo para amostragem de pontos de dados (com ou sem reposi????o)
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
            print(f'floresta--~~->>{fl:.3f}')
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

        # ****************verificamos a presen??a de posi????es abertas*****************************
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

            # preparamos a solicita????o
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
                    # enviamos a solicita????o de negocia????o
                    result = mt5.order_check(request)
                    resultOrdby = mt5.order_send(request)
                    print(result);
                    print(resultOrdby);
                    # verificamos o resultado da execu????o
                    # print("1. order_send(): by {} {} lots at {} with deviation={} points".format(symbol, lot, price,
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
                    # enviamos a solicita????o de negocia????o
                    result = mt5.order_send(request)
                    print(result)
                    # verificamos o resultado da execu????o
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
        print(TICKET)
        print(f'fl {fl:.3f} - ult_valor {last:.3f}--~~->>{dif:.2f} pontos')
        print("controle")

if __name__ == "__main__":
    main()
'''

'''    
    # This is what is important to you!
    # if(type == mt5.ORDER_TYPE_BUY):
    #    order_type = mt5.ORDER_TYPE_SELL
    #    price = mt5.symbol_info_tick(symbol).bid
    # else:
    #    order_type = mt5.ORDER_TYPE_BUY
    #    price = mt5.symbol_info_tick(symbol).ask
    # importance ends here.

    # sltp_request = {
    #    "action": mt5.TRADE_ACTION_SLTP,
    #    "symbol": symbol,
    #    "volume": float(volume),
    #    "type": order_type,
    #    "position": deal_id,
    #    "sl": sl,
    #    "price": price,
    #    "magic": 234000,
    #    "comment": "Change stop loss",
    #    "type_time": mt5.ORDER_TIME_GTC,
    #   "type_filling": mt5.ORDER_FILLING_IOC,
    # }

    # result = mt5.order_send(sltp_request)
    '''
'''
    request = {'action': mt5.TRADE_ACTION_SLTP, 'position': posicao[0][6], 'sl': 0.99700 }
    result = mt5.order_send(request)
    print(result)
'''
