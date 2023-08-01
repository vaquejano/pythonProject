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

import MetaTrader5 as mt5
import time
'''
mt5.initialize()
#Input
TRAIL_AMOUNT =0.5 # stop
MAX_DIST_SL = 50.0  #max distancai
DEFAULT_SL = 3.0 # if posicao

symbol = "WDOX22"

def on_init(s):
    bilhete = mt5.positions_get(symbol=symbol)[0]
    bilhete.symbol
    bilhete.ticket
    bilhete.type
    bilhete.magic
    bilhete.price_open
    bilhete.price_current
    bilhete.sl
    bilhete.profit
    return(bilhete)
res_bilhete = on_init(symbol)

def info_tick(sybol):
    ultimo_tick = mt5.symbol_info_tick(symbol)
    print(ultimo_tick)
    ask = ultimo_tick.ask
    print(ask)
    bid = ultimo_tick.bid
    print(bid)
    ultimoPreco = ultimo_tick.last
    return(ultimoPreco)

ultimoTick = info_tick(symbol)
print(ultimoTick)
mt5.ra
from Trailingstop_1 import trail_sl

#mt5.initialize()
#symbol = "WDOX22"
#bilhete = mt5.positions_get(symbol=symbol)
#ticket = mt5.positions_get(symbol=symbol)[0][0]
#posicao = mt5.positions_get(symbol=symbol)[0][5]
#price_open = mt5.positions_get(symbol=symbol)[0][10]
#price_current = mt5.positions_get(symbol=symbol)[0][13]
#print(bilhete)
#print(f'ticket {ticket}, posicao {posicao}, price_open {price_open},')
'''

#****************Bibliotecas*************************************
#****************biblioteca matematica*************************************
import math
#import BreackEvan
import trallingstop
#import Envio_ordem_C_V
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
#import trallingstop
#from BKE_threading import BreackEvan
#from trallingstop import trallingstop

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
            spread = symbol_info.name
            return symbol_info, last, ativo, posicao, spread

        def status(last, ativo, posicao, spread):
            print(f'Ativo-------->> {ativo}')
            print(f'qtde_Ifo----->> {posicao}')
            print(f'ultimo_Preco->> {last:.3f}')
            print(f'spread11->> {spread}')
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
        symbol = "WDOZ22"

        #****************funcao criar data frame *************************************
        wdo_m1 = get_ohlc(ativo, mt5.TIMEFRAME_M1, 99999)
        wdo_m1.to_csv('WDO.CSV')

        #****************Informacao do bilhete *************************************
        #simbolo = bilhetagem(symbol)
        #print(f'symbolo', simbolo)
       # print('#####' * 2, 'Informacao do ativo', '#####' * 2)
       # last = simbolo[0]
       # ativo = simbolo[2]
       # posicao = simbolo[3]
       # spread = simbolo[3]
       # status(last, ativo, posicao, spread)

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

        tabela.loc[:, 'media7(s)'] = (tabela['close'].rolling(7).mean())
        tabela.loc[:, 'media7(md)'] = (tabela['close'].rolling(7).median())
        tabela.loc[:, 'media7'] = (tabela['close'].ewm(span=7, min_periods=7).mean())  # media exponencial
        tabela.loc[:, 'media21'] = (tabela['close'].ewm(span=21, min_periods=21).mean())
        tabela.loc[:, 'media36'] = (tabela['close'].ewm(span=36, min_periods=36).mean())
        tabela.loc[:, 'momento'] = (tabela['close'] - tabela['close'].rolling(6).mean())
        tabela["OBV"] = ((np.sign(tabela["close"].diff()) * tabela["real_volume"]-1)/1000).fillna(0).cumsum()
        #Calculando o Estocástico Rápido
        tabela.loc[:, 'n_highest_high'] = tabela["high"].rolling(8).max()
        tabela.loc[:, 'n_lowest_low'] = tabela["low"].rolling(8).min()
        tabela.loc[:, '%K'] = (tabela["close"] - tabela["n_lowest_low"]) / (
                tabela["n_highest_high"] - tabela["n_lowest_low"]) * 100
        tabela["%D"] = tabela['%K'].rolling(3).mean()
        tabela.dropna(inplace=True)
        tabela["Slow %K"] = tabela["%D"]
        tabela["Slow %D"] = tabela["Slow %K"].rolling(3).mean()
        tabela.dropna(inplace=True)
        tabela.head()
        tabela = tabela.drop(["spread", "tick_volume", "n_highest_high", "n_lowest_low", "Slow %K"], axis=1)
        # "high", "low", "spread", "tick_volume", "real_volume", "media7(s)",'media7(md)', "media7", "media21", "media36"
        #tabela["OBV"] = (np.sign(tabela["close"].diff()) * tabela["real_volume"]).fillna(0).cumsum()
        #****************Preenche o valor vazio com o valor 0**********************************************
        tabela = tabela.fillna(0)  # preenche o valor vazio com o valor 0]
        #print(f'>', tabela.iloc[-30:])
        # print(tabela.head())
        # print(tabela.shape)
        # ****************Verifica a correlacao**********************************************
        tabela.corr()
        #****************Grafico correlacao*********************************************
        #sns.heatmap(tabela.corr(), annot=True, vmin=-1, vmax=1, cmap='Reds')
        #plt.show()

        #****************Separa dados de X e y*********************************************
        X = tabela.drop(["close"], axis=1)
        X = X.fillna(0)
        #****************Grafico correlacao de X *********************************************
        # sns.heatmap(X.corr(), annot=True, vmin=-1, vmax=1, cmap='Reds')
        # plt.show()
        y = tabela['close'].values

        #****************Separa dados de trino e dados de teste*********************************************
        X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.618, random_state=42)
        # ****************Separa dados de trino e dados de teste*********************************************
        # print(X_teste)

        # para teste(10, 20, 30, 40, 50, 60, 70, 80, 90, 100)
        #****************Parametros da IA Grid Saerch**********************************************
        # Número de árvores na floresta aleatória
        n_stimators = [int(x) for x in np.linspace(start=1, stop=1000,
                                                   num=1)]  # Numero de Arvores[int(x) for x in np.linspace(start=1, stop=180, num=10)]
        max_samples = [None]
        # Número de feições a serem consideradas em cada divisão
        max_features = ['sqrt',
                        'log2']  # número máximo de recursos (variáveis independentes) a serem considerados ao dividir um nó
        max_features.append(1.0)
        # Número máximo de níveis na árvore
        max_depth = [int(x) for x in np.linspace(10, 200, num=11)]  # 11-[None, 30] 25
        max_depth.append(None)
        # Número mínimo de amostras necessárias para dividir um nó
        min_samples_split = [int(x) for x in np.linspace(1.0, 200.00, num=50)]  # numero 2 e default--float usa ponto(número mínimo de amostras em um subconjunto (também conhecido como nó) para dividi-lo em mais dois subconjuntos)2, 3, 5, 8, 13, 21, 34, 55, 89, 134
        min_samples_split.append(2)
        # Número mínimo de amostras necessárias em cada nó folha
        min_samples_leaf = [int(x) for x in np.linspace(2.0, 200.0, num=8)]  # 8 '76' 1 = default-float usa ponto 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 134
        min_samples_leaf.append(1)
        # Método de seleção de amostras para treinamento de cada árvore
        bootstrap = [True]
        oob_score = [False]
        #oob_score.append(bool)
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
            'oob_score': oob_score, #estimar a pontuação de generalização
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
        #****************Treina o modelo com os Parametros da IA Grid Saerch**********************************************
        rf_RandomGrid.fit(X_treino, y_treino)

        #****************Melhores Parametros da IA Grid Saerch**********************************************
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
        #print(rf_RandomGrid.score(X_teste, y_teste))
        #accuracy_score(y_teste, y_pred_rf_rg)
        rf_RandomGrid.best_estimator_

        #****************Treino minha IA**********************************************
        floresta = rf_RandomGrid.best_estimator_
        floresta.fit(X_treino, y_treino)

        #****************Faco minha predicao **********************************************
        p = floresta.predict(X_teste)
        # accuracy_score(p, X_treino)

        #****************Decisao***********************************************
        fl = (mean_squared_error(y_teste, p) * 1000)
        print(f'floresta--~~->>{fl:.3f}')
        dif = (last - fl)/10
        # print(f'Ticket da posicao: {str(ticket)}')
        print(f'fl {fl:.3f} - ult_valor {last:.3f} =--~~->>{dif:.2f} dif. pontos')
        #fl1 = np.sqrt(mean_squared_error(y_teste, p))
        #print(f'floresta_Quadrada--~~->>{fl1:.3f}')
        #****************Printa os dados**********************************************
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
'''
    #****************verificamos a presença de posições abertas*****************************
        positions_total = mt5.positions_total()
        if positions_total > 0:
            print("Total positions=", positions_total)
            print('<<Nao pode Negociar tem posicao aberta')
            bilhete = mt5.positions_get(symbol=symbol)[0]
            print(bilhete)
            bilhete_neg = bilhete.ticket
            price_open = bilhete.price_open
            sl = bilhete.sl
            profit = bilhete.profit
            type = bilhete.type
            print("t", type)
            #print(bilhete.ticket)
            price_open = bilhete.price_open
            #print(bilhete.price_open)
            if (type == 0):
                print('posição negativo')
                if (profit <=2):
                    print('BE ativo')
                    print('breack vaquero')
                    print('Price breack', price_open+0.1)
                    trallingstop
                elif (profit >= 10 ):
                    print('Traling ativo')
                    print('traling Stop v@quejano')
                    print('Price traling', price_open - 2.6)

            else:
                if (type == 1):
                    if (profit == 2):
                        print('BE ativo')
                        print('breack vaquero')
                        print('Price breack', price_open-2.0)
                        trallingstop
                    elif (profit >= 10 ):
                            print('Traling ativo')
                            print('traling Stop v@quejano')
                            print('Price traling', price_open - 2.6)
        else:
            print("<<>>Nao tem posicao aberta, pode Negociar!!!")
            #Preparamos a solicitação
        #****************funcao simples compra e venda**********************************************
            def neg(fl):
                print(fl)
                if (fl >= last):
                    print('Compra ^')
                    volume = 1.0
                    stp = 8500
                    tkp = 15000
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
                    print(result);
                    print(resultOrdby);
                elif (fl <= last):
                    print('Venda V')
                    volume = 1.0
                    stp = 8500
                    tkp = 15000
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
'''
if __name__ == "__main__":
    main()

'''
__name__ == '__trailingstop__'
import MetaTrader5 as mt5
import time
mt5.initialize()
symbol = "WDOH23"
ticket = mt5.positions_get(symbol=symbol)[0][0]
print(ticket)
digits = mt5.symbol_info(symbol)
TICKET = ticket
MAX_DIST_SL = 5.0  # max distancai
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
    if order_type == 0:
        dist_from_sl = round(price_current-sl, 6)  # proft
        print('dist_from_sl ', dist_from_sl)
        # Calcula o trailing
        if dist_from_sl >= MAX_DIST_SL:
            print('sobe')
            new_sl = sl + TRAIL_AMOUNT
            print(f'Buy', new_sl)
            request = {
                "action": mt5.TRADE_ACTION_SLTP,
                "symbol": symbol,
                "position": TICKET,
                "sl": new_sl,
                "comment":'^'"V@-~~~~~>>chicote estrala",
                }
            # verificamos e exibimos o resultado como está
            result = mt5.order_check(request)
            resultOrd = mt5.order_send(request)
            print(result)
            print(resultOrd)
            return (result)
    elif order_type == 1:
        dist_from_sl = round(sl-price_current, 6)  # proft
        print('dist_from_sl ', dist_from_sl)
        #Calcula o trailing
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
            #verificamos e exibimos o resultado como está
            result = mt5.order_check(request)
            resultOrd = mt5.order_send(request)
            print(result)
            print(resultOrd)
            return (result)
    #else:
    #    print('FIM')

if __name__ == '__main__':
    print('Start trailing Stoploss..')
    print(f'Position: {str(TICKET)}')

    while True:
        result = trail_sl(TICKET, MAX_DIST_SL, TRAIL_AMOUNT, DEFAULT_SL)

        if result:
            print(result)
            # print(resultOrd);

        time.sleep(1)

'''
'''
Implementar uma estratégia de negociação bem sucedida é um empreendimento complexo que requer habilidades em análise de mercado, estratégias de negociação e gerenciamento de risco. Embora a tarefa possa ser facilitada com a ajuda de código Python, é importante lembrar que o desenvolvimento de uma estratégia de negociação bem-sucedida leva tempo, esforço e testes adequados.

Dito isso, para refatorar o código Python para usar um ponto de equilíbrio e parada móvel e obter um lucro de 10.000 por dia, é possível seguir as etapas abaixo:

#1-Defina funções separadas para funcionalidades de ponto de equilíbrio e parada móvel. Essas funções podem receber a posição atual e o preço como entrada e retornar o nível de stop loss com base nos critérios especificados.

def calc_stop_loss(curr_pos, price):
    # código para calcular o nível de stop loss com base na posição atual e preço
    # retornar o nível de stop loss
    return stop_loss

def calc_break_even(curr_pos, price):
    # código para calcular o nível de ponto de equilíbrio com base na posição atual e preço
    # retornar o nível de ponto de equilíbrio
    return break_even
#2-Defina uma função para calcular o lucro e a perda de uma determinada posição. Esta função pode receber o preço de entrada, o preço de saída e o tamanho da posição como entrada e retornar o valor de lucro/prejuízo.
def calc_profit_loss(entry_price, exit_price, position_size):
    # código para calcular o lucro ou prejuízo com base no preço de entrada, preço de saída e tamanho da posição
    # retornar o valor de lucro/prejuízo
    return profit_loss
#3-Defina uma função para verificar as condições atuais do mercado e determinar se deve entrar ou sair de uma posição. Esta função pode usar indicadores técnicos, como médias móveis, osciladores estocásticos ou quaisquer outros sinais relevantes para gerar sinais de compra/venda.
def check_market_conditions():
    # código para verificar as condições do mercado e gerar sinais de compra/venda
    # retornar um sinal de compra/venda
    return signal
#4-Defina uma função para fazer pedidos e gerenciar a posição. Esta função pode receber a posição atual, o preço e o nível de stop-loss como entrada e colocar uma ordem apropriada com base na estratégia de negociação desejada.
def place_order(curr_pos, price, stop_loss):
    # código para colocar uma ordem de compra/venda com base na posição atual, preço e nível de stop-loss
    # retornar o resultado da ordem
    return order_result
#5Defina uma função principal que integra todas as funções acima e execute uma estratégia de negociação. Esta função pode ser executada em loop, verificando periodicamente as condições do mercado, colocando ordens e gerenciando a posição. Use o controle apropriado de tratamento de erros e registro para garantir que o código seja executado sem problemas e lide com eventos inesperados normalmente.
def trading_strategy():
    # código para execut


'''

