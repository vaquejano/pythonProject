#****************Bibliotecas*************************************
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
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
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
    datetime.now()

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
        print('hoje:', hoje)

        #****************Informacoes do ativo***********************************
        ativo = 'WDO$'
        symbol = 'WDOQ23'
        #****************funcao criar data frame *************************************
        wdo_m1 = get_ohlc(ativo, mt5.TIMEFRAME_M1, 99999)
        wdo_m1.to_csv('WDO.CSV')
        #print(wdo_m1)
        tabela=wdo_m1

        #tabela.to_csv('WDO.CSV')  # criar um banco de dados em Excel
        #tabela = pd.read_csv('WDO.CSV', index_col=0)
        tabela.dropna(inplace=True)
        ## Preparar dados para o modelo
        tabela['RSI1'] = tabela['close'].rolling(2).median()
        tabela['RSI2'] = tabela['close'].rolling(7).median()
        tabela['RSI3'] = tabela['close'].rolling(21).median()
        tabela.dropna(inplace=True)
        tabela = tabela.drop(["spread", "tick_volume", "real_volume", "high", "low", "open", ], axis=1)
        tabela['signal'] = np.where(
            (tabela['RSI1'] < tabela['close']) &
            (tabela['RSI2'] < tabela['close']) &
            (tabela['RSI3'] < tabela['close']),  # Sell signal condition
            -1,  # Sell signal
            np.where(
                (tabela['RSI1'] > tabela['close']) &
                (tabela['RSI2'] > tabela['close']) &
                (tabela['RSI3'] > tabela['close']),  # Buy signal condition
                1,  # Buy signal
                0  # No signal
            )
        )
                ## Prepare features and target variable for the model
        X = tabela.drop(['signal'], axis=1)

        y = tabela['signal'].values

        ## Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        ## Train a Random Forest Classifier
        rf_classifier = RandomForestClassifier(random_state=42)
        rf_classifier.fit(X_train, y_train)

        y_pred = rf_classifier.predict(X_test)
        floresta = y_pred
        #****************Faco minha predicao **********************************************
        p = floresta
        fl = p[-1]
        print(fl)
        '''
        dif = (last - fl)
        defx = (fl+last)/2
        print('last->', last)
        print('fl->', fl)
        print('dif->', dif, 'last', '-', 'fl')
        print(defx)
        print('dif2->', last-defx)

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
                if (profit <= 0 and profit<= 25.0):
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
                elif profit >0 and profit>= 01.0:
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
                    if (profit == 25.0 ):
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
                    elif profit > 30.0:
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
                        #parametro da posicao'''
'''
        else:
            print("<<>>Nao tem posicao aberta, pode Negociar!!!")
            #Preparamos a solicitação
        #****************funcao simples compra e venda**********************************************
            def neg(fl):
                print('-->',fl)
                if (fl <= last):
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
                elif (fl >= last):
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
'''
if __name__ == "__main__":
    main()