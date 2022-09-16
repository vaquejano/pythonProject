# ****************Bibliotecas*************************************
import math
import MetaTrader5 as mt5
import MetaTrader5 as trade
from datetime import datetime
import time
import pandas as pd
import numpy as np
# ****************Modulo sklearn*************************************
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score

# ****************Setando as colunas das planilhas*************************************
pd.set_option('display.max_columns', 400)  # número de colunas mostradas
pd.set_option('display.width', 1500)  # max. largura máxima da tabela exibida

#****************Informacao do Ativo*************************************
ativo = 'EURUSD'
symbol = "EURUSD"
volume = 1.0

#symbol_info = mt5.symbol_info(symbol)
# ****************Conectar ao sistema Meta Trader 5*************************************
# Passo 1: Conectar ao sistema Meta Trader 5
#while True:
# conecte-se ao MetaTrader 5
if not mt5.initialize():
    print("initialize() failed")
    mt5.shutdown()
    print('******************')
    print('*   conectado    *')
    print('******************')

#****************funcao criar data frame *************************************
def get_ohlc(ativo, timeframe, n=10):
    ativo = mt5.copy_rates_from_pos(ativo, timeframe, 0, n)
    ativo = pd.DataFrame(ativo)
    ativo['time'] = pd.to_datetime(ativo['time'], unit='s')
    ativo.set_index('time', inplace=True)
    return ativo
wdo_m1 = get_ohlc(ativo, mt5.TIMEFRAME_M1,99999)
# ****************Transforma em arquivo CSV***********************************
wdo_m1.to_csv('forex.CSV')

# ****************Data atual**************************************************
hoje = datetime.now()

# ****************Pega informacao do tick*************************************
ticks = mt5.copy_ticks_from(ativo, hoje, 8000, mt5.COPY_TICKS_INFO)
lasttick = mt5.symbol_info_tick(symbol)
#last = mt5.symbol_info_tick(symbol).last
ask = mt5.symbol_info_tick(symbol).ask
ask = ask
price = ask
bid = mt5.symbol_info_tick(symbol).bid
bid = bid
last = (bid+ask)/2
total = mt5.positions_total()
posicao = mt5.positions_get()
orderBuy = mt5.ORDER_TYPE_BUY
orderSell = mt5.ORDER_TYPE_SELL
beAtivo = False
print(type(total))
#print(orderSell)

# ****************Dados do ativo**********************************************
# Dados do ativo
tabela = wdo_m1
# print(tabela)

# ****************Reset index**********************************************
tabela.reset_index('time', inplace=True)
tabela = tabela.drop(["time"], axis=1)

# ****************Tabela manipulaco**********************************************
tabela = tabela.drop(["spread", "real_volume"], axis=1)
tabela.loc[:, 'media1'] = (tabela['close'].rolling(2).median())
tabela.loc[:, 'media7'] = (tabela['close'].rolling(7).median())
tabela.loc[:, 'media21'] = (tabela['close'].rolling(21).median())
tabela.loc[:, 'media36'] = (tabela['close'].rolling(36).median())
tabela.loc[:, 'media200'] = (tabela['close'].rolling(200).median())
# print(tabela)

# ****************Preenche o valor vazio com o valor 0**********************************************
tabela = tabela.fillna(0)  # preenche o valor vazio com o valor 0
#print(tabela)

# ****************Separa dados de X e y*********************************************
X = tabela.drop("close", axis=1)
y = tabela['close'].values
# print(X)
# print(y)

# ****************Separa dados de trino e dados de teste*********************************************
X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.9, random_state=42)

# ****************Seleciono a IA**********************************************
floresta = RandomForestRegressor(bootstrap       =True,
                                 n_estimators    =25,
                                 min_samples_leaf=5,
                                 max_leaf_nodes  =5,
                                 max_depth       =30,
                                 random_state    =42,
                                 verbose         =0,
                                 warm_start      =True,
                                 n_jobs          =-1
                                 )
# floresta = RandomForestRegressor(bootstrap=True, n_estimators=1, min_samples_leaf=5, max_leaf_nodes=5,
#                                 random_state=42, n_jobs=-1)

# ****************Treino minha IA**********************************************
floresta.fit(X_treino, y_treino)

# ****************Faco minha predicao **********************************************
p = floresta.predict(X_teste)
# print(tabela)
# print(accuracy_score(y_teste, p))

# ****************Decisao***********************************************
fl = mean_squared_error(y_teste, p)
#ac = accuracy_score(y_teste, p)
# print(f'Floresta------------>>{fl}')

# ****************Erro medio quadrado*********************************************
ss = np.sqrt(mean_squared_error(y_teste, p))
# print(f'Erro medio quadrado->>{ss}')

# ****************Metrica de precisao**********************************************
r_square = metrics.r2_score(y_teste, p)
# print(f'Metrica------------->>{r_square}')

#****************Acuracia da floresta**********************************************
# acc = accuracy_score(y_teste, p)
# <<<<<<<<<<Nova IA >>>>>>>>>>>>>>>>>>>
x1 = X_teste
y1 = y_teste

# ****************Treino minha IA**********************************************
floresta.fit(x1, y1)

# ****************Faco minha predicao **********************************************
p1 = floresta.predict(x1)
# print(tabela)
# print(accuracy_score(y_teste, p))

# ****************Decisao***********************************************
fl1 = mean_squared_error(y1, p1)

#****************Fim Floresta de decisao***********************************************
#######################################################################################
#****************Mostra os valores***********************************************
print('*'*10, 'Tick','*' *10)
print(f'Floresta------------>>{fl1}%5.2f')
print(f'ask----------------->>{ask}')
print(f'bid----------------->>{bid}')
print(f'last---------------->>{last}')
print(f'pos----------------->>{total}')
#print(f'posticket------------>>{posi}')
print('*'*35)
#######################################################################################
#****************verificamos a presença de posições abertas*****************************
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
        if (ult_valor >= f):
            print('Compra ^')
            volume = 1.0
            stp = 150
            tkp = 250
            point = mt5.symbol_info(symbol).point
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": 1.0,
                 "price": mt5.symbol_info_tick(symbol).ask,
                 "sl": mt5.symbol_info_tick(symbol).ask - stp * point,
                 "tp": mt5.symbol_info_tick(symbol).ask + tkp * point,
                 "deviation": 10,
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
            #print("1. order_send(): by {} {} lots at {} with deviation={} points".format(symbol, lot, price,
             #                                                                                deviation))

        elif (ult_valor <= f):
            print('Venda V')
            volume = 1.0
            stp = 150
            tkp = 250
            point = mt5.symbol_info(symbol).point
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": 1.0,
                "price": mt5.symbol_info_tick(symbol).bid,
                "sl": mt5.symbol_info_tick(symbol).bid + stp * point,
                "tp": mt5.symbol_info_tick(symbol).bid - tkp * point,
                "deviation": 10,
                "type": mt5.ORDER_TYPE_SELL,
                "type_filling": mt5.ORDER_FILLING_RETURN,
                "comment": "Boa V@",
                "magic": 234000,
            }
            # enviamos a solicitação de negociação
            result = mt5.order_send(request)
            # verificamos o resultado da execução
            #print("1. order_send(): by {} {} lots at {} with deviation={} points".format(symbol, volume, price,
            #                                                                                 deviation))
        else:
            print('Ordem pendente')
    flor = (fl)

    #****************ultimo valor do tick**********************************************
    ult_valor = last

    # ****************Valor da condicao BUY e SELL**********************************************
    a1 = neg(flor)

ticket = mt5.positions_get(symbol=symbol)[0][0]
print(ticket)
digits = mt5.symbol_info(symbol)
TICKET = ticket
MAX_DIST_SL = 1.0  # max distancai
TRAIL_AMOUNT = 0.0001  # amaont
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
    dist_from_sl = round(proft, 6)
    print('dist_from_sl ', dist_from_sl)

    # calcula o trailing
    if dist_from_sl > MAX_DIST_SL:
        # calculo novo sl
        if sl != 0.0:
            print('1 order')
            if order_type == 0:
                print('pos')
                new_sl = sl + TRAIL_AMOUNT
                print(new_sl)
            elif order_type == 1:
                new_sl = sl - TRAIL_AMOUNT
        else:
            print(2)
            new_sl = price_open + DEFAULT_SL if order_type == 0 else price_open + DEFAULT_SL  # seting
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "symbol": symbol,
            "position": TICKET,
            "sl": new_sl,
            "comment": "V@ -~~~~~>> chicote estrala",
        }
       
        # request = {
        #        "action": mt5.TRADE_ACTION_SLTP,
        #        "ticket": ticket,
        #        "symbol": symbol,
        #        "magic": magic,
        #        "sl": sl,
        #        "tp": tp,
        #        # "pisiton": ticket,
        #        "magic": magic,
        #        "comment": "B sai fora boiadeiro V@",
        #    }


        # verificamos e exibimos o resultado como está
        result = mt5.order_check(request)
        resultOrd = mt5.order_send(request)
        print(result);
        # print(resultOrd);
        return (result)


if __name__ == '__main__':
    print('Start trailing Stoploss..')
    print(f'Position: {str(TICKET)}')

    while True:
        result = trail_sl(TICKET, MAX_DIST_SL, TRAIL_AMOUNT, DEFAULT_SL)

        if result:
            print(result)
            # print(resultOrd);

        time.sleep(1)
