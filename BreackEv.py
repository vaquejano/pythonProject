__name__ == '__BreackEv__'
#****************Bibliotecas*************************************
import MetaTrader5
import MetaTrader5 as mt5
import time
import threading

mt5.initialize()
symbol = 'EURUSD'
bilhete = mt5.positions_get(symbol=symbol)
ticket = mt5.positions_get(symbol=symbol)[0][0]
posicao = mt5.positions_get(symbol=symbol)[0][5]
price_open = mt5.positions_get(symbol=symbol)[0][10]
print(bilhete)
print(f'ticket {ticket}, posicao {posicao}, price_open {price_open},')

#****************Bibliotecas*************************************
gatilho = 0.0005
anda = 0.5


def BreackEvan(posicao, gatilho):
    while posicao == 0 or 1:
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
            request = {'action': mt5.TRADE_ACTION_SLTP, 'position': ticket, 'sl': disparo, "symbol": symbol}
            result = mt5.order_send(request)
            return
    else:
        print('fim BE')


t = threading.Thread(target=BreackEvan, args=(posicao, gatilho))
t.start()
t.join()
