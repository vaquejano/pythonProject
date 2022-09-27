__name__ == '__BreackEvan__'
# ****************Bibliotecas*************************************
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

# ****************Bibliotecas*************************************
gatilho = 0.25
spread = 0.20
k2 = 1
def BreackEvan(posicao, gatilho, k2):

        if posicao == 0:
            k2
            print('gatilho acionado')
            disparo = price_open + gatilho - spread
            print(f'compra=', disparo)
            request = {'action': mt5.TRADE_ACTION_SLTP, 'position': ticket, 'sl': disparo, "symbol": symbol}
            result = mt5.order_send(request)
            print(result)
            return
        else:
            print('gatilho acionado')
            disparo = (price_open - gatilho) - spread
            print(f'venda=', disparo)
            k2
            request = {'action': mt5.TRADE_ACTION_SLTP, 'position': ticket, 'sl': disparo, "symbol": symbol}
            result = mt5.order_send(request)
            print(result)
            return

#t1 = threading.Thread(target=BreackEvan, args=(posicao, gatilho))
#t1.start()
#t1.join()

if __name__ == '__main__':
    print('BreackEvan..')
    print(f'Position: {str(posicao)}')

    while gatilho > price_open:
        result = BreackEvan(posicao, gatilho)
        t1 = threading.Thread(target=BreackEvan, args=(posicao, gatilho, k2))
        t1.start()
        t1.join()

        if result:
           print(result)
            # print(resultOrd);

        time.sleep(1)
        break
#t1 = threading.Thread(target=BreackEvan, args=(posicao, gatilho))
#t1.start()
#t1.join()

print(k)

#Fim T1
digits = mt5.symbol_info(symbol)
TICKET = ticket
MAX_DIST_SL = 2.7  # max distancai
TRAIL_AMOUNT = 0.5  # amaont
DEFAULT_SL = 1.01  # if posicao

'''
#Inicio T2
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
    dist_from_sl = round(price_current-sl, 6) #proft
    print('dist_from_sl ', dist_from_sl)

    # calcula o trailing
    if dist_from_sl >= MAX_DIST_SL:
        # calculo novo sl
        if sl != 0.0:
            print('1 order')
            if order_type == 0:
                print('pos')
                new_sl = sl + TRAIL_AMOUNT
                print(f'buy', new_sl)
            elif order_type == 1:
                new_sl = sl - TRAIL_AMOUNT
                print(f'sell', new_sl)

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
'''


#if __name__ == '__main__':

#    t1 = threading.Thread(target=BreackEvan, args=(posicao, gatilho))
    #t2 = threading.Thread(target=trail_sl, args=(TICKET, MAX_DIST_SL, TRAIL_AMOUNT, DEFAULT_SL))
#    t1.start()
    #t2.start()
 #   t1.join()
    #t2.join()



