#****************Bibliotecas*************************************
import MetaTrader5
import MetaTrader5 as mt5
import multiprocessing as mp
import time

#from mercado_forex import X_teste

#****************funcao Breack Evan*************************************
'''#funciona'''
def BreackEvan(g):
    #gatilho = 0.5
    mt5.initialize()
    symbol = "WDOX22"
    bilhete = mt5.positions_get(symbol=symbol)
    ticket = mt5.positions_get(symbol=symbol)[0][0]
    posicao = mt5.positions_get(symbol=symbol)[0][5]
    price_open = mt5.positions_get(symbol=symbol)[0][10]
    price_current = mt5.positions_get(symbol=symbol)[0][13]
    print(bilhete)
    print(f'ticket {ticket}, posicao {posicao}, price_open {price_open},')
    while price_open > gatilho:
        if posicao == 0:
            print('gatilho acionado compra')
            disparo = price_open + gatilho
            print(f'compra=', disparo)
            request = {'action': mt5.TRADE_ACTION_SLTP, 'position': ticket, 'sl': disparo, "symbol": symbol}
            result = mt5.order_send(request)
            result_check = mt5.order_check(request)
            print(result)
            print(result_check)
        else:
            print('gatilho acionado venda')
            disparo = price_open - gatilho
            print(f'venda=', disparo)
            request = {'action': mt5.TRADE_ACTION_SLTP, 'position': ticket, 'sl': disparo, "symbol": symbol}
            result = mt5.order_check(request)
            resultOrd = mt5.order_send(request)
            #trail_sl(asdf)
            print(result)
            print(resultOrd)
        return result

#BreackEvan(0.5)
if __name__ == '__main__':
    gatilho = 0.5
    p = mp.Pool()
    res = p.map(BreackEvan, gatilho)
    p.close()





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