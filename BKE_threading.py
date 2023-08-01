#__name__ == '__BreackEvan__'
#****************Bibliotecas*************************************
import MetaTrader5
import MetaTrader5 as mt5
import time
import threading
#gatilho = 0.5
#mt5.initialize()
#symbol = "WDOX22"
#bilhete = mt5.positions_get(symbol=symbol)
#ticket = mt5.positions_get(symbol=symbol)[0][0]
#posicao = mt5.positions_get(symbol=symbol)[0][5]
#price_open = mt5.positions_get(symbol=symbol)[0][10]
#price_current = mt5.positions_get(symbol=symbol)[0][13]
#print(bilhete)
#print(f'ticket {ticket}, posicao {posicao}, price_open {price_open},')
#****************funcao Breack Evan*************************************
'''
#funciona
def BreackEvan(gatilho):
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
            g_t = 1
            print('gatilho acionadocompra')
            disparo = price_open + gatilho
            print(f'compra=', disparo)
            request = {'action': mt5.TRADE_ACTION_SLTP, 'position': ticket, 'sl': disparo, "symbol": symbol}
            result = mt5.order_send(request)
            result_check = mt5.order_check(request)
            print(result)
            print(result_check)
            #asdf = price_open + 10
            #trail_sl(asdf)
            #print('as', asdf)
            #print(result)
            #return
        else:
            print('gatilho acionado')
            disparo = price_open - gatilho
            print(f'venda=', disparo)
            g_t = 1
            request = {'action': mt5.TRADE_ACTION_SLTP, 'position': ticket, 'sl': disparo, "symbol": symbol}
            result = mt5.order_check(request)
            resultOrd = mt5.order_send(request)
            #trail_sl(asdf)
            print(result)
            print(resultOrd)
            asdf = price_open - 10
           # trail_sl(asdf)
            #print('as', asdf)
            #print(result)
            #return
        time.sleep(2)
        #digits = mt5.symbol_info(symbol)

t1 = threading.Thread(target=BreackEvan, args=(posicao, gatilho))
t1.start()
t1.join()
'''

mt5.initialize()
symbol = "WDOX22"
#bilhete = mt5.positions_get(symbol=symbol)
#ticket = mt5.positions_get(symbol=symbol)[0][0]
#posicao = mt5.positions_get(symbol=symbol)[0][5]
#price_open = mt5.positions_get(symbol=symbol)[0][10]
#profit = mt5.positions_get(symbol=symbol)[0][15]
#print("b", bilhete)
#print(f'ticket {ticket}, posicao {posicao}, price_open {price_open},')
gatilho = 0.5
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
        print(type)
        print('aguardando')
        if (profit <= 6):
            print('porrada')
            def BreackEvan(gatilho):
                if positions_total == 1:
                    print('gatilho acionado compra')
                    disparo = price_open + gatilho
                    print(f'compra=', disparo)
                    request = {'action': mt5.TRADE_ACTION_SLTP, 'position': bilhete_neg, 'sl': disparo,
                               "symbol": symbol}
                    result = mt5.order_send(request)
                    print(result)
                    return
                else:
                    print('gatilho acionado venda')
            t1 = threading.Thread(target=BreackEvan, args=(gatilho,))
            t1.start()
            t1.join()
    else:
        if (type == 1):
            if (profit >= 6):
                print('porrada')
                def BreackEvan(gatilho):
                    if positions_total == 1:
                        print('gatilho acionado compra')
                        disparo = price_open - gatilho
                        print(f'compra=', disparo)
                        request = {'action': mt5.TRADE_ACTION_SLTP, 'position':  bilhete_neg, 'sl': disparo, "symbol": symbol}
                        result = mt5.order_send(request)
                        print(result)
                        return
                    else:
                       print('gatilho acionado venda')
                #t1 = threading.Thread(target=BreackEvan, args=(gatilho,))
                #t1.start()
                #t1.join()
else:
    print("<<>>Nao tem posicao aberta, Aguarde!!!")
mt5.shutdown()
#Preparamos a solicitação
#Breack Evan Thread funciona
'''
def BreackEvan(gatilho):
    if posicao == 0:
        print('gatilho acionado compra')
        disparo = price_open + gatilho
        print(f'compra=', disparo)
        request = {'action': mt5.TRADE_ACTION_SLTP, 'position': ticket, 'sl': disparo, "symbol": symbol}
        result = mt5.order_send(request)
        print(result)
        return
    else:
        print('gatilho acionado venda')
        disparo = (price_open - gatilho)
        print(f'venda=', disparo)
        request = {'action': mt5.TRADE_ACTION_SLTP, 'position': ticket, 'sl': disparo, "symbol": symbol}
        result = mt5.order_send(request)
        print(result)
        return
t1 = threading.Thread(target=BreackEvan, args=(gatilho,))
t1.start()
t1.join()
'''
'''    
if __name__ == '__main__':
     #print(f'Position: {str(posicao)}')
    while profit<=6:
          #result = BreackEvan(posicao, gatilho)
          #print(result)
          gatilho = 0.5
          t1 = threading.Thread(target=BreackEvan, args=(gatilho,))
          t1.start()
          t1.join()
          #if result:
          #   print(result)
              #print(resultOrd);
          time.sleep(2)

#t1 = threading.Thread(target=BreackEvan, args=(posicao, gatilho))
#t1.start()
#t1.join()

#print(k)

#Fim T1

'''