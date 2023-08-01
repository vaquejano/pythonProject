#****************Bibliotecas*************************************
import MetaTrader5
import MetaTrader5 as mt5
import time

gatilho = 0.5
mt5.initialize()
symbol = "WDOX22"
#****************verificamos a presença de posições abertas*****************************
symbol = "WDOX22"
bilhete = mt5.positions_get(symbol=symbol)
ticket = mt5.positions_get(symbol=symbol)[0][0]
posicao = mt5.positions_get(symbol=symbol)[0][5]
price_open = mt5.positions_get(symbol=symbol)[0][10]
price_current = mt5.positions_get(symbol=symbol)[0][13]
print(bilhete)
print(f'ticket {ticket}, posicao {posicao}, price_open {price_open},')
#****************verificamos a presença de posições abertas*****************************
positions_total = mt5.positions_total()
if positions_total > 0:
    print("Total positions=", positions_total)
    print('<<Nao pode Negociar tem posicao aberta')
    beAtivo = False
    traling = True
    if (positions_total & beAtivo == 0):
        print('BE ativo')
        if price_open < gatilho+1:
           print('breack vaquero')
           print(beAtivo)
           gatilho = 0.5
            #BreackEvan()
            #breakEvan = (last)
            #print('verdade')
    elif(positions_total & traling == 1):
        print('traling vaquero')
        print(traling)
else:
    print("<<>>Nao tem posicao aberta, pode Negociar!!!")



#mt5.initialize()
#****************funcao Breack Evan*************************************
def BreackEvan(price_open, gatilho):
    #mt5.initialize()
    #symbol = "WDOX22"
    #bilhete = mt5.positions_get(symbol=symbol)
    #ticket = mt5.positions_get(symbol=symbol)[0][0]
    #posicao = mt5.positions_get(symbol=symbol)[0][5]
    #price_open = mt5.positions_get(symbol=symbol)[0][10]
    #price_current = mt5.positions_get(symbol=symbol)[0][13]
    #print(bilhete)
    #rint(f'ticket {ticket}, posicao {posicao}, price_open {price_open},')

    while price_open > gatilho:
        if posicao == 0:
            g_t = 1
            print('gatilho acionadocompra')
            disparo = price_open + gatilho
            print(f'compra=', disparo)
            request = {'action': mt5.TRADE_ACTION_SLTP, 'position': ticket, 'sl': disparo, "symbol": symbol}
            result = mt5.order_send(request)
            result = mt5.order_check(request)
            resultOrd = mt5.order_send(request)
            print(result)
            print(resultOrd)
            return
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
            return
        time.sleep(2)
        #digits = mt5.symbol_info(symbol)

BreackEvan(price_open, gatilho)
