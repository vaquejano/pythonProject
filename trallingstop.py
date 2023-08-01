__name__ == '__trallingstop__'
import MetaTrader5 as mt5
import time
#mt5.initialize()
#symbol = "WDOF23"
#ticket = mt5.positions_get(symbol=symbol)[0][0]
#print(ticket)
#digits = mt5.symbol_info(symbol)
#TICKET = ticket
#MAX_DIST_SL = 5.0  # max distancai
#TRAIL_AMOUNT = 0.5  # amaont
#DEFAULT_SL = 1.01  # if posicao

def trallingstop(TICKET ):
    mt5.initialize()
    symbol = "WDOG23"
    ticket = mt5.positions_get(symbol=symbol)[0][0]
    print(ticket)
    digits = mt5.symbol_info(symbol)
    TICKET = ticket
    MAX_DIST_SL = 5.0  # max distancai
    TRAIL_AMOUNT = 0.5  # amaont
    DEFAULT_SL = 1.01  # if posicao
    MAX_DIST_SL = 5.0  # max distancai
    TRAIL_AMOUNT = 0.5  # amaont
    DEFAULT_SL = 1.01  # if posicao
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
    else:
        print('FIM')
        mt5.shutdown()
#if __name__ == '__main__':
#    print('Start trailing Stoploss..')
    #print(f'Position: {str(TICKET)}')

#    while True:
        #result = trallingstop(TICKET)

        #if result:
            #print(result)
            #print(resultOrd);

#        time.sleep(1)


