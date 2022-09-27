import MetaTrader5 as mt5
import time
mt5.initialize()
symbol = 'EURUSD'
ticket = mt5.positions_get(symbol=symbol)[0][0]
print(ticket)

digits = mt5.symbol_info(symbol)
TICKET = ticket
MAX_DIST_SL = 0.7  # max distancai
TRAIL_AMOUNT = 0.00000005  # amaont
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
