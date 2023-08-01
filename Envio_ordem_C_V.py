import MetaTrader5
import MetaTrader5 as mt5
import time
import threading
mt5.initialize()
def neg(fl):
    symbol = 'WDON23'
    # **
    symbol_info = mt5.symbol_info(symbol)
    print(symbol_info)
      # **
    last = symbol_info.last
   #  print(last)
    bilhete = mt5.positions_get(symbol=symbol)

    '''
    ticket = mt5.positions_get(symbol=symbol)[0][0]
    bilhete = mt5.positions_get(symbol=symbol)
    posicao = mt5.positions_get(symbol=symbol)[0][5]
    price_open = mt5.positions_get(symbol=symbol)[0][10]
    price_current = mt5.positions_get(symbol=symbol)[0][13]
    print(bilhete)
    print(f'ticket {ticket}, posicao {posicao}, price_open {price_open},')

    if (fl >= last):
        print('Compra ^')
        volume = 1.0
        stp = 5000
        tkp = 10000
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
            "magic": 234000,
        }
        # enviamos a solicitação de negociação
        result = mt5.order_check(request)
        resultOrdby = mt5.order_send(request)
        print(result);
        print(resultOrdby);
        return
    elif (fl <= last):
        print('Venda V')
        volume = 1.0
        stp = 5000
        tkp = 10000
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
            "magic": 234000,
        }
        # Enviamos a solicitação de negociação
        result = mt5.order_send(request)
        print(result)
        return
    else:
        print('Ordem pendente')
        return
# ****************Valor da condicao BUY e SELL**********************************************
'''

fl = 4898
neg(fl)
#print(a1)

mt5.shutdown()