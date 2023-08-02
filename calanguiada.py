import MetaTrader5 as mt5
if not mt5.initialize():
    print("initialize() failed")
    mt5.shutdown()
    if not mt5.initialize():
        print("initialize() failed")

    # mt5.shutdown()
print('******************')
print('*   conectado    *')
print('******************')
symbol = 'WDOU23'
'''parametros da ordem compra'''
point = mt5.symbol_info(symbol).point
bilhete = mt5.positions_get(symbol=symbol)[0]
print(bilhete)
'''parametros da ordem'''
ticket = bilhete.ticket
symbol = 'WDOU23'
price_open = bilhete.price_open
sl = bilhete.sl
profit = bilhete.profit
type = bilhete.type
volume = bilhete.volume
offset = 500000

if type == 1:
    def calanguiada(offset):
        print("8" * 100)
        print('calanguiada acionada')
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "ticket": ticket,
            "symbol": symbol,
            "volume": volume,
            "price": mt5.symbol_info_tick(symbol).bid,
            "sl": mt5.symbol_info_tick(symbol).bid - offset * point,
            "tp": mt5.symbol_info_tick(symbol).bid + offset * point,
            "type": mt5.ORDER_TYPE_BUY,
            "type_filling": mt5.ORDER_FILLING_RETURN,
            "deviation": 10,
            "comment": "Boa V@",
            "magic": 171,
        }
        # Enviamos a solicitação de negociação
        result = mt5.order_send(request)
        resultOrdby = mt5.order_send(request)
        print(result)
        print(resultOrdby)

    calanguiada(offset)

if type == 0:
    def calanguiada(offset):
        print("8" * 100)
        print('calanguiada acionada')
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "ticket": ticket,
            "symbol": symbol,
            "volume": volume,
            "price": mt5.symbol_info_tick(symbol).ask,
            "sl": mt5.symbol_info_tick(symbol).ask + offset * point,
            "tp": mt5.symbol_info_tick(symbol).ask - offset * point,
            "type": mt5.ORDER_TYPE_SELL,
            "type_filling": mt5.ORDER_FILLING_RETURN,
            "deviation": 10,
            "comment": "Boa V@",
            "magic": 171,
        }
        # Enviamos a solicitação de negociação
        result = mt5.order_send(request)
        resultOrdby = mt5.order_send(request)
        print(result)
        print(resultOrdby)

    calanguiada(offset)
