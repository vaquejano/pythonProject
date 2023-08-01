import sched
import threading

import MetaTrader5 as mt5
import time
import pandas as pd
import numpy as np
from datetime import datetime

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor

# Inicialização do MetaTrader 5
mt5.initialize()

def equilibrio(custo, spread, c_fix, lucro):
    p_equilibrio = ((custo + spread) / (lucro - c_fix))
    entrada = custo + p_equilibrio
    return entrada

def repeat_at_interval(scheduler, event, interval=60, add_n=10, start_t=None):
    """Adiciona 'add_n' chamadas adicionais para 'event' a cada 'interval' segundos"""
    if start_t is None:
        t = time.time()
        t = t - (t % interval) + interval
    else:
        t = start_t

    for i in range(add_n):
        scheduler.enterabs(t, 0, event)
        t += interval

    scheduler.enterabs(t - interval, 0, repeat_at_interval, kwargs={
        "scheduler": scheduler,
        "event": event,
        "interval": interval,
        "add_n": add_n,
        "start_t": t
    })

def main():
    # Inicializa o scheduler
    scheduler = sched.scheduler(time.time, time.sleep)

    # Função de teste
    def test():
        print(datetime.now())

    # Repete o teste a cada 60 segundos
    repeat_at_interval(scheduler, test, interval=60)

    # Inicia o thread do scheduler
    thread = threading.Thread(target=scheduler.run)
    thread.start()

    # Loop principal
    while True:
        # Código principal do seu programa aqui
        ## ****************Cria minhas funcoes*************************************
        # ****************funcao criar data frame *************************************
        def get_ohlc(ativo, timeframe, n=10):
            ativo = mt5.copy_rates_from_pos(ativo, timeframe, 0, n)
            ativo = pd.DataFrame(ativo)
            ativo['time'] = pd.to_datetime(ativo['time'], unit='s')
            ativo.set_index('time', inplace=True)
            return ativo

        def bilhetagem(Symbol):
            symbol_info = mt5.symbol_info(symbol)
            posicao = (len(symbol_info))
            last = symbol_info.last
            ativo = symbol_info.name
            spread = symbol_info.spread
            return symbol_info, last, ativo, posicao, spread
        def status(last, ativo, posicao, spread):
            print(f'Ativo-------->> {ativo}')
            print(f'qtde_Ifo----->> {posicao}')
            print(f'ultimo_Preco->> {last:.3f}')
            print(f'spread->> {spread:.3f}')

            #print(f'bid->> {bid:.3f}\n')
# ****************Final da funcoes*************************************

        if not mt5.initialize():
            print("initialize() failed")
            mt5.shutdown()
            if not mt5.initialize():
                print("initialize() failed")

        # mt5.shutdown()
        print('******************')
        print('*   conectado    *')
        print('******************')

        #**************** Fim data atual**************************************************
        hoje = datetime.now()
        print(hoje)

        #****************Informacoes do ativo***********************************
        ativo = 'WDO$'
        symbol = 'WDON23'
        #****************funcao criar data frame *************************************
        wdo_m1 = get_ohlc(ativo, mt5.TIMEFRAME_M1, 99999)
        wdo_m1.to_csv('WDO.CSV')
        #****************Informacao do bilhete *************************************
        simbolo = bilhetagem(symbol)
        point = mt5.symbol_info(symbol).point
        simbolo1 = mt5.symbol_info(symbol).name
        print('symb=>', simbolo1)
        print(point)
        print(f'symboloxx', simbolo)
        print('#####' * 2, 'Informacao do ativo', '#####' * 2)
        last = simbolo[1]
        ativo = simbolo[2]
        posicao = simbolo[3]
        spread = simbolo[3]
        status(last, ativo, posicao, spread)

        #***************Seta a planilha dos dados ************************************
        '''Setando as colunas das planilhas'''
        pd.set_option('display.max_columns', 400)  # número de colunas mostradas
        pd.set_option('display.width', 1500)  # max. largura máxima da tabela exibida

        #***************Busca os dados Historicos************************************
        tabela = pd.read_csv('WDO.CSV', index_col=0)
        # tabela.reset_index('time', inplace=True)#reseta index
        #print(tabela)
        #****************Tabela manipulaco**********************************************
        # tabela = tabela.drop(["open", "high", "low", "spread", "tick_volume", "real_volume"], axis=1)
        # Calculando o Estocástico Rápido
        #tabela = tabela.drop(["open","spread", "tick_volume", "real_volume"], axis=1)
        #print(tabela)
        '''tabela de manipulacao de medias'''
        tabela.loc[:, 'media(s)'] = tabela['close'].rolling(21).mean()
        tabela.loc[:, 'media(md)'] = tabela['close'].rolling(21).median()
        tabela.loc[:, 'media7'] = tabela['close'].ewm(span=7, min_periods=7).mean()  # media exponencial
        tabela.loc[:, 'media14'] = tabela['close'].ewm(span=14, min_periods=14).mean()  # media exponencial
        tabela.loc[:, 'media20'] = tabela['close'].ewm(span=21, min_periods=21).mean()
        tabela.loc[:, 'momento'] = tabela['close'] - tabela['close'].rolling(20).mean()
        tabela.loc[:, 'OBV'] = (np.sign(tabela["close"].diff()) * tabela["real_volume"]).fillna(0).cumsum()
        tabela.loc[:, 'rsi_c'] = 100 - 100 / (1 + (tabela["close"].diff().fillna(0).rolling(window=7).mean() /
                                                 tabela["close"].diff().fillna(0).rolling(window=7).std()))
        tabela.loc[:, 'rsi_m'] = 100 - 100 / (1 + (tabela["close"].diff().fillna(0).rolling(window=14).mean() /
                                                 tabela["close"].diff().fillna(0).rolling(window=14).std()))
        tabela.loc[:, 'rsi_l'] = 100 - 100 / (1 + (tabela["close"].diff().fillna(0).rolling(window=21).mean() /
                                                 tabela["close"].diff().fillna(0).rolling(window=21).std()))
        # Calculando o Estocástico Rápido
        tabela.loc[:, 'n_highest_high'] = tabela["high"].rolling(21).max()
        tabela.loc[:, 'n_lowest_low'] = tabela["low"].rolling(21).min()
        tabela.loc[:, '%K'] = (tabela["close"] - tabela["n_lowest_low"]) / (
                tabela["n_highest_high"] - tabela["n_lowest_low"]) * 100
        tabela["%D"] = tabela['%K'].rolling(21).mean()
        tabela.dropna(inplace=True)
        tabela["Slow %K"] = (tabela["%D"])
        tabela["Slow %D"] = (tabela["Slow %K"].rolling(21).mean()) * 100
        # Calcular indicadores de análise técnica
        tabela["returns"] = (tabela["close"] - tabela["open"]) / tabela["open"]
        tabela["ma20"] = tabela["close"].rolling(window=20).mean()
        tabela["upper_band"] = tabela["ma20"] + 2 * tabela["close"].rolling(window=20).std()
        tabela["lower_band"] = tabela["ma20"] - 2 * tabela["close"].rolling(window=20).std()
        # Análise de entrada de compra
        tabela["buy_signal"] = 0
        tabela.loc[(tabela["returns"] > 0) & (tabela["close"] > tabela["ma20"]), "buy_signal"] = 1
        # Análise de saída de venda
        tabela["sell_signal"] = 0
        tabela.loc[(tabela["returns"] < 0) & (tabela["close"] < tabela["ma20"]), "sell_signal"] = 1
        # Análise de stop loss
        tabela["stop_loss"] = tabela["ma20"] - tabela["returns"].abs().rolling(window=20).mean()
        # Análise de break even
        tabela["break_even"] = tabela["open"] + tabela["returns"].abs().rolling(window=20).mean()
        # Análise de trailing stop
        tabela["trailing_stop"] = tabela["close"].rolling(window=20).max() - tabela["returns"].abs().rolling(
            window=20).mean()
        tabela.dropna(inplace=True)
        tabela.head()
        tabela = tabela.drop(["spread", "tick_volume", "n_highest_high", "n_lowest_low", "real_volume"], axis=1)
        #print(tabela)
        #print(tabela.shape)
        '''Verifica a correlacao'''
        #        tabela.corr()
        '''Grafico correlacao '''
        #sns.heatmap(tabela.corr(), annot=True, vmin=-1, vmax=1, cmap='Reds')
        #plt.show()
        '''Separa dados de X e y '''
        X = tabela.drop(columns=["close"], axis=1).fillna(0)
        y = tabela['close'].values
        #****************Grafico correlacao de X *********************************************
        # sns.heatmap(X.corr(), annot=True, vmin=-1, vmax=1, cmap='Reds')
        # plt.show()
        '''Separa dados de trino e teste '''
        X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.618, random_state=42)
        '''Parametros da IA Grid Saerch '''
        '''Número de árvores na floresta aleatória '''
        n_stimators = [int(x) for x in np.linspace(start=1, stop=2000,
                                                   num=1)]#'''Numero de Arvores[int(x) for x in np.linspace(start=1, stop=180, num=10)]'''
        max_samples = [None]
        '''Número de feições a serem consideradas em cada divisão'''
        max_features = ['sqrt',
                        'log2']  # número máximo de recursos (variáveis independentes) a serem considerados ao dividir um nó
        max_features.append(1.0)
        '''Número máximo de níveis na árvore'''
        max_depth = [int(x) for x in np.linspace(20, 1000, num=20)]  # 11-[None, 30] 25
        max_depth.append(None)
        '''Número mínimo de amostras necessárias para dividir um nó'''
        min_samples_split = [int(x) for x in np.linspace(2.0, 350.0, num=14)]  # numero 2 e default--float usa ponto(número mínimo de amostras em um subconjunto (também conhecido como nó) para dividi-lo em mais dois subconjuntos)2, 3, 5, 8, 13, 21, 34, 55, 89, 134
        min_samples_split.append(2)
        '''Número mínimo de amostras necessárias em cada nó folha'''
        min_samples_leaf = [int(x) for x in np.linspace(2.0, 250.0, num=10)]  # 8 '76' 1 = default-float usa ponto 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 134
        min_samples_leaf.append(1)
        '''Número mínimo de amostras necessárias em cada nó folha'''
        bootstrap = [True]
        oob_score = [False]
        '''oob_score.append(bool)'''
        random_state = [42]
        n_jobs = [-1]
        ''' Grid search'''
        param_grid = {
            'n_estimators': n_stimators,  # número de árvores no foreset
            'max_samples': max_samples,
            'max_features': max_features,  # número máximo de recursos considerados para dividir um nó
            'max_depth': max_depth,  # número máximo de níveis em cada árvore de decisão
            'min_samples_split': min_samples_split,
            # número mínimo de pontos de dados colocados em um nó antes que o nó seja dividido
            'min_samples_leaf': min_samples_leaf,  # número mínimo de pontos de dados permitidos em um nó folha
            'bootstrap': bootstrap,  # método para amostragem de pontos de dados (com ou sem reposição)
            'oob_score': oob_score, #estimar a pontuação de generalização
            'random_state': random_state,
            'n_jobs': n_jobs,
        }
        # print(param_grid)
        '''Random Frorest'''
        floresta = RandomForestRegressor()
        # %timeit
        rf_RandomGrid = RandomizedSearchCV(estimator=floresta,
                                           param_distributions=param_grid,
                                           n_iter=10,
                                           cv=3,
                                           verbose=0,
                                           random_state=42,
                                           n_jobs=-1
                                           )
        'Treina o modelo com os Parametros da IA Grid Saerch'
        rf_RandomGrid.fit(X_treino, y_treino)
        '''Melhores Parametros da IA Grid Saerch*'''
        k = rf_RandomGrid.best_params_
        # print(k["max_depth"])
        rf_RandomGrid.best_score_
        rf_RandomGrid.cv_results_
        df = pd.DataFrame(rf_RandomGrid.cv_results_)
        # print(df)
        # print(rf_RandomGrid.cv_results_)
        y_pred_rf_rg = rf_RandomGrid.predict(X_teste)
        print("y_pred->>>>", y_pred_rf_rg[-1])
        rf_RandomGrid.score(X_teste,y_teste)
       #print(rf_RandomGrid.score(X_teste, y_teste))
        #accuracy_score(y_teste, y_pred_rf_rg)
        print("estimetor", rf_RandomGrid.best_estimator_)
        #****************Treino minha IA**********************************************
        floresta = rf_RandomGrid.best_estimator_
        floresta.fit(X_treino, y_treino)
        #****************Faco minha predicao **********************************************
        p= floresta.predict(X_teste)
        # accuracy_score(p, X_treino)
        print("p-->>", p[-1])

        #****************Decisao***********************************************
        fl1 = (mean_squared_error(y_teste, p) * 1000)
        fl2 = np.sqrt(mean_squared_error(y_teste, p) * 1000)
        print(f'floresta----~~->>{fl1:.3f}')
        print(f'floresta_2--~~->>{fl2:.3f}')
        #dif = (fl1 - last)
        avg_price = (fl1 + last) / 2
        fl = avg_price
        dif = (avg_price - last)
        print(f"avg", avg_price)
        # print(f'Ticket da posicao: {str(ticket)}')
        print(f'fl {fl:.3f} - ult_valor {last:.3f} =--~~->>{dif:.2f} dif. pontos')
        #fl1 = np.sqrt(mean_squared_error(y_teste, p))
        #print(f'floresta_Quadrada--~~->>{fl1:.3f}')
        #****************Printa os dados**********************************************
        print("*" * 100)
        print('***melhor scor_params***')
        print(f'{rf_RandomGrid.best_score_:.2f},"%"')
        print('*$*' * 30)
        #****************verificamos a presença de posições abertas*****************************
        positions_total = mt5.positions_total()
        if positions_total > 0:
            print("Total positions=", positions_total)
            print('<<Nao pode Negociar tem posicao aberta>>')
            bilhete = mt5.positions_get(symbol=symbol)[0]
            ticket = mt5.positions_get(symbol=symbol)[0][0]
            print(bilhete)
            bilhete_neg = bilhete.ticket
            price_open = bilhete.price_open
            sl = bilhete.sl
            profit = bilhete.profit
            type = bilhete.type
            print("t", type)
            price_open = bilhete.price_open
            if (type == 0):
                if (profit >= 25.0 and profit <= 30.0):
                    print('BE ativo')
                    def break_even():
                        positions = mt5.positions_get(symbol=symbol)
                        print
                        if len(positions) > 0:
                            position = positions[0]
                            ticket = position[0]
                            posicao = position[5]
                            price_open = position[10]
                            current_price = mt5.symbol_info_tick(symbol).last
                            if posicao == 0:
                                disparo = price_open + (current_price - price_open) / 2
                                request = {
                                    'action': mt5.TRADE_ACTION_SLTP,
                                    'position': ticket,
                                    'sl': disparo,
                                    'symbol': symbol
                                }
                                result = mt5.order_send(request)
                                if result.retcode != mt5.TRADE_RETCODE_DONE:
                                    print(f"Erro ao enviar o break even: {result.comment}")
                                else:
                                    print("Break even enviado com sucesso")
                    print('breack vaquero')
                elif (profit >= 30.0):
                    print('traling Stop v@quejano')
                    # Parâmetros de configuração
                    TICKET = ticket
                    TRAIL_AMOUNT = 0.5
                    MAX_DIST_SL = 25.0
                    DEFAULT_SL = 3.0

                    def trail_sl(TICKET, MAX_DIST_SL, TRAIL_AMOUNT, DEFAULT_SL):
                        # Obtém a posição pelo ticket
                        position = mt5.positions_get(ticket=TICKET)[0]
                        symbol = position.symbol
                        order_type = position.type
                        price_current = position.price_current
                        price_open = position.price_open
                        profit = position.profit
                        sl = position.sl

                        # Calcula a distância para o stop loss atual
                        dist_from_sl = round(profit, 6)

                        if dist_from_sl > MAX_DIST_SL:
                            # Calcula o novo stop loss
                            if sl != 0.0:
                                if order_type == 0:
                                    new_sl = sl + TRAIL_AMOUNT
                                elif order_type == 1:
                                    new_sl = sl - TRAIL_AMOUNT
                            else:
                                new_sl = price_open - DEFAULT_SL if order_type == 0 else price_open + DEFAULT_SL

                            request = {
                                "action": mt5.TRADE_ACTION_SLTP,
                                "position": TICKET,
                                "sl": new_sl,
                                "comment": "V@ -~~>> chicote estrala",
                                "symbol": symbol,
                            }

                            # Verifica e exibe o resultado
                            result = mt5.order_check(request)
                            resultOrd = mt5.order_send(request)
                            return result
                    # Chama a função trail_sl
                    trail_sl(TICKET, MAX_DIST_SL, TRAIL_AMOUNT, DEFAULT_SL)
                    print('traling Stop v@quejano')
            else:
                if (type == 1):
                    if (profit >=25.0 and profit <= 30.0):
                        print('BE ativo')
                        def break_even():
                            positions = mt5.positions_get(symbol=symbol)
                            prin(positions)
                            if len(positions) > 0:
                                position = positions[0]
                                ticket = position[0]
                                posicao = position[5]
                                price_open = position[10]
                                current_price = mt5.symbol_info_tick(symbol).last
                                print(current_price)
                                if posicao == 0:
                                    disparo = price_open + (current_price - price_open) / 2
                                    request = {
                                        'action': mt5.TRADE_ACTION_SLTP,
                                        'position': ticket,
                                        'sl': disparo,
                                        'symbol': symbol
                                    }
                                    result = mt5.order_send(request)
                                    if result.retcode != mt5.TRADE_RETCODE_DONE:
                                        print(f"Erro ao enviar o break even: {result.comment}")
                                    else:
                                        print("Break even enviado com sucesso")

                        print('breack vaquero')
                    elif (profit >= 30.0):
                        print('traling Stop v@quejano')
                        TICKET = ticket
                        MAX_DIST_SL = 3.0 # max distancai
                        TRAIL_AMOUNT = 0.5  # amaont
                        # Parâmetros de configuração
                        TICKET = ticket
                        TRAIL_AMOUNT = 0.5
                        MAX_DIST_SL = 50.0
                        DEFAULT_SL = 3.0

                        def trail_sl(TICKET, MAX_DIST_SL, TRAIL_AMOUNT, DEFAULT_SL):
                            # Obtém a posição pelo ticket
                            position = mt5.positions_get(ticket=TICKET)[0]
                            symbol = position.symbol
                            order_type = position.type
                            price_current = position.price_current
                            price_open = position.price_open
                            profit = position.profit
                            sl = position.sl

                            # Calcula a distância para o stop loss atual
                            dist_from_sl = round(profit, 6)

                            if dist_from_sl > MAX_DIST_SL:
                                # Calcula o novo stop loss
                                if sl != 0.0:
                                    if order_type == 0:
                                        new_sl = sl + TRAIL_AMOUNT
                                    elif order_type == 1:
                                        new_sl = sl - TRAIL_AMOUNT
                                else:
                                    new_sl = price_open - DEFAULT_SL if order_type == 0 else price_open + DEFAULT_SL

                                request = {
                                    "action": mt5.TRADE_ACTION_SLTP,
                                    "position": TICKET,
                                    "sl": new_sl,
                                    "comment": "V@ -~~>> chicote estrala",
                                    "symbol": symbol,
                                }

                                # Verifica e exibe o resultado
                                result = mt5.order_check(request)
                                resultOrd = mt5.order_send(request)
                                return result

                        # Chama a função trail_sl
                        trail_sl(TICKET, MAX_DIST_SL, TRAIL_AMOUNT, DEFAULT_SL)
                        print('traling Stop v@quejano')
        else:
            print("<<>>Nao tem posicao aberta, pode Negociar!!!")
            #Preparamos a solicitação
        #****************funcao simples compra e venda**********************************************
            def neg(fl):
                print('-->',fl)
                if (fl >= last):
                    print('Compra ^')
                    volume = 1.0
                    stp = 8000
                    tkp = 20000
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
                        "magic": 171,
                    }
                    # enviamos a solicitação de negociação
                    result = mt5.order_check(request)
                    resultOrdby = mt5.order_send(request)
                    print(result)
                    print(resultOrdby)
                elif (fl <= last):
                    print('Venda V')
                    volume = 1.0
                    stp = 8000
                    tkp = 20000
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
                        "magic": 171,
                    }
                    # Enviamos a solicitação de negociação
                    result = mt5.order_send(request)
                    print(result)
                else:
                   print('Ordem pendente')
            # ****************Valor da condicao BUY e SELL**********************************************
            a1=neg(fl)
            #Envio_ordem_C_V.neg(fl)
            print(a1)
            # ****************Ultimo valor do tick**********************************************
            ult_valor = last
            print("V@quejano IA, aqui o sistema e sertanejo...")
            print("V@quejada IA valeu o Boiii..."
                  " Se voce tem um sonho nao deixe ninguem deboxar dele nao..."
                  "Seu sonho e grande..."
                  " Foi Deus quem botou no seu Coração..."
                  "Meu Deus eu confio no sonho que o Senhor tem p minha vida..."
                  " Levanta cedo pra labuta que eu to pronto...By: Joao Gomes")
            time.sleep(10)

if __name__ == '__main__':
    #print('Iniciando o trailing Stoploss..')
    #print(f'Posição: {str(TICKET)}')
    main()
