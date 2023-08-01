import numpy as np
import pandas as pd
import MetaTrader5 as mt5
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn.ensemble
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, classification_report, confusion_matrix

## Configurar colunas do DataFrame
pd.set_option('display.max_columns', 400)  # número de colunas mostradas
pd.set_option('display.width', 1500)  # largura máxima da tabela exibida

## Função para obter dados OHLC
def get_ohlc(ativo, timeframe, n=10):
    ativo = mt5.copy_rates_from_pos(ativo, timeframe, 0, n)
    ativo = pd.DataFrame(ativo)
    ativo['time'] = pd.to_datetime(ativo['time'], unit='s')
    ativo.set_index('time', inplace=True)
    return ativo

## Informações do ativo
ativo = 'WDO$'
symbol = "WDOQ23"

## Obter dados OHLC e salvar em um arquivo CSV
#tabela = get_ohlc(ativo, mt5.TIMEFRAME_M1, 99999)
#tabela.to_csv('WDO.CSV')  # criar um banco de dados em Excel
tabela = pd.read_csv('WDO.CSV', index_col=0)
tabela.dropna(inplace=True)
## Preparar dados para o modelo
tabela['RSI1'] = tabela['close'].rolling(2).median()
tabela['RSI2'] = tabela['close'].rolling(7).median()
tabela['RSI3'] = tabela['close'].rolling(21).median()
tabela.dropna(inplace=True)
tabela = tabela.drop(["spread", "tick_volume", "real_volume", "high", "low", "open",], axis=1)
tabela['signal'] = np.where(
    (tabela['RSI1'] < tabela['close']) &
    (tabela['RSI2'] < tabela['close']) &
    (tabela['RSI3'] < tabela['close']),  # Sell signal condition
    -1,  # Sell signal
    np.where(
        (tabela['RSI1'] > tabela['close']) &
        (tabela['RSI2'] > tabela['close']) &
        (tabela['RSI3'] > tabela['close']),  # Buy signal condition
        1,  # Buy signal
        0  # No signal
    )
)
#print(tabela)
## Prepare features and target variable for the model
X = tabela.drop(['signal'], axis=1)

y = tabela['signal'].values

## Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

## Train a Random Forest Classifier
rf_classifier = RandomForestClassifier(random_state=42)
rf_classifier.fit(X_train, y_train)

y_pred = rf_classifier.predict(X_test)

# Backtest: Calcular o resultado financeiro da estratégia
def calcular_resultado_financeiro(dados, previsoes, custo_operacao=0.0):
    # Inicialização de variáveis
    saldo = 0
    acoes = 0
    historico_saldo = [saldo]

    # Iterar sobre os dados e tomar decisões com base nas previsões do modelo
    for i in range(len(dados)):
        sinal = previsoes[i]
        preco_fechamento = dados.iloc[i]['close']

        if sinal == 1:  # Comprar
            saldo -= preco_fechamento  # Deduzir o preço de compra do saldo
            acoes += 1  # Adicionar uma ação à carteira

        elif sinal == -1:  # Vender
            saldo += preco_fechamento * acoes  # Adicionar o preço de venda ao saldo
            acoes = 0  # Limpar a carteira de ações

        saldo -= custo_operacao  # Deduzir custos de operação (opcional)
        #historico_saldo.append(saldo)
        historico_saldo.append(saldo)

    return historico_saldo

# Realizar o backtest apenas no conjunto de teste
historico_saldo = calcular_resultado_financeiro(X_test, y_pred)
# Resultado final após o backtest
resultado_final = historico_saldo[-1]
print(f'resultado_final: {resultado_final :.2f}')
# Verificar se houve lucro ou prejuízo
if resultado_final >= 0:
    print("Parabéns! A estratégia gerou lucro.")
else:
    print("A estratégia resultou em prejuízo.")

'''grafico'''
'''
#tabela.reset_index('time', inplace=True)#reseta index
tabela1 = tabela.iloc[-50:]
print(tabela1.index)
#print(tabela1)
figure = plt.figure(figsize=(12, 6))
plt.plot(tabela1.index, tabela1['close'], label='Preço de Fechamento', color='blue')
plt.plot(tabela1.index, tabela1['RSI1'], label=f'RSI1', color='black')
plt.plot(tabela1.index, tabela1['RSI2'], label=f'RSI2', color='orange')
plt.plot(tabela1.index, tabela1['RSI3'], label=f'RSI3', color='red')
## Marca os pontos de compra e venda no gráfico
dados1 = tabela1

plt.plot(dados1[dados1['signal'] == -1].index, dados1['close'][dados1['signal'] == -1], '^', markersize=10, color='g', label='Sinal de Compra')
plt.plot(dados1[dados1['signal'] == 1].index, dados1['close'][dados1['signal'] == 1], 'v', markersize=10, color='r', label='Sinal de Venda')
'''
'''
x = tabela1['time']
y =tabela1["RSI1" ]
x1 = tabela1['time']
y1 =tabela1["RSI2"]
x2 = tabela1['time']
y2 =tabela1["RSI3"]
x3 = tabela1['time']
y3 =tabela1["close"]

plt.plot(x, y, label='RSI1', linewidth=2)
plt.plot(x1, y1, label='RSI2', linewidth=2)
plt.plot(x2, y2, label='RSI3', linewidth=2)
plt.plot(x3, y3, label='close', linewidth=2)
## Marca os pontos de compra e venda no gráfico
plt.plot(dados[dados['sinal_compra'] == 1].index, dados['close'][dados['sinal_compra'] == 1], '^', markersize=10, color='g', label='Sinal de Compra')
plt.plot(dados[dados['sinal_venda'] == -1].index, dados['close'][dados['sinal_venda'] == -1], 'v', markersize=10, color='r', label='Sinal de Venda')
'''
plt.title('info')
plt.ylabel('RSI')
plt.xlabel("time")
plt.legend()
plt.show()

'''
#X = tabela.drop(columns=["close"], axis=1).fillna(0)
#X = tabela.drop(columns=["close"], axis=1).fillna(0)
X = tabela[['RSI1', 'RSI2', 'RSI3']]

#y = tabela['close']  # coluna de labels para treinamento
y = tabela['close'].values

## Dividir dados em treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

## Treinar modelo Random Forest (Regressão)
model = sklearn.ensemble.RandomForestRegressor()
model.fit(X_train, y_train)
## Fazer previsões
y_pred = model.predict(X_test)
print("predicao:", y_pred[-1])
## Avaliar desempenho do modelo de regressão
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print("Coeficiente de Determinação (R2):", r2)
print("Erro Quadrático Médio (MSE):", mse)
## Criar gráfico de boxplot para os dados de fechamento do ativo



## Prepare features and target variable for the model
X = tabela.drop(['signal'], axis=1)
y = tabela['signal']

## Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

## Train a Random Forest Classifier
rf_classifier = RandomForestClassifier(random_state=42)
rf_classifier.fit(X_train, y_train)

## Make predictions on the test set
y_pred = rf_classifier.predict(X_test)


## Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
'''