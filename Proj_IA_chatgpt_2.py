'''
import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

response = openai.Completion.create(
  model="text-davinci-003",
  prompt="Decide whether a Tweet's sentiment is positive, neutral, or negative.\n\nTweet: \"I loved the new Batman movie!\"\nSentiment:",
  temperature=0,
  max_tokens=60,
  top_p=1,
  frequency_penalty=0.5,
  presence_penalty=0
)
'''
'''
import numpy as np

#Aqui está um exemplo de código python para analisar e tomar decisões de negociação com base nos valores
# Open, High, Low, Close, Tick_Volume, Spread, Real_Volume de um título e definir um Stop Loss, Break Even e Trailing Stop 
# para proteger o lucro de 10.000 reais:

def trade_analysis(open_price, high, low, close, tick_volume, spread, real_volume):
    # Calculate the average price
    avg_price = (high + low) / 2

    # Buy and Sell entry
    buy_entry = avg_price - spread
    sell_entry = avg_price + spread

    # Stop Loss
    stop_loss = avg_price - 2 * spread

    # Break Even
    break_even = avg_price

    # Trailing Stop
    trailing_stop = np.maximum(avg_price - spread, close - (spread * 2))

    # Calculate profit
    profit = close - buy_entry

    if profit >= 10000:
        return f"Best Buy Entry: {buy_entry:.2f}, Best Sell Entry: {sell_entry:.2f}, Stop Loss: {stop_loss:.2f}, Break Even: {break_even:.2f}, Trailing Stop: {trailing_stop:.2f}, Profit: {profit:.2f}"
    else:
        return "Not a good trade opportunity"
   
 #   Nota: Este código assume que os cálculos de spread, stop loss, break even e trailing stop são baseados em alguma estratégia de negociação 
  #  e os valores e cálculos usados podem variar dependendo das necessidades e estratégias específicas do indivíduo.
'''

'''
class BTreeNode:
  def __init__(self, order):
    self.keys = []
    self.values = []
    self.children = []
    self.order = order
    self.parent = None

  def is_leaf(self):
    return len(self.children) == 0

  def is_full(self):
    return len(self.keys) == self.order - 1

  def insert_key_value(self, key, value):
    i = len(self.keys) - 1
    while i >= 0 and self.keys[i] > key:
      i -= 1
    self.keys.insert(i + 1, key)
    self.values.insert(i + 1, value)

  def insert_child(self, child):
    self.children.append(child)
    child.parent = self
'''

