import plotly.plotly as py
#from plotly.grid_objs import Grid, Column
from plotly.tools import FigureFactory as FF

import pandas as pd
import time

tabela = pd.read_csv('WDO.CSV', index_col=0)
