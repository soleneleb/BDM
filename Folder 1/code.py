# Visualization
import matplotlib.pyplot as plt
import pandas as pd


%matplotlib inline
plt.style.use('fivethirtyeight')
plt.rcParams['font.size'] = 18
plt.rcParams['patch.edgecolor'] = 'k'

pd.options.display.max_columns = 150

# Read in data
train = pd.read_csv('./train.csv')
test = pd.read_csv('./test.csv')
train.head()

train.info()

