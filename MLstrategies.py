# -*- coding: utf-8 -*-
"""
Created on Sat Mar 19 16:28:51 2022

@author: qinyan
"""
import pandas as pd
import yfinance as yf
import plotly.express as px
import datetime
#from datetime import date
import numpy as np
from scipy.stats import norm
from abc import ABC

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm

class DIRECTION:
    BUY = 1
    SELL = -1
    HOLD = 0


class BaseStrategy(ABC):

    def fit(self, price):
        pass

    def predict(self, price):
        pass



class ForLoopBackTester:

    def __init__(self, strategy=None, commission=0.00025):
        self.list_position = []
        self.list_cash = []
        self.list_holdings = []
        self.list_total = []

        self.long_signal = False
        # default two sides commissions
        self.commission = commission
        self.position = 0
        self.cash = 100000
        self.total = 0
        self.holdings = 0

        self.market_data_count = 0
        self.prev_price = None
        self.statistical_model = None
        # self.historical_data = pd.DataFrame(columns=['Trade', 'Price', 'OpenClose', 'HighLow'])
        self.strategy = strategy

    def on_market_data_received(self, price_update):
        #print("into")
        if self.strategy:
            #print(price_update)
            self.strategy.fit(price_update)
            predicted_value = self.strategy.predict(price_update)
        else:
            predicted_value = DIRECTION.HOLD

        if predicted_value == DIRECTION.BUY:
            return 'buy'
        if predicted_value == DIRECTION.SELL:
            return 'sell'
        return 'hold'

    def buy_sell_or_hold_something(self, price_update, action):
        if action == 'buy':
            cash_needed = 10 * price_update['Close'] * (1 + self.commission)  # commission
            if self.cash - cash_needed >= 0:
                #print(str(price_update['Date']) +
                      #" send buy order for 10 shares price=%.2f" % (price_update['Close']))
                self.position += 10
                self.cash -= cash_needed
            #else:
                #print('buy impossible because not enough cash')

        if action == 'sell':
            position_allowed = 10
            if self.position - position_allowed >= 0:  # -position_allowed??
                #(str(price_update['Date']) +
                     # " send sell order for 10 shares price=%.2f" % (price_update['Close']))
                self.position -= position_allowed
                self.cash -= -position_allowed * price_update['Close'] * (1 + self.commission)  # commission
            #else:
                #('sell impossible because not enough position')

        self.holdings = self.position * price_update['Close']
        self.total = (self.holdings + self.cash)
        # print('%s total=%d, holding=%d, cash=%d' %
        #       (str(price_update['date']),self.total, self.holdings, self.cash))

        self.list_position.append(self.position)
        self.list_cash.append(self.cash)
        self.list_holdings.append(self.holdings)
        self.list_total.append(self.holdings + self.cash)
def price_volume_deviation(price, volume, win=10):
    """
    deviation of price and volume
    = -1 * corr(price(last 10 days), volume(last 10 days))
    :param price:
    :param volume:
    :param win:
    :returnd
    """
    factor = price.rolling(win).corr(volume)
    return factor


def opening_price_gap(close_price, open_price):
    """
    gap between today's open price and previous day's close price
    = open_price(today) / close_price(last day)
    :param close_price:
    :param open_price:
    :return:
    """
    factor = open_price / close_price.shift(1)
    return factor


def abnormal_volume(volume, win=10):
    """
    abnormal volume within previous 10 days
    = -1 volume(today) / mean(volume (last 10 days))
    :param win:
    :param volume:
    :return:
    """
    factor = volume / volume.rolling(win).mean()
    return factor


def volume_swing_deviation(volume, high, low, win=10):
    """
    = -1 * corr(high(last 10 days) / low(last 10 days), volume(last  10 days))
    :param volume:
    :param high:
    :param low:
    :param win:
    :return:
    """
    amplitude = high / low
    factor = -1 * amplitude.rolling(win).corr(volume)
    return factor


def volume_reverse(volume, win=10):
    """
    = ts_min(-volume, 10)
    :param win:
    :param volume:
    :return:
    """
    factor = -volume.rolling(win).min()
    return factor


def price_reverse(high, close, win=10):
    """
    ts_corr(-high, log(close), 5)
    :param high:
    :param close:
    :param win:
    :return:
    """
    factor = high.rolling(win).corr(close)
    return factor

def WVAD(close,op,high,low,volume):
    return ((close-op)+(high-low)*volume)
#%% Naive Strategies



class NaiveStrategy(BaseStrategy):

    def __init__(self):
        super().__init__()
        self.buy = True
        self.count = -1

    def fit(self, price):
        pass

    def predict(self, price):
        self.count += 1
        if ((self.count % 5) == 0) and (self.buy is True):
            self.buy = False
            return DIRECTION.BUY
        elif ((self.count % 5) == 0) and (self.buy is False):
            self.buy = True
            return DIRECTION.SELL
        else:
            return DIRECTION.HOLD
#%% Other Strategies
class LogisticStrategy(BaseStrategy):

    def __init__(self):
        super().__init__()
        self.dataframe = pd.DataFrame()
        self.prev_price = None
        self.buy = True
        self.model = None
        self.prediction = None

    def fit(self, price_update):
        price = pd.DataFrame.from_records(
            [
                {
                    'label': 1 if self.prev_price is None or price_update['Close'] > self.prev_price else -1,
                    'price': price_update['Close'],
                    'volume': price_update['Volume'],
                    'open_close': price_update['Open'] - price_update['Close'],
                    'high_low': price_update['High'] - price_update['Low'],
                    'price_volume_deviation': price_update['price_vol_deviation'],
                    'opening_price_gap': price_update['opening_price_gap'],
                    'abnormal_volume': price_update['abnormal_volume'],
                    'volume_swing_deviation': price_update['volume_swing_deviation'],
                    'volume_reverse': price_update['volume_reverse'],
                    'price_reverse': price_update['price_reverse']
                }
            ]
        )

        self.dataframe = pd.concat([self.dataframe, price])
        self.prev_price = price.iloc[-1]['price']

        # fit model with data until previous day
        if len(self.dataframe) > 150:
            self.model = LogisticRegression().fit(
                self.dataframe.drop(['label'], axis=1).iloc[:-1], self.dataframe['label'].iloc[:-1])

    def predict(self, price_update):
        price = pd.DataFrame.from_records(
            [
                {
                    'label': 1 if price_update['Close'] > self.prev_price else -1,
                    'price': price_update['Close'],
                    'volume': price_update['Volume'],
                    'open_close': price_update['Open'] - price_update['Close'],
                    'high_low': price_update['High'] - price_update['Low'],
                    'price_volume_deviation': price_update['price_vol_deviation'],
                    'opening_price_gap': price_update['opening_price_gap'],
                    'abnormal_volume': price_update['abnormal_volume'],
                    'volume_swing_deviation': price_update['volume_swing_deviation'],
                    'volume_reverse': price_update['volume_reverse'],
                    'price_reverse': price_update['price_reverse']
                    }
            ]
        )

        if self.model is not None:
            self.prediction = self.model.predict(price.drop(['label'], axis=1))
            if (self.prediction == 1) and (self.buy is True):
                self.buy = False
                return DIRECTION.BUY
            elif (self.prediction == -1) and (self.buy is False):
                self.buy = True
                return DIRECTION.SELL
            else:
                return DIRECTION.HOLD
        else:
            return DIRECTION.HOLD

class SVMStrategy_sigmoid(BaseStrategy):

    def __init__(self):
        super().__init__()
        self.dataframe = pd.DataFrame()
        self.prev_price = None
        self.buy = True
        self.model = None
        self.prediction = None

    def fit(self, price_update):
        price = pd.DataFrame.from_records(
            [
                {
                    'label': 1 if self.prev_price is None or price_update['Close'] > self.prev_price else -1,
                    'price': price_update['Close'],
                    'volume': price_update['Volume'],
                    'open_close': price_update['Open'] - price_update['Close'],
                    'high_low': price_update['High'] - price_update['Low'],
                    'price_volume_deviation': price_update['price_vol_deviation'],
                    'opening_price_gap': price_update['opening_price_gap'],
                    'abnormal_volume': price_update['abnormal_volume'],
                    'volume_swing_deviation': price_update['volume_swing_deviation'],
                    'volume_reverse': price_update['volume_reverse'],
                    'price_reverse': price_update['price_reverse']
                }
            ]
        )

        self.dataframe = pd.concat([self.dataframe, price])
        self.prev_price = price.iloc[-1]['price']

        # fit model with data until previous day
        if len(self.dataframe) >150:
            self.model = svm.SVC(kernel='sigmoid',).fit(
                self.dataframe.drop(['label'], axis=1).iloc[:-1], self.dataframe['label'].iloc[:-1])

    def predict(self, price_update):
        price = pd.DataFrame.from_records(
            [
                {
                    'label': 1 if price_update['Close'] > self.prev_price else -1,
                    'price': price_update['Close'],
                    'volume': price_update['Volume'],
                    'open_close': price_update['Open'] - price_update['Close'],
                    'high_low': price_update['High'] - price_update['Low'],
                    'price_volume_deviation': price_update['price_vol_deviation'],
                    'opening_price_gap': price_update['opening_price_gap'],
                    'abnormal_volume': price_update['abnormal_volume'],
                    'volume_swing_deviation': price_update['volume_swing_deviation'],
                    'volume_reverse': price_update['volume_reverse'],
                    'price_reverse': price_update['price_reverse']
                    }
            ]
        )

        if self.model is not None:
            self.prediction = self.model.predict(price.drop(['label'], axis=1))
            if (self.prediction == 1) and (self.buy is True):
                self.buy = False
                return DIRECTION.BUY
            elif (self.prediction == -1) and (self.buy is False):
                self.buy = True
                return DIRECTION.SELL
            else:
                return DIRECTION.HOLD
        else:
            return DIRECTION.HOLD

class SVMStrategy_rbf(BaseStrategy):

    def __init__(self):
        super().__init__()
        self.dataframe = pd.DataFrame()
        self.prev_price = None
        self.buy = True
        self.model = None
        self.prediction = None

    def fit(self, price_update):
        price = pd.DataFrame.from_records(
            [
                {
                    'label': 1 if self.prev_price is None or price_update['Close'] > self.prev_price else -1,
                    'price': price_update['Close'],
                    'volume': price_update['Volume'],
                    'open_close': price_update['Open'] - price_update['Close'],
                    'high_low': price_update['High'] - price_update['Low'],
                    'price_volume_deviation': price_update['price_vol_deviation'],
                    'opening_price_gap': price_update['opening_price_gap'],
                    'abnormal_volume': price_update['abnormal_volume'],
                    'volume_swing_deviation': price_update['volume_swing_deviation'],
                    'volume_reverse': price_update['volume_reverse'],
                    'price_reverse': price_update['price_reverse']
                }
            ]
        )

        self.dataframe = pd.concat([self.dataframe, price])
        self.prev_price = price.iloc[-1]['price']

        # fit model with data until previous day
        if len(self.dataframe) > 150:
            self.model = svm.SVC(kernel='rbf',).fit(
                self.dataframe.drop(['label'], axis=1).iloc[:-1], self.dataframe['label'].iloc[:-1])

    def predict(self, price_update):
        price = pd.DataFrame.from_records(
            [
                {
                    'label': 1 if price_update['Close'] > self.prev_price else -1,
                    'price': price_update['Close'],
                    'volume': price_update['Volume'],
                    'open_close': price_update['Open'] - price_update['Close'],
                    'high_low': price_update['High'] - price_update['Low'],
                    'price_volume_deviation': price_update['price_vol_deviation'],
                    'opening_price_gap': price_update['opening_price_gap'],
                    'abnormal_volume': price_update['abnormal_volume'],
                    'volume_swing_deviation': price_update['volume_swing_deviation'],
                    'volume_reverse': price_update['volume_reverse'],
                    'price_reverse': price_update['price_reverse']
                    }
            ]
        )

        if self.model is not None:
            self.prediction = self.model.predict(price.drop(['label'], axis=1))
            if (self.prediction == 1) and (self.buy is True):
                self.buy = False
                return DIRECTION.BUY
            elif (self.prediction == -1) and (self.buy is False):
                self.buy = True
                return DIRECTION.SELL
            else:
                return DIRECTION.HOLD
        else:
            return DIRECTION.HOLD

class GradientBootStrategy(BaseStrategy):

    def __init__(self):
        super().__init__()
        self.dataframe = pd.DataFrame()
        self.prev_price = None
        self.buy = True
        self.model = None
        self.prediction = None

    def fit(self, price_update):
        price = pd.DataFrame.from_records(
            [
                {
                    'label': 1 if self.prev_price is None or price_update['Close'] > self.prev_price else -1,
                    'price': price_update['Close'],
                    'volume': price_update['Volume'],
                    'open_close': price_update['Open'] - price_update['Close'],
                    'high_low': price_update['High'] - price_update['Low'],
                    'price_volume_deviation': price_update['price_vol_deviation'],
                    'opening_price_gap': price_update['opening_price_gap'],
                    'abnormal_volume': price_update['abnormal_volume'],
                    'volume_swing_deviation': price_update['volume_swing_deviation'],
                    'volume_reverse': price_update['volume_reverse'],
                    'price_reverse': price_update['price_reverse']
                }
            ]
        )

        self.dataframe = pd.concat([self.dataframe, price])
        self.prev_price = price.iloc[-1]['price']

        # fit model with data until previous day
        if len(self.dataframe) > 150:
            self.model = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,max_depth=1, random_state=0).fit(
                self.dataframe.drop(['label'], axis=1).iloc[:-1], self.dataframe['label'].iloc[:-1])

    def predict(self, price_update):
        price = pd.DataFrame.from_records(
            [
                {
                    'label': 1 if price_update['Close'] > self.prev_price else -1,
                    'price': price_update['Close'],
                    'volume': price_update['Volume'],
                    'open_close': price_update['Open'] - price_update['Close'],
                    'high_low': price_update['High'] - price_update['Low'],
                    'price_volume_deviation': price_update['price_vol_deviation'],
                    'opening_price_gap': price_update['opening_price_gap'],
                    'abnormal_volume': price_update['abnormal_volume'],
                    'volume_swing_deviation': price_update['volume_swing_deviation'],
                    'volume_reverse': price_update['volume_reverse'],
                    'price_reverse': price_update['price_reverse']
                    }
            ]
        )

        if self.model is not None:
            self.prediction = self.model.predict(price.drop(['label'], axis=1))
            if (self.prediction == 1) and (self.buy is True):
                self.buy = False
                return DIRECTION.BUY
            elif (self.prediction == -1) and (self.buy is False):
                self.buy = True
                return DIRECTION.SELL
            else:
                return DIRECTION.HOLD
        else:
            return DIRECTION.HOLD


class RandomForestStrategy(BaseStrategy):

    def __init__(self):
        super().__init__()
        self.dataframe = pd.DataFrame()
        self.prev_price = None
        self.buy = True
        self.model = None
        self.prediction = None

    def fit(self, price_update):
        price = pd.DataFrame.from_records(
            [
                {
                    'label': 1 if self.prev_price is None or price_update['Close'] > self.prev_price else -1,
                    'price': price_update['Close'],
                    'volume': price_update['Volume'],
                    'open_close': price_update['Open'] - price_update['Close'],
                    'high_low': price_update['High'] - price_update['Low'],
                    'price_volume_deviation': price_update['price_vol_deviation'],
                    'opening_price_gap': price_update['opening_price_gap'],
                    'abnormal_volume': price_update['abnormal_volume'],
                    'volume_swing_deviation': price_update['volume_swing_deviation'],
                    'volume_reverse': price_update['volume_reverse'],
                    'price_reverse': price_update['price_reverse']
                }
            ]
        )

        self.dataframe = pd.concat([self.dataframe, price])
        self.prev_price = price.iloc[-1]['price']

        # fit model with data until previous day
        if len(self.dataframe) > 150:
            self.model = RandomForestClassifier(max_depth=5, random_state=0).fit(
                self.dataframe.drop(['label'], axis=1).iloc[:-1], self.dataframe['label'].iloc[:-1])

    def predict(self, price_update):
        price = pd.DataFrame.from_records(
            [
                {
                    'label': 1 if price_update['Close'] > self.prev_price else -1,
                    'price': price_update['Close'],
                    'volume': price_update['Volume'],
                    'open_close': price_update['Open'] - price_update['Close'],
                    'high_low': price_update['High'] - price_update['Low'],
                    'price_volume_deviation': price_update['price_vol_deviation'],
                    'opening_price_gap': price_update['opening_price_gap'],
                    'abnormal_volume': price_update['abnormal_volume'],
                    'volume_swing_deviation': price_update['volume_swing_deviation'],
                    'volume_reverse': price_update['volume_reverse'],
                    'price_reverse': price_update['price_reverse']
                    }
            ]
        )

        if self.model is not None:
            self.prediction = self.model.predict(price.drop(['label'], axis=1))
            if (self.prediction == 1) and (self.buy is True):
                self.buy = False
                return DIRECTION.BUY
            elif (self.prediction == -1) and (self.buy is False):
                self.buy = True
                return DIRECTION.SELL
            else:
                return DIRECTION.HOLD
        else:
            return DIRECTION.HOLD
#%% Test all of the strategies
def mystrategy(start_date,symbol="FB",strategy=NaiveStrategy()):
    """
    back testing
    :param title: str; strategy name
    :param symbol: str; ticker name
    :param nb_of_rows: int; number of rows in training data
    :param strategy: Class; i.e. NaiveStrategy
    :return: plot and matrix
    """
    # initialize
    # nb_of_rows = 1800
    
    back_tester = ForLoopBackTester(strategy)

    # select price volume data of facebook from yahoo finance
    
    ticker = yf.Ticker(symbol)

    # initialize single input data (of 1 min frequency)
    df_ticker = ticker.history(
        #period='400d',
        interval='1d',
        start=start_date,
        end=None,
        actions=True,
        auto_adjust=True,
        back_adjust=False).drop(["Dividends", "Stock Splits"], axis=1)
    
    # initialize evaluation matrix for strategy performance
    matrix = pd.DataFrame(columns=["signal", "position", "close_price", "cash", "holdings", "total", "pnl"],dtype='object')
   
    # initialize dataframes for factors or parameters
    df_ticker["price_vol_deviation"] = price_volume_deviation(df_ticker.Close, df_ticker.Volume)
    df_ticker["opening_price_gap"] = opening_price_gap(df_ticker.Close, df_ticker.Open)
    df_ticker["abnormal_volume"] = abnormal_volume(df_ticker.Volume)
    df_ticker["volume_swing_deviation"] = volume_swing_deviation(df_ticker.Volume, df_ticker.High, df_ticker.Low)
    df_ticker["volume_reverse"] = volume_reverse(df_ticker.Volume)
    df_ticker["price_reverse"] = price_reverse(df_ticker.High, df_ticker.Close)
    df_ticker = df_ticker.dropna().reset_index()
    #("First:",df_ticker.Date)
    # initialize single input data (of 1 min frequency)
    #print(df_ticker)
    for date in df_ticker.Date:
        #print("11111")
     
        price_information = dict(df_ticker[df_ticker.Date== date].iloc[0])
        
        #print("price:",price_information)
        # set buy or sell signal and run the strategy
        action = back_tester.on_market_data_received(price_information)
        back_tester.buy_sell_or_hold_something(price_information, action)

        # fill the data
        matrix.loc[date] = [
            action,
            back_tester.list_position[-1],
            price_information["Close"],
            back_tester.list_cash[-1],
            back_tester.list_holdings[-1],
            back_tester.list_total[-1],
            back_tester.list_total[-1] - 100000
        ]
    pnl=(back_tester.list_total[-1] - 10000)
    #print("PNL:",pnl)
    pnl_list=[(i-10000) for i in back_tester.list_total]

    # sharpe ratio for the strategy
    sharpe_ratio = matrix['total'].pct_change().mean() / (matrix['total'].pct_change().std()+0.00000000001)

    # initialize decay matrix
    dcy_mat = pd.DataFrame()
    buy_idx = matrix[(matrix["signal"] == "buy")].index
    for idx in buy_idx:
        idx_num = matrix.index.get_loc(idx)
        dcy_win = matrix.iloc[idx_num: idx_num + 5]["pnl"] - matrix.iloc[idx_num]["pnl"]
        dcy_win = dcy_win.reset_index()["pnl"]
        dcy_mat = pd.concat([dcy_mat, dcy_win], axis=1)
    sell_idx = matrix[(matrix["signal"] == "sell")].index
    for idx in sell_idx:
        idx_num = matrix.index.get_loc(idx)
        dcy_win = matrix.iloc[idx_num: idx_num + 5]["pnl"] - matrix.iloc[idx_num]["pnl"]
        dcy_win = -dcy_win.reset_index()["pnl"]
        dcy_mat = pd.concat([dcy_mat, dcy_win], axis=1)
    '''
    # plot decay
    dcy_mat.plot(
        legend=False, figsize=(7, 5), grid=True, title="Signal Decay" + title)#, xlabel="minute", ylabel="PnL"
    '''
    #plot pnl
    fig_pnl=px.line(pnl_list)
    return fig_pnl,matrix, dcy_mat, sharpe_ratio,pnl,pnl_list

def find_strategy_class(name_of_strategy="NaiveStrategy"):
    strategy_dict={"NaiveStrategy":NaiveStrategy(),"LogisticStrategy":LogisticStrategy(),"SVMStrategy_sigmoid":SVMStrategy_sigmoid(),"SVMStrategy_rbf":SVMStrategy_rbf(),"GradientBootStrategy":GradientBootStrategy(),"RandomForestStrategy":RandomForestStrategy()}
    return strategy_dict[name_of_strategy]