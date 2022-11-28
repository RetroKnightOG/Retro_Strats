# --- Do not remove these libs ---
from functools import reduce
import re
from typing import Optional, Union
import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy as np
# --------------------------------
import talib.abstract as ta
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
from datetime import datetime, timedelta
from freqtrade.persistence import Trade
from pandas import DataFrame, Series, DatetimeIndex, merge
from freqtrade.strategy import merge_informative_pair, CategoricalParameter, DecimalParameter, IntParameter, stoploss_from_open
from user_data.strategies.Custom_inidcators3 import RMI, TKE, laguerre, osc, vfi, vwmacd, mmar, VIDYA, madrid_sqz
from user_data.strategies.Custom_indicators2 import TA2
from user_data.strategies.custom_indicators import WaveTrend
import user_data.strategies.Custom_inidcators3 as CT3

#=========================================================================================================================================================|
# Circles a strat based on live manual trading most effeicent.                                                                                            |
# 1. Madrid Ribbon Default numbers for indicating uptrend                                                                                                 |
# 2. Vidya for providing a more rounded MA however more to help with finding more of a for sure. {18,close}                                                |
# 3. RMI (Relative Momentum Index) in the name for reason {25,3}                                                                                          |
# 4. Bollinger Bands main focus for this is to use the Moving average in the middle to provide the use of 1h timeframe to check 15 to 1h difference {50,2}|
# ALL IN THE 15m TIMEFRAME                                                                                                                                |
# Made by $NAZMONEY$ RetroKnight                                                                                                                          |
#=========================================================================================================================================================|


class Circles(IStrategy):

    # ROI table:
    minimal_roi = {
      "0": 0.99
    #   "40": 0.09,
    #   "88": 0.023,
    #   "193": 0
    }

    stoploss = -0.99

    # Timeframes:
    timeframe = '15m'
    inf_timeframe = '15m'
    

    # Trailing stoploss
    trailing_stop = False
    trailing_stop_positive = 0.337
    trailing_stop_positive_offset = 0.433
    trailing_only_offset_is_reached = True  # Disabled / not configured


    use_sell_signal = True
    sell_profit_only = False
    sell_profit_offset = 0.001 # it doesn't meant anything, just to guarantee there is a minimal profit.
    ignore_roi_if_buy_signal = False

    # Using Custom Stoploss
    use_custom_stoploss = False


    # Number of candles the strategy requires before producing valid signals 6 hours of data before providing vaild solid signals.
    startup_candle_count: int = 24

    # Optional order type mapping.
    order_types = {
        'buy': 'limit',
        'sell': 'limit',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }

    # FOR PLOTTIN PURPOSES:
    plot_config = {
        'main_plot': {
            'SAR': {'color': 'green'},
            # 'bbands_mid': {'color': 'green'},
            # 'bbands_upper': {'color': 'red'},
            # 'bb_lowerband1': {'color': 'red'},
            # 'bb_middleband1': {'color': 'green'},
        },
        'subplots': {
            'MMAR' :{
                # 'sqz_cma_c': {'color': 'yellow'},
                # 'sqz_rma_c': {'color': 'blue'},
                # 'sqz_sma_c': {'color': 'green'},
                # 'rsi': {'color': 'green'},
                'KaufOSCI': {'color': 'yellow'},
                # 'leadMA': {},
            }
        }
    }

    ############################################################################

    """
    Informative Pair Definitions
    """

    def informative_pairs(self):
        pairs = self.dp.current_whitelist()
        informative_pairs = []
        infs = {}
        for pair in pairs:
            inf_pair = self.getInformative(pair)
            # informative_pairs += [(pair, self.inf_timeframe)]
            if (inf_pair != ""):
                infs[inf_pair] = (inf_pair, self.inf_timeframe)

        informative_pairs = list(infs.values())

        # print("informative_pairs: ", informative_pairs)

        return informative_pairs

    ############################################################################


    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        # Base pair informative timeframe indicators
        curr_pair = metadata['pair']

        # only process if long or short (not 'normal')
        if (self.isBull(curr_pair)) or (self.isBear(curr_pair)):
            inf_pair = self.getInformative(curr_pair)
            # print("pair: ", curr_pair, " inf_pair: ", inf_pair)

            inf_slow = self.dp.get_pair_dataframe(pair=inf_pair, timeframe=self.inf_timeframe)
            inf_fast = self.dp.get_pair_dataframe(pair=inf_pair, timeframe=self.timeframe)

            dataframe['VIDYA'] = VIDYA(dataframe, length=18)
            dataframe['KaufOSCI'] = TA2.ER(dataframe, period=20)
            dataframe['SAR'] = TA2.SAR(dataframe, af=0.0061,amax=0.2) # This indicator is great since when (af) is less than 0.01 and amax as default
            dataframe['RMI'] = RMI(dataframe, length=25, mom=3)

            dataframe['rmi'] = RMI(dataframe, length=25, mom=3)
            dataframe['rmi-up'] = np.where(dataframe['rmi'] >= dataframe['rmi'].shift(), 1, 0)
            dataframe['rmi-up-trend'] = np.where(dataframe['rmi-up'].rolling(5).sum() >= 3, 1, 0)
            dataframe['rmi-dn-trend'] = np.where(dataframe['rmi-up'].rolling(5).sum() <= 2, 1, 0)

        # # CT indicators 2
        # # dataframe['APZ'] = TA2.APZ(dataframe)
        # dataframe['SQZ'] = TA2.SQZMI(dataframe)
        # dataframe['VPT'] = TA2.VPT(dataframe)
        # dataframe['VFI'] = TA2.VFI(dataframe)

        # bollinger1 = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        # dataframe['bb_lowerband1'] = bollinger1['lower']
        # dataframe['bb_middleband1'] = bollinger1['mid']
        # dataframe['bb_upperband1'] = bollinger1['upper']

        # WaveTrend = TA2.WTO(dataframe, channel_length=10, average_length=47)
        # dataframe['wt1'] = WaveTrend['WT1.']
        # dataframe['wt2'] = WaveTrend['WT2.']

        # # dataframe["sqz_cma_c"], dataframe["sqz_rma_c"], dataframe["sqz_sma_c"] = CT3.madrid_sqz(dataframe)

        dataframe["leadMA"], dataframe["ma10_c"], dataframe["ma20_c"],dataframe["ma30_c"], dataframe["ma40_c"],dataframe["ma50_c"],dataframe["ma60_c"],dataframe["ma70_c"],dataframe["ma80_c"],dataframe["ma90_c"] = CT3.mmar(dataframe)
       
        # dataframe['basic_ub'], dataframe['basic_lb'], dataframe['final_ub'], dataframe['final_lb'] = CT3.PMAX(dataframe)
        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        short_conditions = []
        long_conditions = []
        dataframe.loc[:, 'enter_tag'] = ''

        # 'Bull'/long leveraged token
        if self.isBull(metadata['pair']):

            # volume check
            long_conditions.append(dataframe['volume'] > 0)

            # # Trend
            # long_conditions.append(dataframe['inf_candle-dn-trend'] == 1)

            # # DWT triggers in long_dwt this is a variable that can be called if conditions are appended according to if.isBull
            # long_dwt_cond = (

            #     qtpylib.crossed_above(dataframe['dwt_model_diff'], self.entry_long_dwt_diff.value)
            # )

            # # DWTs will spike on big gains, so try to constrain
            # long_spike_cond = (
            #         dataframe['dwt_model_diff'] < 2.0 * self.entry_long_dwt_diff.value
            # )

            # long_conditions.append(long_dwt_cond)
            # long_conditions.append(long_spike_cond)

            # # set entry tags
            # dataframe.loc[long_dwt_cond, 'enter_tag'] += 'long_dwt_entry ' # naming the variable that makes the trade

            if long_conditions:
                dataframe.loc[reduce(lambda x, y: x & y, long_conditions), 'enter_long'] = 0

        # 'Bear'/short leveraged token
        elif self.isBear(metadata['pair']):

            # volume check
            short_conditions.append(dataframe['volume'] > 0)

            # # Trend
            # short_conditions.append(dataframe['inf_candle-up-trend'] == 1)

            # # DWT triggers
            # short_dwt_cond = (
            #         qtpylib.crossed_below(dataframe['dwt_model_diff'], self.entry_short_dwt_diff.value)
            # )


            # # DWTs will spike on big gains, so try to constrain
            # short_spike_cond = (
            #         dataframe['dwt_model_diff'] > 2.0 * self.entry_short_dwt_diff.value
            # )

            # short_conditions.append(short_dwt_cond)
            # short_conditions.append(short_spike_cond)

            # # set entry tags
            # dataframe.loc[short_dwt_cond, 'enter_tag'] += 'short_dwt_entry '

            if short_conditions:
                dataframe.loc[reduce(lambda x, y: x & y, short_conditions), 'enter_long'] = 0

        else:
            dataframe.loc[(dataframe['close'].notnull()), 'enter_long'] = 0

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        short_conditions = []
        long_conditions = []
        dataframe.loc[:, 'exit_tag'] = ''

        # 'Bull'/long leveraged token
        if self.isBull(metadata['pair']):

            # # DWT triggers
            # long_dwt_cond = (
            #         qtpylib.crossed_below(dataframe['dwt_model_diff'], self.exit_long_dwt_diff.value)
            # )

            # # DWTs will spike on big gains, so try to constrain
            # long_spike_cond = (
            #         dataframe['dwt_model_diff'] > 2.0 * self.exit_long_dwt_diff.value
            # )

            # long_conditions.append(long_dwt_cond)
            # long_conditions.append(long_spike_cond)

            # # set exit tags
            # dataframe.loc[long_dwt_cond, 'exit_tag'] += 'long_dwt_exit '

            if long_conditions:
                dataframe.loc[reduce(lambda x, y: x & y, long_conditions), 'exit_long'] = 0

        # 'Bear'/short leveraged token
        elif self.isBear(metadata['pair']):

            # note that these aren't true 'short' pairs, they just leverage in the short direction.
            # In other words, the conditions are the same as or bull/long pairs, just with independent hyperparameters

            # DWT triggers
            # short_dwt_cond = (
            #     qtpylib.crossed_above(dataframe['dwt_model_diff'], self.exit_short_dwt_diff.value)
            # )


            # # DWTs will spike on big gains, so try to constrain
            # short_spike_cond = (
            #         dataframe['dwt_model_diff'] < 2.0 * self.exit_short_dwt_diff.value
            # )

            # # conditions.append(trend_cond)
            # short_conditions.append(short_dwt_cond)
            # short_conditions.append(short_spike_cond)

            # # set exit tags
            # dataframe.loc[short_dwt_cond, 'exit_tag'] += 'short_dwt_exit '

            if short_conditions:
                dataframe.loc[reduce(lambda x, y: x & y, short_conditions), 'exit_long'] = 0

        else:
            dataframe.loc[(dataframe['close'].notnull()), 'exit_long'] = 0

        return dataframe


    ############################################################################

    def isBull(self, pair):
        return re.search(".*(BULL|UP|[235]L)", pair)

    def isBear(self, pair):
        return re.search(".*(BEAR|DOWN|[235]S)", pair)

    def getInformative(self, pair) -> str:
        inf_pair = ""
        if self.isBull(pair):
            inf_pair = re.sub('(BULL|UP|[235]L)', '', pair)
        elif self.isBear(pair):
            inf_pair = re.sub('(BEAR|DOWN|[235]S)', '', pair)

        # print(pair, " -> ", inf_pair)
        return inf_pair

    ############################################################################

# Indidcatrs that work VIDYA,
# NOTE: VERY IMPORTANT FOR LATER ISSUES:
# IF YOU ARE GETTING ERROR OF Expected a 1D array, got an array with shape (1173, 2)
# THIS MEANS IN pd.concat THAT YOU NEED TO REFER TO THE VARIABLES EACH THEY ARE NOT DEFINED ALONE AND YOU HAVE TO GO TO EACH ONE LIKE BBANDS
# EXAMPLE:

# CALL WHATVR U WANT = CLASS OF FILE, FUNCTION, PARAMETERS
        # bollinger1 = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        # dataframe['bb_lowerband1'] = bollinger1['lower']
        # dataframe['bb_middleband1'] = bollinger1['mid']
        # dataframe['bb_upperband1'] = bollinger1['upper']

# MIMIC HERE THE SAME REFER TO THE FILE OF CT FOR THE PARAM NAMES.
        # BBNADS = TA2.BBANDS(dataframe, period=20, std_multiplier=2)
        # dataframe['bbands_lower'] = BBNADS['BB_LOWER']
        # dataframe['bbands_mid'] = BBNADS['BB_MIDDLE']
        # dataframe['bbands_upper'] = BBNADS['BB_UPPER']
        # dataframe['fish'] = TA2.FISH(dataframe)


    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime, current_rate: float,
                        current_profit: float, **kwargs) -> float:

        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()
        trade_dur = int((current_time.timestamp() - trade.open_date_utc.timestamp()) // 60)
        in_trend = self.custom_trade_info[trade.pair]['had-trend']

        # NOTE For stoploss we have to check indicator implementation and time implementation checks,
        # If time was surpassed and indicator is false then sell.

        # Step 1: If we surpass checkpoint one of stoploss amount like 3%,
        # Step 2: Then we check the SAR_AVG above or below,
        # Step 3: The ROC is negative over a period of time,
        # Step 4: A certain amount of time was passed like over 8hrs,
        # Step 5: Finally Max_stoploss reached and just sell.

        # ADD PROTECTION PARAMETERS OR FIGURE OUT HOW TO LIMIT BUYS ON PAIRS.

        # First Check on stoploss surpassed or not.
        # if current_profit <  self.cstop_max_stoploss.value:
        #     return 0.01 
            

        # Time based Stoploss
        # if current_time - timedelta(minutes=self.cstop_time1.value) > trade.open_date_utc:
        #     return -0.05
        # elif current_time - timedelta(minutes=self.cstop_time2.value) > trade.open_date_utc:
        #     return -0.10

        # Determine how we exit when we are in a loss
        # if current_profit < self.cstop_loss_threshold.value:
        #     if self.cstop_bail_how.value == 'SAR_AVG' or self.cstop_bail_how.value == 'any':
        #         if last_candle['SAR_AVG'] >= last_candle['close']:
        #             if current_time - timedelta(minutes=self.cstop_time1.value) > trade.open_date_utc:
        #                 return 0.01
                # Dynamic bailout based on rate of change
                # Edit this to perfect accuracy.
            # if self.cstop_bail_how.value == 'ROC' or self.cstop_bail_how.value == 'any':
            #     if last_candle['ROC'] == self.cstop_bail_roc.value:
            #         return 0.01
            #     if last_candle['SAR-Down1h'] == self.SAR_stoploss_dn.value:
            #         if last_candle['WT-dn-trend2'] == self.WT_dn_value.value:
            #             # if last_candle['RMIx'] <= self.RMIx_stoploss.value:
            #             return 0.01

            # return 0.01        
        # if self.cstop_bail_how.value == 'time' or self.cstop_bail_how.value == 'any':
        #     # Dynamic bailout based on time, unless time_trend is true and there is a potential reversal
        #     if trade_dur > self.cstop_bail_time.value:
        #         if self.cstop_bail_time_trend.value == True and in_trend == True:
        #             return 1
        #         else:
        #             return 0.01
        return 1
