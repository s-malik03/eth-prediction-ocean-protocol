import datetime
import numpy as np
from pathlib import Path
import os
import time
import matplotlib
import matplotlib.pyplot as plt
from ocean_lib.example_config import ExampleConfig
from ocean_lib.ocean.ocean import Ocean
from ocean_lib.web3_internal.wallet import Wallet
import pandas as pd
import statsmodels.api as sm
import ccxt


def create_ocean_instance() -> Ocean:
  config = ExampleConfig.get_config(
    "https://polygon-rpc.com")  # points to Polygon mainnet
  config["BLOCK_CONFIRMATIONS"] = 1  #faster
  ocean = Ocean(config)
  return ocean


def create_alice_wallet(ocean: Ocean) -> Wallet:
  config = ocean.config_dict
  alice_private_key = ""
  alice_wallet = Wallet(ocean.web3, alice_private_key,
                        config["BLOCK_CONFIRMATIONS"],
                        config["TRANSACTION_TIMEOUT"])
  bal = ocean.from_wei(alice_wallet.web3.eth.get_balance(alice_wallet.address))
  print(f"alice_wallet.address={alice_wallet.address}. bal={bal}")
  # Fix f-string is missing placeholders
  #assert bal > 0, f"Alice needs MATIC"
  assert bal > 0, f"Alice needs MATIC"
  return alice_wallet


#helper functions: time
def to_unixtime(dt: datetime.datetime):
  return time.mktime(dt.timetuple())


def to_unixtimes(dts: list) -> list:
  return [to_unixtime(dt) for dt in dts]


def to_datetime(ut) -> datetime.datetime:
  return datetime.datetime.utcfromtimestamp(ut)


def to_datetimes(uts: list) -> list:
  return [to_datetime(ut) for ut in uts]


def round_to_nearest_hour(dt: datetime.datetime) -> datetime.datetime:
  return (dt.replace(second=0, microsecond=0, minute=0, hour=dt.hour) +
          datetime.timedelta(hours=dt.minute // 30))


def pretty_time(dt: datetime.datetime) -> str:
  return dt.strftime('%Y/%m/%d, %H:%M:%S')


def print_datetime_info(descr: str, uts: list):
  dts = to_datetimes(uts)
  print(descr + ":")
  print(f"  starts on: {pretty_time(dts[0])}")
  print(f"    ends on: {pretty_time(dts[-1])}")
  print(f"  {len(dts)} datapoints")
  print(f"  time interval between datapoints: {(dts[1]-dts[0])}")


def target_12h_unixtimes(start_dt: datetime.datetime) -> list:
  target_dts = [start_dt + datetime.timedelta(hours=h) for h in range(12)]
  target_uts = to_unixtimes(target_dts)
  return target_uts


#helper-functions: higher level
def load_from_ohlc_data(file_name: str) -> tuple:
  """Returns (list_of_unixtimes, list_of_close_prices)"""
  with open(file_name, "r") as file:
    data_str = file.read().rstrip().replace('"', '')
  x = eval(data_str)  #list of lists
  uts = [xi[0] / 1000 for xi in x]
  vals = [xi[4] for xi in x]
  return (uts, vals)


def filter_to_target_uts(target_uts: list, unfiltered_uts: list,
                         unfiltered_vals: list) -> list:
  """Return filtered_vals -- values at at the target timestamps"""
  filtered_vals = [None] * len(target_uts)
  for i, target_ut in enumerate(target_uts):
    time_diffs = np.abs(np.asarray(unfiltered_uts) - target_ut)
    tol_s = 1  #should always align within e.g. 1 second
    target_ut_s = pretty_time(to_datetime(target_ut))
    assert min(time_diffs) <= tol_s, \
        f"Unfiltered times is missing target time: {target_ut_s}"
    j = np.argmin(time_diffs)
    filtered_vals[i] = unfiltered_vals[j]
  return filtered_vals


#helpers: save/load list
def save_list(list_: list, file_name: str):
  """Save a file shaped: [1.2, 3.4, 5.6, ..]"""
  p = Path(file_name)
  p.write_text(str(list_))


def load_list(file_name: str) -> list:
  """Load from a file shaped: [1.2, 3.4, 5.6, ..]"""
  p = Path(file_name)
  s = p.read_text()
  list_ = eval(s)
  return list_


#helpers: prediction performance
def calc_nmse(y, yhat) -> float:
  assert len(y) == len(yhat)
  mse_xy = np.sum(np.square(np.asarray(y) - np.asarray(yhat)))
  mse_x = np.sum(np.square(np.asarray(y)))
  nmse = mse_xy / mse_x
  return nmse


def plot_prices(cex_vals, pred_vals):
  matplotlib.rcParams.update({'font.size': 22})
  x = [h for h in range(0, 12)]
  assert len(x) == len(cex_vals) == len(pred_vals)
  fig, ax = plt.subplots()
  ax.plot(x, cex_vals, '--', label="CEX values")
  ax.plot(x, pred_vals, '-', label="Pred. values")
  ax.legend(loc='lower right')
  plt.ylabel("ETH price")
  plt.xlabel("Hour")
  fig.set_size_inches(18, 18)
  plt.xticks(x)
  #add
  plt.savefig('my_plot.png')
  plt.show()



# Get 10000 Bittsamp data points and reverse Ordeer

# 1. Dataset manupilation
df = pd.read_csv("Bitstamp_ETHUSD_1h.csv")
df = df[:10000]
df = df.iloc[::-1]
df2 = df.set_index('date')

# 2. Use the last 12 hours of testing set, all the previous data is used as training
train_data = list(df2[0:-12]['close'])
test_data = list(df2[-12:]['close'])
n_test_obser = len(test_data)
pred_vals = []

# 3. Modeling
for i in range(n_test_obser):
  #model = sm.tsa.arima.ARIMA(train_data, order=(4, 1, 0))
  model = sm.tsa.arima.ARIMA(train_data, order=(5,1,0)) 
  model_fit = model.fit()
  output = model_fit.forecast()
  pred_vals.append(output[0])
  actual_test_valute = test_data[i]
  train_data.append(actual_test_valute)
print(pred_vals)

# 4. Get the time range we want to test for
start_dt = datetime.datetime.now() - datetime.timedelta(
  hours=24)  #must be >= 12h ago
start_dt = round_to_nearest_hour(start_dt)  # so that times line up
target_uts = target_12h_unixtimes(start_dt)
print_datetime_info("target times", target_uts)

# 5. get the actual ETH values at that time
allcex_x = ccxt.binance().fetch_ohlcv('ETH/USDT', '1h')
allcex_uts = [xi[0] / 1000 for xi in allcex_x]
allcex_vals = [xi[4] for xi in allcex_x]
print_datetime_info("allcex times", allcex_uts)
cex_vals = filter_to_target_uts(target_uts, allcex_uts, allcex_vals)

# 6. now, we have predicted and actual values. Let's find error, and plot!
nmse = calc_nmse(cex_vals, pred_vals)
mse = np.square(np.subtract(cex_vals,pred_vals)).mean()
print(f"NMSE = {nmse}")
print(f"mse = {mse}")
plot_prices(cex_vals, pred_vals)

# plot will be save to my_plot.png

#Evaluation
file_name = "spreadspoke_scores"
save_list(pred_vals, file_name)




